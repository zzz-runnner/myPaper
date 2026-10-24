#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
BASIC = ROOT / "external" / "BasicIRSTD"
sys.path.insert(0, str(BASIC))
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))

from metrics import PD_FA, mIoU  # noqa: E402
from net import Net  # noqa: E402
from utils import Normalized, PadImg, get_img_norm_cfg, random_crop, seed_pytorch  # noqa: E402


class MultiTrainSet(Dataset):
    def __init__(self, dataset_dir: Path, dataset_name: str, patch_size: int) -> None:
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.root = dataset_dir / dataset_name
        self.patch_size = patch_size
        self.norm = get_img_norm_cfg(dataset_name, str(dataset_dir))
        self.names = (self.root / "img_idx" / f"train_{dataset_name}.txt").read_text(encoding="utf-8").splitlines()

    def __len__(self) -> int:
        return len(self.names)

    def _open(self, name: str):
        for ext in (".png", ".bmp", ".jpg"):
            image_path = self.root / "images" / f"{name}{ext}"
            mask_path = self.root / "masks" / f"{name}{ext}"
            if image_path.exists() and mask_path.exists():
                return Image.open(image_path).convert("I"), Image.open(mask_path)
        raise FileNotFoundError(f"{self.dataset_name}/{name}")

    def __getitem__(self, idx: int):
        image_pil, mask_pil = self._open(self.names[idx])
        image = Normalized(np.array(image_pil, dtype=np.float32), self.norm)
        mask = np.array(mask_pil, dtype=np.float32) / 255.0
        if mask.ndim > 2:
            mask = mask[:, :, 0]
        image, mask = random_crop(image, mask, self.patch_size, pos_prob=0.5)
        if np.random.rand() < 0.5:
            image = image[::-1, :]
            mask = mask[::-1, :]
        if np.random.rand() < 0.5:
            image = image[:, ::-1]
            mask = mask[:, ::-1]
        if np.random.rand() < 0.5:
            image = image.transpose(1, 0)
            mask = mask.transpose(1, 0)
        image = torch.from_numpy(np.ascontiguousarray(image[None, :, :])).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask[None, :, :])).float()
        return image, mask


class TestSet(Dataset):
    def __init__(self, dataset_dir: Path, dataset_name: str, norm_dataset: str) -> None:
        self.dataset_name = dataset_name
        self.root = dataset_dir / dataset_name
        self.norm = get_img_norm_cfg(norm_dataset, str(dataset_dir))
        self.names = (self.root / "img_idx" / f"test_{dataset_name}.txt").read_text(encoding="utf-8").splitlines()

    def __len__(self) -> int:
        return len(self.names)

    def _open(self, name: str):
        for ext in (".png", ".bmp", ".jpg"):
            image_path = self.root / "images" / f"{name}{ext}"
            mask_path = self.root / "masks" / f"{name}{ext}"
            if image_path.exists() and mask_path.exists():
                return Image.open(image_path).convert("I"), Image.open(mask_path)
        raise FileNotFoundError(f"{self.dataset_name}/{name}")

    def __getitem__(self, idx: int):
        image_pil, mask_pil = self._open(self.names[idx])
        image = Normalized(np.array(image_pil, dtype=np.float32), self.norm)
        mask = np.array(mask_pil, dtype=np.float32) / 255.0
        if mask.ndim > 2:
            mask = mask[:, :, 0]
        height, width = image.shape
        image = PadImg(image)
        mask = PadImg(mask)
        return (
            torch.from_numpy(np.ascontiguousarray(image[None, :, :])).float(),
            torch.from_numpy(np.ascontiguousarray(mask[None, :, :])).float(),
            [height, width],
            self.names[idx],
        )


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def forward_last(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    pred = model(image)
    if isinstance(pred, (tuple, list)):
        pred = pred[-1]
    return pred


def safe_get_pdfa(metric: PD_FA) -> tuple[float, float]:
    pd = float(metric.PD / metric.target) if metric.target else 0.0
    fa = float(metric.dismatch_pixel / metric.all_pixel) if metric.all_pixel else 0.0
    return pd, fa


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataset_dir: Path, dataset_name: str, norm_dataset: str, device: torch.device, threshold: float):
    loader = DataLoader(TestSet(dataset_dir, dataset_name, norm_dataset), batch_size=1, shuffle=False, num_workers=1)
    model.eval()
    miou = mIoU()
    pdfa = PD_FA()
    for image, mask, size, _name in loader:
        image = image.to(device)
        pred = forward_last(model, image)
        h, w = int(size[0]), int(size[1])
        pred = pred[:, :, :h, :w]
        mask = mask[:, :, :h, :w]
        miou.update((pred > threshold).cpu(), mask)
        pdfa.update((pred[0, 0] > threshold).cpu(), mask[0, 0], [h, w])
    pix_acc, score_miou = miou.get()
    pd, fa = safe_get_pdfa(pdfa)
    return {"pixAcc": float(pix_acc), "mIoU": float(score_miou), "Pd": float(pd), "Fa": float(fa)}


def save_checkpoint(path: Path, model: torch.nn.Module, epoch: int, metrics: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "state_dict": model.state_dict(), "metrics": metrics}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BasicIRSTD models on the full multi-dataset pool.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--train_datasets", default="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3")
    parser.add_argument("--eval_datasets", default="IRSTD-1K,NUAA-SIRST")
    parser.add_argument("--norm_dataset", default="NUAA-SIRST")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default=str(ROOT / "runs" / "basicirstd_multidata"))
    args = parser.parse_args()

    seed_pytorch(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dataset_dir = Path(args.dataset_dir)
    train_sets = [MultiTrainSet(dataset_dir, name, args.patch_size) for name in parse_csv(args.train_datasets)]
    train_loader = DataLoader(
        ConcatDataset(train_sets),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )

    model = Net(args.model_name, mode="train").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[max(args.epochs // 2, 1), max(args.epochs * 3 // 4, 1)], gamma=0.2)

    run_dir = Path(args.output_dir) / f"{args.model_name}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    history = []
    best_mean = -1.0
    best_metrics: dict[str, object] | None = None
    with log_path.open("w", encoding="utf-8") as log:
        log.write(json.dumps(vars(args), indent=2) + "\n")
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            started = time.time()
            for image, mask in train_loader:
                image = image.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                pred = forward_last(model, image)
                loss = model.loss(pred, mask)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            scheduler.step()
            row = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else 0.0, "seconds": time.time() - started}
            if epoch % args.eval_interval == 0 or epoch == args.epochs:
                evals = {
                    name: evaluate(model, dataset_dir, name, args.norm_dataset, device, args.threshold)
                    for name in parse_csv(args.eval_datasets)
                }
                row["eval"] = evals
                mean_miou = float(np.mean([item["mIoU"] for item in evals.values()]))
                if mean_miou > best_mean:
                    best_mean = mean_miou
                    best_metrics = {"epoch": epoch, "mean_mIoU": mean_miou, "eval": evals}
                    save_checkpoint(run_dir / "best_mean_mIoU.pth.tar", model, epoch, best_metrics)
            history.append(row)
            line = json.dumps(row, sort_keys=True)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
    (run_dir / "summary.json").write_text(
        json.dumps({"args": vars(args), "best": best_metrics, "history": history}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
