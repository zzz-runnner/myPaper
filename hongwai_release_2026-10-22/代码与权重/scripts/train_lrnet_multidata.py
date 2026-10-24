#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
LRNET = ROOT / "external" / "LR-Net"
sys.path.insert(0, str(LRNET / "model"))
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))

from LRNet import LRNet  # noqa: E402
from metrics import PD_FA, mIoU  # noqa: E402
from utils import Normalized, PadImg, get_img_norm_cfg, random_crop  # noqa: E402


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def open_pair(root: Path, name: str) -> tuple[Image.Image, Image.Image]:
    for ext in (".png", ".bmp", ".jpg"):
        image = root / "images" / f"{name}{ext}"
        mask = root / "masks" / f"{name}{ext}"
        if image.exists() and mask.exists():
            return Image.open(image).convert("I"), Image.open(mask)
    raise FileNotFoundError(name)


class LRTrainSet(Dataset):
    def __init__(self, dataset_dir: Path, dataset_name: str, patch_size: int, pos_prob: float, max_samples: int = 0) -> None:
        self.root = dataset_dir / dataset_name
        self.dataset_name = dataset_name
        self.patch_size = patch_size
        self.pos_prob = pos_prob
        self.norm = get_img_norm_cfg(dataset_name, str(dataset_dir))
        self.names = (self.root / "img_idx" / f"train_{dataset_name}.txt").read_text(encoding="utf-8").splitlines()
        if max_samples > 0:
            self.names = self.names[:max_samples]

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        image_pil, mask_pil = open_pair(self.root, self.names[idx])
        image = Normalized(np.asarray(image_pil, dtype=np.float32), self.norm)
        mask = np.asarray(mask_pil, dtype=np.float32) / 255.0
        if mask.ndim > 2:
            mask = mask[:, :, 0]
        image, mask = random_crop(image, mask, self.patch_size, pos_prob=self.pos_prob)
        if random.random() < 0.5:
            image = image[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
        if random.random() < 0.5:
            image = image[::-1, :].copy()
            mask = mask[::-1, :].copy()
        image = torch.from_numpy(np.ascontiguousarray(image)).float().repeat(3, 1, 1)
        mask = torch.from_numpy(np.ascontiguousarray(mask[None, :, :])).float()
        return image, mask


class LRTestSet(Dataset):
    def __init__(self, dataset_dir: Path, norm_dataset: str, dataset_name: str, max_samples: int = 0) -> None:
        self.root = dataset_dir / dataset_name
        self.dataset_name = dataset_name
        self.norm = get_img_norm_cfg(norm_dataset, str(dataset_dir))
        self.names = (self.root / "img_idx" / f"test_{dataset_name}.txt").read_text(encoding="utf-8").splitlines()
        if max_samples > 0:
            self.names = self.names[:max_samples]

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        image_pil, mask_pil = open_pair(self.root, self.names[idx])
        image = Normalized(np.asarray(image_pil, dtype=np.float32), self.norm)
        mask = np.asarray(mask_pil, dtype=np.float32) / 255.0
        if mask.ndim > 2:
            mask = mask[:, :, 0]
        height, width = image.shape
        image = PadImg(image)
        mask = PadImg(mask)
        return (
            torch.from_numpy(np.ascontiguousarray(image)).float().repeat(3, 1, 1),
            torch.from_numpy(np.ascontiguousarray(mask[None, :, :])).float(),
            [height, width],
            self.names[idx],
        )


def soft_iou_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target - pred * target).sum(dim=(1, 2, 3))
    return 1.0 - ((inter + 1.0) / (union + 1.0)).mean()


def loss_fn(logits: torch.Tensor, target: torch.Tensor, pos_weight: float) -> torch.Tensor:
    pw = torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
    return bce + 0.65 * soft_iou_loss(logits, target)


def safe_get_pdfa(metric: PD_FA) -> tuple[float, float]:
    pd = float(metric.PD / metric.target) if metric.target else 0.0
    fa = float(metric.dismatch_pixel / metric.all_pixel) if metric.all_pixel else 0.0
    return pd, fa


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataset_dir: Path, norm_dataset: str, dataset_name: str, device: torch.device, threshold: float, max_samples: int) -> dict[str, float]:
    loader = DataLoader(LRTestSet(dataset_dir, norm_dataset, dataset_name, max_samples=max_samples), batch_size=1, shuffle=False, num_workers=1)
    model.eval()
    miou = mIoU()
    pdfa = PD_FA()
    for image, mask, size, _name in loader:
        image = image.to(device, non_blocking=True)
        logits = model(image)
        prob = torch.sigmoid(logits)
        h, w = int(size[0]), int(size[1])
        prob = prob[:, :, :h, :w].cpu()
        mask = mask[:, :, :h, :w]
        miou.update(prob > threshold, mask)
        pdfa.update(prob[0, 0] > threshold, mask[0, 0], [h, w])
    pix_acc, score_miou = miou.get()
    pd, fa = safe_get_pdfa(pdfa)
    return {"pixAcc": float(pix_acc), "mIoU": float(score_miou), "Pd": pd, "Fa": fa}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LR-Net on the local full multi-dataset IRSTD pool.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--train_datasets", default="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3,SIRST-UAVB")
    parser.add_argument("--eval_datasets", default="IRSTD-1K,NUAA-SIRST")
    parser.add_argument("--norm_dataset", default="NUAA-SIRST")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--pos_weight", type=float, default=60.0)
    parser.add_argument("--pos_prob", type=float, default=0.65)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--eval_max_samples", type=int, default=0)
    parser.add_argument("--max_train_samples_per_dataset", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    dataset_dir = Path(args.dataset_dir)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    train_sets = [
        LRTrainSet(dataset_dir, name, args.patch_size, args.pos_prob, args.max_train_samples_per_dataset)
        for name in parse_csv(args.train_datasets)
    ]
    loader = DataLoader(
        ConcatDataset(train_sets),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=3,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    model = LRNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    best = None
    history = []
    with (run_dir / "train.log").open("w", encoding="utf-8") as log:
        log.write(json.dumps(vars(args), indent=2) + "\n")
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            model.train()
            losses = []
            for images, masks in loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                logits = model(images)
                loss = loss_fn(logits, masks, args.pos_weight)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            scheduler.step()
            row = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else 0.0, "seconds": time.time() - start}
            if epoch % args.eval_interval == 0 or epoch == args.epochs:
                evals = {
                    name: evaluate(model, dataset_dir, args.norm_dataset, name, device, args.threshold, args.eval_max_samples)
                    for name in parse_csv(args.eval_datasets)
                }
                row["eval"] = evals
                mean_miou = float(np.mean([item["mIoU"] for item in evals.values()]))
                if best is None or mean_miou > best["mean_mIoU"]:
                    best = {"epoch": epoch, "mean_mIoU": mean_miou, "eval": evals}
                    torch.save({"epoch": epoch, "state_dict": model.state_dict(), "best": best}, run_dir / "best.pth.tar")
            history.append(row)
            line = json.dumps(row, sort_keys=True)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
    (run_dir / "summary.json").write_text(json.dumps({"args": vars(args), "best": best, "history": history}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
