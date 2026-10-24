#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
DWT = ROOT / "external" / "DWTFreqNet"
sys.path.insert(0, str(DWT))
sys.path.insert(0, str(ROOT / "scripts"))

from model.Config import get_DWTFreqNet_config  # noqa: E402
from model.DWTFreqNet import DWTFreqNet  # noqa: E402
from utils import Normalized, get_img_norm_cfg, random_crop, weights_init_kaiming  # noqa: E402
from eval_component_fusion_learner import metric_from_maps  # noqa: E402


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


class DWTTrainSet(Dataset):
    def __init__(self, dataset_dir: Path, dataset_name: str, patch_size: int, max_samples: int = 0) -> None:
        self.root = dataset_dir / dataset_name
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.patch_size = patch_size
        self.names = (self.root / "img_idx" / f"train_{dataset_name}.txt").read_text(encoding="utf-8").splitlines()
        if max_samples > 0:
            self.names = self.names[:max_samples]
        self.norm = get_img_norm_cfg(dataset_name, str(dataset_dir))

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        image_pil, mask_pil = open_pair(self.root, self.names[idx])
        image = Normalized(np.asarray(image_pil, dtype=np.float32), self.norm)
        mask = np.asarray(mask_pil, dtype=np.float32) / 255.0
        if mask.ndim > 2:
            mask = mask[:, :, 0]
        image, mask = random_crop(image, mask, self.patch_size, pos_prob=0.5)
        if random.random() < 0.5:
            image = image[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
        if random.random() < 0.5:
            image = image[::-1, :].copy()
            mask = mask[::-1, :].copy()
        return (
            torch.from_numpy(np.ascontiguousarray(image[None, :, :])).float(),
            torch.from_numpy(np.ascontiguousarray(mask[None, :, :])).float(),
        )


class DWTTestSet(Dataset):
    def __init__(self, dataset_dir: Path, train_dataset_name: str, dataset_name: str, max_samples: int = 0) -> None:
        self.root = dataset_dir / dataset_name
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.names = (self.root / "img_idx" / f"test_{dataset_name}.txt").read_text(encoding="utf-8").splitlines()
        if max_samples > 0:
            self.names = self.names[:max_samples]
        self.norm = get_img_norm_cfg(train_dataset_name, str(dataset_dir))

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        image_pil, mask_pil = open_pair(self.root, self.names[idx])
        image = Normalized(np.asarray(image_pil, dtype=np.float32), self.norm)
        mask = np.asarray(mask_pil, dtype=np.float32) / 255.0
        if mask.ndim > 2:
            mask = mask[:, :, 0]
        height, width = image.shape
        pad_h = (32 - height % 32) % 32
        pad_w = (32 - width % 32) % 32
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="constant")
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
        return (
            torch.from_numpy(np.ascontiguousarray(image[None, :, :])).float(),
            torch.from_numpy(np.ascontiguousarray(mask[None, :, :])).float(),
            [height, width],
            self.names[idx],
        )


def dwt_loss(preds, masks: torch.Tensor) -> torch.Tensor:
    if isinstance(preds, tuple):
        return sum(F.binary_cross_entropy(pred, masks) for pred in preds) / float(len(preds))
    if isinstance(preds, list):
        return sum(F.binary_cross_entropy(pred, masks) for pred in preds) / float(len(preds))
    return F.binary_cross_entropy(preds, masks)


@torch.no_grad()
def evaluate(model: DWTFreqNet, dataset_dir: Path, norm_dataset: str, dataset_name: str, device: torch.device, threshold: float, max_samples: int) -> dict[str, float]:
    old_mode = model.mode
    model.mode = "test"
    model.eval()
    dataset = DWTTestSet(dataset_dir, norm_dataset, dataset_name, max_samples=max_samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    samples = []
    maps = []
    for image, mask, size, name in loader:
        image = image.to(device)
        pred = model(image)
        height, width = int(size[0]), int(size[1])
        prob = pred[0, 0, :height, :width].detach().cpu().numpy().astype(np.float32)
        gt = mask[0, 0, :height, :width].numpy().astype(np.float32)
        samples.append({"name": str(name[0]), "mask": gt, "size": [height, width]})
        maps.append(prob)
    model.mode = old_mode
    return metric_from_maps(samples, maps, threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DWTFreqNet on full multi-dataset IRSTD pool.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--train_datasets", default="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3")
    parser.add_argument("--eval_datasets", default="IRSTD-1K,NUAA-SIRST")
    parser.add_argument("--norm_dataset", default="NUAA-SIRST")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eta_min", type=float, default=1e-5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--eval_max_samples", type=int, default=0)
    parser.add_argument("--max_train_samples_per_dataset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default=str(ROOT / "runs" / "dwtfreq_multidata"))
    args = parser.parse_args()

    seed_everything(args.seed)
    dataset_dir = Path(args.dataset_dir)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    train_sets = [
        DWTTrainSet(dataset_dir, name, args.patch_size, args.max_train_samples_per_dataset)
        for name in parse_csv(args.train_datasets)
    ]
    loader = DataLoader(
        ConcatDataset(train_sets),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    model = DWTFreqNet(get_DWTFreqNet_config(), mode="train", deepsuper=True).to(device)
    model.apply(weights_init_kaiming)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.eta_min)
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best = None
    with (run_dir / "train.log").open("w", encoding="utf-8") as log:
        log.write(json.dumps(vars(args), indent=2) + "\n")
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            model.train()
            model.mode = "train"
            losses = []
            for images, masks in loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                preds = model(images)
                loss = dwt_loss(preds, masks)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            scheduler.step()
            row = {
                "epoch": epoch,
                "loss": float(np.mean(losses)) if losses else 0.0,
                "lr": float(scheduler.get_last_lr()[0]),
                "seconds": time.time() - start,
            }
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
