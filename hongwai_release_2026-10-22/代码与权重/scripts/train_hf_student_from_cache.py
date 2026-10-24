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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))

from eval_cached_prediction_fusion import load_cache  # noqa: E402
from eval_component_fusion_learner import metric_from_maps  # noqa: E402


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def robust01(x: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(x, [low, high])
    if hi <= lo + 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def open_image(dataset_dir: Path, dataset_name: str, name: str) -> np.ndarray:
    root = dataset_dir / dataset_name
    for ext in (".png", ".bmp", ".jpg"):
        path = root / "images" / f"{name}{ext}"
        if path.exists():
            return np.asarray(Image.open(path).convert("I"), dtype=np.float32)
    raise FileNotFoundError(f"{dataset_name}/{name}")


def image_features(image: np.ndarray) -> np.ndarray:
    src = robust01(image)
    mean = cv2.blur(src, (9, 9))
    sq = cv2.blur(src * src, (9, 9))
    std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    z = robust01((src - mean) / (std + 1e-3), 50.0, 99.8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tophat = robust01(cv2.morphologyEx(src, cv2.MORPH_TOPHAT, kernel), 50.0, 99.8)
    blur1 = cv2.GaussianBlur(src, (0, 0), sigmaX=1.0, sigmaY=1.0)
    blur2 = cv2.GaussianBlur(src, (0, 0), sigmaX=2.2, sigmaY=2.2)
    dog = robust01(blur1 - blur2, 50.0, 99.8)
    log = robust01(cv2.Laplacian(blur1, cv2.CV_32F, ksize=3), 50.0, 99.8)
    return np.stack([src, z, tophat, dog, log], axis=0).astype(np.float32)


def pseudo_from_preds(preds: dict[str, np.ndarray], names: list[str], mode: str) -> np.ndarray:
    stack = np.stack([preds[name] for name in names], axis=0).astype(np.float32)
    if mode == "avg":
        return stack.mean(axis=0)
    if mode == "max":
        return stack.max(axis=0)
    if mode in names:
        return preds[mode].astype(np.float32)
    raise ValueError(mode)


class CacheTrainSet(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        cache_path: Path,
        dataset_name: str,
        pseudo_mode: str,
        patch_size: int,
        max_samples: int,
        seed: int,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.samples, self.names = load_cache(cache_path)
        if max_samples > 0:
            self.samples = self.samples[:max_samples]
        self.pseudo_mode = pseudo_mode
        self.patch_size = patch_size
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        h, w = [int(x) for x in sample["size"]]
        image = open_image(self.dataset_dir, self.dataset_name, str(sample["name"]))[:h, :w]
        feat = image_features(image)
        gt = sample["mask"].astype(np.float32)[:h, :w]
        pseudo = pseudo_from_preds(sample["preds"], self.names, self.pseudo_mode)[:h, :w]
        if h < self.patch_size or w < self.patch_size:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            feat = np.pad(feat, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
            gt = np.pad(gt, ((0, pad_h), (0, pad_w)), mode="constant")
            pseudo = np.pad(pseudo, ((0, pad_h), (0, pad_w)), mode="constant")
            h, w = gt.shape
        for _ in range(20):
            y0 = self.rng.randint(0, h - self.patch_size)
            x0 = self.rng.randint(0, w - self.patch_size)
            gt_patch = gt[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
            if self.rng.random() > 0.55 or gt_patch.sum() > 0:
                break
        feat_patch = feat[:, y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
        pseudo_patch = pseudo[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
        if self.rng.random() < 0.5:
            feat_patch = feat_patch[:, :, ::-1].copy()
            gt_patch = gt_patch[:, ::-1].copy()
            pseudo_patch = pseudo_patch[:, ::-1].copy()
        if self.rng.random() < 0.5:
            feat_patch = feat_patch[:, ::-1, :].copy()
            gt_patch = gt_patch[::-1, :].copy()
            pseudo_patch = pseudo_patch[::-1, :].copy()
        return (
            torch.from_numpy(np.ascontiguousarray(feat_patch)).float(),
            torch.from_numpy(np.ascontiguousarray(gt_patch[None, :, :])).float(),
            torch.from_numpy(np.ascontiguousarray(pseudo_patch[None, :, :])).float(),
        )


class CacheTestSet(Dataset):
    def __init__(self, dataset_dir: Path, cache_path: Path, dataset_name: str, max_samples: int = 0) -> None:
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.samples, self.names = load_cache(cache_path)
        if max_samples > 0:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        h, w = [int(x) for x in sample["size"]]
        image = open_image(self.dataset_dir, self.dataset_name, str(sample["name"]))[:h, :w]
        feat = image_features(image)
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h or pad_w:
            feat = np.pad(feat, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
        return (
            torch.from_numpy(np.ascontiguousarray(feat)).float(),
            sample["mask"].astype(np.float32)[:h, :w],
            [h, w],
            str(sample["name"]),
        )


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HFStudent(nn.Module):
    def __init__(self, in_ch: int = 5, base: int = 24) -> None:
        super().__init__()
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.mid = ConvBlock(base * 4, base * 4)
        self.d2 = ConvBlock(base * 6, base * 2)
        self.d1 = ConvBlock(base * 3, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        mid = self.mid(e3)
        d2 = F.interpolate(mid, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()


def loss_fn(logits: torch.Tensor, gt: torch.Tensor, pseudo: torch.Tensor, pseudo_weight: float) -> torch.Tensor:
    gt_loss = F.binary_cross_entropy_with_logits(logits, gt) + dice_loss(logits, gt)
    if pseudo_weight <= 0:
        return gt_loss
    soft = pseudo.clamp(1e-4, 1.0 - 1e-4)
    pseudo_loss = F.binary_cross_entropy_with_logits(logits, soft)
    return gt_loss + pseudo_weight * pseudo_loss


@torch.no_grad()
def evaluate(model: nn.Module, dataset: CacheTestSet, device: torch.device, thresholds: list[float]) -> list[dict[str, float]]:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    samples = []
    maps = []
    model.eval()
    for feat, mask, size, name in loader:
        feat = feat.to(device)
        prob = torch.sigmoid(model(feat))[0, 0].detach().cpu().numpy()
        h, w = int(size[0]), int(size[1])
        maps.append(prob[:h, :w].astype(np.float32))
        samples.append({"name": str(name[0]), "mask": mask.numpy()[0].astype(np.float32), "size": [h, w]})
    rows = [{"threshold": th, **metric_from_maps(samples, maps, th)} for th in thresholds]
    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a high-frequency image student from cached teacher pseudo labels.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--train_cache_spec", action="append", required=True, help="DATASET|CACHE")
    parser.add_argument("--test_cache_spec", action="append", required=True, help="DATASET|CACHE")
    parser.add_argument("--pseudo_mode", default="avg")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=24)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--pseudo_weight", type=float, default=0.5)
    parser.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60")
    parser.add_argument("--eval_interval", type=int, default=2)
    parser.add_argument("--max_train_samples_per_cache", type=int, default=0)
    parser.add_argument("--eval_max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dataset_dir = Path(args.dataset_dir)
    train_sets = []
    for raw in args.train_cache_spec:
        dataset_name, cache = raw.split("|", 1)
        train_sets.append(
            CacheTrainSet(
                dataset_dir,
                Path(cache),
                dataset_name,
                args.pseudo_mode,
                args.patch_size,
                args.max_train_samples_per_cache,
                args.seed + len(train_sets),
            )
        )
    test_sets = []
    for raw in args.test_cache_spec:
        dataset_name, cache = raw.split("|", 1)
        test_sets.append((dataset_name, CacheTestSet(dataset_dir, Path(cache), dataset_name, args.eval_max_samples)))
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    loader = DataLoader(
        ConcatDataset(train_sets),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    model = HFStudent(base=args.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)
    thresholds = [float(x) for x in parse_csv(args.thresholds)]
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
            for feat, gt, pseudo in loader:
                feat = feat.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)
                pseudo = pseudo.to(device, non_blocking=True)
                logits = model(feat)
                loss = loss_fn(logits, gt, pseudo, args.pseudo_weight)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu()))
            sched.step()
            row = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else 0.0, "lr": float(sched.get_last_lr()[0]), "seconds": time.time() - start}
            if epoch % args.eval_interval == 0 or epoch == args.epochs:
                evals = {name: evaluate(model, ds, device, thresholds)[0] for name, ds in test_sets}
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
