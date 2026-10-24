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
from torch.utils.data import ConcatDataset, DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_teacher_anchor_refiner_from_cache import (  # noqa: E402
    AnchorResidualUNet,
    dice_loss,
    image_feature_stack,
    logit_np,
    open_image,
    parse_csv,
    teacher_maps,
)
from eval_cached_prediction_fusion import load_cache  # noqa: E402
from eval_component_fusion_learner import metric_from_maps  # noqa: E402


def boundary_map(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    src = (mask > 0.5).astype(np.uint8)
    if src.max() == 0:
        return np.zeros_like(mask, dtype=np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    dil = cv2.dilate(src, kernel)
    ero = cv2.erode(src, kernel)
    return ((dil - ero) > 0).astype(np.float32)


def teacher_boundary_features(teacher: np.ndarray) -> np.ndarray:
    prob = teacher.astype(np.float32)
    uncertainty = 4.0 * prob * (1.0 - prob)
    gx = cv2.Sobel(prob, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(prob, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(gx * gx + gy * gy)
    edge = edge / (float(edge.max()) + 1e-6)
    seed = (prob > 0.25).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    band = cv2.dilate(seed, kernel).astype(np.float32) - cv2.erode(seed, kernel).astype(np.float32)
    return np.stack([uncertainty, edge, np.clip(band, 0.0, 1.0)], axis=0).astype(np.float32)


class UBTrainSet(Dataset):
    def __init__(self, dataset_dir: Path, dataset_name: str, cache_path: Path, teacher_mode: str, patch_size: int, max_samples: int, seed: int) -> None:
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.samples, self.names = load_cache(cache_path)
        if max_samples > 0:
            self.samples = self.samples[:max_samples]
        self.teacher_mode = teacher_mode
        self.patch_size = patch_size
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        h, w = [int(x) for x in sample["size"]]
        image = open_image(self.dataset_dir, self.dataset_name, str(sample["name"]))[:h, :w]
        img_feat = image_feature_stack(image)
        teacher, t_feat = teacher_maps(sample["preds"], self.names, self.teacher_mode)
        teacher = teacher[:h, :w]
        gt = sample["mask"].astype(np.float32)[:h, :w]
        ub_feat = teacher_boundary_features(teacher)
        gt_boundary = boundary_map(gt, radius=2)
        feat = np.concatenate([img_feat, t_feat[:, :h, :w], ub_feat[:, :h, :w]], axis=0)
        tlogit = logit_np(teacher)
        if h < self.patch_size or w < self.patch_size:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            feat = np.pad(feat, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
            gt = np.pad(gt, ((0, pad_h), (0, pad_w)), mode="constant")
            gt_boundary = np.pad(gt_boundary, ((0, pad_h), (0, pad_w)), mode="constant")
            tlogit = np.pad(tlogit, ((0, pad_h), (0, pad_w)), mode="constant")
            h, w = gt.shape
        for _ in range(40):
            y0 = self.rng.randint(0, h - self.patch_size)
            x0 = self.rng.randint(0, w - self.patch_size)
            gt_patch = gt[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
            if self.rng.random() > 0.35 or gt_patch.sum() > 0:
                break
        feat_patch = feat[:, y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
        gt_patch = gt[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
        boundary_patch = gt_boundary[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
        tlogit_patch = tlogit[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
        if self.rng.random() < 0.5:
            feat_patch = feat_patch[:, :, ::-1].copy()
            gt_patch = gt_patch[:, ::-1].copy()
            boundary_patch = boundary_patch[:, ::-1].copy()
            tlogit_patch = tlogit_patch[:, ::-1].copy()
        if self.rng.random() < 0.5:
            feat_patch = feat_patch[:, ::-1, :].copy()
            gt_patch = gt_patch[::-1, :].copy()
            boundary_patch = boundary_patch[::-1, :].copy()
            tlogit_patch = tlogit_patch[::-1, :].copy()
        return (
            torch.from_numpy(np.ascontiguousarray(feat_patch)).float(),
            torch.from_numpy(np.ascontiguousarray(gt_patch[None])).float(),
            torch.from_numpy(np.ascontiguousarray(tlogit_patch[None])).float(),
            torch.from_numpy(np.ascontiguousarray(boundary_patch[None])).float(),
        )


class UBTestSet(Dataset):
    def __init__(self, dataset_dir: Path, dataset_name: str, cache_path: Path, teacher_mode: str, max_samples: int = 0) -> None:
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.samples, self.names = load_cache(cache_path)
        if max_samples > 0:
            self.samples = self.samples[:max_samples]
        self.teacher_mode = teacher_mode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        h, w = [int(x) for x in sample["size"]]
        image = open_image(self.dataset_dir, self.dataset_name, str(sample["name"]))[:h, :w]
        img_feat = image_feature_stack(image)
        teacher, t_feat = teacher_maps(sample["preds"], self.names, self.teacher_mode)
        teacher = teacher[:h, :w]
        ub_feat = teacher_boundary_features(teacher)
        feat = np.concatenate([img_feat, t_feat[:, :h, :w], ub_feat[:, :h, :w]], axis=0)
        tlogit = logit_np(teacher)
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h or pad_w:
            feat = np.pad(feat, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
            tlogit = np.pad(tlogit, ((0, pad_h), (0, pad_w)), mode="constant")
        return (
            torch.from_numpy(np.ascontiguousarray(feat)).float(),
            torch.from_numpy(np.ascontiguousarray(tlogit[None])).float(),
            sample["mask"].astype(np.float32)[:h, :w],
            [h, w],
            str(sample["name"]),
        )


def weighted_dice_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (weight * prob * target).sum(dim=(1, 2, 3))
    denom = (weight * prob).sum(dim=(1, 2, 3)) + (weight * target).sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an uncertainty/boundary weighted teacher-anchored residual refiner from cached predictions.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--train_cache_spec", action="append", required=True, help="DATASET|CACHE")
    parser.add_argument("--test_cache_spec", action="append", required=True, help="DATASET|CACHE")
    parser.add_argument("--teacher_mode", default="avg")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--residual_scale", type=float, default=0.45)
    parser.add_argument("--anchor_weight", type=float, default=0.02)
    parser.add_argument("--confident_anchor_weight", type=float, default=0.03)
    parser.add_argument("--pos_weight", type=float, default=25.0)
    parser.add_argument("--uncertainty_weight", type=float, default=1.0)
    parser.add_argument("--boundary_weight", type=float, default=2.0)
    parser.add_argument("--false_positive_weight", type=float, default=1.5)
    parser.add_argument("--thresholds", default="0.30,0.35,0.38,0.40,0.42,0.45,0.50,0.55,0.60")
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
        train_sets.append(UBTrainSet(dataset_dir, dataset_name, Path(cache), args.teacher_mode, args.patch_size, args.max_train_samples_per_cache, args.seed + len(train_sets)))
    test_sets = []
    for raw in args.test_cache_spec:
        dataset_name, cache = raw.split("|", 1)
        test_sets.append((dataset_name, UBTestSet(dataset_dir, dataset_name, Path(cache), args.teacher_mode, args.eval_max_samples)))
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    loader = DataLoader(ConcatDataset(train_sets), batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=device.type == "cuda")
    in_ch = train_sets[0][0][0].shape[0]
    model = AnchorResidualUNet(in_ch=in_ch, base=args.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)
    thresholds = [float(x) for x in parse_csv(args.thresholds)]
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    best = None
    history = []
    pos_weight = torch.tensor([args.pos_weight], dtype=torch.float32, device=device).view(1, 1, 1, 1)

    with (run_dir / "train.log").open("w", encoding="utf-8") as log:
        log.write(json.dumps(vars(args), indent=2) + "\n")
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            model.train()
            losses = []
            for feat, gt, tlogit, gt_boundary in loader:
                feat = feat.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)
                tlogit = tlogit.to(device, non_blocking=True)
                gt_boundary = gt_boundary.to(device, non_blocking=True)
                teacher_prob = torch.sigmoid(tlogit)
                uncertainty = 4.0 * teacher_prob * (1.0 - teacher_prob)
                teacher_fp = torch.relu(teacher_prob - 0.35) * (1.0 - gt)
                pixel_weight = 1.0 + args.uncertainty_weight * uncertainty + args.boundary_weight * gt_boundary + args.false_positive_weight * teacher_fp
                residual = args.residual_scale * torch.tanh(model(feat))
                logits = tlogit + residual
                bce = F.binary_cross_entropy_with_logits(logits, gt, weight=pixel_weight, pos_weight=pos_weight, reduction="mean")
                confident = 1.0 - uncertainty
                anchor = (residual.square() * (1.0 + args.confident_anchor_weight * confident)).mean()
                loss = bce + weighted_dice_loss(logits, gt, pixel_weight) + args.anchor_weight * anchor + 0.15 * dice_loss(logits, gt)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu()))
            sched.step()
            row = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else 0.0, "lr": float(sched.get_last_lr()[0]), "seconds": time.time() - start}
            if epoch % args.eval_interval == 0 or epoch == args.epochs:
                evals = {}
                model.eval()
                for name, ds in test_sets:
                    samples = []
                    maps = []
                    for feat, tlogit, mask, size, sample_name in DataLoader(ds, batch_size=1, shuffle=False, num_workers=1):
                        feat = feat.to(device)
                        tlogit = tlogit.to(device)
                        with torch.no_grad():
                            residual = args.residual_scale * torch.tanh(model(feat))
                            prob = torch.sigmoid(tlogit + residual)[0, 0].detach().cpu().numpy()
                        h, w = int(size[0]), int(size[1])
                        maps.append(prob[:h, :w].astype(np.float32))
                        samples.append({"name": str(sample_name[0]), "mask": mask.numpy()[0].astype(np.float32), "size": [h, w]})
                    rows = [{"threshold": th, **metric_from_maps(samples, maps, th)} for th in thresholds]
                    rows.sort(key=lambda r: r["mIoU"], reverse=True)
                    evals[name] = rows[0]
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
