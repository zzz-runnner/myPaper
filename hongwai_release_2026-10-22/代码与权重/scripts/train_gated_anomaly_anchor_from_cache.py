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

from eval_cached_prediction_fusion import load_cache  # noqa: E402
from eval_component_fusion_learner import metric_from_maps  # noqa: E402
from train_teacher_anchor_refiner_from_cache import AnchorResidualUNet, dice_loss, logit_np, parse_csv, robust01, teacher_maps  # noqa: E402


def open_image(dataset_dir: Path, dataset_name: str, name: str) -> np.ndarray:
    root = dataset_dir / dataset_name
    for ext in (".png", ".bmp", ".jpg"):
        path = root / "images" / f"{name}{ext}"
        if path.exists():
            return np.asarray(Image.open(path).convert("I"), dtype=np.float32)
    raise FileNotFoundError(f"{dataset_name}/{name}")


def anomaly_feature_stack(image: np.ndarray) -> np.ndarray:
    src = robust01(image)
    feats = [src]
    for k in (5, 9, 17, 31):
        mean = cv2.blur(src, (k, k))
        sq = cv2.blur(src * src, (k, k))
        std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
        feats.append(robust01(src - mean, 50.0, 99.8))
        feats.append(robust01((src - mean) / (std + 1e-3), 50.0, 99.8))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        feats.append(robust01(cv2.morphologyEx(src, cv2.MORPH_TOPHAT, kernel), 50.0, 99.8))
    for sigma in (0.8, 1.2, 2.0, 3.2):
        small = cv2.GaussianBlur(src, (0, 0), sigmaX=sigma, sigmaY=sigma)
        large = cv2.GaussianBlur(src, (0, 0), sigmaX=sigma * 2.7, sigmaY=sigma * 2.7)
        feats.append(robust01(small - large, 50.0, 99.8))
        feats.append(robust01(np.abs(cv2.Laplacian(small, cv2.CV_32F, ksize=3)), 50.0, 99.8))
    h, w = src.shape
    small_bg = cv2.resize(src, (max(8, w // 20), max(8, h // 20)), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(small_bg, (w, h), interpolation=cv2.INTER_CUBIC)
    feats.append(robust01(src - bg, 50.0, 99.8))
    feats.append(robust01(bg, 0.5, 99.5))
    return np.stack(feats, axis=0).astype(np.float32)


class GatedTrainSet(Dataset):
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
        img_feat = anomaly_feature_stack(image)
        teacher, t_feat = teacher_maps(sample["preds"], self.names, self.teacher_mode)
        teacher = teacher[:h, :w]
        gt = sample["mask"].astype(np.float32)[:h, :w]
        teacher_unc = 4.0 * teacher * (1.0 - teacher)
        teacher_support = np.stack([teacher, teacher_unc], axis=0).astype(np.float32)
        feat = np.concatenate([img_feat, t_feat[:, :h, :w], teacher_support], axis=0)
        image_only = img_feat
        tlogit = logit_np(teacher)
        if h < self.patch_size or w < self.patch_size:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            feat = np.pad(feat, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
            image_only = np.pad(image_only, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
            gt = np.pad(gt, ((0, pad_h), (0, pad_w)), mode="constant")
            tlogit = np.pad(tlogit, ((0, pad_h), (0, pad_w)), mode="constant")
            h, w = gt.shape
        for _ in range(40):
            y0 = self.rng.randint(0, h - self.patch_size)
            x0 = self.rng.randint(0, w - self.patch_size)
            gt_patch = gt[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
            if self.rng.random() > 0.30 or gt_patch.sum() > 0:
                break
        feat_patch = feat[:, y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
        img_patch = image_only[:, y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
        gt_patch = gt[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
        tlogit_patch = tlogit[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
        if self.rng.random() < 0.5:
            feat_patch = feat_patch[:, :, ::-1].copy()
            img_patch = img_patch[:, :, ::-1].copy()
            gt_patch = gt_patch[:, ::-1].copy()
            tlogit_patch = tlogit_patch[:, ::-1].copy()
        if self.rng.random() < 0.5:
            feat_patch = feat_patch[:, ::-1, :].copy()
            img_patch = img_patch[:, ::-1, :].copy()
            gt_patch = gt_patch[::-1, :].copy()
            tlogit_patch = tlogit_patch[::-1, :].copy()
        return (
            torch.from_numpy(np.ascontiguousarray(feat_patch)).float(),
            torch.from_numpy(np.ascontiguousarray(img_patch)).float(),
            torch.from_numpy(np.ascontiguousarray(gt_patch[None])).float(),
            torch.from_numpy(np.ascontiguousarray(tlogit_patch[None])).float(),
        )


class GatedTestSet(Dataset):
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
        img_feat = anomaly_feature_stack(image)
        teacher, t_feat = teacher_maps(sample["preds"], self.names, self.teacher_mode)
        teacher = teacher[:h, :w]
        teacher_unc = 4.0 * teacher * (1.0 - teacher)
        feat = np.concatenate([img_feat, t_feat[:, :h, :w], np.stack([teacher, teacher_unc], axis=0)], axis=0)
        tlogit = logit_np(teacher)
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h or pad_w:
            feat = np.pad(feat, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
            img_feat = np.pad(img_feat, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
            tlogit = np.pad(tlogit, ((0, pad_h), (0, pad_w)), mode="constant")
        return (
            torch.from_numpy(np.ascontiguousarray(feat)).float(),
            torch.from_numpy(np.ascontiguousarray(img_feat)).float(),
            torch.from_numpy(np.ascontiguousarray(tlogit[None])).float(),
            sample["mask"].astype(np.float32)[:h, :w],
            [h, w],
            str(sample["name"]),
        )


class GatedAnomalyAnchor(nn.Module):
    def __init__(self, fused_ch: int, image_ch: int, base: int = 64, anomaly_base: int = 32) -> None:
        super().__init__()
        self.residual = AnchorResidualUNet(fused_ch, base)
        self.anomaly = AnchorResidualUNet(image_ch, anomaly_base)
        self.gate = nn.Sequential(
            nn.Conv2d(fused_ch + 3, base, 3, padding=1),
            nn.GroupNorm(max(1, min(16, base // 4)), base),
            nn.SiLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.GroupNorm(max(1, min(16, base // 4)), base),
            nn.SiLU(inplace=True),
            nn.Conv2d(base, 1, 1),
        )

    def forward(self, feat: torch.Tensor, image_feat: torch.Tensor, tlogit: torch.Tensor, residual_scale: float, anomaly_scale: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = residual_scale * torch.tanh(self.residual(feat))
        anomaly_logit = anomaly_scale * torch.tanh(self.anomaly(image_feat))
        gate_in = torch.cat([feat, tlogit, residual, anomaly_logit], dim=1)
        gate = torch.sigmoid(self.gate(gate_in))
        logits = tlogit + residual + gate * anomaly_logit
        return logits, residual, anomaly_logit, gate


def weighted_dice_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (weight * prob * target).sum(dim=(1, 2, 3))
    denom = (weight * prob).sum(dim=(1, 2, 3)) + (weight * target).sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a gated anomaly candidate branch anchored to cached teacher logits.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--train_cache_spec", action="append", required=True, help="DATASET|CACHE")
    parser.add_argument("--test_cache_spec", action="append", required=True, help="DATASET|CACHE")
    parser.add_argument("--teacher_mode", default="avg")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--base_channels", type=int, default=56)
    parser.add_argument("--anomaly_base_channels", type=int, default=28)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--residual_scale", type=float, default=0.45)
    parser.add_argument("--anomaly_scale", type=float, default=0.65)
    parser.add_argument("--anchor_weight", type=float, default=0.02)
    parser.add_argument("--anomaly_weight", type=float, default=0.25)
    parser.add_argument("--gate_weight", type=float, default=0.01)
    parser.add_argument("--pos_weight", type=float, default=25.0)
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
        train_sets.append(GatedTrainSet(dataset_dir, dataset_name, Path(cache), args.teacher_mode, args.patch_size, args.max_train_samples_per_cache, args.seed + len(train_sets)))
    test_sets = []
    for raw in args.test_cache_spec:
        dataset_name, cache = raw.split("|", 1)
        test_sets.append((dataset_name, GatedTestSet(dataset_dir, dataset_name, Path(cache), args.teacher_mode, args.eval_max_samples)))
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    loader = DataLoader(ConcatDataset(train_sets), batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=device.type == "cuda")
    first_feat, first_img, _gt, _tlogit = train_sets[0][0]
    model = GatedAnomalyAnchor(first_feat.shape[0], first_img.shape[0], args.base_channels, args.anomaly_base_channels).to(device)
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
            for feat, img_feat, gt, tlogit in loader:
                feat = feat.to(device, non_blocking=True)
                img_feat = img_feat.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)
                tlogit = tlogit.to(device, non_blocking=True)
                teacher_prob = torch.sigmoid(tlogit)
                uncertainty = 4.0 * teacher_prob * (1.0 - teacher_prob)
                hard = torch.relu(teacher_prob - 0.35) * (1.0 - gt)
                pixel_weight = 1.0 + 0.8 * uncertainty + 1.2 * hard
                logits, residual, anomaly_logit, gate = model(feat, img_feat, tlogit, args.residual_scale, args.anomaly_scale)
                bce = F.binary_cross_entropy_with_logits(logits, gt, weight=pixel_weight, pos_weight=pos_weight)
                main_loss = bce + weighted_dice_loss(logits, gt, pixel_weight) + 0.15 * dice_loss(logits, gt)
                anomaly_loss = F.binary_cross_entropy_with_logits(anomaly_logit, gt, pos_weight=pos_weight) + dice_loss(anomaly_logit, gt)
                reg_loss = residual.square().mean() + args.gate_weight * gate.mean()
                loss = main_loss + args.anomaly_weight * anomaly_loss + args.anchor_weight * reg_loss
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
                    for feat, img_feat, tlogit, mask, size, sample_name in DataLoader(ds, batch_size=1, shuffle=False, num_workers=1):
                        feat = feat.to(device)
                        img_feat = img_feat.to(device)
                        tlogit = tlogit.to(device)
                        with torch.no_grad():
                            logits, _residual, _anomaly, _gate = model(feat, img_feat, tlogit, args.residual_scale, args.anomaly_scale)
                            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
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
