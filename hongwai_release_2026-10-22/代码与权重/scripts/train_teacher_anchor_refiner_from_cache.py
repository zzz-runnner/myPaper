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
from search_nuaa_official_mamba_local_rules import build_maps as build_nuaa_local_rule  # noqa: E402


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def robust01(x: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(x, [low, high])
    if hi <= lo + 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def logit_np(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float32), 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p)).astype(np.float32)


def open_image(dataset_dir: Path, dataset_name: str, name: str) -> np.ndarray:
    root = dataset_dir / dataset_name
    for ext in (".png", ".bmp", ".jpg"):
        path = root / "images" / f"{name}{ext}"
        if path.exists():
            return np.asarray(Image.open(path).convert("I"), dtype=np.float32)
    raise FileNotFoundError(f"{dataset_name}/{name}")


def image_feature_stack(image: np.ndarray) -> np.ndarray:
    src = robust01(image)
    feats = [src]
    for k in (7, 15, 31):
        mean = cv2.blur(src, (k, k))
        sq = cv2.blur(src * src, (k, k))
        std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
        feats.append(robust01(src - mean, 50.0, 99.8))
        feats.append(robust01((src - mean) / (std + 1e-3), 50.0, 99.8))
    for k in (9, 17):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        feats.append(robust01(cv2.morphologyEx(src, cv2.MORPH_TOPHAT, kernel), 50.0, 99.8))
    for sigma in (1.0, 2.2):
        small = cv2.GaussianBlur(src, (0, 0), sigmaX=sigma, sigmaY=sigma)
        large = cv2.GaussianBlur(src, (0, 0), sigmaX=sigma * 2.5, sigmaY=sigma * 2.5)
        feats.append(robust01(small - large, 50.0, 99.8))
        feats.append(robust01(np.abs(cv2.Laplacian(small, cv2.CV_32F, ksize=3)), 50.0, 99.8))
    # Cheap low-rank/background proxy: aggressively downsample then reconstruct.
    h, w = src.shape
    small = cv2.resize(src, (max(8, w // 16), max(8, h // 16)), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    feats.append(robust01(src - bg, 50.0, 99.8))
    feats.append(robust01(bg, 0.5, 99.5))
    return np.stack(feats, axis=0).astype(np.float32)


def teacher_maps(preds: dict[str, np.ndarray], names: list[str], mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "local_rule":
        if "official_mamba_fused" in preds and "bgm_nuaa" in preds:
            teacher = build_nuaa_local_rule({"preds": preds}, {
                "bg_th": 0.08,
                "support_th": 0.22,
                "rank_th": 0.985,
                "suppress_scale": 0.85,
                "agree_th": 0.1,
                "mx_th": 0.5,
                "restore_cap": 0.58,
                "range_th": 0.05,
                "restore_mix": 0.0,
                "gamma": 1.0,
            }).astype(np.float32)
        else:
            fallback = [name for name in ("strong_fusion", "weighted_raw", "tpc_a", "tpc_b", "lca") if name in preds]
            if not fallback:
                fallback = names
            teacher = np.stack([preds[name].astype(np.float32) for name in fallback], axis=0).mean(axis=0)
        base_order = [
            "official_mamba_fused",
            "strong_fusion",
            "weighted_raw",
            "tpc_a",
            "tpc_b",
            "lca",
            "bgm_nuaa",
            "rpcanet_sirst_s3",
            "bg_anomaly",
        ]
        existing = [preds[name].astype(np.float32) if name in preds else teacher for name in base_order]
        stack = np.stack(existing, axis=0)
        stats = [teacher, stack.max(axis=0), stack.min(axis=0), stack.std(axis=0)]
        stats.extend([stack[i] for i in range(stack.shape[0])])
        return teacher.astype(np.float32), np.stack(stats, axis=0).astype(np.float32)
    if "," in mode:
        for item in parse_csv(mode):
            if item in preds or item in {"avg", "max"}:
                teacher, _ = teacher_maps(preds, names, item)
                base_order = ["strong_fusion", "weighted_raw", "tpc_a", "tpc_b", "lca", "bgm_nuaa", "rpcanet_sirst_s3", "bg_anomaly"]
                base_names = [name for name in base_order if name in preds]
                if not base_names:
                    base_names = [name for name in names if name != item]
                _, stats = teacher_maps(preds, base_names, "avg")
                return teacher, stats
        raise ValueError(mode)
    stack = np.stack([preds[name].astype(np.float32) for name in names], axis=0)
    if mode == "avg":
        teacher = stack.mean(axis=0)
    elif mode == "max":
        teacher = stack.max(axis=0)
    elif mode in names:
        teacher = preds[mode].astype(np.float32)
    else:
        raise ValueError(mode)
    stats = [teacher, stack.max(axis=0), stack.min(axis=0), stack.std(axis=0)]
    stats.extend([stack[i] for i in range(stack.shape[0])])
    return teacher.astype(np.float32), np.stack(stats, axis=0).astype(np.float32)


class AnchorTrainSet(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        dataset_name: str,
        cache_path: Path,
        teacher_mode: str,
        patch_size: int,
        max_samples: int,
        seed: int,
        focus_dataset: str = "",
        focus_patch_prob: float = 0.0,
        focus_hard_threshold: float = 0.45,
        focus_miss_threshold: float = 0.18,
        copy_paste_prob: float = 0.0,
        copy_paste_max_area: int = 96,
        gamma_prob: float = 0.0,
        noise_std: float = 0.0,
        image_aug_mode: str = "image",
    ) -> None:
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.samples, self.names = load_cache(cache_path)
        if max_samples > 0:
            self.samples = self.samples[:max_samples]
        self.teacher_mode = teacher_mode
        self.patch_size = patch_size
        self.rng = random.Random(seed)
        self.focus_enabled = bool(focus_dataset) and dataset_name == focus_dataset and focus_patch_prob > 0.0
        self.focus_patch_prob = focus_patch_prob
        self.focus_hard_threshold = focus_hard_threshold
        self.focus_miss_threshold = focus_miss_threshold
        self.copy_paste_prob = copy_paste_prob
        self.copy_paste_max_area = copy_paste_max_area
        self.gamma_prob = gamma_prob
        self.noise_std = noise_std
        self.image_aug_mode = image_aug_mode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        h, w = [int(x) for x in sample["size"]]
        image = open_image(self.dataset_dir, self.dataset_name, str(sample["name"]))[:h, :w]
        if self.image_aug_mode == "image" and self.gamma_prob > 0.0 and self.rng.random() < self.gamma_prob:
            min_v = float(image.min())
            rng = float(image.max() - min_v)
            if rng > 1e-6:
                gamma = self.rng.uniform(0.65, 1.55)
                x = np.clip((image - min_v) / rng, 0.0, 1.0)
                image = np.power(x, gamma) * rng + min_v
        if self.image_aug_mode == "image" and self.noise_std > 0.0 and self.rng.random() < 0.7:
            image = image + np.random.normal(0.0, self.noise_std, size=image.shape).astype(np.float32)
        img_feat = image_feature_stack(image)
        if self.image_aug_mode == "feature":
            if self.gamma_prob > 0.0 and self.rng.random() < self.gamma_prob:
                gamma = self.rng.uniform(0.75, 1.35)
                img_feat = np.clip(np.power(np.clip(img_feat, 0.0, 1.0), gamma), 0.0, 1.0).astype(np.float32)
            if self.noise_std > 0.0 and self.rng.random() < 0.7:
                img_feat = np.clip(
                    img_feat + np.random.normal(0.0, self.noise_std, size=img_feat.shape).astype(np.float32),
                    0.0,
                    1.0,
                ).astype(np.float32)
        img_feat_channels = int(img_feat.shape[0])
        teacher, t_feat = teacher_maps(sample["preds"], self.names, self.teacher_mode)
        gt = sample["mask"].astype(np.float32)[:h, :w]
        feat = np.concatenate([img_feat, t_feat[:, :h, :w]], axis=0)
        tlogit = logit_np(teacher[:h, :w])
        if h < self.patch_size or w < self.patch_size:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            feat = np.pad(feat, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
            gt = np.pad(gt, ((0, pad_h), (0, pad_w)), mode="constant")
            tlogit = np.pad(tlogit, ((0, pad_h), (0, pad_w)), mode="constant")
            h, w = gt.shape
        if self.focus_enabled and self.rng.random() < self.focus_patch_prob:
            teacher_prob = 1.0 / (1.0 + np.exp(-np.clip(tlogit, -20.0, 20.0)))
            hard = (teacher_prob > self.focus_hard_threshold) & (gt < 0.5)
            miss = (teacher_prob < self.focus_miss_threshold) & (gt > 0.5)
            focus = hard | miss | (gt > 0.5)
            ys, xs = np.where(focus)
            if len(ys) > 0:
                pick = self.rng.randrange(len(ys))
                cy, cx = int(ys[pick]), int(xs[pick])
                y0 = min(max(0, cy - self.rng.randint(0, self.patch_size - 1)), h - self.patch_size)
                x0 = min(max(0, cx - self.rng.randint(0, self.patch_size - 1)), w - self.patch_size)
            else:
                y0 = self.rng.randint(0, h - self.patch_size)
                x0 = self.rng.randint(0, w - self.patch_size)
        else:
            for _ in range(30):
                y0 = self.rng.randint(0, h - self.patch_size)
                x0 = self.rng.randint(0, w - self.patch_size)
                gt_patch = gt[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
                if self.rng.random() > 0.45 or gt_patch.sum() > 0:
                    break
        feat_patch = feat[:, y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
        gt_patch = gt[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
        tlogit_patch = tlogit[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
        if self.copy_paste_prob > 0.0 and self.rng.random() < self.copy_paste_prob and gt_patch.sum() > 0:
            binary = (gt_patch > 0.5).astype(np.uint8)
            num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
            candidate_labels = []
            for label in range(1, num):
                area = int(stats[label, cv2.CC_STAT_AREA])
                width = int(stats[label, cv2.CC_STAT_WIDTH])
                height = int(stats[label, cv2.CC_STAT_HEIGHT])
                if 1 <= area <= self.copy_paste_max_area and width <= 24 and height <= 24:
                    candidate_labels.append(label)
            if candidate_labels:
                label = self.rng.choice(candidate_labels)
                sx = int(stats[label, cv2.CC_STAT_LEFT])
                sy = int(stats[label, cv2.CC_STAT_TOP])
                sw = int(stats[label, cv2.CC_STAT_WIDTH])
                sh = int(stats[label, cv2.CC_STAT_HEIGHT])
                obj_mask = labels[sy : sy + sh, sx : sx + sw] == label
                for _ in range(24):
                    dx = self.rng.randint(0, self.patch_size - sw)
                    dy = self.rng.randint(0, self.patch_size - sh)
                    dst_gt = gt_patch[dy : dy + sh, dx : dx + sw] > 0.5
                    if not bool((dst_gt & obj_mask).any()):
                        src_feat = feat_patch[:img_feat_channels, sy : sy + sh, sx : sx + sw]
                        dst_feat = feat_patch[:img_feat_channels, dy : dy + sh, dx : dx + sw]
                        dst_feat[:, obj_mask] = src_feat[:, obj_mask]
                        gt_patch[dy : dy + sh, dx : dx + sw][obj_mask] = 1.0
                        break
        if self.rng.random() < 0.5:
            feat_patch = feat_patch[:, :, ::-1].copy()
            gt_patch = gt_patch[:, ::-1].copy()
            tlogit_patch = tlogit_patch[:, ::-1].copy()
        if self.rng.random() < 0.5:
            feat_patch = feat_patch[:, ::-1, :].copy()
            gt_patch = gt_patch[::-1, :].copy()
            tlogit_patch = tlogit_patch[::-1, :].copy()
        return (
            torch.from_numpy(np.ascontiguousarray(feat_patch)).float(),
            torch.from_numpy(np.ascontiguousarray(gt_patch[None])).float(),
            torch.from_numpy(np.ascontiguousarray(tlogit_patch[None])).float(),
        )


class AnchorTestSet(Dataset):
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
        feat = np.concatenate([img_feat, t_feat[:, :h, :w]], axis=0)
        tlogit = logit_np(teacher[:h, :w])
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


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(max(1, min(16, cout // 4)), cout),
            nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(max(1, min(16, cout // 4)), cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AnchorResidualUNet(nn.Module):
    def __init__(self, in_ch: int, base: int = 48) -> None:
        super().__init__()
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.e4 = ConvBlock(base * 4, base * 8)
        self.mid = ConvBlock(base * 8, base * 8)
        self.d3 = ConvBlock(base * 12, base * 4)
        self.d2 = ConvBlock(base * 6, base * 2)
        self.d1 = ConvBlock(base * 3, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        mid = self.mid(e4)
        d3 = F.interpolate(mid, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()


def weighted_dice_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (weight * prob * target).sum(dim=(1, 2, 3))
    denom = (weight * prob).sum(dim=(1, 2, 3)) + (weight * target).sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a teacher-anchored residual refiner from cached predictions plus low-rank/high-frequency image cues.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--train_cache_spec", action="append", required=True, help="DATASET|CACHE")
    parser.add_argument("--test_cache_spec", action="append", required=True, help="DATASET|CACHE")
    parser.add_argument("--teacher_mode", default="avg")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--residual_scale", type=float, default=1.0)
    parser.add_argument("--anchor_weight", type=float, default=0.05)
    parser.add_argument("--pos_weight", type=float, default=25.0)
    parser.add_argument("--hard_clutter_weight", type=float, default=0.0)
    parser.add_argument("--hard_clutter_threshold", type=float, default=0.35)
    parser.add_argument("--miss_weight", type=float, default=0.0)
    parser.add_argument("--miss_threshold", type=float, default=0.20)
    parser.add_argument("--uncertainty_weight", type=float, default=0.0)
    parser.add_argument("--weighted_dice", action="store_true")
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
        train_sets.append(AnchorTrainSet(dataset_dir, dataset_name, Path(cache), args.teacher_mode, args.patch_size, args.max_train_samples_per_cache, args.seed + len(train_sets)))
    test_sets = []
    for raw in args.test_cache_spec:
        dataset_name, cache = raw.split("|", 1)
        test_sets.append((dataset_name, AnchorTestSet(dataset_dir, dataset_name, Path(cache), args.teacher_mode, args.eval_max_samples)))
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
            for feat, gt, tlogit in loader:
                feat = feat.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)
                tlogit = tlogit.to(device, non_blocking=True)
                residual = args.residual_scale * torch.tanh(model(feat))
                logits = tlogit + residual
                with torch.no_grad():
                    teacher_prob = torch.sigmoid(tlogit)
                    hard_clutter = (teacher_prob > args.hard_clutter_threshold).float() * (1.0 - gt)
                    missed_target = (teacher_prob < args.miss_threshold).float() * gt
                    uncertainty = 4.0 * teacher_prob * (1.0 - teacher_prob)
                    pixel_weight = (
                        1.0
                        + args.hard_clutter_weight * hard_clutter
                        + args.miss_weight * missed_target
                        + args.uncertainty_weight * uncertainty
                    )
                bce = F.binary_cross_entropy_with_logits(logits, gt, weight=pixel_weight, pos_weight=pos_weight)
                dloss = weighted_dice_loss(logits, gt, pixel_weight) if args.weighted_dice else dice_loss(logits, gt)
                loss = bce + dloss + args.anchor_weight * residual.square().mean()
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
