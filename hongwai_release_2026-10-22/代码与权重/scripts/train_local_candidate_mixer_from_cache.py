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
    h, w = src.shape
    small = cv2.resize(src, (max(8, w // 16), max(8, h // 16)), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    feats.append(robust01(src - bg, 50.0, 99.8))
    return np.stack(feats, axis=0).astype(np.float32)


def candidate_stack(preds: dict[str, np.ndarray], names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    stack = np.stack([preds[name].astype(np.float32) for name in names], axis=0)
    avg = stack.mean(axis=0)
    stats = [avg, stack.max(axis=0), stack.min(axis=0), stack.std(axis=0)]
    stats.extend([stack[i] for i in range(stack.shape[0])])
    return stack.astype(np.float32), np.stack(stats, axis=0).astype(np.float32)


def prob_to_logit_tensor(prob: torch.Tensor) -> torch.Tensor:
    prob = prob.clamp(1e-4, 1.0 - 1e-4)
    return torch.log(prob / (1.0 - prob))


class MixerTrainSet(Dataset):
    def __init__(self, dataset_dir: Path, dataset_name: str, cache_path: Path, patch_size: int, max_samples: int, seed: int) -> None:
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.samples, self.names = load_cache(cache_path)
        if max_samples > 0:
            self.samples = self.samples[:max_samples]
        self.patch_size = patch_size
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        h, w = [int(x) for x in sample["size"]]
        image = open_image(self.dataset_dir, self.dataset_name, str(sample["name"]))[:h, :w]
        img_feat = image_feature_stack(image)
        cands, c_feat = candidate_stack(sample["preds"], self.names)
        gt = sample["mask"].astype(np.float32)[:h, :w]
        feat = np.concatenate([img_feat, c_feat[:, :h, :w]], axis=0)
        cand_logits = logit_np(cands[:, :h, :w])
        if h < self.patch_size or w < self.patch_size:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            feat = np.pad(feat, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
            cand_logits = np.pad(cand_logits, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
            gt = np.pad(gt, ((0, pad_h), (0, pad_w)), mode="constant")
            h, w = gt.shape
        for _ in range(30):
            y0 = self.rng.randint(0, h - self.patch_size)
            x0 = self.rng.randint(0, w - self.patch_size)
            gt_patch = gt[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
            if self.rng.random() > 0.45 or gt_patch.sum() > 0:
                break
        feat_patch = feat[:, y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
        logit_patch = cand_logits[:, y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
        gt_patch = gt[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
        if self.rng.random() < 0.5:
            feat_patch = feat_patch[:, :, ::-1].copy()
            logit_patch = logit_patch[:, :, ::-1].copy()
            gt_patch = gt_patch[:, ::-1].copy()
        if self.rng.random() < 0.5:
            feat_patch = feat_patch[:, ::-1, :].copy()
            logit_patch = logit_patch[:, ::-1, :].copy()
            gt_patch = gt_patch[::-1, :].copy()
        return (
            torch.from_numpy(np.ascontiguousarray(feat_patch)).float(),
            torch.from_numpy(np.ascontiguousarray(logit_patch)).float(),
            torch.from_numpy(np.ascontiguousarray(gt_patch[None])).float(),
        )


class MixerTestSet(Dataset):
    def __init__(self, dataset_dir: Path, dataset_name: str, cache_path: Path, max_samples: int = 0) -> None:
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
        img_feat = image_feature_stack(image)
        cands, c_feat = candidate_stack(sample["preds"], self.names)
        feat = np.concatenate([img_feat, c_feat[:, :h, :w]], axis=0)
        cand_logits = logit_np(cands[:, :h, :w])
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h or pad_w:
            feat = np.pad(feat, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
            cand_logits = np.pad(cand_logits, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant")
        return (
            torch.from_numpy(np.ascontiguousarray(feat)).float(),
            torch.from_numpy(np.ascontiguousarray(cand_logits)).float(),
            sample["mask"].astype(np.float32)[:h, :w],
            [h, w],
            str(sample["name"]),
        )


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        groups = max(g for g in range(1, min(16, cout) + 1) if cout % g == 0)
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(groups, cout),
            nn.SiLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(groups, cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalCandidateMixer(nn.Module):
    def __init__(self, in_ch: int, candidates: int, base: int = 48) -> None:
        super().__init__()
        self.candidates = candidates
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = ConvBlock(base, base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.e4 = ConvBlock(base * 4, base * 8)
        self.mid = ConvBlock(base * 8, base * 8)
        self.d3 = ConvBlock(base * 12, base * 4)
        self.d2 = ConvBlock(base * 6, base * 2)
        self.d1 = ConvBlock(base * 3, base)
        self.weight_head = nn.Conv2d(base, candidates, 1)
        self.residual_head = nn.Conv2d(base, 1, 1)
        self.temp_head = nn.Conv2d(base, 1, 1)
        nn.init.zeros_(self.weight_head.weight)
        nn.init.zeros_(self.weight_head.bias)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        nn.init.zeros_(self.temp_head.weight)
        nn.init.constant_(self.temp_head.bias, 0.0)

    def forward(self, feat: torch.Tensor, cand_logits: torch.Tensor, residual_scale: float, init_bias: torch.Tensor | None = None):
        e1 = self.e1(feat)
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
        weight_logits = self.weight_head(d1)
        if init_bias is not None:
            weight_logits = weight_logits + init_bias.view(1, -1, 1, 1)
        temp = 0.75 + F.softplus(self.temp_head(d1))
        weights = F.softmax(weight_logits / temp, dim=1)
        # Mix calibrated probabilities, not logits. Prior best cached fusions were
        # probability-space averages; logit averaging over-suppresses weak targets.
        mixed_prob = (weights * torch.sigmoid(cand_logits)).sum(dim=1, keepdim=True)
        mixed = prob_to_logit_tensor(mixed_prob)
        residual = residual_scale * torch.tanh(self.residual_head(d1))
        return mixed + residual, weights, residual


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()


def entropy_loss(weights: torch.Tensor) -> torch.Tensor:
    ent = -(weights.clamp_min(1e-6) * weights.clamp_min(1e-6).log()).sum(dim=1)
    return ent.mean()


def total_variation(x: torch.Tensor) -> torch.Tensor:
    return (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean() + (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()


def init_weight_bias(names: list[str], preferred: str, device: torch.device) -> torch.Tensor:
    bias = torch.zeros(len(names), dtype=torch.float32, device=device)
    if preferred:
        prefs = [item.strip() for item in preferred.split(",") if item.strip()]
        for pref in prefs:
            if pref == "avg":
                continue
            for idx, name in enumerate(names):
                if pref == name or pref.lower() in name.lower():
                    bias[idx] += 0.7
    return bias


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a local per-pixel candidate mixer from cached prediction maps.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--train_cache_spec", action="append", required=True, help="DATASET|CACHE")
    parser.add_argument("--test_cache_spec", action="append", required=True, help="DATASET|CACHE")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--residual_scale", type=float, default=0.20)
    parser.add_argument("--anchor_weight", type=float, default=0.02)
    parser.add_argument("--entropy_weight", type=float, default=0.004)
    parser.add_argument("--tv_weight", type=float, default=0.002)
    parser.add_argument("--pos_weight", type=float, default=25.0)
    parser.add_argument("--preferred", default="", help="comma-separated candidate name substrings to bias at initialization")
    parser.add_argument("--thresholds", default="0.30,0.35,0.368,0.38,0.40,0.42,0.45,0.50,0.55,0.60")
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
    first_names: list[str] | None = None
    for raw in args.train_cache_spec:
        dataset_name, cache = raw.split("|", 1)
        ds = MixerTrainSet(dataset_dir, dataset_name, Path(cache), args.patch_size, args.max_train_samples_per_cache, args.seed + len(train_sets))
        if first_names is None:
            first_names = ds.names
        elif ds.names != first_names:
            raise ValueError(f"candidate names differ: {ds.names} vs {first_names}")
        train_sets.append(ds)
    test_sets = []
    for raw in args.test_cache_spec:
        dataset_name, cache = raw.split("|", 1)
        ds = MixerTestSet(dataset_dir, dataset_name, Path(cache), args.eval_max_samples)
        if ds.names != first_names:
            raise ValueError(f"candidate names differ in test cache: {ds.names} vs {first_names}")
        test_sets.append((dataset_name, ds))

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    loader = DataLoader(ConcatDataset(train_sets), batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=device.type == "cuda")
    sample_feat, sample_logits, _ = train_sets[0][0]
    model = LocalCandidateMixer(sample_feat.shape[0], sample_logits.shape[0], base=args.base_channels).to(device)
    bias = init_weight_bias(first_names or [], args.preferred, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)
    thresholds = [float(x) for x in parse_csv(args.thresholds)]
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    best = None
    history = []
    pos_weight = torch.tensor([args.pos_weight], dtype=torch.float32, device=device).view(1, 1, 1, 1)

    with (run_dir / "train.log").open("w", encoding="utf-8") as log:
        header = {"args": vars(args), "candidate_names": first_names}
        log.write(json.dumps(header, indent=2) + "\n")
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            model.train()
            losses = []
            for feat, cand_logits, gt in loader:
                feat = feat.to(device, non_blocking=True)
                cand_logits = cand_logits.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)
                logits, weights, residual = model(feat, cand_logits, args.residual_scale, bias)
                bce = F.binary_cross_entropy_with_logits(logits, gt, pos_weight=pos_weight)
                loss = (
                    bce
                    + dice_loss(logits, gt)
                    + args.anchor_weight * residual.square().mean()
                    - args.entropy_weight * entropy_loss(weights)
                    + args.tv_weight * total_variation(weights)
                )
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
                    for feat, cand_logits, mask, size, sample_name in DataLoader(ds, batch_size=1, shuffle=False, num_workers=1):
                        feat = feat.to(device)
                        cand_logits = cand_logits.to(device)
                        with torch.no_grad():
                            logits, _, _ = model(feat, cand_logits, args.residual_scale, bias)
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
                    torch.save({"epoch": epoch, "state_dict": model.state_dict(), "best": best, "candidate_names": first_names}, run_dir / "best.pth.tar")
            history.append(row)
            line = json.dumps(row, sort_keys=True)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
    (run_dir / "summary.json").write_text(json.dumps({"args": vars(args), "candidate_names": first_names, "best": best, "history": history}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
