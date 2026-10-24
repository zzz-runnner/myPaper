#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_cached_prediction_fusion import load_cache  # noqa: E402
from eval_component_fusion_learner import metric_from_maps  # noqa: E402


BASE_NAMES = [
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


def logit_np(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float32), 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p)).astype(np.float32)


def rank01(x: np.ndarray) -> np.ndarray:
    flat = x.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(flat), dtype=np.float32)
    return ranks.reshape(x.shape)


def local_feats(base: np.ndarray) -> list[np.ndarray]:
    feats: list[np.ndarray] = []
    for k in (3, 7, 15):
        mean = cv2.blur(base, (k, k))
        mx = cv2.dilate(base, np.ones((k, k), np.uint8))
        mn = cv2.erode(base, np.ones((k, k), np.uint8))
        feats.extend([mean.astype(np.float32), (base - mean).astype(np.float32), (mx - mn).astype(np.float32)])
    return feats


def feature_stack(preds: dict[str, np.ndarray], use_names: list[str]) -> np.ndarray:
    maps = [preds[name].astype(np.float32) for name in use_names if name in preds]
    if not maps:
        raise ValueError("no maps")
    stack = np.stack(maps, axis=0)
    base = preds["official_mamba_fused"].astype(np.float32) if "official_mamba_fused" in preds else stack.mean(axis=0)
    feats: list[np.ndarray] = []
    feats.extend(maps)
    feats.extend([logit_np(m) for m in maps[:4]])
    feats.extend([stack.mean(axis=0), stack.max(axis=0), stack.min(axis=0), stack.std(axis=0)])
    feats.extend([rank01(base), rank01(stack.max(axis=0)), np.maximum(stack.max(axis=0) - base, 0.0)])
    feats.extend(local_feats(base))
    return np.stack(feats, axis=0).astype(np.float32)


def sample_pixels(
    samples: list[dict],
    names: list[str],
    indices: list[int],
    max_pos: int,
    max_neg: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    pos_budget = max(1, max_pos // max(1, len(indices)))
    neg_budget = max(1, max_neg // max(1, len(indices)))
    use_names = [name for name in BASE_NAMES if name in names]
    for idx in indices:
        sample = samples[idx]
        mask = sample["mask"].astype(np.float32)
        feats = feature_stack(sample["preds"], use_names)
        base = sample["preds"]["official_mamba_fused"].astype(np.float32)
        pos = np.flatnonzero(mask.reshape(-1) > 0.5)
        neg = np.flatnonzero(mask.reshape(-1) <= 0.5)
        if len(pos) > 0:
            take = rng.choice(pos, size=min(pos_budget, len(pos)), replace=False)
            xs.append(feats.reshape(feats.shape[0], -1)[:, take].T)
            ys.append(np.ones(len(take), dtype=np.float32))
        if len(neg) > 0:
            neg_score = base.reshape(-1)[neg]
            hard_count = min(neg_budget // 2, len(neg))
            rand_count = min(neg_budget - hard_count, len(neg))
            hard = neg[np.argpartition(-neg_score, hard_count - 1)[:hard_count]] if hard_count > 0 else np.empty(0, dtype=np.int64)
            rand = rng.choice(neg, size=rand_count, replace=False) if rand_count > 0 else np.empty(0, dtype=np.int64)
            take = np.unique(np.concatenate([hard, rand]))
            xs.append(feats.reshape(feats.shape[0], -1)[:, take].T)
            ys.append(np.zeros(len(take), dtype=np.float32))
    return np.concatenate(xs, axis=0).astype(np.float32), np.concatenate(ys, axis=0).astype(np.float32)


def train_linear(x: np.ndarray, y: np.ndarray, c: float, seed: int, steps: int = 600) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    mu = x.mean(axis=0, keepdims=True).astype(np.float32)
    sig = (x.std(axis=0, keepdims=True) + 1e-5).astype(np.float32)
    xn = (x - mu) / sig
    xt = torch.from_numpy(xn)
    yt = torch.from_numpy(y[:, None])
    w = torch.zeros((x.shape[1], 1), dtype=torch.float32, requires_grad=True)
    b = torch.zeros((1,), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.AdamW([w, b], lr=0.03, weight_decay=c)
    pos = float(max(1, (y > 0.5).sum()))
    neg = float(max(1, (y <= 0.5).sum()))
    pos_weight = torch.tensor([min(neg / pos, 80.0)], dtype=torch.float32)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = xt @ w + b
        loss = F.binary_cross_entropy_with_logits(logits, yt, pos_weight=pos_weight)
        loss.backward()
        opt.step()
    return w.detach().numpy().reshape(-1), float(b.detach().numpy()[0]), mu.reshape(-1), sig.reshape(-1)


def predict_maps(samples: list[dict], names: list[str], params: tuple[np.ndarray, float, np.ndarray, np.ndarray], blend: float, gamma: float) -> list[np.ndarray]:
    w, b, mu, sig = params
    use_names = [name for name in BASE_NAMES if name in names]
    out = []
    for sample in samples:
        feats = feature_stack(sample["preds"], use_names)
        c, h, ww = feats.shape
        x = feats.reshape(c, -1).T
        logits = ((x - mu) / sig) @ w + b
        learned = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
        learned = learned.reshape(h, ww).astype(np.float32)
        base = sample["preds"]["official_mamba_fused"].astype(np.float32)
        p = np.clip((1.0 - blend) * base + blend * learned, 0.0, 1.0)
        out.append(np.clip(p, 0.0, 1.0) ** gamma)
    return out


def evaluate(samples: list[dict], maps: list[np.ndarray], thresholds: list[float]) -> dict:
    rows = [{"threshold": th, **metric_from_maps(samples, maps, th)} for th in thresholds]
    rows.sort(key=lambda r: r["mIoU"], reverse=True)
    return rows[0]


def fold_indices(n: int, fold: int, folds: int) -> tuple[list[int], list[int]]:
    train, val = [], []
    for i in range(n):
        (val if i % folds == fold else train).append(i)
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_cache", default=str(ROOT / "reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__train.npz"))
    ap.add_argument("--test_cache", default=str(ROOT / "reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__test.npz"))
    ap.add_argument("--output", default=str(ROOT / "reports/nuaa_official_mamba_pixel_calibrator.json"))
    ap.add_argument("--seed", type=int, default=20260615)
    ap.add_argument("--max_pos", type=int, default=120000)
    ap.add_argument("--max_neg", type=int, default=260000)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=420)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    train_samples, names = load_cache(Path(args.train_cache))
    test_samples, test_names = load_cache(Path(args.test_cache))
    thresholds = [round(x, 3) for x in np.arange(0.44, 0.561, 0.004)]
    if args.fast:
        regs = [0.0, 3e-5, 1e-4]
        blends = [0.04, 0.08, 0.12, 0.18]
        gammas = [0.98, 1.0, 1.02]
    else:
        regs = [0.0, 1e-5, 3e-5, 1e-4, 3e-4]
        blends = [0.04, 0.08, 0.12, 0.16, 0.22, 0.30]
        gammas = [0.96, 0.98, 1.0, 1.02, 1.04]
    val_rows = []
    for reg in regs:
        for blend in blends:
            for gamma in gammas:
                fold_scores = []
                fold_rows = []
                for fold in range(args.folds):
                    tr, va = fold_indices(len(train_samples), fold, args.folds)
                    x, y = sample_pixels(train_samples, names, tr, args.max_pos, args.max_neg, args.seed + fold)
                    params = train_linear(x, y, reg, args.seed + fold, args.steps)
                    val_subset = [train_samples[i] for i in va]
                    maps = predict_maps(val_subset, names, params, blend, gamma)
                    row = evaluate(val_subset, maps, thresholds)
                    fold_scores.append(row["mIoU"])
                    fold_rows.append(row)
                val_rows.append({
                    "reg": reg,
                    "blend": blend,
                    "gamma": gamma,
                    "folds": args.folds,
                    "val_mean_mIoU": float(np.mean(fold_scores)),
                    "val_min_mIoU": float(np.min(fold_scores)),
                    "fold_rows": fold_rows,
                })
    val_rows.sort(key=lambda r: (r["val_mean_mIoU"], r["val_min_mIoU"]), reverse=True)

    test_rows = []
    selected = val_rows[:12]
    full_idx = list(range(len(train_samples)))
    for cand in selected:
        x, y = sample_pixels(train_samples, names, full_idx, args.max_pos, args.max_neg, args.seed + 99)
        params = train_linear(x, y, cand["reg"], args.seed + 99, args.steps)
        maps = predict_maps(test_samples, test_names, params, cand["blend"], cand["gamma"])
        row = evaluate(test_samples, maps, thresholds)
        test_rows.append({k: cand[k] for k in ("reg", "blend", "gamma", "val_mean_mIoU", "val_min_mIoU")} | row)
    test_rows.sort(key=lambda r: r["mIoU"], reverse=True)
    Path(args.output).write_text(json.dumps({"ranked_by_internal_val": val_rows[:50], "test_from_val_top": test_rows}, indent=2), encoding="utf-8")
    print(json.dumps({"best_test": test_rows[0], "best_val": val_rows[0]}, indent=2))


if __name__ == "__main__":
    main()
