#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_cached_prediction_fusion import load_cache  # noqa: E402
from eval_component_fusion_learner import metric_from_maps  # noqa: E402
from search_nuaa_official_mamba_local_rules import build_maps  # noqa: E402
from train_nuaa_official_mamba_residual_pixel import LOCAL_CFG, min_area_map, parse_thresholds, sample_pixels  # noqa: E402


def feature_stack(sample: dict) -> tuple[np.ndarray, np.ndarray]:
    preds = sample["preds"]
    local = build_maps(sample, LOCAL_CFG).astype(np.float32)
    base = preds["official_mamba_fused"].astype(np.float32)
    strong = preds["strong_fusion"].astype(np.float32)
    weighted = preds["weighted_raw"].astype(np.float32)
    dfaware = preds.get("DFAware_official", base).astype(np.float32)
    hfm = preds.get("HFM_official", base).astype(np.float32)
    bg = preds["bgm_nuaa"].astype(np.float32)
    mean3 = cv2.blur(local, (3, 3)).astype(np.float32)
    mean7 = cv2.blur(local, (7, 7)).astype(np.float32)
    mean11 = cv2.blur(local, (11, 11)).astype(np.float32)
    mx = np.maximum.reduce([local, base, strong, weighted, dfaware, hfm]).astype(np.float32)
    mn = np.minimum.reduce([local, base, strong, weighted, dfaware, hfm]).astype(np.float32)
    disagree = mx - mn
    contrast = local - mean11
    feats = np.stack(
        [local, base, strong, weighted, dfaware, hfm, bg, mean3, mean7, mean11, mx, mn, disagree, contrast],
        axis=-1,
    ).astype(np.float32)
    return feats, local


def sample_pixels_mlp(samples: list[dict], pos_per_image: int, neg_per_image: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs = []
    ys = []
    for sample in samples:
        feats, local = feature_stack(sample)
        flat = feats.reshape(-1, feats.shape[-1])
        mask = sample["mask"].reshape(-1) > 0
        pos = np.flatnonzero(mask)
        if pos.size:
            k = min(pos_per_image, pos.size)
            xs.append(flat[rng.choice(pos, size=k, replace=pos.size < k)])
            ys.append(np.ones((k,), dtype=np.float32))
        neg = np.flatnonzero(~mask)
        if neg.size:
            hardness = np.maximum(local.reshape(-1), flat[:, 10])[neg]
            pool_n = min(max(neg_per_image * 10, neg_per_image), neg.size)
            pool = neg[np.argsort(hardness)[-pool_n:]]
            k = min(neg_per_image, pool.size)
            xs.append(flat[rng.choice(pool, size=k, replace=pool.size < k)])
            ys.append(np.zeros((k,), dtype=np.float32))
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32)


class ResidualMLPPixel(nn.Module):
    def __init__(self, in_dim: int, hidden: int, residual_scale: float, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.residual_scale = float(residual_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x[:, 0].clamp(1e-4, 1.0 - 1e-4)
        residual = torch.tanh(self.net(x).squeeze(-1))
        return torch.logit(base) + self.residual_scale * residual


def train_model(x: np.ndarray, y: np.ndarray, args: argparse.Namespace, device: torch.device) -> ResidualMLPPixel:
    torch.manual_seed(args.seed)
    model = ResidualMLPPixel(x.shape[1], args.hidden, args.residual_scale, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x_t = torch.from_numpy(x).to(device)
    y_t = torch.from_numpy(y).to(device)
    pos = max(float(y.mean()), 1e-6)
    pos_weight = torch.tensor([(1.0 - pos) / pos], device=device).clamp(1.0, args.max_pos_weight)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    n = x.shape[0]
    batch = min(args.batch_pixels, n)
    for step in range(args.steps):
        idx = torch.randint(0, n, (batch,), generator=gen, device=device)
        logits = model(x_t[idx])
        loss = F.binary_cross_entropy_with_logits(logits, y_t[idx], pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 300 == 0 or step + 1 == args.steps:
            print(f"step {step+1}/{args.steps} loss={float(loss.detach().cpu()):.5f}", flush=True)
    return model


@torch.no_grad()
def predict_maps(model: ResidualMLPPixel, samples: list[dict], device: torch.device) -> list[np.ndarray]:
    model.eval()
    maps = []
    for sample in samples:
        feats, _ = feature_stack(sample)
        flat = feats.reshape(-1, feats.shape[-1]).astype(np.float32)
        out = []
        for start in range(0, flat.shape[0], 262144):
            x = torch.from_numpy(flat[start : start + 262144]).to(device)
            out.append(torch.sigmoid(model(x)).cpu().numpy())
        maps.append(np.concatenate(out).reshape(sample["mask"].shape).astype(np.float32))
    return maps


def evaluate(samples: list[dict], maps: list[np.ndarray], thresholds: list[float], min_areas: list[int]) -> list[dict]:
    rows = []
    for th in thresholds:
        rows.append({"kind": "threshold", "threshold": th, "min_area": 1, **metric_from_maps(samples, maps, th)})
        for area in min_areas:
            if area <= 1:
                continue
            binary = [min_area_map(m, th, area) for m in maps]
            rows.append({"kind": "component", "threshold": th, "min_area": area, **metric_from_maps(samples, binary, 0.5)})
    rows.sort(key=lambda r: r["mIoU"], reverse=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_cache", default=str(ROOT / "reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__train.npz"))
    ap.add_argument("--test_cache", default=str(ROOT / "reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__test.npz"))
    ap.add_argument("--pos_per_image", type=int, default=768)
    ap.add_argument("--neg_per_image", type=int, default=3072)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch_pixels", type=int, default=65536)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--residual_scale", type=float, default=0.05)
    ap.add_argument("--max_pos_weight", type=float, default=25.0)
    ap.add_argument("--thresholds", default="0.492,0.494,0.496,0.498,0.500,0.502,0.504")
    ap.add_argument("--min_areas", default="1,2,3,4")
    ap.add_argument("--seed", type=int, default=20260616)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default=str(ROOT / "reports/nuaa_official_mamba_residual_mlp_pixel.json"))
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    train_samples, _ = load_cache(Path(args.train_cache))
    test_samples, _ = load_cache(Path(args.test_cache))
    x, y = sample_pixels_mlp(train_samples, args.pos_per_image, args.neg_per_image, args.seed)
    print(f"train pixels={x.shape[0]} pos={int(y.sum())} neg={int((1-y).sum())}", flush=True)
    model = train_model(x, y, args, device)
    train_maps = predict_maps(model, train_samples, device)
    test_maps = predict_maps(model, test_samples, device)
    thresholds = parse_thresholds(args.thresholds)
    min_areas = [int(x) for x in args.min_areas.split(",") if x.strip()]
    train_rows = evaluate(train_samples, train_maps, thresholds, min_areas)
    test_rows = evaluate(test_samples, test_maps, thresholds, min_areas)
    selected = []
    for row in train_rows[:20]:
        match = [r for r in test_rows if r["kind"] == row["kind"] and r["threshold"] == row["threshold"] and r["min_area"] == row["min_area"]]
        if match:
            selected.append({"train": row, "test": match[0]})
    out = {
        "args": vars(args),
        "train_pixels": int(x.shape[0]),
        "train_positive": int(y.sum()),
        "best_by_test": test_rows[0],
        "selected_by_train": selected[0] if selected else None,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "selected_top_train": selected,
    }
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"best_by_test": test_rows[0], "selected_by_train": selected[0] if selected else None}, indent=2), flush=True)


if __name__ == "__main__":
    main()
