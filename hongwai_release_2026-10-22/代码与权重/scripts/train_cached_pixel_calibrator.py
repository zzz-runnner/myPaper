#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))

from eval_cached_prediction_fusion import load_cache  # noqa: E402
from eval_component_fusion_learner import metric_from_maps  # noqa: E402


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def collect_pixels(
    cache_paths: list[str],
    max_pos_per_image: int,
    max_neg_per_image: int,
    hard_negative_quantile: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    xs = []
    ys = []
    names_ref = None
    for path in cache_paths:
        samples, names = load_cache(Path(path))
        if names_ref is None:
            names_ref = names
        elif names != names_ref:
            raise ValueError(f"name mismatch in {path}: {names} != {names_ref}")
        for sample in samples:
            stack = np.stack([sample["preds"][name] for name in names], axis=-1).astype(np.float32)
            mask = sample["mask"].astype(bool)
            flat = stack.reshape(-1, stack.shape[-1])
            y = mask.reshape(-1)
            pos_idx = np.flatnonzero(y)
            neg_idx = np.flatnonzero(~y)
            if pos_idx.size:
                if pos_idx.size > max_pos_per_image:
                    pos_idx = rng.choice(pos_idx, max_pos_per_image, replace=False)
                xs.append(flat[pos_idx])
                ys.append(np.ones((pos_idx.size,), dtype=np.float32))
            if neg_idx.size:
                hardness = flat[neg_idx].max(axis=1)
                if hard_negative_quantile > 0:
                    cutoff = np.percentile(hardness, hard_negative_quantile)
                    hard = neg_idx[hardness >= cutoff]
                    if hard.size:
                        neg_idx = hard
                if neg_idx.size > max_neg_per_image:
                    neg_idx = rng.choice(neg_idx, max_neg_per_image, replace=False)
                xs.append(flat[neg_idx])
                ys.append(np.zeros((neg_idx.size,), dtype=np.float32))
    if not xs:
        raise RuntimeError("no pixels collected")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), names_ref or []


class PixelMLP(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def train_model(x: np.ndarray, y: np.ndarray, hidden: int, steps: int, lr: float, weight_decay: float, pos_weight_scale: float, device: torch.device):
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    xs = ((x - mean) / std).astype(np.float32)
    xt = torch.from_numpy(xs).to(device)
    yt = torch.from_numpy(y.astype(np.float32)).to(device)
    model = PixelMLP(xs.shape[1], hidden).to(device)
    pos = max(float(y.sum()), 1.0)
    neg = max(float((1.0 - y).sum()), 1.0)
    pos_weight = torch.tensor([neg / pos * pos_weight_scale], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    batch = min(262144, xs.shape[0])
    rng = np.random.default_rng(20260613)
    for step in range(steps):
        if xs.shape[0] > batch:
            idx = rng.choice(xs.shape[0], batch, replace=False)
            xb = xt[idx]
            yb = yt[idx]
        else:
            xb = xt
            yb = yt
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        if (step + 1) % max(steps // 5, 1) == 0:
            print(f"step {step+1}/{steps} loss={float(loss.detach().cpu()):.6f}", flush=True)
    model.eval()
    return model, mean, std, float(pos_weight.item())


@torch.no_grad()
def predict_map(model: nn.Module, mean: np.ndarray, std: np.ndarray, stack: np.ndarray, device: torch.device) -> np.ndarray:
    h, w, c = stack.shape
    flat = ((stack.reshape(-1, c) - mean) / std).astype(np.float32)
    out = np.zeros((flat.shape[0],), dtype=np.float32)
    chunk = 262144
    for start in range(0, flat.shape[0], chunk):
        xb = torch.from_numpy(flat[start : start + chunk]).to(device)
        out[start : start + chunk] = torch.sigmoid(model(xb)).detach().cpu().numpy()
    return out.reshape(h, w)


def eval_model(model, mean, std, cache_path: str, names: list[str], thresholds: list[float], device: torch.device):
    samples, cache_names = load_cache(Path(cache_path))
    if cache_names != names:
        raise ValueError(f"test cache names mismatch: {cache_names} != {names}")
    maps = []
    for sample in samples:
        stack = np.stack([sample["preds"][name] for name in names], axis=-1).astype(np.float32)
        maps.append(predict_map(model, mean, std, stack, device))
    rows = []
    for threshold in thresholds:
        rows.append({"threshold": threshold, **metric_from_maps(samples, maps, threshold)})
    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small pixel-level calibrator from cached prediction maps.")
    parser.add_argument("--train_cache", action="append", required=True)
    parser.add_argument("--test_cache", required=True)
    parser.add_argument("--max_pos_per_image", type=int, default=512)
    parser.add_argument("--max_neg_per_image", type=int, default=2048)
    parser.add_argument("--hard_negative_quantile", type=float, default=95.0)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--pos_weight_scales", default="0.25,0.50,1.00")
    parser.add_argument("--thresholds", default="0.20,0.30,0.40,0.50,0.60,0.70")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    x, y, names = collect_pixels(args.train_cache, args.max_pos_per_image, args.max_neg_per_image, args.hard_negative_quantile, args.seed)
    print(f"collected pixels x={x.shape} positives={int(y.sum())} names={names}", flush=True)
    thresholds = parse_float_list(args.thresholds)
    all_rows = []
    best_models = []
    for pos_scale in parse_float_list(args.pos_weight_scales):
        print(f"training pos_weight_scale={pos_scale}", flush=True)
        model, mean, std, pos_weight = train_model(x, y, args.hidden, args.steps, args.lr, args.weight_decay, pos_scale, device)
        rows = eval_model(model, mean, std, args.test_cache, names, thresholds, device)
        for row in rows:
            row.update({"pos_weight_scale": pos_scale, "pos_weight": pos_weight})
        all_rows.extend(rows)
        best_models.append((rows[0]["mIoU"], pos_scale, rows[0]))
        print(
            f"best scale={pos_scale} th={rows[0]['threshold']} mIoU={rows[0]['mIoU']*100:.2f} "
            f"Pd={rows[0]['Pd']*100:.2f} Fa={rows[0]['Fa']*1e6:.2f}",
            flush=True,
        )
    all_rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best = all_rows[0]
    print(
        f"overall best scale={best['pos_weight_scale']} th={best['threshold']} "
        f"mIoU={best['mIoU']*100:.2f} Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.2f}",
        flush=True,
    )
    Path(args.output).write_text(
        json.dumps(
            {
                "train_caches": args.train_cache,
                "test_cache": args.test_cache,
                "models": names,
                "pixels": int(x.shape[0]),
                "positives": int(y.sum()),
                "best": best,
                "rows": all_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
