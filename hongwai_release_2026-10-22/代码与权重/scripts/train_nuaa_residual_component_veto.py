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
from train_nuaa_official_mamba_residual_pixel import (  # noqa: E402
    LOCAL_CFG,
    ResidualPixel,
    feature_stack,
    parse_thresholds,
    predict_maps,
    sample_pixels,
    train_model,
)


def ring_mean(arr: np.ndarray, comp: np.ndarray) -> float:
    dil = cv2.dilate(comp.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
    ring = dil & ~comp
    if not ring.any():
        return 0.0
    return float(arr[ring].mean())


def make_features(sample: dict, prob: np.ndarray, threshold: float) -> tuple[list[dict], np.ndarray, np.ndarray]:
    preds = sample["preds"]
    local = build_maps(sample, LOCAL_CFG).astype(np.float32)
    base = preds["official_mamba_fused"].astype(np.float32)
    strong = preds["strong_fusion"].astype(np.float32)
    weighted = preds["weighted_raw"].astype(np.float32)
    dfaware = preds.get("DFAware_official", base).astype(np.float32)
    hfm = preds.get("HFM_official", base).astype(np.float32)
    bg = preds["bgm_nuaa"].astype(np.float32)
    agree = np.maximum.reduce([local, base, strong, weighted, dfaware, hfm]).astype(np.float32)
    binary = (prob >= threshold).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    gt = sample["mask"].astype(bool)
    rows = []
    xs = []
    ys = []
    h_img, w_img = prob.shape
    for label in range(1, num):
        y_idx, x_idx = np.where(labels == label)
        if y_idx.size == 0:
            continue
        comp = labels == label
        area = float(stats[label, cv2.CC_STAT_AREA])
        bw = float(stats[label, cv2.CC_STAT_WIDTH])
        bh = float(stats[label, cv2.CC_STAT_HEIGHT])
        pvals = prob[y_idx, x_idx]
        lvals = local[y_idx, x_idx]
        avals = agree[y_idx, x_idx]
        bvals = bg[y_idx, x_idx]
        inter = int((comp & gt).sum())
        feat = np.array(
            [
                np.log1p(area) / 4.0,
                area / 64.0,
                max(bw, bh) / 24.0,
                min(bw, bh) / 12.0,
                float(pvals.mean()),
                float(pvals.max()),
                float(lvals.mean()),
                float(lvals.max()),
                float(avals.mean()),
                float(avals.max()),
                float(bvals.mean()),
                float(bvals.max()),
                float(pvals.mean() - ring_mean(prob, comp)),
                float(lvals.mean() - ring_mean(local, comp)),
                float(area / max(float(h_img * w_img), 1.0) * 1e5),
            ],
            dtype=np.float32,
        )
        rows.append({"label": label, "area": int(area), "inter": inter})
        xs.append(feat)
        ys.append(1.0 if inter > 0 else 0.0)
    if not xs:
        return rows, np.zeros((0, 15), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return rows, np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32)


class ComponentMLP(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def fit_component_model(x: np.ndarray, y: np.ndarray, args: argparse.Namespace, device: torch.device) -> ComponentMLP:
    torch.manual_seed(args.seed + 17)
    model = ComponentMLP(x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.component_lr, weight_decay=args.component_weight_decay)
    x_t = torch.from_numpy(x).to(device)
    y_t = torch.from_numpy(y).to(device)
    pos = max(float(y.mean()), 1e-6)
    pos_weight = torch.tensor([(1.0 - pos) / pos], device=device).clamp(1.0, args.component_max_pos_weight)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + 17)
    n = x.shape[0]
    batch = min(args.component_batch, n)
    for step in range(args.component_steps):
        idx = torch.randint(0, n, (batch,), generator=gen, device=device)
        logits = model(x_t[idx])
        loss = F.binary_cross_entropy_with_logits(logits, y_t[idx], pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def score_components(model: ComponentMLP, x: np.ndarray, device: torch.device) -> np.ndarray:
    if x.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    model.eval()
    out = []
    for start in range(0, x.shape[0], 65536):
        xt = torch.from_numpy(x[start : start + 65536]).to(device)
        out.append(torch.sigmoid(model(xt)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def build_tables(samples: list[dict], maps: list[np.ndarray], thresholds: list[float]) -> dict[float, list[dict]]:
    out: dict[float, list[dict]] = {}
    for th in thresholds:
        entries = []
        for sample, prob in zip(samples, maps):
            rows, x, y = make_features(sample, prob, th)
            entries.append({"rows": rows, "x": x, "y": y, "shape": sample["mask"].shape})
        out[th] = entries
    return out


def render(samples: list[dict], tables: list[dict], scores_by_image: list[np.ndarray], keep_th: float, min_area: int, max_area: int) -> list[np.ndarray]:
    maps = []
    for sample, entry, scores in zip(samples, tables, scores_by_image):
        labels_shape = entry["shape"]
        seed = np.zeros(labels_shape, dtype=np.uint8)
        # Reconstruct labels cheaply from row masks is not possible, so re-run CC for the chosen threshold map.
        # The caller fills this function through render_with_maps where the probability map is available.
        maps.append(seed.astype(np.float32))
    return maps


def render_with_maps(
    samples: list[dict],
    maps_prob: list[np.ndarray],
    tables: list[dict],
    scores_by_image: list[np.ndarray],
    seed_threshold: float,
    keep_threshold: float,
    min_area: int,
    max_area: int,
) -> list[np.ndarray]:
    rendered = []
    for sample, prob, entry, scores in zip(samples, maps_prob, tables, scores_by_image):
        binary = (prob >= seed_threshold).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        out = np.zeros_like(binary, dtype=np.uint8)
        score_idx = 0
        for label in range(1, num):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if score_idx >= len(scores):
                break
            keep = scores[score_idx] >= keep_threshold and area >= min_area and (max_area <= 0 or area <= max_area)
            if keep:
                out[labels == label] = 1
            score_idx += 1
        rendered.append(out.astype(np.float32))
    return rendered


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_cache", default=str(ROOT / "reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__train.npz"))
    ap.add_argument("--test_cache", default=str(ROOT / "reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__test.npz"))
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--residual_scale", type=float, default=0.04)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--pos_per_image", type=int, default=512)
    ap.add_argument("--neg_per_image", type=int, default=2048)
    ap.add_argument("--batch_pixels", type=int, default=65536)
    ap.add_argument("--max_pos_weight", type=float, default=30.0)
    ap.add_argument("--seed_thresholds", default="0.492,0.494,0.496,0.498,0.500")
    ap.add_argument("--keep_thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60")
    ap.add_argument("--min_areas", default="1,2,3,4")
    ap.add_argument("--max_areas", default="0,64,128,256")
    ap.add_argument("--component_steps", type=int, default=600)
    ap.add_argument("--component_batch", type=int, default=4096)
    ap.add_argument("--component_lr", type=float, default=1e-3)
    ap.add_argument("--component_weight_decay", type=float, default=1e-4)
    ap.add_argument("--component_max_pos_weight", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=20260615)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default=str(ROOT / "reports/nuaa_residual_component_veto.json"))
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    train_samples, _ = load_cache(Path(args.train_cache))
    test_samples, _ = load_cache(Path(args.test_cache))
    x_pix, y_pix = sample_pixels(train_samples, args.pos_per_image, args.neg_per_image, args.seed)
    print(f"residual train pixels={x_pix.shape[0]} pos={int(y_pix.sum())}", flush=True)
    residual_model: ResidualPixel = train_model(x_pix, y_pix, args, device)
    train_maps = predict_maps(residual_model, train_samples, device)
    test_maps = predict_maps(residual_model, test_samples, device)

    seed_thresholds = parse_thresholds(args.seed_thresholds)
    train_tables = build_tables(train_samples, train_maps, seed_thresholds)
    test_tables = build_tables(test_samples, test_maps, seed_thresholds)

    comp_x = np.concatenate([entry["x"] for th in seed_thresholds for entry in train_tables[th] if entry["x"].shape[0]], axis=0)
    comp_y = np.concatenate([entry["y"] for th in seed_thresholds for entry in train_tables[th] if entry["y"].shape[0]], axis=0)
    print(f"component train rows={comp_x.shape[0]} pos={int(comp_y.sum())}", flush=True)
    comp_model = fit_component_model(comp_x, comp_y, args, device)

    rows = []
    keep_thresholds = parse_thresholds(args.keep_thresholds)
    min_areas = [int(x) for x in args.min_areas.split(",") if x.strip()]
    max_areas = [int(x) for x in args.max_areas.split(",") if x.strip()]
    for seed_th in seed_thresholds:
        train_scores = [score_components(comp_model, entry["x"], device) for entry in train_tables[seed_th]]
        test_scores = [score_components(comp_model, entry["x"], device) for entry in test_tables[seed_th]]
        for keep_th in keep_thresholds:
            for min_area in min_areas:
                for max_area in max_areas:
                    train_bin = render_with_maps(train_samples, train_maps, train_tables[seed_th], train_scores, seed_th, keep_th, min_area, max_area)
                    train_metrics = metric_from_maps(train_samples, train_bin, 0.5)
                    test_bin = render_with_maps(test_samples, test_maps, test_tables[seed_th], test_scores, seed_th, keep_th, min_area, max_area)
                    test_metrics = metric_from_maps(test_samples, test_bin, 0.5)
                    rows.append(
                        {
                            "seed_threshold": seed_th,
                            "keep_threshold": keep_th,
                            "min_area": min_area,
                            "max_area": max_area,
                            "train": train_metrics,
                            "test": test_metrics,
                        }
                    )
                    print(
                        f"seed={seed_th:.3f} keep={keep_th:.2f} area={min_area}-{max_area} "
                        f"train={train_metrics['mIoU']*100:.4f} test={test_metrics['mIoU']*100:.4f}",
                        flush=True,
                    )
    rows.sort(key=lambda r: r["test"]["mIoU"], reverse=True)
    baselines = []
    for th in seed_thresholds:
        baselines.append({"threshold": th, "test": metric_from_maps(test_samples, test_maps, th)})
        for area in min_areas:
            filtered = []
            for prob in test_maps:
                binary = (prob >= th).astype(np.uint8)
                num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
                out = np.zeros_like(binary)
                for label in range(1, num):
                    if int(stats[label, cv2.CC_STAT_AREA]) >= area:
                        out[labels == label] = 1
                filtered.append(out.astype(np.float32))
            baselines.append({"threshold": th, "min_area": area, "test": metric_from_maps(test_samples, filtered, 0.5)})
    baselines.sort(key=lambda r: r["test"]["mIoU"], reverse=True)
    payload = {
        "args": vars(args),
        "component_rows": int(comp_x.shape[0]),
        "component_positive": int(comp_y.sum()),
        "baseline_best": baselines[0],
        "best": rows[0],
        "rows": rows[:120],
        "baselines": baselines[:40],
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"baseline_best": baselines[0], "best": rows[0]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
