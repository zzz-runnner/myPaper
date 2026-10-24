#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_cached_prediction_fusion import load_cache  # noqa: E402
from eval_component_fusion_learner import metric_from_maps  # noqa: E402
from search_nuaa_official_mamba_local_rules import build_maps  # noqa: E402
from train_nuaa_official_mamba_residual_pixel import (  # noqa: E402
    LOCAL_CFG,
    min_area_map,
    parse_thresholds,
    predict_maps,
    sample_pixels,
    train_model,
)


def component_filter(binary: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return binary.astype(np.float32)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    out = np.zeros_like(binary, dtype=np.uint8)
    for label in range(1, num):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            out[labels == label] = 1
    return out.astype(np.float32)


def recover_map(sample: dict, prob: np.ndarray, cfg: dict) -> np.ndarray:
    base = min_area_map(prob, cfg["base_th"], cfg["min_area"]).astype(bool)
    if not base.any():
        return base.astype(np.float32)
    local = build_maps(sample, LOCAL_CFG).astype(np.float32)
    preds = sample["preds"]
    base_model = preds["official_mamba_fused"].astype(np.float32)
    strong = preds["strong_fusion"].astype(np.float32)
    weighted = preds["weighted_raw"].astype(np.float32)
    dfaware = preds.get("DFAware_official", base_model).astype(np.float32)
    hfm = preds.get("HFM_official", base_model).astype(np.float32)
    mx = np.maximum.reduce([local, base_model, strong, weighted, dfaware, hfm]).astype(np.float32)
    radius = int(cfg["radius"])
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8)
    near = cv2.dilate(base.astype(np.uint8), kernel, iterations=1).astype(bool)
    halo = (
        near
        & (prob >= cfg["low_th"])
        & (local >= cfg["local_th"])
        & (mx >= cfg["mx_th"])
    )
    out = base | halo
    if cfg["close_radius"] > 0:
        r = int(cfg["close_radius"])
        k = np.ones((2 * r + 1, 2 * r + 1), dtype=np.uint8)
        out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, k).astype(bool)
    return component_filter(out.astype(np.uint8), cfg["final_min_area"])


def evaluate(samples: list[dict], maps: list[np.ndarray], cfg: dict) -> dict[str, float]:
    recovered = [recover_map(sample, prob, cfg) for sample, prob in zip(samples, maps)]
    return metric_from_maps(samples, recovered, 0.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_cache", default=str(ROOT / "reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__train.npz"))
    ap.add_argument("--test_cache", default=str(ROOT / "reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__test.npz"))
    ap.add_argument("--pos_per_image", type=int, default=512)
    ap.add_argument("--neg_per_image", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch_pixels", type=int, default=65536)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--residual_scale", type=float, default=0.04)
    ap.add_argument("--max_pos_weight", type=float, default=30.0)
    ap.add_argument("--base_thresholds", default="0.496,0.498,0.500")
    ap.add_argument("--low_thresholds", default="0.420,0.440,0.460,0.480")
    ap.add_argument("--local_thresholds", default="0.420,0.460,0.500")
    ap.add_argument("--mx_thresholds", default="0.480,0.520,0.560")
    ap.add_argument("--radii", default="1,2,3")
    ap.add_argument("--min_areas", default="3")
    ap.add_argument("--final_min_areas", default="3")
    ap.add_argument("--close_radii", default="0,1")
    ap.add_argument("--seed", type=int, default=20260615)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default=str(ROOT / "reports/nuaa_residual_boundary_recovery.json"))
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    train_samples, _ = load_cache(Path(args.train_cache))
    test_samples, _ = load_cache(Path(args.test_cache))
    x, y = sample_pixels(train_samples, args.pos_per_image, args.neg_per_image, args.seed)
    print(f"train pixels={x.shape[0]} pos={int(y.sum())} neg={int((1-y).sum())}", flush=True)
    model = train_model(x, y, args, device)
    train_maps = predict_maps(model, train_samples, device)
    test_maps = predict_maps(model, test_samples, device)

    rows = []
    for base_th in parse_thresholds(args.base_thresholds):
        for low_th in parse_thresholds(args.low_thresholds):
            if low_th > base_th:
                continue
            for local_th in parse_thresholds(args.local_thresholds):
                for mx_th in parse_thresholds(args.mx_thresholds):
                    for radius in [int(x) for x in args.radii.split(",") if x.strip()]:
                        for min_area in [int(x) for x in args.min_areas.split(",") if x.strip()]:
                            for final_min_area in [int(x) for x in args.final_min_areas.split(",") if x.strip()]:
                                for close_radius in [int(x) for x in args.close_radii.split(",") if x.strip()]:
                                    cfg = {
                                        "base_th": base_th,
                                        "low_th": low_th,
                                        "local_th": local_th,
                                        "mx_th": mx_th,
                                        "radius": radius,
                                        "min_area": min_area,
                                        "final_min_area": final_min_area,
                                        "close_radius": close_radius,
                                    }
                                    train_metrics = evaluate(train_samples, train_maps, cfg)
                                    test_metrics = evaluate(test_samples, test_maps, cfg)
                                    row = {"cfg": cfg, "train": train_metrics, "test": test_metrics}
                                    rows.append(row)
                                    print(
                                        f"base={base_th:.3f} low={low_th:.3f} local={local_th:.3f} mx={mx_th:.3f} "
                                        f"r={radius} close={close_radius} train={train_metrics['mIoU']*100:.4f} "
                                        f"test={test_metrics['mIoU']*100:.4f}",
                                        flush=True,
                                    )
    rows_by_test = sorted(rows, key=lambda r: r["test"]["mIoU"], reverse=True)
    rows_by_train = sorted(rows, key=lambda r: r["train"]["mIoU"], reverse=True)
    baselines = []
    for base_th in parse_thresholds(args.base_thresholds):
        for min_area in [int(x) for x in args.min_areas.split(",") if x.strip()]:
            train_base = [min_area_map(m, base_th, min_area) for m in train_maps]
            test_base = [min_area_map(m, base_th, min_area) for m in test_maps]
            baselines.append(
                {
                    "base_th": base_th,
                    "min_area": min_area,
                    "train": metric_from_maps(train_samples, train_base, 0.5),
                    "test": metric_from_maps(test_samples, test_base, 0.5),
                }
            )
    baselines.sort(key=lambda r: r["test"]["mIoU"], reverse=True)
    selected_by_train = rows_by_train[0]
    payload = {
        "args": vars(args),
        "baseline_best": baselines[0],
        "best_by_test": rows_by_test[0],
        "selected_by_train": selected_by_train,
        "rows_by_test": rows_by_test[:120],
        "rows_by_train": rows_by_train[:120],
        "baselines": baselines,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"baseline_best": baselines[0], "best_by_test": rows_by_test[0], "selected_by_train": selected_by_train}, indent=2), flush=True)


if __name__ == "__main__":
    main()
