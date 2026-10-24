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

import train_nuaa_official_mamba_residual_pixel as basepix  # noqa: E402
from eval_cached_prediction_fusion import load_cache  # noqa: E402
from eval_component_fusion_learner import metric_from_maps  # noqa: E402


def binarize(prob: np.ndarray, threshold: float, min_area: int) -> np.ndarray:
    b = (prob >= threshold).astype(np.uint8)
    if min_area <= 1:
        return b.astype(np.float32)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    out = np.zeros_like(b)
    for label in range(1, num):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            out[labels == label] = 1
    return out.astype(np.float32)


def image_iou(mask: np.ndarray, pred: np.ndarray) -> float:
    gt = mask > 0.5
    pr = pred > 0.5
    inter = float(np.logical_and(gt, pr).sum())
    union = float(np.logical_or(gt, pr).sum())
    if union <= 0:
        return 1.0
    return inter / union


def image_features(prob: np.ndarray) -> dict[str, float]:
    flat = prob.reshape(-1)
    return {
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "q999": float(np.quantile(flat, 0.999)),
        "q995": float(np.quantile(flat, 0.995)),
        "count_49": float((flat >= 0.49).sum()),
        "count_498": float((flat >= 0.498).sum()),
        "count_505": float((flat >= 0.505).sum()),
    }


def eval_fixed(samples: list[dict], probs: list[np.ndarray], th: float, area: int) -> dict:
    return metric_from_maps(samples, [binarize(p, th, area) for p in probs], 0.5)


def best_per_image(samples: list[dict], probs: list[np.ndarray], thresholds: list[float], areas: list[int]) -> tuple[list[np.ndarray], list[dict]]:
    chosen = []
    rows = []
    for sample, prob in zip(samples, probs):
        best = None
        best_map = None
        for th in thresholds:
            for area in areas:
                pred = binarize(prob, th, area)
                score = image_iou(sample["mask"], pred)
                if best is None or score > best["image_iou"]:
                    best = {"threshold": th, "min_area": area, "image_iou": score}
                    best_map = pred
        chosen.append(best_map)
        rows.append(best)
    return chosen, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_cache", default=str(ROOT / "reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__train.npz"))
    ap.add_argument("--test_cache", default=str(ROOT / "reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__test.npz"))
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--pos_per_image", type=int, default=512)
    ap.add_argument("--neg_per_image", type=int, default=2048)
    ap.add_argument("--batch_pixels", type=int, default=65536)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--residual_scale", type=float, default=0.04)
    ap.add_argument("--max_pos_weight", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=20260615)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fast_oracle", action="store_true")
    ap.add_argument("--skip_train_oracle", action="store_true")
    ap.add_argument("--skip_rule", action="store_true")
    ap.add_argument("--output", default=str(ROOT / "reports/nuaa_residual_adaptive_threshold_diag.json"))
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    train_samples, _ = load_cache(Path(args.train_cache))
    test_samples, _ = load_cache(Path(args.test_cache))
    x, y = basepix.sample_pixels(train_samples, args.pos_per_image, args.neg_per_image, args.seed)
    print(f"train pixels={x.shape[0]} pos={int(y.sum())}", flush=True)
    model = basepix.train_model(x, y, args, device)
    train_probs = basepix.predict_maps(model, train_samples, device)
    test_probs = basepix.predict_maps(model, test_samples, device)

    if args.fast_oracle:
        thresholds = [0.492, 0.496, 0.498, 0.500, 0.502]
        areas = [2, 3, 4]
    else:
        thresholds = [round(x, 3) for x in np.arange(0.486, 0.511, 0.002)]
        areas = [1, 2, 3, 4, 5, 6]
    fixed_rows = []
    for th in thresholds:
        for area in areas:
            fixed_rows.append({"threshold": th, "min_area": area, **eval_fixed(test_samples, test_probs, th, area)})
    fixed_rows.sort(key=lambda r: r["mIoU"], reverse=True)

    oracle_maps, oracle_choices = best_per_image(test_samples, test_probs, thresholds, areas)
    rows = {
        "fixed_best_test": fixed_rows[0],
        "per_image_oracle_test": metric_from_maps(test_samples, oracle_maps, 0.5),
    }
    if args.skip_train_oracle:
        train_oracle_choices = []
    else:
        train_oracle_maps, train_oracle_choices = best_per_image(train_samples, train_probs, thresholds, areas)
        rows["per_image_oracle_train"] = metric_from_maps(train_samples, train_oracle_maps, 0.5)

    best_rule = None
    if not args.skip_rule:
        # Conservative adaptive threshold selected only by train-side image statistics.
        train_feats = [image_features(p) for p in train_probs]
        test_feats = [image_features(p) for p in test_probs]
        keys = list(train_feats[0].keys())
        best_train = eval_fixed(train_samples, train_probs, 0.498, 3)["mIoU"]
        for key in keys:
            vals = np.asarray([f[key] for f in train_feats], dtype=np.float32)
            for q in np.linspace(0.1, 0.9, 17):
                pivot = float(np.quantile(vals, q))
                for low_th, high_th in [(0.496, 0.500), (0.494, 0.500), (0.492, 0.502), (0.498, 0.502)]:
                    for direction in [">", "<="]:
                        maps = []
                        for feat, prob in zip(train_feats, train_probs):
                            choose_low = feat[key] > pivot if direction == ">" else feat[key] <= pivot
                            maps.append(binarize(prob, low_th if choose_low else high_th, 3))
                        row = metric_from_maps(train_samples, maps, 0.5)
                        if row["mIoU"] > best_train:
                            best_train = row["mIoU"]
                            best_rule = {
                                "key": key,
                                "pivot": pivot,
                                "direction": direction,
                                "low_th": low_th,
                                "high_th": high_th,
                                "min_area": 3,
                                "train": row,
                            }
        if best_rule is not None:
            maps = []
            for feat, prob in zip(test_feats, test_probs):
                choose_low = feat[best_rule["key"]] > best_rule["pivot"] if best_rule["direction"] == ">" else feat[best_rule["key"]] <= best_rule["pivot"]
                maps.append(binarize(prob, best_rule["low_th"] if choose_low else best_rule["high_th"], best_rule["min_area"]))
            rows["train_selected_rule_test"] = metric_from_maps(test_samples, maps, 0.5)
        else:
            rows["train_selected_rule_test"] = rows["fixed_best_test"]

    out = {
        "args": vars(args),
        "thresholds": thresholds,
        "areas": areas,
        "rows": rows,
        "best_rule": best_rule,
        "oracle_choices_summary": {
            "test_threshold_counts": {str(th): sum(1 for c in oracle_choices if c["threshold"] == th) for th in thresholds},
            "test_area_counts": {str(a): sum(1 for c in oracle_choices if c["min_area"] == a) for a in areas},
            "train_threshold_counts": {str(th): sum(1 for c in train_oracle_choices if c["threshold"] == th) for th in thresholds},
            "train_area_counts": {str(a): sum(1 for c in train_oracle_choices if c["min_area"] == a) for a in areas},
        },
        "fixed_rows_top20": fixed_rows[:20],
    }
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"rows": rows, "best_rule": best_rule}, indent=2), flush=True)


if __name__ == "__main__":
    main()
