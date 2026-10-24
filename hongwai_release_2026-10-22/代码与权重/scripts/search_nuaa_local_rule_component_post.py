#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_cached_prediction_fusion import load_cache  # noqa: E402
from eval_component_fusion_learner import metric_from_maps  # noqa: E402
from search_nuaa_official_mamba_local_rules import build_maps  # noqa: E402


BASE_CFG = {
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
}


def component_post(prob: np.ndarray, th: float, min_area: int, min_mean: float, min_max: float) -> np.ndarray:
    binary = (prob >= th).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros_like(binary)
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        ys, xs = np.where(labels == label)
        vals = prob[ys, xs]
        if float(vals.mean()) < min_mean:
            continue
        if float(vals.max()) < min_max:
            continue
        out[ys, xs] = 1
    return out.astype(np.float32)


def eval_cfg(samples: list[dict], maps: list[np.ndarray], cfg: dict) -> dict:
    out_maps = [component_post(m, cfg["threshold"], cfg["min_area"], cfg["min_mean"], cfg["min_max"]) for m in maps]
    # The maps are already binary, so evaluate at 0.5.
    row = metric_from_maps(samples, out_maps, 0.5)
    return {"output_threshold": cfg["threshold"], **row}


def fold_indices(n: int, fold: int, folds: int) -> list[int]:
    return [i for i in range(n) if i % folds == fold]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_cache", default=str(ROOT / "reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__train.npz"))
    ap.add_argument("--test_cache", default=str(ROOT / "reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__test.npz"))
    ap.add_argument("--output", default=str(ROOT / "reports/nuaa_local_rule_component_post_train_selected.json"))
    args = ap.parse_args()

    train_samples, _ = load_cache(Path(args.train_cache))
    test_samples, _ = load_cache(Path(args.test_cache))
    train_maps = [build_maps(s, BASE_CFG) for s in train_samples]
    test_maps = [build_maps(s, BASE_CFG) for s in test_samples]

    cfgs = []
    for threshold in [0.476, 0.480, 0.484, 0.488, 0.492, 0.496, 0.500]:
        for min_area in [1, 2, 3, 4, 5, 6]:
            for min_mean in [0.0, 0.40, 0.44, 0.48]:
                for min_max in [0.0, 0.50, 0.54, 0.58]:
                    cfgs.append({"threshold": threshold, "min_area": min_area, "min_mean": min_mean, "min_max": min_max})

    ranked = []
    for cfg in cfgs:
        scores = []
        rows = []
        for fold in range(3):
            idx = fold_indices(len(train_samples), fold, 3)
            row = eval_cfg([train_samples[i] for i in idx], [train_maps[i] for i in idx], cfg)
            scores.append(row["mIoU"])
            rows.append(row)
        ranked.append(cfg | {"val_mean_mIoU": float(np.mean(scores)), "val_min_mIoU": float(np.min(scores)), "fold_rows": rows})
    ranked.sort(key=lambda r: (r["val_mean_mIoU"], r["val_min_mIoU"]), reverse=True)
    test_rows = []
    for cand in ranked[:20]:
        cfg = {k: cand[k] for k in ["threshold", "min_area", "min_mean", "min_max"]}
        row = eval_cfg(test_samples, test_maps, cfg)
        test_rows.append(cfg | {"val_mean_mIoU": cand["val_mean_mIoU"], "val_min_mIoU": cand["val_min_mIoU"]} | row)
    test_rows.sort(key=lambda r: r["mIoU"], reverse=True)
    out = {"base_rule": BASE_CFG, "ranked_by_internal_val": ranked[:50], "test_from_val_top": test_rows}
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"best_test": test_rows[0], "best_val": ranked[0]}, indent=2))


if __name__ == "__main__":
    main()
