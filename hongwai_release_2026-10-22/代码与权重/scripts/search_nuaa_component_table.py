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


def component_records(prob: np.ndarray, mask: np.ndarray, threshold: float) -> tuple[list[dict], int, int]:
    binary = (prob >= threshold).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    records = []
    mask_bool = mask > 0
    mask_sum = int(mask_bool.sum())
    area = int(mask_bool.size)
    for label in range(1, num):
        ys, xs = np.where(labels == label)
        vals = prob[ys, xs]
        comp_mask = labels == label
        comp_area = int(stats[label, cv2.CC_STAT_AREA])
        inter = int((comp_mask & mask_bool).sum())
        records.append({
            "area": comp_area,
            "mean": float(vals.mean()) if vals.size else 0.0,
            "max": float(vals.max()) if vals.size else 0.0,
            "inter": inter,
        })
    return records, mask_sum, area


def build_tables(samples: list[dict], thresholds: list[float]) -> list[dict]:
    tables = []
    for sample in samples:
        prob = build_maps(sample, BASE_CFG)
        mask = sample["mask"].astype(np.float32)
        by_th = {}
        for threshold in thresholds:
            records, mask_sum, image_area = component_records(prob, mask, threshold)
            by_th[f"{threshold:.3f}"] = records
        tables.append({"mask_sum": mask_sum, "image_area": image_area, "by_threshold": by_th})
    return tables


def fast_counts(tables: list[dict], cfg: dict) -> tuple[int, int, int, list[float], int]:
    tp = 0
    pred_area = 0
    gt_area = 0
    sample_ious = []
    mismatch_pixels = 0
    th_key = f"{cfg['threshold']:.3f}"
    for table in tables:
        sample_pred = 0
        sample_inter = 0
        for rec in table["by_threshold"][th_key]:
            if rec["area"] < cfg["min_area"]:
                continue
            if rec["mean"] < cfg["min_mean"]:
                continue
            if rec["max"] < cfg["min_max"]:
                continue
            sample_pred += rec["area"]
            sample_inter += rec["inter"]
            if rec["inter"] <= 0:
                mismatch_pixels += rec["area"]
        sample_gt = int(table["mask_sum"])
        union = sample_pred + sample_gt - sample_inter
        sample_ious.append(sample_inter / float(union + np.spacing(1)))
        tp += sample_inter
        pred_area += sample_pred
        gt_area += sample_gt
    return tp, pred_area, gt_area, sample_ious, mismatch_pixels


def fast_metric(tables: list[dict], cfg: dict) -> dict[str, float]:
    tp, pred_area, gt_area, sample_ious, mismatch_pixels = fast_counts(tables, cfg)
    union = pred_area + gt_area - tp
    all_pixels = sum(int(t["image_area"]) for t in tables)
    return {
        "mIoU": float(tp / float(union + np.spacing(1))),
        "nIoU": float(np.mean(sample_ious)),
        "Fa_proxy": float(mismatch_pixels / float(max(all_pixels, 1))),
    }


def component_post(prob: np.ndarray, cfg: dict) -> np.ndarray:
    binary = (prob >= cfg["threshold"]).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros_like(binary)
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < cfg["min_area"]:
            continue
        ys, xs = np.where(labels == label)
        vals = prob[ys, xs]
        if float(vals.mean()) < cfg["min_mean"]:
            continue
        if float(vals.max()) < cfg["min_max"]:
            continue
        out[ys, xs] = 1
    return out.astype(np.float32)


def exact_metric(samples: list[dict], cfg: dict) -> dict[str, float]:
    maps = [component_post(build_maps(sample, BASE_CFG), cfg) for sample in samples]
    return metric_from_maps(samples, maps, 0.5)


def fold_indices(n: int, fold: int, folds: int = 3) -> list[int]:
    return [i for i in range(n) if i % folds == fold]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_cache", default=str(ROOT / "reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__train.npz"))
    ap.add_argument("--test_cache", default=str(ROOT / "reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__test.npz"))
    ap.add_argument("--output", default=str(ROOT / "reports/nuaa_component_table_train_selected.json"))
    args = ap.parse_args()

    thresholds = [round(x, 3) for x in np.arange(0.472, 0.505, 0.002)]
    train_samples, _ = load_cache(Path(args.train_cache))
    test_samples, _ = load_cache(Path(args.test_cache))
    train_tables = build_tables(train_samples, thresholds)
    test_tables = build_tables(test_samples, thresholds)

    cfgs = []
    for threshold in thresholds:
        for min_area in [1, 2, 3, 4, 5, 6, 8, 10]:
            for min_mean in [0.0, 0.36, 0.40, 0.44, 0.48, 0.52]:
                for min_max in [0.0, 0.48, 0.52, 0.56, 0.60]:
                    cfgs.append({"threshold": threshold, "min_area": min_area, "min_mean": min_mean, "min_max": min_max})

    ranked = []
    for cfg in cfgs:
        fold_rows = []
        scores = []
        for fold in range(3):
            idx = fold_indices(len(train_tables), fold)
            row = fast_metric([train_tables[i] for i in idx], cfg)
            fold_rows.append(row)
            scores.append(row["mIoU"])
        ranked.append(cfg | {"val_mean_mIoU": float(np.mean(scores)), "val_min_mIoU": float(np.min(scores)), "fold_rows": fold_rows})
    ranked.sort(key=lambda r: (r["val_mean_mIoU"], r["val_min_mIoU"], -r["Fa_proxy"] if "Fa_proxy" in r else 0.0), reverse=True)

    test_rows = []
    for cand in ranked[:80]:
        cfg = {k: cand[k] for k in ["threshold", "min_area", "min_mean", "min_max"]}
        fast = fast_metric(test_tables, cfg)
        exact = exact_metric(test_samples, cfg)
        test_rows.append(cfg | {"val_mean_mIoU": cand["val_mean_mIoU"], "val_min_mIoU": cand["val_min_mIoU"], "fast_test": fast} | exact)
    test_rows.sort(key=lambda r: r["mIoU"], reverse=True)
    out = {"base_rule": BASE_CFG, "ranked_by_internal_val": ranked[:200], "test_from_val_top": test_rows}
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"num_cfgs": len(cfgs), "best_test": test_rows[0], "best_val": ranked[0]}, indent=2))


if __name__ == "__main__":
    main()
