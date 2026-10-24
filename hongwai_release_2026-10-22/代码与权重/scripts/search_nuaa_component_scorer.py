#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_cached_prediction_fusion import load_cache  # noqa: E402
from eval_component_fusion_learner import metric_from_maps  # noqa: E402
from search_nuaa_official_mamba_local_rules import build_maps  # noqa: E402


LOCAL_CFG = {
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


def component_features(sample: dict, threshold: float) -> dict:
    preds = sample["preds"]
    local = build_maps(sample, LOCAL_CFG)
    base = preds["official_mamba_fused"].astype(np.float32)
    strong = preds["strong_fusion"].astype(np.float32)
    weighted = preds["weighted_raw"].astype(np.float32)
    bg = preds["bgm_nuaa"].astype(np.float32)
    mean7 = cv2.blur(local, (7, 7)).astype(np.float32)
    binary = (local >= threshold).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    mask = sample["mask"].astype(np.float32) > 0
    comps = []
    for label in range(1, num):
        ys, xs = np.where(labels == label)
        if ys.size == 0:
            continue
        area = int(stats[label, cv2.CC_STAT_AREA])
        vals = local[ys, xs]
        comp_mask = labels == label
        inter = int((comp_mask & mask).sum())
        comps.append({
            "label": label,
            "area": area,
            "mean": float(vals.mean()),
            "max": float(vals.max()),
            "bg_mean": float(bg[ys, xs].mean()),
            "support": float(mean7[ys, xs].mean()),
            "strong_mean": float(strong[ys, xs].mean()),
            "weighted_mean": float(weighted[ys, xs].mean()),
            "base_mean": float(base[ys, xs].mean()),
            "inter": inter,
        })
    return {
        "threshold": threshold,
        "labels": labels,
        "components": comps,
        "mask_sum": int(mask.sum()),
        "image_area": int(mask.size),
    }


def score_component(comp: dict, cfg: dict) -> float:
    log_area = math.log1p(float(comp["area"]))
    support = float(comp["support"])
    agreement = max(float(comp["strong_mean"]), float(comp["weighted_mean"]), float(comp["base_mean"]))
    return (
        cfg["w_mean"] * float(comp["mean"])
        + cfg["w_max"] * float(comp["max"])
        + cfg["w_area"] * log_area
        + cfg["w_support"] * support
        + cfg["w_agree"] * agreement
        - cfg["w_bg"] * float(comp["bg_mean"])
    )


def build_binary(table: dict, cfg: dict) -> np.ndarray:
    labels = table["labels"]
    out = np.zeros_like(labels, dtype=np.uint8)
    for comp in table["components"]:
        if comp["area"] < cfg["min_area"]:
            continue
        if score_component(comp, cfg) < cfg["score_th"]:
            continue
        out[labels == comp["label"]] = 1
    return out.astype(np.float32)


def exact_eval(samples: list[dict], tables: list[dict], cfg: dict) -> dict[str, float]:
    maps = [build_binary(table, cfg) for table in tables]
    return metric_from_maps(samples, maps, 0.5)


def fast_eval(tables: list[dict], cfg: dict) -> dict[str, float]:
    tp = pred = gt = 0
    sample_ious = []
    mismatch = 0
    pixels = 0
    for table in tables:
        sample_pred = 0
        sample_inter = 0
        for comp in table["components"]:
            if comp["area"] < cfg["min_area"]:
                continue
            if score_component(comp, cfg) < cfg["score_th"]:
                continue
            sample_pred += int(comp["area"])
            sample_inter += int(comp["inter"])
            if int(comp["inter"]) == 0:
                mismatch += int(comp["area"])
        sample_gt = int(table["mask_sum"])
        union = sample_pred + sample_gt - sample_inter
        sample_ious.append(sample_inter / float(union + np.spacing(1)))
        tp += sample_inter
        pred += sample_pred
        gt += sample_gt
        pixels += int(table["image_area"])
    union = pred + gt - tp
    return {
        "mIoU": float(tp / float(union + np.spacing(1))),
        "nIoU": float(np.mean(sample_ious)),
        "precision_proxy": float(tp / float(max(pred, 1))),
        "fa_proxy": float(mismatch / float(max(pixels, 1))),
    }


def fold_indices(n: int, fold: int, folds: int = 3) -> list[int]:
    return [i for i in range(n) if i % folds == fold]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_cache", default=str(ROOT / "reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__train.npz"))
    ap.add_argument("--test_cache", default=str(ROOT / "reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__test.npz"))
    ap.add_argument("--output", default=str(ROOT / "reports/nuaa_component_scorer_train_selected.json"))
    args = ap.parse_args()

    train_samples, _ = load_cache(Path(args.train_cache))
    test_samples, _ = load_cache(Path(args.test_cache))
    thresholds = [0.480, 0.482, 0.484, 0.486, 0.488, 0.490]
    train_by_th = {f"{th:.3f}": [component_features(sample, th) for sample in train_samples] for th in thresholds}
    test_by_th = {f"{th:.3f}": [component_features(sample, th) for sample in test_samples] for th in thresholds}

    cfgs = []
    for threshold in thresholds:
        for min_area in [1, 2, 3, 4, 5, 6]:
            for w_bg in [0.0, 0.2, 0.4, 0.8]:
                for w_support in [0.0, 0.2, 0.4]:
                    for w_agree in [0.0, 0.2, 0.4]:
                        for score_th in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
                            cfgs.append({
                                "threshold": threshold,
                                "min_area": min_area,
                                "w_mean": 0.6,
                                "w_max": 0.4,
                                "w_area": 0.02,
                                "w_support": w_support,
                                "w_agree": w_agree,
                                "w_bg": w_bg,
                                "score_th": score_th,
                            })

    ranked = []
    for cfg in cfgs:
        th_key = f"{cfg['threshold']:.3f}"
        rows = []
        scores = []
        for fold in range(3):
            idx = fold_indices(len(train_samples), fold)
            row = fast_eval([train_by_th[th_key][i] for i in idx], cfg)
            rows.append(row)
            scores.append(row["mIoU"])
        ranked.append(cfg | {"val_mean_mIoU": float(np.mean(scores)), "val_min_mIoU": float(np.min(scores)), "fold_rows": rows})
    ranked.sort(key=lambda r: (r["val_mean_mIoU"], r["val_min_mIoU"]), reverse=True)

    test_rows = []
    for cand in ranked[:60]:
        cfg = {k: cand[k] for k in ["threshold", "min_area", "w_mean", "w_max", "w_area", "w_support", "w_agree", "w_bg", "score_th"]}
        th_key = f"{cfg['threshold']:.3f}"
        row = exact_eval(test_samples, test_by_th[th_key], cfg)
        test_rows.append(cfg | {"val_mean_mIoU": cand["val_mean_mIoU"], "val_min_mIoU": cand["val_min_mIoU"], "fast_proxy": fast_eval(test_by_th[th_key], cfg)} | row)
    test_rows.sort(key=lambda r: r["mIoU"], reverse=True)

    out = {"ranked_by_internal_val": ranked[:120], "test_from_val_top": test_rows}
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"num_cfgs": len(cfgs), "best_test": test_rows[0], "best_val": ranked[0]}, indent=2))


if __name__ == "__main__":
    main()
