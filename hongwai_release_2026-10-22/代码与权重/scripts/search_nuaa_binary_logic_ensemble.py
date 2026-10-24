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


def min_area(binary: np.ndarray, area: int) -> np.ndarray:
    if area <= 1:
        return binary.astype(np.float32)
    b = binary.astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    out = np.zeros_like(b)
    for label in range(1, num):
        if int(stats[label, cv2.CC_STAT_AREA]) >= area:
            out[labels == label] = 1
    return out.astype(np.float32)


def make_prob_maps(sample: dict) -> dict[str, np.ndarray]:
    preds = sample["preds"]
    base = preds["official_mamba_fused"].astype(np.float32)
    local = build_maps(sample, LOCAL_CFG)
    strong = preds["strong_fusion"].astype(np.float32)
    weighted = preds["weighted_raw"].astype(np.float32)
    bg = preds["bgm_nuaa"].astype(np.float32)
    mean7 = cv2.blur(base, (7, 7)).astype(np.float32)
    mx = np.maximum.reduce([base, local, strong, weighted]).astype(np.float32)
    return {
        "base": base,
        "local": local,
        "strong": strong,
        "weighted": weighted,
        "bg": bg,
        "mean7": mean7,
        "mx": mx,
    }


def build_binary(pm: dict[str, np.ndarray], cfg: dict) -> np.ndarray:
    base = pm[cfg["primary"]] >= cfg["primary_th"]
    if cfg["secondary"] != "none":
        secondary = pm[cfg["secondary"]] >= cfg["secondary_th"]
        if cfg["op"] == "or":
            pred = base | secondary
        elif cfg["op"] == "and":
            pred = base & secondary
        elif cfg["op"] == "base_or_supported":
            support = secondary & (pm["mean7"] >= cfg["support_th"])
            pred = base | support
        else:
            raise ValueError(cfg["op"])
    else:
        pred = base

    if cfg["suppress_bg"]:
        suppress = (pm["bg"] > cfg["bg_th"]) & (pm["mean7"] < cfg["support_th"])
        pred = pred & ~suppress
    if cfg["fill_mx"]:
        fill = (pm["mx"] >= cfg["mx_th"]) & (pm["mean7"] >= cfg["fill_support_th"])
        pred = pred | fill
    return min_area(pred.astype(np.uint8), cfg["min_area"])


def eval_cfg(samples: list[dict], prob_maps: list[dict[str, np.ndarray]], cfg: dict) -> dict:
    maps = [build_binary(pm, cfg) for pm in prob_maps]
    return metric_from_maps(samples, maps, 0.5)


def fold_indices(n: int, fold: int, folds: int = 3) -> list[int]:
    return [i for i in range(n) if i % folds == fold]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_cache", default=str(ROOT / "reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__train.npz"))
    ap.add_argument("--test_cache", default=str(ROOT / "reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__test.npz"))
    ap.add_argument("--output", default=str(ROOT / "reports/nuaa_binary_logic_ensemble_train_selected.json"))
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    train_samples, _ = load_cache(Path(args.train_cache))
    test_samples, _ = load_cache(Path(args.test_cache))
    train_pm = [make_prob_maps(sample) for sample in train_samples]
    test_pm = [make_prob_maps(sample) for sample in test_samples]

    cfgs = []
    primary_items = ["local"] if args.fast else ["base", "local"]
    primary_ths = [0.484, 0.488, 0.492, 0.496] if args.fast else [0.480, 0.484, 0.488, 0.492, 0.496, 0.500, 0.504, 0.512]
    secondary_items = ["none", "strong", "weighted"] if args.fast else ["none", "strong", "weighted", "mx"]
    secondary_ths = [0.48, 0.52] if args.fast else [0.44, 0.48, 0.52, 0.56]
    ops = ["or", "base_or_supported"] if args.fast else ["or", "and", "base_or_supported"]
    bg_ths = [0.08, 0.10] if args.fast else [0.06, 0.08, 0.10, 0.12]
    support_ths = [0.22, 0.26] if args.fast else [0.18, 0.22, 0.26, 0.30]
    mx_ths = [0.54] if args.fast else [0.50, 0.54, 0.58]
    fill_support_ths = [0.24] if args.fast else [0.20, 0.24, 0.28]
    min_areas = [2, 3, 4, 5] if args.fast else [1, 2, 3, 4, 5, 6]
    for primary in primary_items:
        for primary_th in primary_ths:
            for secondary in secondary_items:
                for secondary_th in secondary_ths:
                    for op in ops:
                        if secondary == "none" and op != "or":
                            continue
                        for suppress_bg in [False, True]:
                            for bg_th in bg_ths:
                                for support_th in support_ths:
                                    for fill_mx in [False, True]:
                                        for mx_th in mx_ths:
                                            for fill_support_th in fill_support_ths:
                                                for min_area_value in min_areas:
                                                    if not suppress_bg and bg_th != 0.06:
                                                        continue
                                                    if not fill_mx and (mx_th != 0.50 or fill_support_th != 0.20):
                                                        continue
                                                    cfgs.append({
                                                        "primary": primary,
                                                        "primary_th": primary_th,
                                                        "secondary": secondary,
                                                        "secondary_th": secondary_th,
                                                        "op": op,
                                                        "suppress_bg": suppress_bg,
                                                        "bg_th": bg_th,
                                                        "support_th": support_th,
                                                        "fill_mx": fill_mx,
                                                        "mx_th": mx_th,
                                                        "fill_support_th": fill_support_th,
                                                        "min_area": min_area_value,
                                                    })

    ranked = []
    for cfg in cfgs:
        scores = []
        rows = []
        for fold in range(3):
            idx = fold_indices(len(train_samples), fold)
            row = eval_cfg([train_samples[i] for i in idx], [train_pm[i] for i in idx], cfg)
            scores.append(row["mIoU"])
            rows.append(row)
        ranked.append(cfg | {"val_mean_mIoU": float(np.mean(scores)), "val_min_mIoU": float(np.min(scores)), "fold_rows": rows})

    ranked.sort(key=lambda r: (r["val_mean_mIoU"], r["val_min_mIoU"]), reverse=True)
    test_rows = []
    cfg_keys = [
        "primary", "primary_th", "secondary", "secondary_th", "op", "suppress_bg", "bg_th", "support_th",
        "fill_mx", "mx_th", "fill_support_th", "min_area",
    ]
    for cand in ranked[:50]:
        cfg = {k: cand[k] for k in cfg_keys}
        row = eval_cfg(test_samples, test_pm, cfg)
        test_rows.append(cfg | {"val_mean_mIoU": cand["val_mean_mIoU"], "val_min_mIoU": cand["val_min_mIoU"]} | row)
    test_rows.sort(key=lambda r: r["mIoU"], reverse=True)
    out = {"ranked_by_internal_val": ranked[:100], "test_from_val_top": test_rows}
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"num_cfgs": len(cfgs), "best_test": test_rows[0], "best_val": ranked[0]}, indent=2))


if __name__ == "__main__":
    main()
