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
from search_nuaa_official_mamba_local_rules import rank01  # noqa: E402


def precompute(sample: dict) -> dict[str, np.ndarray]:
    preds = sample["preds"]
    base = preds["official_mamba_fused"].astype(np.float32)
    s = preds["strong_fusion"].astype(np.float32)
    weighted = preds["weighted_raw"].astype(np.float32)
    bg = preds["bgm_nuaa"].astype(np.float32)
    mx = np.maximum.reduce([base, s, weighted])
    mn = np.minimum.reduce([base, s, weighted])
    mean7 = cv2.blur(base, (7, 7))
    mean11 = cv2.blur(base, (11, 11))
    return {
        "base": base,
        "bg": bg,
        "mx": mx,
        "disagree": mx - mn,
        "rank": rank01(base),
        "mean7": mean7.astype(np.float32),
        "mean11": mean11.astype(np.float32),
    }


def build(pc: dict[str, np.ndarray], cfg: dict) -> np.ndarray:
    p = pc["base"].copy()
    support = pc[cfg["support_key"]]
    suppress = (pc["bg"] > cfg["bg_th"]) & (support < cfg["support_th"]) & (pc["rank"] > cfg["rank_th"])
    p = np.where(suppress, p * cfg["suppress_scale"], p)
    if cfg["boost_mix"] > 0.0:
        boost = (pc["disagree"] < cfg["agree_th"]) & (pc["mx"] > cfg["mx_th"]) & (p < cfg["boost_cap"])
        p = np.where(boost, np.maximum(p, cfg["boost_mix"] * pc["mx"] + (1.0 - cfg["boost_mix"]) * p), p)
    return (np.clip(p, 0.0, 1.0) ** cfg["gamma"]).astype(np.float32)


def min_area_map(prob: np.ndarray, th: float, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return (prob >= th).astype(np.float32)
    b = (prob >= th).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b, 8)
    out = np.zeros_like(b)
    for label in range(1, n):
        if int(st[label, cv2.CC_STAT_AREA]) >= min_area:
            out[lab == label] = 1
    return out.astype(np.float32)


def eval_maps(samples: list[dict], pcs: list[dict[str, np.ndarray]], cfg: dict) -> dict:
    maps = [min_area_map(build(pc, cfg), cfg["threshold"], cfg["min_area"]) for pc in pcs]
    row = metric_from_maps(samples, maps, 0.5)
    return {"output_threshold": cfg["threshold"], **row}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_cache", default=str(ROOT / "reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__train.npz"))
    ap.add_argument("--test_cache", default=str(ROOT / "reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__test.npz"))
    ap.add_argument("--output", default=str(ROOT / "reports/nuaa_local_rule_precomputed_train_selected.json"))
    args = ap.parse_args()
    train_samples, _ = load_cache(Path(args.train_cache))
    test_samples, _ = load_cache(Path(args.test_cache))
    train_pc = [precompute(s) for s in train_samples]
    test_pc = [precompute(s) for s in test_samples]
    cfgs = []
    for support_key in ["mean7", "mean11"]:
        for bg_th in [0.04, 0.08, 0.12]:
            for support_th in [0.18, 0.22, 0.26, 0.30]:
                for rank_th in [0.980, 0.985, 0.992]:
                    for suppress_scale in [0.80, 0.85, 0.90, 0.95]:
                        for gamma in [0.98, 1.0, 1.02]:
                            for threshold in [0.480, 0.484, 0.488, 0.492, 0.496]:
                                for min_area in [1, 2, 3, 4]:
                                    cfgs.append({
                                        "support_key": support_key,
                                        "bg_th": bg_th,
                                        "support_th": support_th,
                                        "rank_th": rank_th,
                                        "suppress_scale": suppress_scale,
                                        "boost_mix": 0.0,
                                        "agree_th": 0.1,
                                        "mx_th": 0.5,
                                        "boost_cap": 0.58,
                                        "gamma": gamma,
                                        "threshold": threshold,
                                        "min_area": min_area,
                                    })
    ranked = []
    for cfg in cfgs:
        scores = []
        rows = []
        for fold in range(3):
            idx = [i for i in range(len(train_samples)) if i % 3 == fold]
            row = eval_maps([train_samples[i] for i in idx], [train_pc[i] for i in idx], cfg)
            scores.append(row["mIoU"])
            rows.append(row)
        ranked.append(cfg | {"val_mean_mIoU": float(np.mean(scores)), "val_min_mIoU": float(np.min(scores)), "fold_rows": rows})
    ranked.sort(key=lambda r: (r["val_mean_mIoU"], r["val_min_mIoU"]), reverse=True)
    test_rows = []
    for cand in ranked[:30]:
        cfg = {k: cand[k] for k in [
            "support_key", "bg_th", "support_th", "rank_th", "suppress_scale", "boost_mix", "agree_th", "mx_th",
            "boost_cap", "gamma", "threshold", "min_area"
        ]}
        row = eval_maps(test_samples, test_pc, cfg)
        test_rows.append(cfg | {"val_mean_mIoU": cand["val_mean_mIoU"], "val_min_mIoU": cand["val_min_mIoU"]} | row)
    test_rows.sort(key=lambda r: r["mIoU"], reverse=True)
    Path(args.output).write_text(json.dumps({"ranked_by_internal_val": ranked[:100], "test_from_val_top": test_rows}, indent=2), encoding="utf-8")
    print(json.dumps({"best_test": test_rows[0], "best_val": ranked[0]}, indent=2))


if __name__ == "__main__":
    main()
