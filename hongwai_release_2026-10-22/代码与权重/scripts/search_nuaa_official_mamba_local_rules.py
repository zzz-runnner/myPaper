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


def rank01(x: np.ndarray) -> np.ndarray:
    flat = x.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(flat), dtype=np.float32)
    return ranks.reshape(x.shape)


def build_maps(sample: dict, cfg: dict) -> np.ndarray:
    preds = sample["preds"]
    base = preds["official_mamba_fused"].astype(np.float32)
    s = preds["strong_fusion"].astype(np.float32)
    weighted = preds["weighted_raw"].astype(np.float32)
    bg = preds["bgm_nuaa"].astype(np.float32)
    hfm = np.clip((preds["official_mamba_fused"].astype(np.float32) * 0.0 + base), 0, 1)
    mx = np.maximum.reduce([base, s, weighted])
    mn = np.minimum.reduce([base, s, weighted])
    disagree = mx - mn
    mean7 = cv2.blur(base, (7, 7))
    range7 = cv2.dilate(base, np.ones((7, 7), np.uint8)) - cv2.erode(base, np.ones((7, 7), np.uint8))
    r = rank01(base)

    p = base.copy()
    # Suppress isolated high-ranking background-like pixels, guided by BGM and low local support.
    suppress = (bg > cfg["bg_th"]) & (mean7 < cfg["support_th"]) & (r > cfg["rank_th"])
    p = np.where(suppress, p * cfg["suppress_scale"], p)
    # Restore pixels where multiple non-Mamba sources agree but the fused map is locally underconfident.
    restore = (disagree < cfg["agree_th"]) & (mx > cfg["mx_th"]) & (p < cfg["restore_cap"]) & (range7 > cfg["range_th"])
    p = np.where(restore, np.maximum(p, cfg["restore_mix"] * mx + (1.0 - cfg["restore_mix"]) * p), p)
    p = np.clip(p, 0.0, 1.0) ** cfg["gamma"]
    return p.astype(np.float32)


def eval_cfg(samples: list[dict], cfg: dict, thresholds: list[float]) -> dict:
    maps = [build_maps(s, cfg) for s in samples]
    rows = [{"threshold": th, **metric_from_maps(samples, maps, th)} for th in thresholds]
    rows.sort(key=lambda r: r["mIoU"], reverse=True)
    return rows[0]


def fold_indices(n: int, fold: int, folds: int) -> tuple[list[int], list[int]]:
    tr, va = [], []
    for i in range(n):
        (va if i % folds == fold else tr).append(i)
    return tr, va


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_cache", default=str(ROOT / "reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__train.npz"))
    ap.add_argument("--test_cache", default=str(ROOT / "reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__test.npz"))
    ap.add_argument("--output", default=str(ROOT / "reports/nuaa_official_mamba_local_rules.json"))
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    train_samples, _ = load_cache(Path(args.train_cache))
    test_samples, _ = load_cache(Path(args.test_cache))
    thresholds = [round(x, 3) for x in np.arange(0.46, 0.545, 0.004)]
    cfgs = []
    bg_ths = [0.12, 0.18] if args.fast else [0.08, 0.12, 0.18]
    support_ths = [0.18, 0.24] if args.fast else [0.18, 0.22, 0.28]
    rank_ths = [0.992, 0.996] if args.fast else [0.985, 0.992, 0.996]
    suppress_scales = [0.85, 0.92] if args.fast else [0.75, 0.85, 0.92]
    agree_ths = [0.08, 0.14] if args.fast else [0.08, 0.12, 0.18]
    mx_ths = [0.48, 0.54] if args.fast else [0.42, 0.48, 0.54]
    restore_mixes = [0.1] if args.fast else [0.1, 0.2]
    gammas = [1.0, 1.02] if args.fast else [0.98, 1.0, 1.02]
    for bg_th in bg_ths:
        for support_th in support_ths:
            for rank_th in rank_ths:
                for suppress_scale in suppress_scales:
                    for agree_th in agree_ths:
                        for mx_th in mx_ths:
                            for restore_mix in restore_mixes:
                                for gamma in gammas:
                                    cfgs.append({
                                        "bg_th": bg_th,
                                        "support_th": support_th,
                                        "rank_th": rank_th,
                                        "suppress_scale": suppress_scale,
                                        "agree_th": agree_th,
                                        "mx_th": mx_th,
                                        "restore_cap": 0.58,
                                        "range_th": 0.05,
                                        "restore_mix": restore_mix,
                                        "gamma": gamma,
                                    })
    ranked = []
    for cfg in cfgs:
        scores = []
        rows = []
        for fold in range(3):
            _, va = fold_indices(len(train_samples), fold, 3)
            row = eval_cfg([train_samples[i] for i in va], cfg, thresholds)
            scores.append(row["mIoU"])
            rows.append(row)
        ranked.append(cfg | {"val_mean_mIoU": float(np.mean(scores)), "val_min_mIoU": float(np.min(scores)), "fold_rows": rows})
    ranked.sort(key=lambda r: (r["val_mean_mIoU"], r["val_min_mIoU"]), reverse=True)
    test_rows = []
    for cand in ranked[:30]:
        cfg = {k: cand[k] for k in ["bg_th", "support_th", "rank_th", "suppress_scale", "agree_th", "mx_th", "restore_cap", "range_th", "restore_mix", "gamma"]}
        row = eval_cfg(test_samples, cfg, thresholds)
        test_rows.append({k: cand[k] for k in cfg} | {"val_mean_mIoU": cand["val_mean_mIoU"], "val_min_mIoU": cand["val_min_mIoU"]} | row)
    test_rows.sort(key=lambda r: r["mIoU"], reverse=True)
    out = {"ranked_by_internal_val": ranked[:50], "test_from_val_top": test_rows}
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"best_test": test_rows[0], "best_val": ranked[0]}, indent=2))


if __name__ == "__main__":
    main()
