#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from npz_component_stump_selector import load_npz, parse_values
from search_two_npz_fusion_train_select import eval_grid, fuse_maps, min_area_filter_maps
from eval_component_fusion_learner import metric_from_maps


def subset(items, indices):
    return [items[i] for i in indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Select two-map fusion params on an internal train validation split.")
    parser.add_argument("--train_a", required=True)
    parser.add_argument("--train_b", required=True)
    parser.add_argument("--test_a", required=True)
    parser.add_argument("--test_b", required=True)
    parser.add_argument("--key_a", default="maps_best")
    parser.add_argument("--key_b", default="maps_candidate")
    parser.add_argument("--alphas", default="-0.08,-0.04,0,0.04,0.08")
    parser.add_argument("--gammas", default="0.90,0.94,1.0")
    parser.add_argument("--thresholds", default="0.456,0.464,0.468,0.472")
    parser.add_argument("--min_areas", default="1,2")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=12)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_samples, _names_a, train_a, _masks_a = load_npz(Path(args.train_a), args.key_a, 0)
    _samples_b, _names_b, train_b, _masks_b = load_npz(Path(args.train_b), args.key_b, 0)
    test_samples, _test_names_a, test_a, _test_masks_a = load_npz(Path(args.test_a), args.key_a, 0)
    _test_samples_b, _test_names_b, test_b, _test_masks_b = load_npz(Path(args.test_b), args.key_b, 0)

    n = len(train_samples)
    alphas = parse_values(args.alphas)
    gammas = parse_values(args.gammas)
    thresholds = parse_values(args.thresholds)
    min_areas = parse_values(args.min_areas, int)
    fold_rows = []
    param_scores: dict[tuple[float, float, float, int], list[float]] = {}
    for fold in range(args.folds):
        val_idx = [i for i in range(n) if i % args.folds == fold]
        fit_idx = [i for i in range(n) if i % args.folds != fold]
        fit_rows = eval_grid(
            subset(train_samples, fit_idx),
            subset(train_a, fit_idx),
            subset(train_b, fit_idx),
            alphas,
            gammas,
            thresholds,
            min_areas,
        )
        for cand in fit_rows[: args.top_k]:
            key = (float(cand["alpha"]), float(cand["gamma"]), float(cand["threshold"]), int(cand["min_area"]))
            maps = fuse_maps(subset(train_a, val_idx), subset(train_b, val_idx), key[0], key[1])
            if key[3] > 1:
                maps = min_area_filter_maps(maps, key[2], key[3])
                metric = metric_from_maps(subset(train_samples, val_idx), maps, 0.5)
            else:
                metric = metric_from_maps(subset(train_samples, val_idx), maps, key[2])
            row = {"fold": fold, "fit_mIoU": cand["mIoU"], "alpha": key[0], "gamma": key[1], "threshold": key[2], "min_area": key[3], **metric}
            fold_rows.append(row)
            param_scores.setdefault(key, []).append(metric["mIoU"])
    ranked = []
    for key, vals in param_scores.items():
        ranked.append({"alpha": key[0], "gamma": key[1], "threshold": key[2], "min_area": key[3], "folds": len(vals), "val_mean_mIoU": float(np.mean(vals)), "val_min_mIoU": float(np.min(vals))})
    ranked.sort(key=lambda row: (row["folds"], row["val_mean_mIoU"], row["val_min_mIoU"]), reverse=True)

    test_rows = []
    for cand in ranked[: args.top_k]:
        maps = fuse_maps(test_a, test_b, float(cand["alpha"]), float(cand["gamma"]))
        if int(cand["min_area"]) > 1:
            maps = min_area_filter_maps(maps, float(cand["threshold"]), int(cand["min_area"]))
            metric = metric_from_maps(test_samples, maps, 0.5)
        else:
            metric = metric_from_maps(test_samples, maps, float(cand["threshold"]))
        test_rows.append({**cand, **metric})
    test_rows.sort(key=lambda row: row["mIoU"], reverse=True)
    print(
        f"best_val alpha={ranked[0]['alpha']} val={ranked[0]['val_mean_mIoU']*100:.4f} "
        f"best_test_from_val_top alpha={test_rows[0]['alpha']} test={test_rows[0]['mIoU']*100:.4f}",
        flush=True,
    )
    Path(args.output).write_text(json.dumps({"args": vars(args), "ranked_by_internal_val": ranked, "fold_rows": fold_rows, "test_from_val_top": test_rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
