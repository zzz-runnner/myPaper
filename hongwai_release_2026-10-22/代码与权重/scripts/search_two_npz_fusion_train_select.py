#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from eval_component_fusion_learner import metric_from_maps
from npz_component_stump_selector import load_npz, parse_values


def min_area_filter_maps(maps: list[np.ndarray], threshold: float, min_area: int) -> list[np.ndarray]:
    out = []
    for prob in maps:
        binary = (prob > threshold).astype(np.uint8)
        if min_area > 1:
            num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
            keep = np.zeros_like(binary)
            for label in range(1, num):
                if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
                    keep[labels == label] = 1
            binary = keep
        out.append(binary.astype(np.float32))
    return out


def fuse_maps(a_maps, b_maps, alpha: float, gamma: float):
    out = []
    for a, b in zip(a_maps, b_maps):
        out.append(np.clip((1.0 - alpha) * a + alpha * b, 0.0, 1.0).astype(np.float32) ** gamma)
    return out


def eval_grid(samples, a_maps, b_maps, alphas, gammas, thresholds, min_areas):
    rows = []
    for alpha in alphas:
        for gamma in gammas:
            maps = fuse_maps(a_maps, b_maps, alpha, gamma)
            for th in thresholds:
                rows.append({"alpha": alpha, "gamma": gamma, "threshold": th, "min_area": 1, **metric_from_maps(samples, maps, th)})
                for area in min_areas:
                    if area <= 1:
                        continue
                    filtered = min_area_filter_maps(maps, th, area)
                    rows.append({"alpha": alpha, "gamma": gamma, "threshold": th, "min_area": area, **metric_from_maps(samples, filtered, 0.5)})
    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Select two-map fusion params on train and apply top candidates to test.")
    parser.add_argument("--train_a", required=True)
    parser.add_argument("--train_b", required=True)
    parser.add_argument("--test_a", required=True)
    parser.add_argument("--test_b", required=True)
    parser.add_argument("--key_a", default="maps_best")
    parser.add_argument("--key_b", default="maps_candidate")
    parser.add_argument("--alphas", default="-0.20,-0.12,-0.08,-0.04,0,0.04,0.08,0.12,0.20")
    parser.add_argument("--gammas", default="0.85,0.90,0.94,1.0,1.06,1.12")
    parser.add_argument("--thresholds", default="0.42,0.44,0.456,0.464,0.468,0.472,0.476,0.48,0.488,0.496,0.504")
    parser.add_argument("--min_areas", default="1,2,3,4,6")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_samples, _train_names, train_a, _train_masks = load_npz(Path(args.train_a), args.key_a, 0)
    _train_samples_b, _train_names_b, train_b, _train_masks_b = load_npz(Path(args.train_b), args.key_b, 0)
    test_samples, _test_names, test_a, _test_masks = load_npz(Path(args.test_a), args.key_a, 0)
    _test_samples_b, _test_names_b, test_b, _test_masks_b = load_npz(Path(args.test_b), args.key_b, 0)

    train_rows = eval_grid(
        train_samples,
        train_a,
        train_b,
        parse_values(args.alphas),
        parse_values(args.gammas),
        parse_values(args.thresholds),
        parse_values(args.min_areas, int),
    )
    test_rows = []
    for cand in train_rows[: args.top_k]:
        alpha = float(cand["alpha"])
        gamma = float(cand["gamma"])
        threshold = float(cand["threshold"])
        min_area = int(cand["min_area"])
        maps = fuse_maps(test_a, test_b, alpha, gamma)
        if min_area > 1:
            maps = min_area_filter_maps(maps, threshold, min_area)
            metric = metric_from_maps(test_samples, maps, 0.5)
        else:
            metric = metric_from_maps(test_samples, maps, threshold)
        test_rows.append({"selected_train_mIoU": cand["mIoU"], "alpha": alpha, "gamma": gamma, "threshold": threshold, "min_area": min_area, **metric})
    test_rows.sort(key=lambda row: row["mIoU"], reverse=True)
    print(
        f"best_train mIoU={train_rows[0]['mIoU']*100:.4f} alpha={train_rows[0]['alpha']} "
        f"test_selected_best mIoU={test_rows[0]['mIoU']*100:.4f} alpha={test_rows[0]['alpha']}",
        flush=True,
    )
    Path(args.output).write_text(
        json.dumps({"args": vars(args), "train_top": train_rows[: args.top_k], "test_from_train_top": test_rows}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
