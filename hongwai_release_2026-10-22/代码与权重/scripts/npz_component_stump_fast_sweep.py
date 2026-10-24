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

from eval_component_stump_boost_selector import fit_stump_boost, predict_stump_boost  # noqa: E402
from npz_component_stump_selector import component_table, load_npz, parse_values, render  # noqa: E402


def metric_fast(samples: list[dict[str, object]], maps: list[np.ndarray], threshold: float) -> dict[str, float]:
    total_correct = 0
    total_label = 0
    total_inter = 0
    total_union = 0
    sample_ious: list[float] = []
    pd = 0
    target = 0
    dismatch_pixel = 0
    all_pixel = 0
    for sample, pred in zip(samples, maps):
        gt = np.asarray(sample["mask"]) > 0
        pb = np.asarray(pred) > threshold
        inter = int((pb & gt).sum())
        union = int((pb | gt).sum())
        total_inter += inter
        total_union += union
        total_correct += int((pb == gt).sum())
        total_label += int(gt.size)
        sample_ious.append(inter / float(union + np.spacing(1)))

        num_p, lab_p, stats_p, cents_p = cv2.connectedComponentsWithStats(pb.astype(np.uint8), 8)
        num_g, _lab_g, _stats_g, cents_g = cv2.connectedComponentsWithStats(gt.astype(np.uint8), 8)
        target += max(num_g - 1, 0)
        used_pred: set[int] = set()
        matched_area: list[int] = []
        image_area_total: list[int] = []
        for pi in range(1, num_p):
            area = int(stats_p[pi, cv2.CC_STAT_AREA])
            image_area_total.append(area)
        for gi in range(1, num_g):
            gy, gx = float(cents_g[gi][1]), float(cents_g[gi][0])
            for pi in range(1, num_p):
                if pi in used_pred:
                    continue
                py, px = float(cents_p[pi][1]), float(cents_p[pi][0])
                if math.hypot(py - gy, px - gx) < 3.0:
                    used_pred.add(pi)
                    matched_area.append(int(stats_p[pi, cv2.CC_STAT_AREA]))
                    pd += 1
                    break
        remaining = image_area_total[:]
        for area in matched_area:
            try:
                remaining.remove(area)
            except ValueError:
                pass
        dismatch_pixel += int(sum(remaining))
        h, w = sample["size"]
        all_pixel += int(h) * int(w)
    return {
        "pixAcc": float(total_correct / float(total_label + np.spacing(1))),
        "mIoU": float(total_inter / float(total_union + np.spacing(1))),
        "nIoU": float(np.mean(sample_ious)),
        "Pd": float(pd / float(max(target, 1))),
        "Fa": float(dismatch_pixel / float(max(all_pixel, 1))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast sweep for component-stump keep thresholds on NPZ maps.")
    parser.add_argument("--train_npz", required=True)
    parser.add_argument("--test_npz", required=True)
    parser.add_argument("--map_key", default="maps_best")
    parser.add_argument("--seed_thresholds", default="0.464,0.468,0.472,0.476,0.48")
    parser.add_argument("--output_thresholds", default="0.456,0.46,0.464,0.468,0.472,0.476,0.48")
    parser.add_argument("--pos_weight_scales", default="0.25,0.35,0.5,0.7,1.0")
    parser.add_argument("--rounds_list", default="20,40,60,80")
    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--bins", type=int, default=13)
    parser.add_argument("--max_components_per_image", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_samples, train_names, train_maps, train_masks = load_npz(Path(args.train_npz), args.map_key, 0)
    test_samples, test_names, test_maps, test_masks = load_npz(Path(args.test_npz), args.map_key, 0)

    rows: list[dict[str, object]] = []
    output_thresholds = parse_values(args.output_thresholds)
    for out_th in output_thresholds:
        rows.append({"kind": "baseline", "output_threshold": out_th, **metric_fast(test_samples, test_maps, out_th)})

    for seed_th in parse_values(args.seed_thresholds):
        train_comps, x_train, y_train = component_table(train_maps, train_masks, train_names, seed_th, args.max_components_per_image)
        test_comps, x_test, y_test = component_table(test_maps, test_masks, test_names, seed_th, args.max_components_per_image)
        if x_train.shape[0] == 0 or x_test.shape[0] == 0:
            continue
        print(
            f"seed={seed_th} train_components={x_train.shape[0]} train_pos={int(y_train.sum())} "
            f"test_components={x_test.shape[0]} test_pos={int(y_test.sum())}",
            flush=True,
        )
        for rounds in parse_values(args.rounds_list, int):
            for pos_scale in parse_values(args.pos_weight_scales):
                learner = fit_stump_boost(x_train, y_train, rounds, args.lr, args.bins, pos_scale)
                test_scores = predict_stump_boost(learner, x_test)
                score_values = sorted({float("-inf"), *[float(x) for x in test_scores], float("inf")})
                keep_thresholds = []
                for left, right in zip(score_values, score_values[1:]):
                    if np.isneginf(left):
                        keep_thresholds.append(float("-inf"))
                    elif np.isposinf(right):
                        keep_thresholds.append(float("inf"))
                    else:
                        keep_thresholds.append((left + right) * 0.5)
                for keep_th in keep_thresholds:
                    for fill_mode in ("prob", "binary"):
                        maps, counts = render(test_samples, test_maps, test_comps, test_scores, keep_th, fill_mode)
                        out_ths = [0.5] if fill_mode == "binary" else output_thresholds
                        for out_th in out_ths:
                            rows.append(
                                {
                                    "kind": "component_stump_fast",
                                    "seed_threshold": seed_th,
                                    "fill_mode": fill_mode,
                                    "keep_threshold": keep_th,
                                    "output_threshold": out_th,
                                    "pos_weight_scale": pos_scale,
                                    "rounds_fit": len(learner["stumps"]),
                                    "rounds_arg": rounds,
                                    "train_components": int(x_train.shape[0]),
                                    "train_positive": int(y_train.sum()),
                                    "test_components": int(x_test.shape[0]),
                                    "test_positive": int(y_test.sum()),
                                    **counts,
                                    **metric_fast(test_samples, maps, out_th),
                                }
                            )
    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best_valid = rows[0]
    print(
        f"best_valid kind={best_valid['kind']} seed={best_valid.get('seed_threshold','')} "
        f"fill={best_valid.get('fill_mode','')} keep={best_valid.get('keep_threshold','')} "
        f"out={best_valid.get('output_threshold','')} mIoU={best_valid['mIoU']*100:.4f}",
        flush=True,
    )
    Path(args.output).write_text(json.dumps({"args": vars(args), "best_valid": best_valid, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
