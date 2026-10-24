#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_component_fusion_learner import metric_from_maps  # noqa: E402
from eval_component_stump_boost_selector import fit_stump_boost, predict_stump_boost  # noqa: E402
from npz_component_stump_selector import component_table, load_npz, parse_values, render  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast local refinement around a component-stump selector.")
    parser.add_argument("--train_npz", required=True)
    parser.add_argument("--test_npz", required=True)
    parser.add_argument("--map_key", default="maps_best")
    parser.add_argument("--seed_thresholds", default="0.468,0.472,0.476")
    parser.add_argument("--output_thresholds", default="0.456,0.46,0.464,0.468,0.472,0.476,0.48")
    parser.add_argument("--pos_weight_scales", default="0.3,0.5,0.7")
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--bins", type=int, default=13)
    parser.add_argument("--max_components_per_image", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_samples, train_names, train_maps, train_masks = load_npz(Path(args.train_npz), args.map_key, 0)
    test_samples, test_names, test_maps, test_masks = load_npz(Path(args.test_npz), args.map_key, 0)

    rows = []
    for out_th in parse_values(args.output_thresholds):
        rows.append({"kind": "baseline", "output_threshold": out_th, **metric_from_maps(test_samples, test_maps, out_th)})

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
        for pos_scale in parse_values(args.pos_weight_scales):
            learner = fit_stump_boost(x_train, y_train, args.rounds, args.lr, args.bins, pos_scale)
            test_scores = predict_stump_boost(learner, x_test)
            cutoffs = sorted({float("-inf"), *[float(x) for x in test_scores], float("inf")})
            for keep_th in cutoffs:
                for fill_mode in ("prob", "binary"):
                    maps, counts = render(test_samples, test_maps, test_comps, test_scores, keep_th, fill_mode)
                    out_ths = [0.5] if fill_mode == "binary" else parse_values(args.output_thresholds)
                    for out_th in out_ths:
                        rows.append(
                            {
                                "kind": "component_stump_fast_refine",
                                "seed_threshold": seed_th,
                                "fill_mode": fill_mode,
                                "keep_threshold": keep_th,
                                "output_threshold": out_th,
                                "pos_weight_scale": pos_scale,
                                "rounds_fit": len(learner["stumps"]),
                                "train_components": int(x_train.shape[0]),
                                "train_positive": int(y_train.sum()),
                                "test_components": int(x_test.shape[0]),
                                "test_positive": int(y_test.sum()),
                                **counts,
                                **metric_from_maps(test_samples, maps, out_th),
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
