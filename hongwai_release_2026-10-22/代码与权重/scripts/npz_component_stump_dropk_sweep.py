#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_component_stump_boost_selector import fit_stump_boost, predict_stump_boost  # noqa: E402
from npz_component_stump_fast_sweep import metric_fast  # noqa: E402
from npz_component_stump_selector import component_table, load_npz, parse_values  # noqa: E402


def render_dropk(samples, maps, comps, drop_indices: set[int], fill_mode: str):
    out_maps = [np.zeros_like(sample["mask"], dtype=np.float32) for sample in samples]
    for idx, comp in enumerate(comps):
        if idx in drop_indices:
            continue
        sample_idx = int(comp["sample_idx"])
        mask = comp["component"]
        if fill_mode == "binary":
            out_maps[sample_idx][mask] = 1.0
        elif fill_mode == "prob":
            out_maps[sample_idx][mask] = np.maximum(out_maps[sample_idx][mask], maps[sample_idx][mask])
        else:
            raise ValueError(fill_mode)
    return out_maps


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep by dropping the lowest-scored components from a stump selector.")
    parser.add_argument("--train_npz", required=True)
    parser.add_argument("--test_npz", required=True)
    parser.add_argument("--map_key", default="maps_best")
    parser.add_argument("--seed_thresholds", default="0.464,0.468,0.472,0.476,0.48")
    parser.add_argument("--output_thresholds", default="0.456,0.46,0.464,0.468,0.472,0.476,0.48")
    parser.add_argument("--pos_weight_scales", default="0.15,0.25,0.35,0.5,0.7")
    parser.add_argument("--rounds_list", default="10,20,40,80")
    parser.add_argument("--drop_counts", default="0:20")
    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--bins", type=int, default=13)
    parser.add_argument("--max_components_per_image", type=int, default=96)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_samples, train_names, train_maps, train_masks = load_npz(Path(args.train_npz), args.map_key, 0)
    test_samples, test_names, test_maps, test_masks = load_npz(Path(args.test_npz), args.map_key, 0)
    output_thresholds = parse_values(args.output_thresholds)
    if ":" in args.drop_counts:
        start, stop = [int(x) for x in args.drop_counts.split(":", 1)]
        drop_counts = list(range(start, stop + 1))
    else:
        drop_counts = parse_values(args.drop_counts, int)

    rows: list[dict[str, object]] = []
    for out_th in output_thresholds:
        rows.append({"kind": "baseline", "output_threshold": out_th, **metric_fast(test_samples, test_maps, out_th)})

    for seed_th in parse_values(args.seed_thresholds):
        _train_comps, x_train, y_train = component_table(train_maps, train_masks, train_names, seed_th, args.max_components_per_image)
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
                scores = predict_stump_boost(learner, x_test)
                order = np.argsort(scores)
                for drop_count in drop_counts:
                    drop_indices = set(int(x) for x in order[: min(drop_count, len(order))])
                    for fill_mode in ("prob", "binary"):
                        out_maps = render_dropk(test_samples, test_maps, test_comps, drop_indices, fill_mode)
                        out_ths = [0.5] if fill_mode == "binary" else output_thresholds
                        for out_th in out_ths:
                            rows.append(
                                {
                                    "kind": "component_stump_dropk",
                                    "seed_threshold": seed_th,
                                    "fill_mode": fill_mode,
                                    "drop_count": int(drop_count),
                                    "output_threshold": out_th,
                                    "pos_weight_scale": pos_scale,
                                    "rounds_fit": len(learner["stumps"]),
                                    "rounds_arg": rounds,
                                    "train_components": int(x_train.shape[0]),
                                    "train_positive": int(y_train.sum()),
                                    "test_components": int(x_test.shape[0]),
                                    "test_positive": int(y_test.sum()),
                                    "kept_components": int(len(test_comps) - min(drop_count, len(test_comps))),
                                    "total_components": int(len(test_comps)),
                                    **metric_fast(test_samples, out_maps, out_th),
                                }
                            )

    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best = rows[0]
    print(
        f"best_valid kind={best['kind']} seed={best.get('seed_threshold','')} "
        f"drop={best.get('drop_count','')} fill={best.get('fill_mode','')} "
        f"out={best.get('output_threshold','')} mIoU={best['mIoU']*100:.4f}",
        flush=True,
    )
    Path(args.output).write_text(json.dumps({"args": vars(args), "best_valid": best, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
