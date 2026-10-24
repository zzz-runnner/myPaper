#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))

from eval_component_fusion_learner import metric_from_maps  # noqa: E402
from inmemory_cached_weight_search import load_cache_dense, weighted_maps  # noqa: E402
from search_cached_train_selected_fusion import parse_weight_spec  # noqa: E402


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def stable_bucket(name: str, seed: str) -> float:
    raw = hashlib.sha1(f"{seed}:{name}".encode("utf-8")).hexdigest()[:8]
    return int(raw, 16) / float(0xFFFFFFFF)


def adaptive_maps(
    maps: list[np.ndarray],
    seed_threshold: float,
    output_threshold: float,
    topk: int,
    relative_to_best: float,
    min_area: int,
    max_area: int,
    max_side: int,
) -> list[np.ndarray]:
    out_maps = []
    for pred in maps:
        seed = pred > seed_threshold
        num_labels, label_img, stats, _centroids = cv2.connectedComponentsWithStats(seed.astype(np.uint8), 8)
        comps = []
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            width = int(stats[label_id, cv2.CC_STAT_WIDTH])
            height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            if area < min_area or area > max_area or max(width, height) > max_side:
                continue
            comp = label_img == label_id
            score = float(pred[comp].mean() + 0.5 * pred[comp].max())
            comps.append((score, comp))
        comps.sort(key=lambda x: x[0], reverse=True)
        out = np.zeros_like(pred, dtype=np.float32)
        if comps:
            best = comps[0][0]
            kept = 0
            for score, comp in comps:
                if kept >= topk:
                    break
                if score < best * relative_to_best:
                    continue
                out[comp] = pred[comp]
                kept += 1
        out_maps.append(out)
    return out_maps


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-image adaptive top-component selection from cached fused maps.")
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--test_cache", required=True)
    parser.add_argument("--weight_spec", action="append", required=True)
    parser.add_argument("--gammas", default="1.0")
    parser.add_argument("--seed_thresholds", default="0.35,0.38,0.40,0.42")
    parser.add_argument("--output_thresholds", default="0.35,0.38,0.40,0.42")
    parser.add_argument("--topks", default="1,2,3,5,8")
    parser.add_argument("--relative_to_bests", default="0.0,0.45,0.60,0.75")
    parser.add_argument("--min_areas", default="1,2")
    parser.add_argument("--max_areas", default="128,512,99999")
    parser.add_argument("--max_sides", default="32,64,99999")
    parser.add_argument("--val_fraction", type=float, default=0.25)
    parser.add_argument("--split_seed", default="20260613")
    parser.add_argument("--val_topk", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_samples, names, train_preds, _train_masks = load_cache_dense(Path(args.train_cache))
    test_samples, test_names, test_preds, _test_masks = load_cache_dense(Path(args.test_cache))
    if test_names != names:
        raise ValueError(f"name mismatch: {test_names} != {names}")
    val_keep = [stable_bucket(str(sample["name"]), args.split_seed) < args.val_fraction for sample in train_samples]
    val_samples = [sample for sample, keep in zip(train_samples, val_keep) if keep]
    val_preds = [pred for pred, keep in zip(train_preds, val_keep) if keep]

    weights = [parse_weight_spec(spec, names) for spec in args.weight_spec]
    val_rows = []
    for label, weight in weights:
        wv = np.asarray([weight[name] for name in names], dtype=np.float32)
        for gamma in parse_float_list(args.gammas):
            base_maps = weighted_maps(val_preds, wv, gamma)
            for seed_th in parse_float_list(args.seed_thresholds):
                for out_th in parse_float_list(args.output_thresholds):
                    for topk in parse_int_list(args.topks):
                        for rel in parse_float_list(args.relative_to_bests):
                            for min_area in parse_int_list(args.min_areas):
                                for max_area in parse_int_list(args.max_areas):
                                    for max_side in parse_int_list(args.max_sides):
                                        maps = adaptive_maps(base_maps, seed_th, out_th, topk, rel, min_area, max_area, max_side)
                                        metrics = metric_from_maps(val_samples, maps, out_th)
                                        val_rows.append(
                                            {
                                                "label": label,
                                                "weights": weight,
                                                "gamma": gamma,
                                                "seed_threshold": seed_th,
                                                "output_threshold": out_th,
                                                "topk": topk,
                                                "relative_to_best": rel,
                                                "min_area": min_area,
                                                "max_area": max_area,
                                                "max_side": max_side,
                                                "selection_score": metrics["mIoU"] - 0.000002 * (metrics["Fa"] * 1e6),
                                                **{f"val_{k}": v for k, v in metrics.items()},
                                            }
                                        )
    val_rows.sort(key=lambda row: row["selection_score"], reverse=True)

    test_rows = []
    for row in val_rows[: max(int(args.val_topk), 1)]:
        wv = np.asarray([row["weights"][name] for name in names], dtype=np.float32)
        base_maps = weighted_maps(test_preds, wv, row["gamma"])
        maps = adaptive_maps(
            base_maps,
            row["seed_threshold"],
            row["output_threshold"],
            int(row["topk"]),
            row["relative_to_best"],
            int(row["min_area"]),
            int(row["max_area"]),
            int(row["max_side"]),
        )
        test_rows.append({**row, **{f"test_{k}": v for k, v in metric_from_maps(test_samples, maps, row["output_threshold"]).items()}})
    selected = test_rows[0]
    best_ref = sorted(test_rows, key=lambda row: row["test_mIoU"], reverse=True)[0]
    print(
        f"selected label={selected['label']} seed={selected['seed_threshold']} out={selected['output_threshold']} "
        f"topk={selected['topk']} rel={selected['relative_to_best']} val_mIoU={selected['val_mIoU']*100:.2f} "
        f"test_mIoU={selected['test_mIoU']*100:.2f} Pd={selected['test_Pd']*100:.2f} Fa={selected['test_Fa']*1e6:.2f}",
        flush=True,
    )
    print(
        f"best_test_in_val_topk label={best_ref['label']} seed={best_ref['seed_threshold']} out={best_ref['output_threshold']} "
        f"topk={best_ref['topk']} rel={best_ref['relative_to_best']} test_mIoU={best_ref['test_mIoU']*100:.2f} "
        f"Pd={best_ref['test_Pd']*100:.2f} Fa={best_ref['test_Fa']*1e6:.2f}",
        flush=True,
    )
    Path(args.output).write_text(
        json.dumps(
            {
                "train_cache": args.train_cache,
                "test_cache": args.test_cache,
                "models": names,
                "selected_by_val": selected,
                "best_test_in_val_topk_reference": best_ref,
                "val_rows": val_rows,
                "test_rows": test_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
