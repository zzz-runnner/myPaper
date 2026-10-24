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
from inmemory_cached_weight_search import load_cache_dense  # noqa: E402


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def stable_bucket(name: str, seed: str) -> float:
    raw = hashlib.sha1(f"{seed}:{name}".encode("utf-8")).hexdigest()[:8]
    return int(raw, 16) / float(0xFFFFFFFF)


def base_map(stack: np.ndarray, names: list[str], mode: str) -> np.ndarray:
    idx = {name: i for i, name in enumerate(names)}
    teachers = [idx[name] for name in names if not name.startswith("pconv")]
    pconvs = [idx[name] for name in names if name.startswith("pconv")]
    if mode == "teacher_avg":
        return stack[teachers].mean(axis=0).astype(np.float32)
    if mode == "teacher_max":
        return stack[teachers].max(axis=0).astype(np.float32)
    if mode == "avg_all":
        return stack.mean(axis=0).astype(np.float32)
    if mode == "max_all":
        return stack.max(axis=0).astype(np.float32)
    if mode == "pconv_avg":
        return stack[pconvs].mean(axis=0).astype(np.float32)
    raise ValueError(mode)


def render_rule_maps(
    preds: list[np.ndarray],
    names: list[str],
    base_mode: str,
    base_threshold: float,
    pconv_threshold: float,
    max_area: int,
    max_side: int,
    min_fill: float,
    max_teacher_mean: float,
    min_pconv_mean: float,
    fill_value: float,
) -> list[np.ndarray]:
    idx = {name: i for i, name in enumerate(names)}
    teachers = [idx[name] for name in names if not name.startswith("pconv")]
    pconvs = [idx[name] for name in names if name.startswith("pconv")]
    maps = []
    for stack in preds:
        base = base_map(stack, names, base_mode)
        out = base.copy()
        teacher_max = stack[teachers].max(axis=0)
        pconv_score = stack[pconvs].max(axis=0)
        seed = (pconv_score > pconv_threshold) & (teacher_max < max_teacher_mean) & (base <= base_threshold)
        num_labels, label_img, stats, _centroids = cv2.connectedComponentsWithStats(seed.astype(np.uint8), 8)
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            width = int(stats[label_id, cv2.CC_STAT_WIDTH])
            height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            if area <= 0 or area > max_area or max(width, height) > max_side:
                continue
            fill = area / float(max(width * height, 1))
            if fill < min_fill:
                continue
            comp = label_img == label_id
            if float(pconv_score[comp].mean()) < min_pconv_mean:
                continue
            out[comp] = np.maximum(out[comp], fill_value)
        maps.append(out.astype(np.float32))
    return maps


def candidate_grid(args: argparse.Namespace):
    for base_mode in [x.strip() for x in args.base_modes.split(",") if x.strip()]:
        for base_threshold in parse_float_list(args.base_thresholds):
            for pconv_threshold in parse_float_list(args.pconv_thresholds):
                for max_area in parse_int_list(args.max_areas):
                    for max_side in parse_int_list(args.max_sides):
                        for min_fill in parse_float_list(args.min_fills):
                            for max_teacher_mean in parse_float_list(args.max_teacher_means):
                                for min_pconv_mean in parse_float_list(args.min_pconv_means):
                                    for fill_value in parse_float_list(args.fill_values):
                                        for out_threshold in parse_float_list(args.output_thresholds):
                                            yield {
                                                "base_mode": base_mode,
                                                "base_threshold": base_threshold,
                                                "pconv_threshold": pconv_threshold,
                                                "max_area": max_area,
                                                "max_side": max_side,
                                                "min_fill": min_fill,
                                                "max_teacher_mean": max_teacher_mean,
                                                "min_pconv_mean": min_pconv_mean,
                                                "fill_value": fill_value,
                                                "output_threshold": out_threshold,
                                            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/val search for PConv-added component filtering from cached predictions.")
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--test_cache", required=True)
    parser.add_argument("--val_fraction", type=float, default=0.25)
    parser.add_argument("--split_seed", default="20260613")
    parser.add_argument("--base_modes", default="teacher_avg,avg_all")
    parser.add_argument("--base_thresholds", default="0.38,0.40,0.42,0.45")
    parser.add_argument("--pconv_thresholds", default="0.35,0.45,0.55")
    parser.add_argument("--max_areas", default="16,32,64,128")
    parser.add_argument("--max_sides", default="8,12,18,28")
    parser.add_argument("--min_fills", default="0.10,0.20,0.35")
    parser.add_argument("--max_teacher_means", default="0.10,0.20,0.30,0.45")
    parser.add_argument("--min_pconv_means", default="0.35,0.50,0.65")
    parser.add_argument("--fill_values", default="0.60,0.80,1.00")
    parser.add_argument("--output_thresholds", default="0.38,0.40,0.42,0.45")
    parser.add_argument("--val_topk", type=int, default=24)
    parser.add_argument("--selection_score", default="mIoU_minus_fa", choices=["mIoU", "mIoU_minus_fa"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_samples, names, train_preds, _train_masks = load_cache_dense(Path(args.train_cache))
    test_samples, test_names, test_preds, _test_masks = load_cache_dense(Path(args.test_cache))
    if test_names != names:
        raise ValueError(f"name mismatch: {test_names} != {names}")
    if not any(name.startswith("pconv") for name in names):
        raise ValueError("cache has no pconv predictions")

    val_keep = [stable_bucket(str(sample["name"]), args.split_seed) < args.val_fraction for sample in train_samples]
    val_samples = [sample for sample, keep in zip(train_samples, val_keep) if keep]
    val_preds = [pred for pred, keep in zip(train_preds, val_keep) if keep]
    if not val_samples:
        raise RuntimeError("empty validation split")

    val_rows = []
    for params in candidate_grid(args):
        maps = render_rule_maps(
            val_preds,
            names,
            params["base_mode"],
            params["base_threshold"],
            params["pconv_threshold"],
            params["max_area"],
            params["max_side"],
            params["min_fill"],
            params["max_teacher_mean"],
            params["min_pconv_mean"],
            params["fill_value"],
        )
        metrics = metric_from_maps(val_samples, maps, params["output_threshold"])
        score = metrics["mIoU"]
        if args.selection_score == "mIoU_minus_fa":
            score = metrics["mIoU"] - 0.000002 * (metrics["Fa"] * 1e6)
        val_rows.append({"selection_score": float(score), **params, **{f"val_{k}": v for k, v in metrics.items()}})
    val_rows.sort(key=lambda row: row["selection_score"], reverse=True)

    test_rows = []
    for row in val_rows[: max(int(args.val_topk), 1)]:
        maps = render_rule_maps(
            test_preds,
            names,
            row["base_mode"],
            row["base_threshold"],
            row["pconv_threshold"],
            row["max_area"],
            row["max_side"],
            row["min_fill"],
            row["max_teacher_mean"],
            row["min_pconv_mean"],
            row["fill_value"],
        )
        test_rows.append({**row, **{f"test_{k}": v for k, v in metric_from_maps(test_samples, maps, row["output_threshold"]).items()}})
    selected = test_rows[0]
    best_ref = sorted(test_rows, key=lambda row: row["test_mIoU"], reverse=True)[0]
    print(
        f"selected base={selected['base_mode']} base_th={selected['base_threshold']} pconv_th={selected['pconv_threshold']} "
        f"out={selected['output_threshold']} val_mIoU={selected['val_mIoU']*100:.2f} "
        f"test_mIoU={selected['test_mIoU']*100:.2f} Pd={selected['test_Pd']*100:.2f} Fa={selected['test_Fa']*1e6:.2f}",
        flush=True,
    )
    print(
        f"best_test_in_val_topk base={best_ref['base_mode']} base_th={best_ref['base_threshold']} "
        f"pconv_th={best_ref['pconv_threshold']} out={best_ref['output_threshold']} "
        f"test_mIoU={best_ref['test_mIoU']*100:.2f} Pd={best_ref['test_Pd']*100:.2f} Fa={best_ref['test_Fa']*1e6:.2f}",
        flush=True,
    )
    Path(args.output).write_text(
        json.dumps(
            {
                "train_cache": args.train_cache,
                "test_cache": args.test_cache,
                "models": names,
                "val_fraction": args.val_fraction,
                "split_seed": args.split_seed,
                "val_samples": len(val_samples),
                "selection_score": args.selection_score,
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
