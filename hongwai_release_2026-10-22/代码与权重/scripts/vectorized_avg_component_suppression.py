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


def avg_teacher_pconv(stack: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = {name: i for i, name in enumerate(names)}
    teachers = [idx[name] for name in names if not name.startswith("pconv")]
    pconvs = [idx[name] for name in names if name.startswith("pconv")]
    return (
        stack.mean(axis=0).astype(np.float32),
        stack[teachers].mean(axis=0).astype(np.float32),
        stack[pconvs].mean(axis=0).astype(np.float32),
    )


def build_table(samples, preds, names, seed_threshold: float, output_threshold: float):
    base_inter = []
    base_union = []
    rows = []
    masks = []
    for sample_idx, (sample, stack) in enumerate(zip(samples, preds)):
        mask = sample["mask"].astype(bool)
        avg, teacher, pconv = avg_teacher_pconv(stack, names)
        base_bin = avg > output_threshold
        base_inter.append(float((base_bin & mask).sum()))
        base_union.append(float((base_bin | mask).sum()))
        seed = avg > seed_threshold
        num_labels, label_img, stats, _centroids = cv2.connectedComponentsWithStats(seed.astype(np.uint8), 8)
        for label_id in range(1, num_labels):
            comp = label_img == label_id
            active = comp & base_bin
            if not active.any():
                continue
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            width = int(stats[label_id, cv2.CC_STAT_WIDTH])
            height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            tp_removed = float((active & mask).sum())
            fp_removed = float((active & ~mask).sum())
            rows.append(
                {
                    "sample_idx": sample_idx,
                    "area": float(area),
                    "max_side": float(max(width, height)),
                    "fill": float(area / max(width * height, 1)),
                    "avg_mean": float(avg[comp].mean()),
                    "avg_max": float(avg[comp].max()),
                    "teacher_mean": float(teacher[comp].mean()),
                    "pconv_mean": float(pconv[comp].mean()),
                    "agreement": float(min(float(teacher[comp].mean()), float(pconv[comp].mean()))),
                    "tp_removed": tp_removed,
                    "fp_removed": fp_removed,
                }
            )
            masks.append((sample_idx, comp))
    if rows:
        table = {key: np.asarray([row[key] for row in rows], dtype=np.float32) for key in rows[0]}
        table["sample_idx"] = table["sample_idx"].astype(np.int32)
    else:
        keys = ["area", "max_side", "fill", "avg_mean", "avg_max", "teacher_mean", "pconv_mean", "agreement", "tp_removed", "fp_removed"]
        table = {key: np.zeros((0,), dtype=np.float32) for key in keys}
        table["sample_idx"] = np.zeros((0,), dtype=np.int32)
    return {
        "base_inter": np.asarray(base_inter, dtype=np.float32),
        "base_union": np.asarray(base_union, dtype=np.float32),
        "table": table,
        "masks": masks,
    }


def keep_mask(table: dict[str, np.ndarray], params: dict[str, float]) -> np.ndarray:
    if table["area"].size == 0:
        return np.zeros((0,), dtype=bool)
    return (
        (table["area"] >= params["min_area"])
        & (table["area"] <= params["max_area"])
        & (table["max_side"] <= params["max_side"])
        & (table["avg_mean"] >= params["min_avg_mean"])
        & (table["avg_max"] >= params["min_avg_max"])
        & (table["teacher_mean"] >= params["min_teacher_mean"])
        & (table["pconv_mean"] >= params["min_pconv_mean"])
        & (table["agreement"] >= params["min_agreement"])
    )


def table_niou(cache: dict[str, object], keep: np.ndarray) -> float:
    table = cache["table"]
    remove = ~keep
    n = len(cache["base_inter"])
    lost_tp = np.zeros((n,), dtype=np.float32)
    lost_fp = np.zeros((n,), dtype=np.float32)
    if remove.size:
        np.add.at(lost_tp, table["sample_idx"][remove], table["tp_removed"][remove])
        np.add.at(lost_fp, table["sample_idx"][remove], table["fp_removed"][remove])
    inter = cache["base_inter"] - lost_tp
    union = cache["base_union"] - lost_fp
    return float(np.mean(inter / (union + np.spacing(1))))


def render_maps(samples, preds, names, cache, keep: np.ndarray):
    maps = [avg_teacher_pconv(stack, names)[0] for stack in preds]
    remove = ~keep
    for comp_idx in np.flatnonzero(remove):
        sample_idx, comp = cache["masks"][int(comp_idx)]
        maps[sample_idx][comp] = 0.0
    return [x.astype(np.float32) for x in maps]


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorized suppression of false-positive components on avg probability maps.")
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--test_cache", required=True)
    parser.add_argument("--val_fraction", type=float, default=0.25)
    parser.add_argument("--split_seed", default="20260613")
    parser.add_argument("--seed_thresholds", default="0.40,0.42")
    parser.add_argument("--output_thresholds", default="0.40,0.42,0.45")
    parser.add_argument("--min_areas", default="1,2")
    parser.add_argument("--max_areas", default="128,512")
    parser.add_argument("--max_sides", default="40,64")
    parser.add_argument("--min_avg_means", default="0.00,0.25")
    parser.add_argument("--min_avg_maxes", default="0.00,0.50")
    parser.add_argument("--min_teacher_means", default="0.00,0.10")
    parser.add_argument("--min_pconv_means", default="0.00,0.20")
    parser.add_argument("--min_agreements", default="0.00,0.10")
    parser.add_argument("--val_topk", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    samples, names, preds, _masks = load_cache_dense(Path(args.train_cache))
    test_samples, test_names, test_preds, _test_masks = load_cache_dense(Path(args.test_cache))
    if test_names != names:
        raise ValueError(f"name mismatch: {test_names} != {names}")
    val_keep = [stable_bucket(str(sample["name"]), args.split_seed) < args.val_fraction for sample in samples]
    val_samples = [sample for sample, keep in zip(samples, val_keep) if keep]
    val_preds = [pred for pred, keep in zip(preds, val_keep) if keep]

    val_rows = []
    for seed_th in parse_float_list(args.seed_thresholds):
        for out_th in parse_float_list(args.output_thresholds):
            val_cache = build_table(val_samples, val_preds, names, seed_th, out_th)
            for min_area in parse_int_list(args.min_areas):
                for max_area in parse_int_list(args.max_areas):
                    for max_side in parse_int_list(args.max_sides):
                        for min_avg_mean in parse_float_list(args.min_avg_means):
                            for min_avg_max in parse_float_list(args.min_avg_maxes):
                                for min_teacher_mean in parse_float_list(args.min_teacher_means):
                                    for min_pconv_mean in parse_float_list(args.min_pconv_means):
                                        for min_agreement in parse_float_list(args.min_agreements):
                                            params = {
                                                "min_area": float(min_area),
                                                "max_area": float(max_area),
                                                "max_side": float(max_side),
                                                "min_avg_mean": float(min_avg_mean),
                                                "min_avg_max": float(min_avg_max),
                                                "min_teacher_mean": float(min_teacher_mean),
                                                "min_pconv_mean": float(min_pconv_mean),
                                                "min_agreement": float(min_agreement),
                                            }
                                            keep = keep_mask(val_cache["table"], params)
                                            val_rows.append(
                                                {
                                                    "seed_threshold": seed_th,
                                                    "output_threshold": out_th,
                                                    **params,
                                                    "val_fast_nIoU": table_niou(val_cache, keep),
                                                    "val_components": int(val_cache["table"]["area"].size),
                                                    "val_kept": int(keep.sum()),
                                                }
                                            )
    val_rows.sort(key=lambda row: row["val_fast_nIoU"], reverse=True)

    test_rows = []
    cache_by_key: dict[tuple[float, float], dict[str, object]] = {}
    for row in val_rows[: max(int(args.val_topk), 1)]:
        key = (float(row["seed_threshold"]), float(row["output_threshold"]))
        if key not in cache_by_key:
            cache_by_key[key] = build_table(test_samples, test_preds, names, key[0], key[1])
        test_cache = cache_by_key[key]
        params = {k: float(row[k]) for k in ["min_area", "max_area", "max_side", "min_avg_mean", "min_avg_max", "min_teacher_mean", "min_pconv_mean", "min_agreement"]}
        keep = keep_mask(test_cache["table"], params)
        maps = render_maps(test_samples, test_preds, names, test_cache, keep)
        test_rows.append(
            {
                **row,
                "test_components": int(test_cache["table"]["area"].size),
                "test_kept": int(keep.sum()),
                **{f"test_{k}": v for k, v in metric_from_maps(test_samples, maps, row["output_threshold"]).items()},
            }
        )
    selected = test_rows[0]
    best_ref = sorted(test_rows, key=lambda row: row["test_mIoU"], reverse=True)[0]
    print(
        f"selected seed={selected['seed_threshold']} out={selected['output_threshold']} "
        f"val_nIoU={selected['val_fast_nIoU']*100:.2f} test_mIoU={selected['test_mIoU']*100:.2f} "
        f"Pd={selected['test_Pd']*100:.2f} Fa={selected['test_Fa']*1e6:.2f}",
        flush=True,
    )
    print(
        f"best_test_in_val_topk seed={best_ref['seed_threshold']} out={best_ref['output_threshold']} "
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
                "selected_by_val_fast_nIoU": selected,
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
