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


def split_indices(samples: list[dict[str, object]], val_fraction: float, seed: str) -> np.ndarray:
    return np.asarray([stable_bucket(str(sample["name"]), seed) < val_fraction for sample in samples], dtype=bool)


def base_score(stack: np.ndarray, names: list[str], mode: str) -> np.ndarray:
    idx = {name: i for i, name in enumerate(names)}
    teachers = [idx[name] for name in names if not name.startswith("pconv")]
    if mode == "avg_all":
        return stack.mean(axis=0).astype(np.float32)
    if mode == "teacher_avg":
        return stack[teachers].mean(axis=0).astype(np.float32)
    if mode == "teacher_max":
        return stack[teachers].max(axis=0).astype(np.float32)
    raise ValueError(mode)


def build_table(samples, preds, names, base_mode: str, base_threshold: float, pconv_threshold: float):
    idx = {name: i for i, name in enumerate(names)}
    teachers = [idx[name] for name in names if not name.startswith("pconv")]
    pconvs = [idx[name] for name in names if name.startswith("pconv")]
    base_inter = []
    base_union = []
    rows = []
    masks = []
    for sample_idx, (sample, stack) in enumerate(zip(samples, preds)):
        mask = sample["mask"].astype(bool)
        base = base_score(stack, names, base_mode)
        base_bin = base > base_threshold
        base_inter.append(float((base_bin & mask).sum()))
        base_union.append(float((base_bin | mask).sum()))
        teacher_max = stack[teachers].max(axis=0)
        pconv_score = stack[pconvs].max(axis=0)
        seed = (pconv_score > pconv_threshold) & ~base_bin
        num_labels, label_img, stats, _centroids = cv2.connectedComponentsWithStats(seed.astype(np.uint8), 8)
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            x0 = int(stats[label_id, cv2.CC_STAT_LEFT])
            y0 = int(stats[label_id, cv2.CC_STAT_TOP])
            width = int(stats[label_id, cv2.CC_STAT_WIDTH])
            height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            comp = label_img == label_id
            add = comp & ~base_bin
            rows.append(
                {
                    "sample_idx": sample_idx,
                    "area": float(area),
                    "max_side": float(max(width, height)),
                    "fill": float(area / max(width * height, 1)),
                    "teacher_mean": float(teacher_max[comp].mean()),
                    "teacher_max": float(teacher_max[comp].max()),
                    "pconv_mean": float(pconv_score[comp].mean()),
                    "pconv_max": float(pconv_score[comp].max()),
                    "add_inter": float((add & mask).sum()),
                    "add_union": float(add.sum()),
                }
            )
            masks.append((sample_idx, comp))
    if rows:
        table = {key: np.asarray([row[key] for row in rows], dtype=np.float32) for key in rows[0]}
        table["sample_idx"] = table["sample_idx"].astype(np.int32)
    else:
        table = {key: np.zeros((0,), dtype=np.float32) for key in ["area", "max_side", "fill", "teacher_mean", "teacher_max", "pconv_mean", "pconv_max", "add_inter", "add_union"]}
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
        (table["area"] <= params["max_area"])
        & (table["max_side"] <= params["max_side"])
        & (table["fill"] >= params["min_fill"])
        & (table["teacher_mean"] <= params["max_teacher_mean"])
        & (table["pconv_mean"] >= params["min_pconv_mean"])
    )


def table_niou(cache: dict[str, object], keep: np.ndarray) -> float:
    n = len(cache["base_inter"])
    add_inter = np.zeros((n,), dtype=np.float32)
    add_union = np.zeros((n,), dtype=np.float32)
    table = cache["table"]
    if keep.size:
        np.add.at(add_inter, table["sample_idx"][keep], table["add_inter"][keep])
        np.add.at(add_union, table["sample_idx"][keep], table["add_union"][keep])
    inter = cache["base_inter"] + add_inter
    union = cache["base_union"] + add_union
    return float(np.mean(inter / (union + np.spacing(1))))


def render_maps(samples, preds, names, cache, base_mode: str, base_threshold: float, keep: np.ndarray, fill_value: float):
    maps = [base_score(stack, names, base_mode) for stack in preds]
    for comp_idx in np.flatnonzero(keep):
        sample_idx, comp = cache["masks"][int(comp_idx)]
        maps[sample_idx][comp] = np.maximum(maps[sample_idx][comp], fill_value)
    return [x.astype(np.float32) for x in maps]


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorized PConv-added component table search.")
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--test_cache", required=True)
    parser.add_argument("--val_fraction", type=float, default=0.25)
    parser.add_argument("--split_seed", default="20260613")
    parser.add_argument("--base_modes", default="avg_all,teacher_avg")
    parser.add_argument("--base_thresholds", default="0.38,0.40,0.42")
    parser.add_argument("--pconv_thresholds", default="0.35,0.45,0.55")
    parser.add_argument("--max_areas", default="16,32,64")
    parser.add_argument("--max_sides", default="8,12,18")
    parser.add_argument("--min_fills", default="0.10,0.25")
    parser.add_argument("--max_teacher_means", default="0.10,0.25,0.40")
    parser.add_argument("--min_pconv_means", default="0.35,0.50")
    parser.add_argument("--fill_values", default="0.80,1.00")
    parser.add_argument("--output_thresholds", default="0.38,0.40,0.42,0.45")
    parser.add_argument("--val_topk", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    samples, names, preds, _masks = load_cache_dense(Path(args.train_cache))
    test_samples, test_names, test_preds, _test_masks = load_cache_dense(Path(args.test_cache))
    if test_names != names:
        raise ValueError(f"name mismatch: {test_names} != {names}")
    val_idx = split_indices(samples, args.val_fraction, args.split_seed)
    val_samples = [sample for sample, keep in zip(samples, val_idx) if keep]
    val_preds = [pred for pred, keep in zip(preds, val_idx) if keep]

    val_rows = []
    for base_mode in [x.strip() for x in args.base_modes.split(",") if x.strip()]:
        for base_threshold in parse_float_list(args.base_thresholds):
            for pconv_threshold in parse_float_list(args.pconv_thresholds):
                val_cache = build_table(val_samples, val_preds, names, base_mode, base_threshold, pconv_threshold)
                for max_area in parse_int_list(args.max_areas):
                    for max_side in parse_int_list(args.max_sides):
                        for min_fill in parse_float_list(args.min_fills):
                            for max_teacher_mean in parse_float_list(args.max_teacher_means):
                                for min_pconv_mean in parse_float_list(args.min_pconv_means):
                                    params = {
                                        "max_area": float(max_area),
                                        "max_side": float(max_side),
                                        "min_fill": float(min_fill),
                                        "max_teacher_mean": float(max_teacher_mean),
                                        "min_pconv_mean": float(min_pconv_mean),
                                    }
                                    keep = keep_mask(val_cache["table"], params)
                                    val_rows.append(
                                        {
                                            "base_mode": base_mode,
                                            "base_threshold": base_threshold,
                                            "pconv_threshold": pconv_threshold,
                                            **params,
                                            "val_fast_nIoU": table_niou(val_cache, keep),
                                            "val_components": int(val_cache["table"]["area"].size),
                                            "val_kept": int(keep.sum()),
                                        }
                                    )
    val_rows.sort(key=lambda row: row["val_fast_nIoU"], reverse=True)

    test_rows = []
    table_cache: dict[tuple[str, float, float], dict[str, object]] = {}
    for row in val_rows[: max(int(args.val_topk), 1)]:
        key = (row["base_mode"], float(row["base_threshold"]), float(row["pconv_threshold"]))
        if key not in table_cache:
            table_cache[key] = build_table(test_samples, test_preds, names, row["base_mode"], row["base_threshold"], row["pconv_threshold"])
        test_cache = table_cache[key]
        params = {k: float(row[k]) for k in ["max_area", "max_side", "min_fill", "max_teacher_mean", "min_pconv_mean"]}
        keep = keep_mask(test_cache["table"], params)
        for fill_value in parse_float_list(args.fill_values):
            maps = render_maps(test_samples, test_preds, names, test_cache, row["base_mode"], row["base_threshold"], keep, fill_value)
            for out_th in parse_float_list(args.output_thresholds):
                test_rows.append(
                    {
                        **row,
                        "fill_value": fill_value,
                        "output_threshold": out_th,
                        "test_components": int(test_cache["table"]["area"].size),
                        "test_kept": int(keep.sum()),
                        **{f"test_{k}": v for k, v in metric_from_maps(test_samples, maps, out_th).items()},
                    }
                )
    selected = test_rows[0]
    best_ref = sorted(test_rows, key=lambda row: row["test_mIoU"], reverse=True)[0]
    print(
        f"selected base={selected['base_mode']} base_th={selected['base_threshold']} pconv_th={selected['pconv_threshold']} "
        f"val_nIoU={selected['val_fast_nIoU']*100:.2f} test_mIoU={selected['test_mIoU']*100:.2f} "
        f"Pd={selected['test_Pd']*100:.2f} Fa={selected['test_Fa']*1e6:.2f}",
        flush=True,
    )
    print(
        f"best_test_in_val_topk base={best_ref['base_mode']} base_th={best_ref['base_threshold']} pconv_th={best_ref['pconv_threshold']} "
        f"out={best_ref['output_threshold']} test_mIoU={best_ref['test_mIoU']*100:.2f} "
        f"Pd={best_ref['test_Pd']*100:.2f} Fa={best_ref['test_Fa']*1e6:.2f}",
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
