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
    avg = stack.mean(axis=0).astype(np.float32)
    teacher = stack[teachers].mean(axis=0).astype(np.float32)
    pconv = stack[pconvs].mean(axis=0).astype(np.float32)
    return avg, teacher, pconv


def suppress_maps(
    preds: list[np.ndarray],
    names: list[str],
    seed_threshold: float,
    min_area: int,
    max_area: int,
    max_side: int,
    min_avg_mean: float,
    min_avg_max: float,
    min_teacher_mean: float,
    min_pconv_mean: float,
    min_agreement: float,
) -> list[np.ndarray]:
    maps = []
    for stack in preds:
        avg, teacher, pconv = avg_teacher_pconv(stack, names)
        out = avg.copy()
        seed = avg > seed_threshold
        num_labels, label_img, stats, _centroids = cv2.connectedComponentsWithStats(seed.astype(np.uint8), 8)
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            width = int(stats[label_id, cv2.CC_STAT_WIDTH])
            height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            if area < min_area or area > max_area or max(width, height) > max_side:
                out[label_img == label_id] = 0.0
                continue
            comp = label_img == label_id
            avg_mean = float(avg[comp].mean())
            avg_max = float(avg[comp].max())
            teacher_mean = float(teacher[comp].mean())
            pconv_mean = float(pconv[comp].mean())
            agreement = min(teacher_mean, pconv_mean)
            keep = (
                avg_mean >= min_avg_mean
                and avg_max >= min_avg_max
                and teacher_mean >= min_teacher_mean
                and pconv_mean >= min_pconv_mean
                and agreement >= min_agreement
            )
            if not keep:
                out[comp] = 0.0
        maps.append(out.astype(np.float32))
    return maps


def main() -> None:
    parser = argparse.ArgumentParser(description="Suppress false-positive components on the dual-PConv average probability map.")
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--test_cache", required=True)
    parser.add_argument("--val_fraction", type=float, default=0.25)
    parser.add_argument("--split_seed", default="20260613")
    parser.add_argument("--seed_thresholds", default="0.38,0.40,0.42")
    parser.add_argument("--output_thresholds", default="0.38,0.40,0.42,0.45")
    parser.add_argument("--min_areas", default="1,2")
    parser.add_argument("--max_areas", default="128,256,512")
    parser.add_argument("--max_sides", default="24,40,64")
    parser.add_argument("--min_avg_means", default="0.00,0.20,0.30")
    parser.add_argument("--min_avg_maxes", default="0.00,0.40,0.55")
    parser.add_argument("--min_teacher_means", default="0.00,0.05,0.10,0.20")
    parser.add_argument("--min_pconv_means", default="0.00,0.10,0.20,0.30")
    parser.add_argument("--min_agreements", default="0.00,0.05,0.10,0.15")
    parser.add_argument("--val_topk", type=int, default=20)
    parser.add_argument("--selection_score", default="mIoU_minus_fa", choices=["mIoU", "mIoU_minus_fa"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_samples, names, train_preds, _train_masks = load_cache_dense(Path(args.train_cache))
    test_samples, test_names, test_preds, _test_masks = load_cache_dense(Path(args.test_cache))
    if test_names != names:
        raise ValueError(f"name mismatch: {test_names} != {names}")
    val_keep = [stable_bucket(str(sample["name"]), args.split_seed) < args.val_fraction for sample in train_samples]
    val_samples = [sample for sample, keep in zip(train_samples, val_keep) if keep]
    val_preds = [pred for pred, keep in zip(train_preds, val_keep) if keep]

    val_rows = []
    for seed_th in parse_float_list(args.seed_thresholds):
        for min_area in parse_int_list(args.min_areas):
            for max_area in parse_int_list(args.max_areas):
                for max_side in parse_int_list(args.max_sides):
                    for min_avg_mean in parse_float_list(args.min_avg_means):
                        for min_avg_max in parse_float_list(args.min_avg_maxes):
                            for min_teacher_mean in parse_float_list(args.min_teacher_means):
                                for min_pconv_mean in parse_float_list(args.min_pconv_means):
                                    for min_agreement in parse_float_list(args.min_agreements):
                                        maps = suppress_maps(
                                            val_preds,
                                            names,
                                            seed_th,
                                            min_area,
                                            max_area,
                                            max_side,
                                            min_avg_mean,
                                            min_avg_max,
                                            min_teacher_mean,
                                            min_pconv_mean,
                                            min_agreement,
                                        )
                                        for out_th in parse_float_list(args.output_thresholds):
                                            metrics = metric_from_maps(val_samples, maps, out_th)
                                            score = metrics["mIoU"]
                                            if args.selection_score == "mIoU_minus_fa":
                                                score = metrics["mIoU"] - 0.000002 * (metrics["Fa"] * 1e6)
                                            val_rows.append(
                                                {
                                                    "selection_score": float(score),
                                                    "seed_threshold": seed_th,
                                                    "output_threshold": out_th,
                                                    "min_area": min_area,
                                                    "max_area": max_area,
                                                    "max_side": max_side,
                                                    "min_avg_mean": min_avg_mean,
                                                    "min_avg_max": min_avg_max,
                                                    "min_teacher_mean": min_teacher_mean,
                                                    "min_pconv_mean": min_pconv_mean,
                                                    "min_agreement": min_agreement,
                                                    **{f"val_{k}": v for k, v in metrics.items()},
                                                }
                                            )
    val_rows.sort(key=lambda row: row["selection_score"], reverse=True)

    test_rows = []
    for row in val_rows[: max(int(args.val_topk), 1)]:
        maps = suppress_maps(
            test_preds,
            names,
            row["seed_threshold"],
            int(row["min_area"]),
            int(row["max_area"]),
            int(row["max_side"]),
            row["min_avg_mean"],
            row["min_avg_max"],
            row["min_teacher_mean"],
            row["min_pconv_mean"],
            row["min_agreement"],
        )
        test_rows.append({**row, **{f"test_{k}": v for k, v in metric_from_maps(test_samples, maps, row["output_threshold"]).items()}})
    selected = test_rows[0]
    best_ref = sorted(test_rows, key=lambda row: row["test_mIoU"], reverse=True)[0]
    print(
        f"selected seed={selected['seed_threshold']} out={selected['output_threshold']} "
        f"val_mIoU={selected['val_mIoU']*100:.2f} test_mIoU={selected['test_mIoU']*100:.2f} "
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
