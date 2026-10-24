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


def postprocess(pred: np.ndarray, threshold: float, min_area: int, max_area: int, close_k: int, dilate_iter: int) -> np.ndarray:
    mask = (pred > threshold).astype(np.uint8)
    if close_k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    num_labels, label_img, stats, _centroids = cv2.connectedComponentsWithStats(mask, 8)
    cleaned = np.zeros_like(mask)
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        cleaned[label_img == label_id] = 1
    if dilate_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.dilate(cleaned, kernel, iterations=dilate_iter)
    return cleaned.astype(np.float32)


def postprocess_maps(maps: list[np.ndarray], threshold: float, min_area: int, max_area: int, close_k: int, dilate_iter: int) -> list[np.ndarray]:
    return [postprocess(pred, threshold, min_area, max_area, close_k, dilate_iter) for pred in maps]


def main() -> None:
    parser = argparse.ArgumentParser(description="Morphological postprocess search on cached fused maps.")
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--test_cache", required=True)
    parser.add_argument("--weight_spec", action="append", required=True)
    parser.add_argument("--thresholds", default="0.38,0.40,0.42,0.45")
    parser.add_argument("--gammas", default="1.0")
    parser.add_argument("--min_areas", default="1,2,3,5")
    parser.add_argument("--max_areas", default="128,512,99999")
    parser.add_argument("--close_ks", default="0,3")
    parser.add_argument("--dilate_iters", default="0,1")
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
            base_val_maps = weighted_maps(val_preds, wv, gamma)
            for threshold in parse_float_list(args.thresholds):
                for min_area in parse_int_list(args.min_areas):
                    for max_area in parse_int_list(args.max_areas):
                        for close_k in parse_int_list(args.close_ks):
                            for dilate_iter in parse_int_list(args.dilate_iters):
                                maps = postprocess_maps(base_val_maps, threshold, min_area, max_area, close_k, dilate_iter)
                                metrics = metric_from_maps(val_samples, maps, 0.5)
                                score = metrics["mIoU"] - 0.000002 * (metrics["Fa"] * 1e6)
                                val_rows.append(
                                    {
                                        "label": label,
                                        "weights": weight,
                                        "gamma": gamma,
                                        "threshold": threshold,
                                        "min_area": min_area,
                                        "max_area": max_area,
                                        "close_k": close_k,
                                        "dilate_iter": dilate_iter,
                                        "selection_score": float(score),
                                        **{f"val_{k}": v for k, v in metrics.items()},
                                    }
                                )
    val_rows.sort(key=lambda row: row["selection_score"], reverse=True)

    test_rows = []
    for row in val_rows[: max(int(args.val_topk), 1)]:
        wv = np.asarray([row["weights"][name] for name in names], dtype=np.float32)
        base_test_maps = weighted_maps(test_preds, wv, row["gamma"])
        maps = postprocess_maps(base_test_maps, row["threshold"], int(row["min_area"]), int(row["max_area"]), int(row["close_k"]), int(row["dilate_iter"]))
        test_rows.append({**row, **{f"test_{k}": v for k, v in metric_from_maps(test_samples, maps, 0.5).items()}})
    selected = test_rows[0]
    best_ref = sorted(test_rows, key=lambda row: row["test_mIoU"], reverse=True)[0]
    print(
        f"selected label={selected['label']} th={selected['threshold']} min_area={selected['min_area']} "
        f"close={selected['close_k']} dilate={selected['dilate_iter']} val_mIoU={selected['val_mIoU']*100:.2f} "
        f"test_mIoU={selected['test_mIoU']*100:.2f} Pd={selected['test_Pd']*100:.2f} Fa={selected['test_Fa']*1e6:.2f}",
        flush=True,
    )
    print(
        f"best_test_in_val_topk label={best_ref['label']} th={best_ref['threshold']} "
        f"test_mIoU={best_ref['test_mIoU']*100:.2f} Pd={best_ref['test_Pd']*100:.2f} Fa={best_ref['test_Fa']*1e6:.2f}",
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
