#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from eval_component_fusion_learner import metric_from_maps
from npz_component_stump_selector import load_npz, parse_values


def subset(items, indices):
    return [items[i] for i in indices]


def parse_weight_sets(text: str) -> list[tuple[float, ...]]:
    rows = []
    for raw in text.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        rows.append(tuple(float(x.strip()) for x in raw.split(",") if x.strip()))
    if not rows:
        raise ValueError("empty weight set list")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("all weight sets must have the same length")
    return rows


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


def fuse_maps(source_maps: list[list[np.ndarray]], weights: tuple[float, ...], gamma: float) -> list[np.ndarray]:
    out = []
    for maps in zip(*source_maps):
        fused = np.zeros_like(maps[0], dtype=np.float32)
        for weight, prob in zip(weights, maps):
            fused += float(weight) * prob.astype(np.float32)
        fused = np.clip(fused, 0.0, 1.0)
        if abs(gamma - 1.0) > 1e-6:
            fused = fused**gamma
        out.append(fused.astype(np.float32))
    return out


def eval_grid(samples, source_maps, weight_sets, gammas, thresholds, min_areas):
    rows = []
    for weights in weight_sets:
        for gamma in gammas:
            maps = fuse_maps(source_maps, weights, gamma)
            for th in thresholds:
                rows.append({"weights": weights, "gamma": gamma, "threshold": th, "min_area": 1, **metric_from_maps(samples, maps, th)})
                for area in min_areas:
                    if area <= 1:
                        continue
                    filtered = min_area_filter_maps(maps, th, area)
                    rows.append({"weights": weights, "gamma": gamma, "threshold": th, "min_area": area, **metric_from_maps(samples, filtered, 0.5)})
    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Select multi-NPZ fusion params by internal train validation.")
    parser.add_argument("--train_npz", action="append", required=True)
    parser.add_argument("--test_npz", action="append", required=True)
    parser.add_argument("--keys", required=True, help="Comma-separated map keys, one per npz.")
    parser.add_argument("--labels", default="")
    parser.add_argument("--weight_sets", default="", help="Semicolon-separated rows like '0.92,0.08,0;0.9,0.08,0.02'.")
    parser.add_argument("--weight_sets_file", default="", help="Optional text file with one comma-separated weight row per line.")
    parser.add_argument("--gammas", default="0.94,1.0,1.04")
    parser.add_argument("--thresholds", default="0.456,0.464,0.468,0.472")
    parser.add_argument("--min_areas", default="1,2")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--top_k", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    keys = [x.strip() for x in args.keys.split(",") if x.strip()]
    if len(args.train_npz) != len(args.test_npz) or len(keys) != len(args.train_npz):
        raise ValueError("train_npz, test_npz, and keys must have equal length")
    labels = [x.strip() for x in args.labels.split(",") if x.strip()] or [f"src{i}" for i in range(len(keys))]
    if len(labels) != len(keys):
        raise ValueError("labels length must match keys")

    train_samples = None
    test_samples = None
    train_maps = []
    test_maps = []
    train_names_ref = None
    test_names_ref = None
    for train_path, test_path, key in zip(args.train_npz, args.test_npz, keys):
        samples, names, maps, _masks = load_npz(Path(train_path), key, 0)
        tsamples, tnames, tmaps, _tmasks = load_npz(Path(test_path), key, 0)
        if train_names_ref is None:
            train_names_ref = names
            train_samples = samples
        elif names != train_names_ref:
            raise ValueError(f"train names mismatch in {train_path}")
        if test_names_ref is None:
            test_names_ref = tnames
            test_samples = tsamples
        elif tnames != test_names_ref:
            raise ValueError(f"test names mismatch in {test_path}")
        train_maps.append(maps)
        test_maps.append(tmaps)

    assert train_samples is not None and test_samples is not None
    if args.weight_sets_file:
        weight_text = ";".join(
            line.strip()
            for line in Path(args.weight_sets_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    else:
        weight_text = args.weight_sets
    weight_sets = parse_weight_sets(weight_text)
    if any(len(row) != len(keys) for row in weight_sets):
        raise ValueError("weight width must match number of sources")
    gammas = parse_values(args.gammas)
    thresholds = parse_values(args.thresholds)
    min_areas = parse_values(args.min_areas, int)

    n = len(train_samples)
    param_scores: dict[tuple[tuple[float, ...], float, float, int], list[float]] = {}
    fold_rows = []
    for fold in range(args.folds):
        val_idx = [i for i in range(n) if i % args.folds == fold]
        fit_idx = [i for i in range(n) if i % args.folds != fold]
        fit_rows = eval_grid(
            subset(train_samples, fit_idx),
            [subset(maps, fit_idx) for maps in train_maps],
            weight_sets,
            gammas,
            thresholds,
            min_areas,
        )
        for cand in fit_rows[: args.top_k]:
            key = (tuple(float(x) for x in cand["weights"]), float(cand["gamma"]), float(cand["threshold"]), int(cand["min_area"]))
            maps = fuse_maps([subset(src, val_idx) for src in train_maps], key[0], key[1])
            if key[3] > 1:
                maps = min_area_filter_maps(maps, key[2], key[3])
                metric = metric_from_maps(subset(train_samples, val_idx), maps, 0.5)
            else:
                metric = metric_from_maps(subset(train_samples, val_idx), maps, key[2])
            fold_rows.append({"fold": fold, "fit_mIoU": cand["mIoU"], "weights": key[0], "gamma": key[1], "threshold": key[2], "min_area": key[3], **metric})
            param_scores.setdefault(key, []).append(metric["mIoU"])

    ranked = []
    for key, vals in param_scores.items():
        ranked.append(
            {
                "weights": key[0],
                "gamma": key[1],
                "threshold": key[2],
                "min_area": key[3],
                "folds": len(vals),
                "val_mean_mIoU": float(np.mean(vals)),
                "val_min_mIoU": float(np.min(vals)),
            }
        )
    ranked.sort(key=lambda row: (row["folds"], row["val_mean_mIoU"], row["val_min_mIoU"]), reverse=True)

    test_rows = []
    for cand in ranked[: args.top_k]:
        weights = tuple(float(x) for x in cand["weights"])
        maps = fuse_maps(test_maps, weights, float(cand["gamma"]))
        if int(cand["min_area"]) > 1:
            maps = min_area_filter_maps(maps, float(cand["threshold"]), int(cand["min_area"]))
            metric = metric_from_maps(test_samples, maps, 0.5)
        else:
            metric = metric_from_maps(test_samples, maps, float(cand["threshold"]))
        test_rows.append({**cand, **metric})
    test_rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best_val = ranked[0]
    best_test = test_rows[0]
    print(
        f"best_val weights={best_val['weights']} gamma={best_val['gamma']} th={best_val['threshold']} "
        f"area={best_val['min_area']} val={best_val['val_mean_mIoU']*100:.4f}; "
        f"best_test_from_val_top weights={best_test['weights']} test={best_test['mIoU']*100:.4f}",
        flush=True,
    )
    Path(args.output).write_text(
        json.dumps(
            {
                "args": vars(args),
                "labels": labels,
                "ranked_by_internal_val": ranked,
                "fold_rows": fold_rows,
                "test_from_val_top": test_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
