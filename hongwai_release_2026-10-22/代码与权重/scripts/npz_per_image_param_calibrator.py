#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def parse_values(text: str, cast=float):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def min_area_filter(prob: np.ndarray, threshold: float, min_area: int) -> np.ndarray:
    binary = (prob > threshold).astype(np.uint8)
    if min_area <= 1:
        return binary.astype(bool)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros_like(binary)
    for label in range(1, num):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            out[labels == label] = 1
    return out.astype(bool)


def metric_binary(preds: list[np.ndarray], masks: list[np.ndarray], sizes: list[list[int]]) -> dict[str, float]:
    total_correct = 0
    total_label = 0
    total_inter = 0
    total_union = 0
    sample_ious = []
    pd = 0
    target = 0
    dismatch_pixel = 0
    all_pixel = 0
    for pred, mask, size in zip(preds, masks, sizes):
        gt = mask > 0.5
        pb = pred.astype(bool)
        inter = int((pb & gt).sum())
        union = int((pb | gt).sum())
        total_inter += inter
        total_union += union
        total_correct += int((pb == gt).sum())
        total_label += int(gt.size)
        sample_ious.append(inter / float(union + np.spacing(1)))

        num_p, _lab_p, stats_p, cents_p = cv2.connectedComponentsWithStats(pb.astype(np.uint8), 8)
        num_g, _lab_g, _stats_g, cents_g = cv2.connectedComponentsWithStats(gt.astype(np.uint8), 8)
        target += max(num_g - 1, 0)
        used = set()
        matched_areas: list[int] = []
        all_areas = [int(stats_p[i, cv2.CC_STAT_AREA]) for i in range(1, num_p)]
        for gi in range(1, num_g):
            gy, gx = float(cents_g[gi][1]), float(cents_g[gi][0])
            for pi in range(1, num_p):
                if pi in used:
                    continue
                py, px = float(cents_p[pi][1]), float(cents_p[pi][0])
                if math.hypot(py - gy, px - gx) < 3.0:
                    used.add(pi)
                    matched_areas.append(int(stats_p[pi, cv2.CC_STAT_AREA]))
                    pd += 1
                    break
        remaining = all_areas[:]
        for area in matched_areas:
            try:
                remaining.remove(area)
            except ValueError:
                pass
        dismatch_pixel += int(sum(remaining))
        all_pixel += int(size[0]) * int(size[1])
    return {
        "pixAcc": float(total_correct / float(total_label + np.spacing(1))),
        "mIoU": float(total_inter / float(total_union + np.spacing(1))),
        "nIoU": float(np.mean(sample_ious)),
        "Pd": float(pd / float(max(target, 1))),
        "Fa": float(dismatch_pixel / float(max(all_pixel, 1))),
    }


def image_features(prob: np.ndarray) -> list[float]:
    flat = prob.reshape(-1).astype(np.float32)
    top = np.sort(flat)
    feats: list[float] = []
    for q in (90, 95, 97, 98, 99, 99.5, 99.9, 99.95):
        feats.append(float(np.percentile(flat, q)))
    for k in (8, 16, 32, 64, 128, 256):
        feats.append(float(top[-min(k, flat.size) :].mean()))
    for th in (0.40, 0.44, 0.456, 0.472, 0.488, 0.50, 0.54):
        binary = flat > th
        feats.append(float(binary.mean()))
        num, _lab, stats, _ = cv2.connectedComponentsWithStats(binary.reshape(prob.shape).astype(np.uint8), 8)
        areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, num)]
        feats.append(float(len(areas)))
        feats.append(float(max(areas) if areas else 0))
        feats.append(float(np.mean(areas) if areas else 0))
    h, w = prob.shape
    feats.extend([float(h), float(w), float(h * w), float(h) / float(max(w, 1))])
    return feats


def load_npz(path: Path, key: str):
    data = np.load(path, allow_pickle=True)
    names = [str(x) for x in list(data["names"])]
    maps = [np.asarray(x, dtype=np.float32) for x in list(data[key])]
    masks = [np.asarray(x, dtype=np.float32) for x in list(data["masks"])]
    sizes = [list(map(int, x)) for x in list(data["sizes"])]
    return names, maps, masks, sizes


def candidate_tables(maps, masks, thresholds, min_areas):
    candidates = [(float(th), int(area)) for th in thresholds for area in min_areas]
    preds_by_candidate: list[list[np.ndarray]] = []
    ious = np.zeros((len(maps), len(candidates)), dtype=np.float32)
    for ci, (th, area) in enumerate(candidates):
        preds = [min_area_filter(prob, th, area) for prob in maps]
        preds_by_candidate.append(preds)
        for i, (pred, mask) in enumerate(zip(preds, masks)):
            gt = mask > 0.5
            inter = float((pred & gt).sum())
            union = float((pred | gt).sum())
            ious[i, ci] = inter / float(union + np.spacing(1))
    return candidates, preds_by_candidate, ious


def standardize(train_x: np.ndarray, test_x: np.ndarray):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    return ((train_x - mean) / std).astype(np.float32), ((test_x - mean) / std).astype(np.float32)


def weighted_vote_indices(train_x, target_x, train_ious, k: int, temp: float, leave_one_out: bool):
    chosen = []
    for i, x in enumerate(target_x):
        d = np.linalg.norm(train_x - x[None, :], axis=1)
        if leave_one_out:
            d[i] = np.inf
        order = np.argsort(d)[:k]
        local_d = d[order]
        scale = float(np.median(local_d[np.isfinite(local_d)])) if np.isfinite(local_d).any() else 1.0
        scale = max(scale, 1e-6)
        weights = np.exp(-local_d / (scale * temp + 1e-6)).astype(np.float32)
        if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
            weights = np.ones_like(local_d, dtype=np.float32)
        scores = (train_ious[order] * weights[:, None]).sum(axis=0) / float(weights.sum())
        chosen.append(int(np.argmax(scores)))
    return chosen


def preds_from_indices(preds_by_candidate, indices):
    return [preds_by_candidate[int(ci)][i] for i, ci in enumerate(indices)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_npz", required=True)
    parser.add_argument("--test_npz", required=True)
    parser.add_argument("--map_key", default="maps_best")
    parser.add_argument("--thresholds", default="0.44,0.448,0.456,0.46,0.464,0.468,0.472,0.476,0.48,0.488,0.496,0.504,0.512")
    parser.add_argument("--min_areas", default="1,2,3,4,5,6,8,10,12")
    parser.add_argument("--ks", default="3,5,9,15,25,41")
    parser.add_argument("--temps", default="0.5,1.0,1.5,2.5")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_names, train_maps, train_masks, train_sizes = load_npz(Path(args.train_npz), args.map_key)
    test_names, test_maps, test_masks, test_sizes = load_npz(Path(args.test_npz), args.map_key)
    thresholds = parse_values(args.thresholds)
    min_areas = parse_values(args.min_areas, int)

    candidates, train_preds_by_candidate, train_ious = candidate_tables(train_maps, train_masks, thresholds, min_areas)
    _test_candidates, test_preds_by_candidate, test_ious = candidate_tables(test_maps, test_masks, thresholds, min_areas)
    train_x_raw = np.asarray([image_features(prob) for prob in train_maps], dtype=np.float32)
    test_x_raw = np.asarray([image_features(prob) for prob in test_maps], dtype=np.float32)
    train_x, test_x = standardize(train_x_raw, test_x_raw)

    rows: list[dict[str, object]] = []
    for ci, (th, area) in enumerate(candidates):
        train_metric = metric_binary(train_preds_by_candidate[ci], train_masks, train_sizes)
        test_metric = metric_binary(test_preds_by_candidate[ci], test_masks, test_sizes)
        rows.append({"kind": "global", "threshold": th, "min_area": area, "train_mIoU": train_metric["mIoU"], **test_metric})

    train_oracle_idx = np.argmax(train_ious, axis=1).astype(int)
    test_oracle_idx = np.argmax(test_ious, axis=1).astype(int)
    train_oracle_metric = metric_binary(preds_from_indices(train_preds_by_candidate, train_oracle_idx), train_masks, train_sizes)
    test_oracle_metric = metric_binary(preds_from_indices(test_preds_by_candidate, test_oracle_idx), test_masks, test_sizes)

    for k in parse_values(args.ks, int):
        for temp in parse_values(args.temps):
            loo_idx = weighted_vote_indices(train_x, train_x, train_ious, min(k, len(train_x) - 1), temp, True)
            loo_metric = metric_binary(preds_from_indices(train_preds_by_candidate, loo_idx), train_masks, train_sizes)
            test_idx = weighted_vote_indices(train_x, test_x, train_ious, min(k, len(train_x)), temp, False)
            test_metric = metric_binary(preds_from_indices(test_preds_by_candidate, test_idx), test_masks, test_sizes)
            rows.append(
                {
                    "kind": "weighted_neighbor_candidate_vote",
                    "k": k,
                    "temp": temp,
                    "train_loo_mIoU": loo_metric["mIoU"],
                    **test_metric,
                }
            )

    rows.sort(key=lambda row: (row.get("train_loo_mIoU", row.get("train_mIoU", -1.0)), row["mIoU"]), reverse=True)
    selected_by_train = rows[0]
    rows_by_test = sorted(rows, key=lambda row: row["mIoU"], reverse=True)
    out = {
        "args": vars(args),
        "selected_by_train": selected_by_train,
        "best_by_test_diagnostic": rows_by_test[0],
        "train_oracle": train_oracle_metric,
        "test_oracle_not_valid": test_oracle_metric,
        "rows_by_train": rows[:100],
        "rows_by_test": rows_by_test[:100],
    }
    print(
        f"selected kind={selected_by_train['kind']} train={selected_by_train.get('train_loo_mIoU', selected_by_train.get('train_mIoU', -1))*100:.4f} "
        f"test={selected_by_train['mIoU']*100:.4f} best_test_diag={rows_by_test[0]['mIoU']*100:.4f} "
        f"test_oracle={test_oracle_metric['mIoU']*100:.4f}",
        flush=True,
    )
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
