#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_component_fusion_learner import metric_from_maps  # noqa: E402
from npz_component_stump_selector import load_npz, parse_values  # noqa: E402


def min_area_filter(prob: np.ndarray, threshold: float, min_area: int) -> np.ndarray:
    binary = (prob > threshold).astype(np.uint8)
    if min_area <= 1:
        return binary.astype(np.float32)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros_like(binary)
    for label in range(1, num):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            out[labels == label] = 1
    return out.astype(np.float32)


def load_source(spec: str):
    label, path, key = spec.split("|", 2)
    samples, names, maps, _masks = load_npz(Path(path), key, 0)
    return label, samples, names, maps


def fuse_maps(source_maps: list[list[np.ndarray]], weights: tuple[float, ...], gamma: float) -> list[np.ndarray]:
    maps = []
    for per_image in zip(*source_maps):
        fused = np.zeros_like(per_image[0], dtype=np.float32)
        for weight, prob in zip(weights, per_image):
            fused += float(weight) * prob.astype(np.float32)
        maps.append((np.clip(fused, 0.0, 1.0) ** gamma).astype(np.float32))
    return maps


def main() -> None:
    parser = argparse.ArgumentParser(description="Search small external-source additions around the retained LPSM pair.")
    parser.add_argument("--source", action="append", required=True, help="LABEL|NPZ|KEY; first source should be LPSM.")
    parser.add_argument("--extra_weight_values", default="-0.04,-0.02,0,0.02,0.04,0.06,0.08,0.10")
    parser.add_argument("--max_extra_abs_sum", type=float, default=0.16)
    parser.add_argument("--gammas", default="0.996,0.998,1.0,1.002")
    parser.add_argument("--thresholds", default="0.499,0.500,0.5005,0.501,0.5015,0.502,0.503")
    parser.add_argument("--min_areas", default="1,2,3,4,5")
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    labels = []
    source_maps = []
    ref_samples = None
    ref_names = None
    for spec in args.source:
        label, samples, names, maps = load_source(spec)
        labels.append(label)
        if ref_names is None:
            ref_names = names
            ref_samples = samples
        elif names != ref_names:
            raise ValueError(f"name mismatch for {label}")
        source_maps.append(maps)
    assert ref_samples is not None

    extra_values = parse_values(args.extra_weight_values)
    gammas = parse_values(args.gammas)
    thresholds = parse_values(args.thresholds)
    min_areas = parse_values(args.min_areas, int)

    rows = []
    n_extra = len(source_maps) - 1
    for extras in itertools.product(extra_values, repeat=n_extra):
        if sum(abs(float(x)) for x in extras) > args.max_extra_abs_sum + 1e-9:
            continue
        base_weight = 1.0 - sum(float(x) for x in extras)
        if base_weight <= 0.0 or base_weight > 1.2:
            continue
        weights = (base_weight, *[float(x) for x in extras])
        for gamma in gammas:
            fused = fuse_maps(source_maps, weights, float(gamma))
            for th in thresholds:
                rows.append({"weights": weights, "gamma": gamma, "threshold": th, "min_area": 1, **metric_from_maps(ref_samples, fused, th)})
                for area in min_areas:
                    if int(area) <= 1:
                        continue
                    filtered = [min_area_filter(prob, float(th), int(area)) for prob in fused]
                    rows.append({"weights": weights, "gamma": gamma, "threshold": th, "min_area": int(area), **metric_from_maps(ref_samples, filtered, 0.5)})

    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best = rows[0]
    print(
        f"best weights={best['weights']} gamma={best['gamma']} th={best['threshold']} "
        f"area={best['min_area']} mIoU={best['mIoU']*100:.4f} nIoU={best['nIoU']*100:.4f} "
        f"Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.4f}",
        flush=True,
    )
    Path(args.output).write_text(
        json.dumps({"args": vars(args), "labels": labels, "best": best, "rows": rows[: args.top_k]}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
