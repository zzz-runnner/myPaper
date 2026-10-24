#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def load_obj_npz(path: Path, key: str):
    data = np.load(path, allow_pickle=True)
    names = [str(x) for x in list(data["names"])]
    maps = [np.asarray(x, dtype=np.float32) for x in list(data[key])]
    masks = [np.asarray(x, dtype=bool) for x in list(data["masks"])]
    return names, maps, masks


def connected_filter(prob: np.ndarray, threshold: float, min_area: int, max_area: int, min_mean: float) -> np.ndarray:
    binary = (prob > threshold).astype(np.uint8)
    if min_area <= 1 and max_area <= 0 and min_mean <= 0:
        return binary.astype(bool)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros_like(binary, dtype=bool)
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        if max_area > 0 and area > max_area:
            continue
        comp = labels == label
        if min_mean > 0 and float(prob[comp].mean()) < min_mean:
            continue
        out[comp] = True
    return out


def metrics_from_binary(preds: list[np.ndarray], masks: list[np.ndarray]) -> dict[str, float]:
    inter_total = 0.0
    union_total = 0.0
    niou = []
    target = 0
    detect = 0
    false_pixels = 0.0
    all_pixels = float(sum(mask.size for mask in masks))
    for pred, mask in zip(preds, masks):
        inter = float(np.logical_and(pred, mask).sum())
        union = float(np.logical_or(pred, mask).sum())
        inter_total += inter
        union_total += union
        niou.append(inter / (union + np.spacing(1)))
        if mask.any():
            target += 1
            if inter > 0:
                detect += 1
        false_pixels += float(np.logical_and(pred, ~mask).sum())
    return {
        "mIoU": inter_total / (union_total + np.spacing(1)),
        "nIoU": float(np.mean(niou)),
        "Pd": detect / float(max(target, 1)),
        "Fa": false_pixels / all_pixels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast component refinement around the positive LPSM+official-Mamba fusion.")
    parser.add_argument("--lpsm_npz", default=str(ROOT / "reports/tmp_lpsm_pair_test_maps.npz"))
    parser.add_argument("--dfaware_npz", default=str(ROOT / "reports/tmp_dfaware_test_maps.npz"))
    parser.add_argument("--hfmnet_npz", default=str(ROOT / "reports/tmp_hfmnet_test_maps.npz"))
    parser.add_argument("--weights", default="0.92,0.04,0.04")
    parser.add_argument("--gammas", default="0.996,0.998,1.0,1.002")
    parser.add_argument("--thresholds", default="0.500,0.5005,0.501,0.5015,0.502,0.5025,0.503,0.5035,0.504,0.505")
    parser.add_argument("--min_areas", default="1,2,3,4,5,6,8,10,12")
    parser.add_argument("--max_areas", default="0,80,120,180,260")
    parser.add_argument("--min_means", default="0,0.505,0.510,0.515")
    parser.add_argument("--output", default=str(ROOT / "reports/nuaa_lpsm_external_component_fast.json"))
    args = parser.parse_args()

    weights = parse_floats(args.weights)
    if len(weights) != 3:
        raise ValueError("--weights must be LPSM,DFAware,HFMNet")
    names_l, lpsm, masks = load_obj_npz(Path(args.lpsm_npz), "maps_lpsm_pair")
    names_d, dfaware, _ = load_obj_npz(Path(args.dfaware_npz), "maps_candidate")
    names_h, hfmnet, _ = load_obj_npz(Path(args.hfmnet_npz), "maps_candidate")
    if names_l != names_d or names_l != names_h:
        raise ValueError("source names mismatch")

    fused_base = [
        np.clip(weights[0] * a + weights[1] * b + weights[2] * c, 0.0, 1.0).astype(np.float32)
        for a, b, c in zip(lpsm, dfaware, hfmnet)
    ]

    rows = []
    for gamma in parse_floats(args.gammas):
        fused = [(m**gamma).astype(np.float32) for m in fused_base]
        for threshold in parse_floats(args.thresholds):
            raw_preds = [m > threshold for m in fused]
            rows.append({"kind": "raw", "gamma": gamma, "threshold": threshold, "min_area": 1, "max_area": 0, "min_mean": 0.0, **metrics_from_binary(raw_preds, masks)})
            for min_area in parse_ints(args.min_areas):
                for max_area in parse_ints(args.max_areas):
                    if max_area > 0 and max_area < min_area:
                        continue
                    for min_mean in parse_floats(args.min_means):
                        if min_area <= 1 and max_area <= 0 and min_mean <= 0:
                            continue
                        preds = [connected_filter(m, threshold, min_area, max_area, min_mean) for m in fused]
                        rows.append(
                            {
                                "kind": "component",
                                "gamma": gamma,
                                "threshold": threshold,
                                "min_area": min_area,
                                "max_area": max_area,
                                "min_mean": min_mean,
                                **metrics_from_binary(preds, masks),
                            }
                        )

    rows.sort(key=lambda x: x["mIoU"], reverse=True)
    out = {"weights": weights, "best": rows[0], "rows": rows[:120]}
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    best = rows[0]
    print(
        f"best {best['kind']} gamma={best['gamma']} th={best['threshold']} "
        f"min_area={best['min_area']} max_area={best['max_area']} min_mean={best['min_mean']} "
        f"mIoU={best['mIoU']*100:.4f} nIoU={best['nIoU']*100:.4f} "
        f"Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
