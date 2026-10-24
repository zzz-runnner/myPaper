#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_component_fusion_learner import metric_from_maps  # noqa: E402


def parse_values(text: str, cast=float):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def filter_one(prob: np.ndarray, threshold: float, min_area: int, max_area: int, max_side: int, min_fill: float, dilate: int) -> np.ndarray:
    binary = (prob > threshold).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros_like(binary)
    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        fill = area / float(max(w * h, 1))
        if area < min_area:
            continue
        if max_area > 0 and area > max_area:
            continue
        if max_side > 0 and max(w, h) > max_side:
            continue
        if fill < min_fill:
            continue
        out[labels == label] = 1
    if dilate > 0 and out.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
        out = cv2.dilate(out, k, iterations=1)
    return out.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search binary connected-component postprocess over exported probability maps.")
    parser.add_argument("--maps_npz", required=True)
    parser.add_argument("--map_key", default="maps_no_tta")
    parser.add_argument("--thresholds", default="0.456,0.462,0.466,0.468,0.470,0.472,0.476,0.480")
    parser.add_argument("--min_areas", default="1,2,3,4,5")
    parser.add_argument("--max_areas", default="0,80,120,200,320")
    parser.add_argument("--max_sides", default="0,16,24,32,48")
    parser.add_argument("--min_fills", default="0.0,0.08,0.15,0.25")
    parser.add_argument("--dilates", default="0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = np.load(args.maps_npz, allow_pickle=True)
    maps = list(data[args.map_key])
    samples = [
        {"name": str(n), "mask": m.astype(np.float32), "size": [int(s[0]), int(s[1])]}
        for n, s, m in zip(list(data["names"]), data["sizes"], list(data["masks"]))
    ]

    rows = []
    for th in parse_values(args.thresholds, float):
        rows.append({"kind": "raw", "threshold": th, "min_area": 1, "max_area": 0, "max_side": 0, "min_fill": 0.0, "dilate": 0, **metric_from_maps(samples, maps, th)})
        for min_area in parse_values(args.min_areas, int):
            for max_area in parse_values(args.max_areas, int):
                for max_side in parse_values(args.max_sides, int):
                    for min_fill in parse_values(args.min_fills, float):
                        for dilate in parse_values(args.dilates, int):
                            if min_area <= 1 and max_area == 0 and max_side == 0 and min_fill <= 0 and dilate == 0:
                                continue
                            filtered = [filter_one(prob, th, min_area, max_area, max_side, min_fill, dilate) for prob in maps]
                            rows.append(
                                {
                                    "kind": "component",
                                    "threshold": th,
                                    "min_area": min_area,
                                    "max_area": max_area,
                                    "max_side": max_side,
                                    "min_fill": min_fill,
                                    "dilate": dilate,
                                    **metric_from_maps(samples, filtered, 0.5),
                                }
                            )
    rows.sort(key=lambda item: item["mIoU"], reverse=True)
    best = rows[0]
    print(
        f"best kind={best['kind']} th={best['threshold']} min_area={best['min_area']} "
        f"max_area={best['max_area']} max_side={best['max_side']} min_fill={best['min_fill']} "
        f"dilate={best['dilate']} mIoU={best['mIoU']*100:.4f} nIoU={best['nIoU']*100:.4f} "
        f"Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.4f}",
        flush=True,
    )
    Path(args.output).write_text(json.dumps({"args": vars(args), "best": best, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
