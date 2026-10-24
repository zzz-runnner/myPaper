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


def parse_floats(spec: str) -> list[float]:
    vals = []
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" in item:
            start, stop, step = [float(x) for x in item.split(":")]
            cur = start
            while cur <= stop + 1e-12:
                vals.append(round(cur, 6))
                cur += step
        else:
            vals.append(float(item))
    return sorted(set(vals))


def parse_ints(spec: str) -> list[int]:
    vals = []
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" in item:
            start, stop, step = [int(x) for x in item.split(":")]
            vals.extend(range(start, stop + 1, step))
        else:
            vals.append(int(item))
    return sorted(set(vals))


def filter_map(prob: np.ndarray, threshold: float, min_area: int, max_area: int, max_side: int, dilate: int) -> np.ndarray:
    binary = prob > threshold
    num_labels, label_img, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    keep = np.zeros_like(binary, dtype=np.uint8)
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        if area < min_area:
            continue
        if max_area > 0 and area > max_area:
            continue
        if max_side > 0 and max(width, height) > max_side:
            continue
        keep[label_img == label_id] = 1
    if dilate > 0:
        k = dilate * 2 + 1
        keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1)
    return keep.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep connected-component filters over saved probability maps.")
    parser.add_argument("--maps_npz", required=True)
    parser.add_argument("--map_key", default="maps_no_tta")
    parser.add_argument("--thresholds", default="0.32:0.40:0.002")
    parser.add_argument("--min_areas", default="1,2,3,4")
    parser.add_argument("--max_areas", default="0,80,160,320")
    parser.add_argument("--max_sides", default="0,16,24,32,48")
    parser.add_argument("--dilates", default="0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = np.load(args.maps_npz, allow_pickle=True)
    maps = list(data[args.map_key])
    names = list(data["names"])
    sizes = data["sizes"]
    masks = list(data["masks"])
    samples = [{"name": str(n), "mask": m.astype(np.float32), "size": [int(s[0]), int(s[1])]} for n, s, m in zip(names, sizes, masks)]
    rows = []
    for th in parse_floats(args.thresholds):
        raw_metrics = metric_from_maps(samples, maps, th)
        rows.append({"kind": "raw", "threshold": th, "min_area": 1, "max_area": 0, "max_side": 0, "dilate": 0, **raw_metrics})
        for min_area in parse_ints(args.min_areas):
            for max_area in parse_ints(args.max_areas):
                for max_side in parse_ints(args.max_sides):
                    for dilate in parse_ints(args.dilates):
                        if min_area <= 1 and max_area <= 0 and max_side <= 0 and dilate <= 0:
                            continue
                        filtered = [filter_map(prob, th, min_area, max_area, max_side, dilate) for prob in maps]
                        metrics = metric_from_maps(samples, filtered, 0.5)
                        rows.append(
                            {
                                "kind": "component_filter",
                                "threshold": th,
                                "min_area": min_area,
                                "max_area": max_area,
                                "max_side": max_side,
                                "dilate": dilate,
                                **metrics,
                            }
                        )
    rows.sort(key=lambda r: r["mIoU"], reverse=True)
    best = rows[0]
    print(
        f"best kind={best['kind']} th={best['threshold']} mIoU={best['mIoU']*100:.4f} "
        f"nIoU={best['nIoU']*100:.4f} Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.4f}",
        flush=True,
    )
    Path(args.output).write_text(json.dumps({"args": vars(args), "best": best, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
