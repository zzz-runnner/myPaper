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
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(cast(part))
    return out


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps_npz", required=True)
    parser.add_argument("--map_key", default="maps_no_tta")
    parser.add_argument("--thresholds", default="0.328,0.34,0.35,0.36,0.368,0.376,0.384")
    parser.add_argument("--min_areas", default="1,2,3,4,5")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = np.load(args.maps_npz, allow_pickle=True)
    maps = list(data[args.map_key])
    names = list(data["names"])
    sizes = data["sizes"]
    masks = list(data["masks"])
    samples = [{"name": str(n), "mask": m.astype(np.float32), "size": [int(s[0]), int(s[1])]} for n, s, m in zip(names, sizes, masks)]

    rows = []
    for th in parse_values(args.thresholds, float):
        rows.append({"kind": "raw", "threshold": th, "min_area": 1, **metric_from_maps(samples, maps, th)})
        for min_area in parse_values(args.min_areas, int):
            if min_area <= 1:
                continue
            filtered = [min_area_filter(prob, th, min_area) for prob in maps]
            rows.append({"kind": "min_area", "threshold": th, "min_area": min_area, **metric_from_maps(samples, filtered, 0.5)})
    rows.sort(key=lambda item: item["mIoU"], reverse=True)
    best = rows[0]
    print(
        f"best kind={best['kind']} th={best['threshold']} min_area={best['min_area']} "
        f"mIoU={best['mIoU']*100:.4f} nIoU={best['nIoU']*100:.4f} Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.4f}",
        flush=True,
    )
    Path(args.output).write_text(json.dumps({"args": vars(args), "best": best, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
