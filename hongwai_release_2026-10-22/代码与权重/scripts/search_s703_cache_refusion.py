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

from eval_cached_prediction_fusion import load_cache  # noqa: E402
from eval_component_fusion_learner import metric_from_maps  # noqa: E402


def parse_values(text: str, cast=float):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


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
    parser = argparse.ArgumentParser(description="Search conservative probability refusion of s703 maps with corrected cache source maps.")
    parser.add_argument("--s703_maps", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--map_key", default="maps_no_tta")
    parser.add_argument("--source_names", default="weighted_raw,strong_fusion,bgm_nuaa,rpcanet_sirst_s3,bg_anomaly")
    parser.add_argument("--alphas", default="-0.08,-0.04,-0.02,0.02,0.04,0.08,0.12")
    parser.add_argument("--gammas", default="0.92,0.96,1.0,1.04,1.08")
    parser.add_argument("--thresholds", default="0.452,0.460,0.468,0.476,0.484,0.492,0.500")
    parser.add_argument("--min_areas", default="1,2,3")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = np.load(args.s703_maps, allow_pickle=True)
    s703_maps = [x.astype(np.float32) for x in list(data[args.map_key])]
    samples = [
        {"name": str(n), "mask": m.astype(np.float32), "size": [int(s[0]), int(s[1])]}
        for n, s, m in zip(list(data["names"]), data["sizes"], list(data["masks"]))
    ]
    cache_samples, _names = load_cache(Path(args.cache))
    cache_by_name = {str(sample["name"]): sample for sample in cache_samples}
    source_names = [x.strip() for x in args.source_names.split(",") if x.strip()]
    rows = []
    thresholds = parse_values(args.thresholds, float)
    min_areas = parse_values(args.min_areas, int)

    for th in thresholds:
        rows.append({"kind": "s703", "source": "", "alpha": 0.0, "gamma": 1.0, "threshold": th, "min_area": 1, **metric_from_maps(samples, s703_maps, th)})
        for min_area in min_areas:
            if min_area <= 1:
                continue
            filtered = [min_area_filter(prob, th, min_area) for prob in s703_maps]
            rows.append({"kind": "s703_component", "source": "", "alpha": 0.0, "gamma": 1.0, "threshold": th, "min_area": min_area, **metric_from_maps(samples, filtered, 0.5)})

    for source in source_names:
        src_maps = []
        for sample in samples:
            cache_sample = cache_by_name[sample["name"]]
            src_maps.append(cache_sample["preds"][source].astype(np.float32)[: sample["size"][0], : sample["size"][1]])
        for alpha in parse_values(args.alphas, float):
            for gamma in parse_values(args.gammas, float):
                maps = []
                for base, src in zip(s703_maps, src_maps):
                    fused = np.clip((1.0 - abs(alpha)) * base + alpha * src if alpha < 0 else (1.0 - alpha) * base + alpha * src, 0.0, 1.0)
                    fused = np.clip(fused, 0.0, 1.0) ** gamma
                    maps.append(fused.astype(np.float32))
                for th in thresholds:
                    rows.append({"kind": "refusion", "source": source, "alpha": alpha, "gamma": gamma, "threshold": th, "min_area": 1, **metric_from_maps(samples, maps, th)})
                    for min_area in min_areas:
                        if min_area <= 1:
                            continue
                        filtered = [min_area_filter(prob, th, min_area) for prob in maps]
                        rows.append({"kind": "refusion_component", "source": source, "alpha": alpha, "gamma": gamma, "threshold": th, "min_area": min_area, **metric_from_maps(samples, filtered, 0.5)})

    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best = rows[0]
    print(
        f"best kind={best['kind']} source={best['source']} alpha={best['alpha']} gamma={best['gamma']} "
        f"th={best['threshold']} min_area={best['min_area']} mIoU={best['mIoU']*100:.4f} "
        f"nIoU={best['nIoU']*100:.4f} Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.4f}",
        flush=True,
    )
    Path(args.output).write_text(json.dumps({"args": vars(args), "best": best, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
