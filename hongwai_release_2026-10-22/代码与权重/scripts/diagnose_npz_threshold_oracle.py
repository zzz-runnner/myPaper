#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
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


def score_binary(preds: list[np.ndarray], masks: list[np.ndarray]) -> dict[str, float]:
    inter_total = 0.0
    union_total = 0.0
    niou = []
    detect = 0
    target = 0
    false_pixels = 0.0
    total_pixels = 0.0
    for pred, mask in zip(preds, masks):
        m = mask > 0.5
        inter = float(np.logical_and(pred, m).sum())
        union = float(np.logical_or(pred, m).sum())
        inter_total += inter
        union_total += union
        niou.append(inter / (union + np.spacing(1)))
        if m.any():
            target += 1
            if inter > 0:
                detect += 1
        false_pixels += float(np.logical_and(pred, ~m).sum())
        total_pixels += float(m.size)
    return {
        "mIoU": inter_total / (union_total + np.spacing(1)),
        "nIoU": float(np.mean(niou)),
        "Pd": float(detect) / float(max(target, 1)),
        "Fa": false_pixels / float(total_pixels),
    }


def image_stats(prob: np.ndarray) -> dict[str, float]:
    flat = prob.reshape(-1)
    return {
        "p99": float(np.percentile(flat, 99.0)),
        "p999": float(np.percentile(flat, 99.9)),
        "mean_top32": float(np.sort(flat)[-min(32, flat.size) :].mean()),
        "mean_top128": float(np.sort(flat)[-min(128, flat.size) :].mean()),
        "active_047": float((flat > 0.47).mean()),
        "active_050": float((flat > 0.50).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps_npz", required=True)
    parser.add_argument("--map_key", default="maps_best")
    parser.add_argument("--thresholds", default="0.44,0.448,0.456,0.464,0.468,0.472,0.476,0.480,0.488,0.496,0.504,0.512,0.52,0.54,0.56")
    parser.add_argument("--min_areas", default="1,2,3,4,5,6,8,10,12")
    parser.add_argument("--skip_adaptive", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = np.load(args.maps_npz, allow_pickle=True)
    maps = [np.asarray(x, dtype=np.float32) for x in list(data[args.map_key])]
    masks = [np.asarray(x, dtype=np.float32) for x in list(data["masks"])]
    names = [str(x) for x in list(data["names"])]
    thresholds = parse_values(args.thresholds, float)
    min_areas = parse_values(args.min_areas, int)

    global_rows = []
    for th in thresholds:
        for area in min_areas:
            preds = [min_area_filter(prob, th, area) for prob in maps]
            global_rows.append({"threshold": th, "min_area": area, **score_binary(preds, masks)})
    global_rows.sort(key=lambda row: row["mIoU"], reverse=True)

    oracle_threshold_preds = []
    per_image_threshold = []
    for name, prob, mask in zip(names, maps, masks):
        rows = []
        for th in thresholds:
            pred = prob > th
            rows.append({"threshold": th, **score_binary([pred], [mask])})
        rows.sort(key=lambda row: row["mIoU"], reverse=True)
        per_image_threshold.append({"name": name, **rows[0], **image_stats(prob)})
        oracle_threshold_preds.append(prob > rows[0]["threshold"])
    oracle_threshold = score_binary(oracle_threshold_preds, masks)

    oracle_component_preds = []
    per_image_component = []
    for name, prob, mask in zip(names, maps, masks):
        rows = []
        for th in thresholds:
            for area in min_areas:
                pred = min_area_filter(prob, th, area)
                rows.append({"threshold": th, "min_area": area, **score_binary([pred], [mask])})
        rows.sort(key=lambda row: row["mIoU"], reverse=True)
        per_image_component.append({"name": name, **rows[0], **image_stats(prob)})
        oracle_component_preds.append(min_area_filter(prob, rows[0]["threshold"], rows[0]["min_area"]))
    oracle_component = score_binary(oracle_component_preds, masks)

    adaptive_rows = []
    if not args.skip_adaptive:
        # Simple adaptive rule: threshold = base + a*(p999 - median_p999) + b*(active_047 - median_active)
        stats = [image_stats(prob) for prob in maps]
        p999 = np.asarray([s["p999"] for s in stats], dtype=np.float32)
        active = np.asarray([s["active_047"] for s in stats], dtype=np.float32)
        p999_c = p999 - float(np.median(p999))
        active_c = active - float(np.median(active))
        for base in thresholds:
            for a in (-0.30, -0.18, -0.10, 0.0, 0.10, 0.18, 0.30):
                for b in (-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0):
                    for area in min_areas:
                        preds = []
                        used = []
                        for idx, prob in enumerate(maps):
                            th = float(np.clip(base + a * p999_c[idx] + b * active_c[idx], 0.35, 0.65))
                            preds.append(min_area_filter(prob, th, area))
                            used.append(th)
                        adaptive_rows.append(
                            {
                                "base": base,
                                "a_p999": a,
                                "b_active": b,
                                "min_area": area,
                                "mean_threshold": float(np.mean(used)),
                                **score_binary(preds, masks),
                            }
                        )
        adaptive_rows.sort(key=lambda row: row["mIoU"], reverse=True)

    out = {
        "args": vars(args),
        "best_global": global_rows[0],
        "oracle_threshold": oracle_threshold,
        "oracle_component": oracle_component,
        "best_adaptive": adaptive_rows[0] if adaptive_rows else None,
        "global_rows": global_rows[:100],
        "adaptive_rows": adaptive_rows[:100],
        "per_image_threshold": per_image_threshold,
        "per_image_component": per_image_component,
    }
    print(
        f"best_global mIoU={out['best_global']['mIoU']*100:.4f} "
        f"oracle_th={oracle_threshold['mIoU']*100:.4f} "
        f"oracle_comp={oracle_component['mIoU']*100:.4f} "
        f"best_adapt={(out['best_adaptive']['mIoU']*100 if out['best_adaptive'] else -1):.4f}",
        flush=True,
    )
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
