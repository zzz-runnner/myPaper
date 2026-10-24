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

from eval_component_fusion_learner import FullSplitSet, metric_from_maps, robust01  # noqa: E402


def parse_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def pmd_score(image: np.ndarray, mode: str) -> np.ndarray:
    src = robust01(image.astype(np.float32), 0.5, 99.5)
    if mode == "bright":
        base = src
    elif mode == "tophat":
        cues = []
        for k in (5, 9, 13, 17):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            cues.append(robust01(cv2.morphologyEx(src, cv2.MORPH_TOPHAT, kernel), 50.0, 99.8))
        base = np.stack(cues, axis=0).max(axis=0)
    elif mode == "gradient":
        gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)
        base = np.sqrt(gx * gx + gy * gy)
    elif mode == "curvature":
        gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)
        gxx = cv2.Sobel(src, cv2.CV_32F, 2, 0, ksize=5)
        gyy = cv2.Sobel(src, cv2.CV_32F, 0, 2, ksize=5)
        gxy = cv2.Sobel(src, cv2.CV_32F, 1, 1, ksize=3)
        denom = np.power(gx * gx + gy * gy, 1.5) + 1e-6
        base = np.abs(gx * gx * gyy - 2.0 * gx * gy * gxy + gy * gy * gxx) / denom
    elif mode == "pmd_mix":
        gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)
        grad = robust01(np.sqrt(gx * gx + gy * gy), 55.0, 99.8)
        lap = robust01(np.abs(cv2.Laplacian(src, cv2.CV_32F, ksize=3)), 55.0, 99.8)
        cues = []
        for k in (5, 9, 13):
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            cues.append(robust01(cv2.morphologyEx(src, cv2.MORPH_TOPHAT, kernel), 55.0, 99.8))
        bright_small = np.stack(cues, axis=0).max(axis=0)
        base = 0.45 * bright_small + 0.35 * lap + 0.20 * grad
    else:
        raise ValueError(f"unknown mode: {mode}")
    return robust01(base, 50.0, 99.8)


def topk_candidate(score: np.ndarray, threshold: float, topk: int, min_area: int, max_area: int, max_side: int) -> np.ndarray:
    seed = score > threshold
    num_labels, label_img, stats, _ = cv2.connectedComponentsWithStats(seed.astype(np.uint8), 8)
    candidates: list[tuple[float, int]] = []
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
        comp = label_img == label_id
        fill = float(area) / max(float(width * height), 1.0)
        compact_penalty = abs(np.log(max(width / float(max(height, 1)), 1e-3)))
        comp_score = float(score[comp].mean() + 0.75 * score[comp].max() + 0.08 * fill - 0.03 * compact_penalty)
        candidates.append((comp_score, label_id))
    out = np.zeros_like(seed, dtype=np.uint8)
    for _score, label_id in sorted(candidates, reverse=True)[: max(0, topk)]:
        out[label_img == label_id] = 1
    if out.any():
        out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return out.astype(np.float32)


def collect(dataset_dir: Path, dataset_name: str, split: str, max_samples: int) -> list[dict[str, object]]:
    dataset = FullSplitSet(dataset_dir, dataset_name, dataset_name, split, max_samples=max_samples)
    samples: list[dict[str, object]] = []
    for idx in range(len(dataset)):
        image_t, mask_t, size, name = dataset[idx]
        height, width = int(size[0]), int(size[1])
        image = image_t[0, :height, :width].numpy().astype(np.float32)
        mask = mask_t[0, :height, :width].numpy().astype(np.float32)
        samples.append({"name": name, "image": image, "mask": mask, "size": [height, width]})
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate IRSAM-PMD-inspired handcrafted candidate maps.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", default="NUAA-SIRST")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--modes", default="pmd_mix,tophat,gradient,curvature")
    parser.add_argument("--thresholds", default="0.78,0.82,0.86,0.90,0.94")
    parser.add_argument("--topks", default="4,8,12,16,24")
    parser.add_argument("--min_areas", default="1,2,3")
    parser.add_argument("--max_areas", default="64,128,256")
    parser.add_argument("--max_sides", default="16,24,32,48")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    samples = collect(Path(args.dataset_dir), args.dataset_name, args.split, args.max_samples)
    rows = []
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    scores_by_mode = {
        mode: [pmd_score(sample["image"], mode) for sample in samples]
        for mode in modes
    }
    metric_samples = [{"name": s["name"], "mask": s["mask"], "size": s["size"]} for s in samples]
    for mode, scores in scores_by_mode.items():
        for threshold in parse_floats(args.thresholds):
            rows.append({"mode": mode, "kind": "dense", "threshold": threshold, **metric_from_maps(metric_samples, scores, threshold)})
            for topk in parse_ints(args.topks):
                for min_area in parse_ints(args.min_areas):
                    for max_area in parse_ints(args.max_areas):
                        for max_side in parse_ints(args.max_sides):
                            maps = [topk_candidate(score, threshold, topk, min_area, max_area, max_side) for score in scores]
                            rows.append(
                                {
                                    "mode": mode,
                                    "kind": "topk_component",
                                    "threshold": threshold,
                                    "topk": topk,
                                    "min_area": min_area,
                                    "max_area": max_area,
                                    "max_side": max_side,
                                    **metric_from_maps(metric_samples, maps, 0.5),
                                }
                            )
    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best = rows[0]
    print(
        f"best mode={best['mode']} kind={best['kind']} th={best['threshold']} "
        f"mIoU={best['mIoU']*100:.4f} nIoU={best['nIoU']*100:.4f} "
        f"Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.4f}",
        flush=True,
    )
    Path(args.output).write_text(
        json.dumps(
            {
                "dataset": args.dataset_name,
                "split": args.split,
                "samples": len(samples),
                "best": best,
                "rows": rows[:200],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
