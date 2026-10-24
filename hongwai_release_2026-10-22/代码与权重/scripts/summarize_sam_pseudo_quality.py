#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def empty_metrics() -> dict[str, int]:
    return {
        "teacher_inter": 0,
        "teacher_union": 0,
        "teacher_pred": 0,
        "teacher_gt": 0,
        "refined_inter": 0,
        "refined_union": 0,
        "refined_pred": 0,
        "refined_gt": 0,
    }


def quality_to_counts(item: dict[str, float]) -> dict[str, int]:
    out = empty_metrics()
    gt = int(round(float(item.get("gt_pixels", 0.0))))
    for prefix in ("teacher", "refined"):
        pred = int(round(float(item.get(f"{prefix}_pred_pixels", 0.0))))
        precision = float(item.get(f"{prefix}_precision", 0.0)) / 100.0
        recall = float(item.get(f"{prefix}_recall", 0.0)) / 100.0
        inter_from_pred = pred * precision
        inter_from_gt = gt * recall
        inter = int(round((inter_from_pred + inter_from_gt) * 0.5))
        union = int(round(inter / max(float(item.get(f"{prefix}_mIoU", 0.0)) / 100.0, 1e-12)))
        out[f"{prefix}_inter"] = inter
        out[f"{prefix}_union"] = union
        out[f"{prefix}_pred"] = pred
        out[f"{prefix}_gt"] = gt
    return out


def add_counts(dst: dict[str, int], src: dict[str, int]) -> None:
    for key, value in src.items():
        dst[key] = dst.get(key, 0) + int(value)


def finalize(metrics: dict[str, int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for prefix in ("teacher", "refined"):
        inter = metrics[f"{prefix}_inter"]
        union = metrics[f"{prefix}_union"]
        pred = metrics[f"{prefix}_pred"]
        gt = metrics[f"{prefix}_gt"]
        out[f"{prefix}_mIoU"] = 100.0 * inter / max(1, union)
        out[f"{prefix}_precision"] = 100.0 * inter / max(1, pred)
        out[f"{prefix}_recall"] = 100.0 * inter / max(1, gt)
        out[f"{prefix}_pred_pixels"] = float(pred)
    out["gt_pixels"] = float(metrics["teacher_gt"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge SAM pseudo-label shard quality summaries.")
    parser.add_argument("pseudo_dir")
    args = parser.parse_args()

    root = Path(args.pseudo_dir)
    summaries = sorted(root.glob("summary_shard*_of_*.json"))
    if not summaries:
        summaries = sorted(root.glob("summary.json"))
    merged: dict[str, dict[str, dict[str, int]]] = {}
    meta = {"pseudo_dir": str(root), "summary_files": [str(path) for path in summaries], "samples": {}}
    for path in summaries:
        with path.open("r") as f:
            data = json.load(f)
        for dataset, sample_info in data.get("samples", {}).items():
            meta["samples"].setdefault(dataset, 0)
            meta["samples"][dataset] += int(sample_info.get("processed_by_this_shard", 0))
        for dataset, by_threshold in data.get("quality", {}).items():
            dataset_bucket = merged.setdefault(dataset, {})
            for threshold, item in by_threshold.items():
                threshold_bucket = dataset_bucket.setdefault(threshold, empty_metrics())
                add_counts(threshold_bucket, quality_to_counts(item))

    out = {
        **meta,
        "quality": {
            dataset: {threshold: finalize(metrics) for threshold, metrics in sorted(by_threshold.items())}
            for dataset, by_threshold in sorted(merged.items())
        },
    }
    out_path = root / "summary_merged.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
