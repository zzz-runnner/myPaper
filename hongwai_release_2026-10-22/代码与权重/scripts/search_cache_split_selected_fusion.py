#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))

from eval_cached_prediction_fusion import load_cache  # noqa: E402
from eval_component_fusion_learner import metric_from_maps  # noqa: E402


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def stable_bucket(name: str, seed: str) -> float:
    raw = hashlib.sha1(f"{seed}:{name}".encode("utf-8")).hexdigest()[:8]
    return int(raw, 16) / float(0xFFFFFFFF)


def select_split(samples: list[dict[str, object]], fraction: float, seed: str) -> list[dict[str, object]]:
    return [s for s in samples if stable_bucket(str(s["name"]), seed) < fraction]


def weight_grid(names: list[str], steps: list[float], mode: str) -> list[dict[str, float]]:
    rows = []
    if mode == "all":
        for values in itertools.product(steps, repeat=len(names)):
            rows.append({name: float(value) for name, value in zip(names, values)})
    elif mode != "pconv_only":
        raise ValueError(f"unknown weight grid mode: {mode}")
    # Keep the grid compact and centered around the known strong avg region.
    if mode == "pconv_only":
        for p43_boost in steps:
            for d05_boost in steps:
                weights = {name: 1.0 for name in names}
                if "pconv_p43" in weights:
                    weights["pconv_p43"] = p43_boost
                if "pconv_d05" in weights:
                    weights["pconv_d05"] = d05_boost
                rows.append(weights)
    rows.append({name: 1.0 for name in names})
    # De-duplicate while preserving order.
    out = []
    seen = set()
    for row in rows:
        key = tuple((name, row[name]) for name in names)
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def render(samples: list[dict[str, object]], names: list[str], weights: dict[str, float]) -> list[np.ndarray]:
    w = np.asarray([weights[name] for name in names], dtype=np.float32)
    w = w / max(float(w.sum()), 1e-6)
    maps = []
    for sample in samples:
        stack = np.stack([sample["preds"][name] for name in names], axis=0).astype(np.float32)
        maps.append(np.tensordot(w, stack, axes=(0, 0)).astype(np.float32))
    return maps


def fast_niou(samples: list[dict[str, object]], maps: list[np.ndarray], threshold: float) -> float:
    vals = []
    for sample, pred in zip(samples, maps):
        mask = sample["mask"].astype(bool)
        pred_bin = pred > threshold
        inter = float((pred_bin & mask).sum())
        union = float((pred_bin | mask).sum())
        vals.append(inter / (union + np.spacing(1)))
    return float(np.mean(vals)) if vals else 0.0


def add_val_metric(row: dict[str, object], samples: list[dict[str, object]], maps: list[np.ndarray], threshold: float) -> dict[str, object]:
    metric = metric_from_maps(samples, maps, threshold)
    return {**row, **{f"val_{key}": value for key, value in metric.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Select fusion weights/thresholds on a train split and evaluate test cache.")
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--test_cache", required=True)
    parser.add_argument("--val_fraction", type=float, default=0.25)
    parser.add_argument("--split_seed", default="20260614")
    parser.add_argument("--thresholds", default="0.30,0.35,0.38,0.40,0.42,0.45,0.48,0.50,0.55")
    parser.add_argument("--weight_steps", default="0.70,0.85,1.00,1.15,1.30")
    parser.add_argument("--weight_mode", default="pconv_only", choices=["pconv_only", "all"])
    parser.add_argument("--select_metric", default="mIoU", choices=["mIoU", "nIoU"])
    parser.add_argument("--diagnostic_topk", type=int, default=20)
    parser.add_argument("--val_metric_topk", type=int, default=60)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_samples, names = load_cache(Path(args.train_cache))
    test_samples, test_names = load_cache(Path(args.test_cache))
    if names != test_names:
        raise ValueError(f"name mismatch: {names} != {test_names}")
    val_samples = select_split(train_samples, args.val_fraction, args.split_seed)
    thresholds = parse_float_list(args.thresholds)
    rows = []
    for weights in weight_grid(names, parse_float_list(args.weight_steps), args.weight_mode):
        val_maps = render(val_samples, names, weights)
        for threshold in thresholds:
            rows.append({"weights": weights, "threshold": threshold, "val_nIoU": fast_niou(val_samples, val_maps, threshold)})
    rows.sort(key=lambda r: r["val_nIoU"], reverse=True)
    if args.select_metric == "mIoU":
        rescored = []
        for row in rows[: args.val_metric_topk]:
            maps = render(val_samples, names, row["weights"])
            rescored.append(add_val_metric(row, val_samples, maps, float(row["threshold"])))
        rescored.sort(key=lambda r: r["val_mIoU"], reverse=True)
        selected = rescored[0]
    else:
        selected = rows[0]
    test_maps = render(test_samples, names, selected["weights"])
    test_metric = metric_from_maps(test_samples, test_maps, float(selected["threshold"]))
    selected_test = {**selected, **{f"test_{k}": v for k, v in test_metric.items()}}
    # Also report the best test row in the validation top-k as a diagnostic only.
    diag = []
    for row in rows[: args.diagnostic_topk]:
        maps = render(test_samples, names, row["weights"])
        metric = metric_from_maps(test_samples, maps, float(row["threshold"]))
        diag.append({**row, **{f"test_{k}": v for k, v in metric.items()}})
    diag.sort(key=lambda r: r["test_mIoU"], reverse=True)
    print(
        f"selected val_nIoU={selected['val_nIoU']*100:.2f} "
        f"th={selected['threshold']} test_mIoU={selected_test['test_mIoU']*100:.2f} "
        f"Pd={selected_test['test_Pd']*100:.2f} Fa={selected_test['test_Fa']*1e6:.2f}",
        flush=True,
    )
    print(
        f"best_test_in_val_top20 th={diag[0]['threshold']} test_mIoU={diag[0]['test_mIoU']*100:.2f}",
        flush=True,
    )
    Path(args.output).write_text(
        json.dumps(
            {
                "train_cache": args.train_cache,
                "test_cache": args.test_cache,
                "models": names,
                "val_fraction": args.val_fraction,
                "split_seed": args.split_seed,
                "weight_mode": args.weight_mode,
                "select_metric": args.select_metric,
                "val_samples": len(val_samples),
                "selected_by_val": selected_test,
                "best_test_in_val_topk_not_selected": diag[0],
                "best_test_in_val_top20_not_selected": diag[0],
                "val_rows_top": rows[:100],
                "test_rows_for_val_top20": diag,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
