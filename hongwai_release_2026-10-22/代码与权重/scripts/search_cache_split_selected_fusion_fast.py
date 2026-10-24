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


def make_weight_grid(names: list[str], steps: list[float], mode: str) -> list[dict[str, float]]:
    if mode == "all":
        rows = [{name: float(value) for name, value in zip(names, values)} for values in itertools.product(steps, repeat=len(names))]
    elif mode == "pconv_only":
        rows = []
        for p43_boost in steps:
            for d05_boost in steps:
                weights = {name: 1.0 for name in names}
                if "pconv_p43" in weights:
                    weights["pconv_p43"] = p43_boost
                if "pconv_d05" in weights:
                    weights["pconv_d05"] = d05_boost
                rows.append(weights)
    else:
        raise ValueError(mode)
    rows.append({name: 1.0 for name in names})
    out = []
    seen = set()
    for row in rows:
        key = tuple((name, row[name]) for name in names)
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def stack_samples(samples: list[dict[str, object]], names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    shapes = {tuple(sample["mask"].shape) for sample in samples}
    if len(shapes) != 1:
        raise ValueError(f"fast path requires same-size samples, got {sorted(shapes)[:5]}")
    preds = np.stack(
        [np.stack([sample["preds"][name] for name in names], axis=0).astype(np.float32) for sample in samples],
        axis=0,
    )
    masks = np.stack([sample["mask"].astype(bool) for sample in samples], axis=0)
    return preds, masks


def fast_mean_niou(preds: np.ndarray, masks: np.ndarray, weights: dict[str, float], names: list[str], threshold: float) -> float:
    w = np.asarray([weights[name] for name in names], dtype=np.float32)
    w = w / max(float(w.sum()), 1e-6)
    fused = np.tensordot(preds, w, axes=([1], [0]))
    pred_bin = fused > threshold
    inter = np.logical_and(pred_bin, masks).sum(axis=(1, 2)).astype(np.float64)
    union = np.logical_or(pred_bin, masks).sum(axis=(1, 2)).astype(np.float64)
    return float(np.mean(inter / (union + np.spacing(1))))


def render_list(samples: list[dict[str, object]], names: list[str], weights: dict[str, float]) -> list[np.ndarray]:
    w = np.asarray([weights[name] for name in names], dtype=np.float32)
    w = w / max(float(w.sum()), 1e-6)
    return [
        np.tensordot(np.stack([sample["preds"][name] for name in names], axis=0).astype(np.float32), w, axes=([0], [0]))
        for sample in samples
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast validation-selected cached fusion search.")
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--test_cache", required=True)
    parser.add_argument("--val_fraction", type=float, default=0.25)
    parser.add_argument("--split_seed", default="20260614")
    parser.add_argument("--thresholds", default="0.36,0.38,0.40,0.42,0.44")
    parser.add_argument("--weight_steps", default="0.85,0.95,1.05,1.15")
    parser.add_argument("--weight_mode", default="all", choices=["all", "pconv_only"])
    parser.add_argument("--diagnostic_topk", type=int, default=30)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_samples, names = load_cache(Path(args.train_cache))
    test_samples, test_names = load_cache(Path(args.test_cache))
    if names != test_names:
        raise ValueError(f"name mismatch: {names} != {test_names}")
    val_samples = select_split(train_samples, args.val_fraction, args.split_seed)
    val_preds, val_masks = stack_samples(val_samples, names)
    thresholds = parse_float_list(args.thresholds)
    weight_rows = make_weight_grid(names, parse_float_list(args.weight_steps), args.weight_mode)

    rows = []
    for weights in weight_rows:
        for threshold in thresholds:
            rows.append(
                {
                    "weights": weights,
                    "threshold": threshold,
                    "val_nIoU": fast_mean_niou(val_preds, val_masks, weights, names, threshold),
                }
            )
    rows.sort(key=lambda r: r["val_nIoU"], reverse=True)
    selected = rows[0]

    selected_maps = render_list(test_samples, names, selected["weights"])
    selected_metric = metric_from_maps(test_samples, selected_maps, float(selected["threshold"]))
    selected_test = {**selected, **{f"test_{k}": v for k, v in selected_metric.items()}}

    diag = []
    for row in rows[: args.diagnostic_topk]:
        maps = render_list(test_samples, names, row["weights"])
        metric = metric_from_maps(test_samples, maps, float(row["threshold"]))
        diag.append({**row, **{f"test_{k}": v for k, v in metric.items()}})
    diag.sort(key=lambda r: r["test_mIoU"], reverse=True)

    print(
        f"selected val_nIoU={selected['val_nIoU']*100:.2f} th={selected['threshold']} "
        f"test_mIoU={selected_test['test_mIoU']*100:.2f} Pd={selected_test['test_Pd']*100:.2f} "
        f"Fa={selected_test['test_Fa']*1e6:.2f}",
        flush=True,
    )
    print(
        f"best_test_in_val_topk th={diag[0]['threshold']} test_mIoU={diag[0]['test_mIoU']*100:.2f}",
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
                "val_samples": len(val_samples),
                "selected_by_val": selected_test,
                "best_test_in_val_topk_not_selected": diag[0],
                "val_rows_top": rows[:100],
                "test_rows_for_val_topk": diag,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
