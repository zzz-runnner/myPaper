#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))

from eval_component_fusion_learner import metric_from_maps  # noqa: E402
from inmemory_cached_weight_search import load_cache_dense, weighted_maps  # noqa: E402
from search_cached_train_selected_fusion import parse_weight_spec  # noqa: E402


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def stable_bucket(name: str, seed: str) -> float:
    raw = hashlib.sha1(f"{seed}:{name}".encode("utf-8")).hexdigest()[:8]
    return int(raw, 16) / float(0xFFFFFFFF)


def subset(items: list, keep: list[bool]) -> list:
    return [item for item, flag in zip(items, keep) if flag]


def main() -> None:
    parser = argparse.ArgumentParser(description="Select cached fusion weights on a held-out train validation split, then evaluate test.")
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--test_cache", required=True)
    parser.add_argument("--weight_spec", action="append", required=True)
    parser.add_argument("--thresholds", default="0.36,0.38,0.40,0.42,0.44,0.45")
    parser.add_argument("--gammas", default="0.95,1.00,1.05")
    parser.add_argument("--val_fraction", type=float, default=0.25)
    parser.add_argument("--split_seed", default="20260613")
    parser.add_argument("--selection_score", default="mIoU_minus_fa", choices=["mIoU", "mIoU_minus_fa"])
    parser.add_argument("--test_topk", type=int, default=12)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_samples, names, train_preds, _train_masks = load_cache_dense(Path(args.train_cache))
    test_samples, test_names, test_preds, _test_masks = load_cache_dense(Path(args.test_cache))
    if test_names != names:
        raise ValueError(f"name mismatch: {test_names} != {names}")

    val_keep = [stable_bucket(str(sample["name"]), args.split_seed) < args.val_fraction for sample in train_samples]
    if not any(val_keep) or all(val_keep):
        raise RuntimeError("bad validation split")
    val_samples = subset(train_samples, val_keep)
    val_preds = subset(train_preds, val_keep)
    fit_count = len(train_samples) - len(val_samples)

    weight_specs = [parse_weight_spec(spec, names) for spec in args.weight_spec]
    thresholds = parse_float_list(args.thresholds)
    gammas = parse_float_list(args.gammas)

    val_rows = []
    for label, weights in weight_specs:
        weights_vec = np.asarray([weights[name] for name in names], dtype=np.float32)
        for gamma in gammas:
            maps = weighted_maps(val_preds, weights_vec, gamma)
            for threshold in thresholds:
                metrics = metric_from_maps(val_samples, maps, threshold)
                score = metrics["mIoU"]
                if args.selection_score == "mIoU_minus_fa":
                    score = metrics["mIoU"] - 0.000002 * (metrics["Fa"] * 1e6)
                val_rows.append(
                    {
                        "label": label,
                        "weights": weights,
                        "gamma": gamma,
                        "threshold": threshold,
                        "selection_score": float(score),
                        **{f"val_{k}": v for k, v in metrics.items()},
                    }
                )
    val_rows.sort(key=lambda row: row["selection_score"], reverse=True)

    test_rows = []
    for row in val_rows[: max(int(args.test_topk), 1)]:
        weights_vec = np.asarray([row["weights"][name] for name in names], dtype=np.float32)
        maps = weighted_maps(test_preds, weights_vec, row["gamma"])
        test_rows.append({**row, **{f"test_{k}": v for k, v in metric_from_maps(test_samples, maps, row["threshold"]).items()}})
    selected = test_rows[0]
    best_test_reference = sorted(test_rows, key=lambda row: row["test_mIoU"], reverse=True)[0]

    print(
        f"selected label={selected['label']} th={selected['threshold']} gamma={selected['gamma']} "
        f"val_mIoU={selected['val_mIoU']*100:.2f} val_Fa={selected['val_Fa']*1e6:.2f} "
        f"test_mIoU={selected['test_mIoU']*100:.2f} Pd={selected['test_Pd']*100:.2f} Fa={selected['test_Fa']*1e6:.2f}",
        flush=True,
    )
    print(
        f"best_test_in_val_topk label={best_test_reference['label']} th={best_test_reference['threshold']} "
        f"gamma={best_test_reference['gamma']} test_mIoU={best_test_reference['test_mIoU']*100:.2f} "
        f"Pd={best_test_reference['test_Pd']*100:.2f} Fa={best_test_reference['test_Fa']*1e6:.2f}",
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
                "fit_samples_not_used_for_selection": fit_count,
                "val_samples": len(val_samples),
                "selection_score": args.selection_score,
                "selected_by_val": selected,
                "best_test_in_val_topk_reference": best_test_reference,
                "val_rows": val_rows,
                "test_rows": test_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
