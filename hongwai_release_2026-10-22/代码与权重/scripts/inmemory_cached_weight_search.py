#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))

from eval_component_fusion_learner import metric_from_maps  # noqa: E402
from search_cached_train_selected_fusion import parse_weight_spec  # noqa: E402


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def load_cache_dense(path: Path):
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta_json"]))
    names = json.loads(str(data["names_json"]))
    samples = []
    preds = []
    masks = []
    for idx, item in enumerate(meta):
        height, width = [int(x) for x in item["size"]]
        mask = data[f"mask_{idx}"][:height, :width].astype(bool)
        stack = np.stack([data[f"pred_{idx}_{name}"][:height, :width].astype(np.float32) for name in names], axis=0)
        masks.append(mask)
        preds.append(stack)
        samples.append({"name": item["name"], "size": [height, width], "mask": mask.astype(np.float32), "preds": {}})
    return samples, names, preds, masks


def weighted_maps(preds: list[np.ndarray], weights_vec: np.ndarray, gamma: float) -> list[np.ndarray]:
    maps = []
    for stack in preds:
        pred = np.tensordot(weights_vec, stack, axes=(0, 0)).astype(np.float32)
        pred = np.clip(pred, 0.0, 1.0)
        if abs(gamma - 1.0) > 1e-6:
            pred = pred**gamma
        maps.append(pred)
    return maps


def fast_niou_from_maps(maps: list[np.ndarray], masks: list[np.ndarray], threshold: float) -> float:
    vals = []
    for pred, mask in zip(maps, masks):
        pred_bin = pred > threshold
        inter = float((pred_bin & mask).sum())
        union = float((pred_bin | mask).sum())
        vals.append(inter / (union + np.spacing(1)))
    return float(np.mean(vals)) if vals else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="In-memory cached weighted search for one train/test cache pair.")
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--test_cache", required=True)
    parser.add_argument("--weight_spec", action="append", required=True)
    parser.add_argument("--thresholds", default="0.36,0.38,0.40,0.42,0.44,0.45")
    parser.add_argument("--gammas", default="0.95,1.00,1.05")
    parser.add_argument("--test_topk", type=int, default=18)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_samples, names, train_preds, train_masks = load_cache_dense(Path(args.train_cache))
    test_samples, test_names, test_preds, test_masks = load_cache_dense(Path(args.test_cache))
    if test_names != names:
        raise ValueError(f"name mismatch: {test_names} != {names}")
    weight_specs = [parse_weight_spec(spec, names) for spec in args.weight_spec]
    thresholds = parse_float_list(args.thresholds)
    gammas = parse_float_list(args.gammas)

    rows = []
    for label, weights in weight_specs:
        weights_vec = np.asarray([weights[name] for name in names], dtype=np.float32)
        for gamma in gammas:
            maps = weighted_maps(train_preds, weights_vec, gamma)
            for threshold in thresholds:
                rows.append(
                    {
                        "label": label,
                        "weights": weights,
                        "gamma": gamma,
                        "threshold": threshold,
                        "train_fast_nIoU": fast_niou_from_maps(maps, train_masks, threshold),
                        "train_samples": len(train_samples),
                    }
                )
    rows.sort(key=lambda row: row["train_fast_nIoU"], reverse=True)

    test_rows = []
    for row in rows[: max(int(args.test_topk), 1)]:
        weights_vec = np.asarray([row["weights"][name] for name in names], dtype=np.float32)
        maps = weighted_maps(test_preds, weights_vec, row["gamma"])
        # Fill preds so metric_from_maps gets normal sample structure.
        test_rows.append({**row, **{f"test_{k}": v for k, v in metric_from_maps(test_samples, maps, row["threshold"]).items()}})
    best_test = sorted(test_rows, key=lambda row: row["test_mIoU"], reverse=True)[0]
    selected = test_rows[0]
    print(
        f"selected label={selected['label']} th={selected['threshold']} gamma={selected['gamma']} "
        f"train_fast_nIoU={selected['train_fast_nIoU']*100:.2f} test_mIoU={selected['test_mIoU']*100:.2f} "
        f"Pd={selected['test_Pd']*100:.2f} Fa={selected['test_Fa']*1e6:.2f}",
        flush=True,
    )
    print(
        f"best_test_in_topk label={best_test['label']} th={best_test['threshold']} gamma={best_test['gamma']} "
        f"test_mIoU={best_test['test_mIoU']*100:.2f} Pd={best_test['test_Pd']*100:.2f} Fa={best_test['test_Fa']*1e6:.2f}",
        flush=True,
    )
    Path(args.output).write_text(
        json.dumps(
            {
                "train_cache": args.train_cache,
                "test_cache": args.test_cache,
                "models": names,
                "selected_by_train_fast_nIoU": selected,
                "best_test_in_train_topk_reference": best_test,
                "train_rows": rows,
                "test_rows": test_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
