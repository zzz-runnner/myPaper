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

from eval_cached_prediction_fusion import load_cache  # noqa: E402
from eval_component_fusion_learner import metric_from_maps  # noqa: E402
from search_cached_train_selected_fusion import parse_weight_spec  # noqa: E402


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def cache_names(path: Path) -> list[str]:
    data = np.load(path, allow_pickle=False)
    return json.loads(str(data["names_json"]))


def score_cache_fast(path: Path, names: list[str], weights: dict[str, float], gamma: float, threshold: float) -> tuple[float, int]:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta_json"]))
    total = 0.0
    count = 0
    for idx, item in enumerate(meta):
        height, width = [int(x) for x in item["size"]]
        mask = data[f"mask_{idx}"][:height, :width].astype(bool)
        pred = np.zeros((height, width), dtype=np.float32)
        for name in names:
            pred += float(weights[name]) * data[f"pred_{idx}_{name}"][:height, :width].astype(np.float32)
        pred = np.clip(pred, 0.0, 1.0)
        if abs(gamma - 1.0) > 1e-6:
            pred = pred**gamma
        pred_bin = pred > threshold
        inter = float((pred_bin & mask).sum())
        union = float((pred_bin | mask).sum())
        total += inter / (union + np.spacing(1))
        count += 1
    return total, count


def render_test_maps(samples: list[dict[str, object]], names: list[str], weights: dict[str, float], gamma: float) -> list[np.ndarray]:
    maps = []
    for sample in samples:
        pred = np.zeros_like(sample["mask"], dtype=np.float32)
        for name in names:
            pred += float(weights[name]) * sample["preds"][name].astype(np.float32)
        pred = np.clip(pred, 0.0, 1.0)
        if abs(gamma - 1.0) > 1e-6:
            pred = pred**gamma
        maps.append(pred.astype(np.float32))
    return maps


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming fast-nIoU train selection over cached weighted fusions.")
    parser.add_argument("--train_cache", action="append", required=True)
    parser.add_argument("--test_cache", required=True)
    parser.add_argument("--weight_spec", action="append", required=True, help="label:name=w,name=w")
    parser.add_argument("--thresholds", default="0.36,0.38,0.40,0.42,0.44,0.45")
    parser.add_argument("--gammas", default="0.95,1.00,1.05")
    parser.add_argument("--test_topk", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    names = cache_names(Path(args.train_cache[0]))
    for raw in args.train_cache[1:] + [args.test_cache]:
        other = cache_names(Path(raw))
        if other != names:
            raise ValueError(f"cache names mismatch {raw}: {other} != {names}")
    weights = [parse_weight_spec(spec, names) for spec in args.weight_spec]
    thresholds = parse_float_list(args.thresholds)
    gammas = parse_float_list(args.gammas)

    rows = []
    for label, weight in weights:
        for gamma in gammas:
            for threshold in thresholds:
                total = 0.0
                count = 0
                for cache in args.train_cache:
                    part_total, part_count = score_cache_fast(Path(cache), names, weight, gamma, threshold)
                    total += part_total
                    count += part_count
                rows.append(
                    {
                        "label": label,
                        "weights": weight,
                        "gamma": gamma,
                        "threshold": threshold,
                        "train_fast_nIoU": total / max(count, 1),
                        "train_samples": count,
                    }
                )
    rows.sort(key=lambda row: row["train_fast_nIoU"], reverse=True)

    test_samples, _names = load_cache(Path(args.test_cache))
    test_rows = []
    for row in rows[: max(int(args.test_topk), 1)]:
        maps = render_test_maps(test_samples, names, row["weights"], row["gamma"])
        test_rows.append({**row, **{f"test_{k}": v for k, v in metric_from_maps(test_samples, maps, row["threshold"]).items()}})
    selected_test = test_rows[0]
    best_test = sorted(test_rows, key=lambda row: row["test_mIoU"], reverse=True)[0]
    print(
        f"selected label={selected_test['label']} th={selected_test['threshold']} gamma={selected_test['gamma']} "
        f"train_fast_nIoU={selected_test['train_fast_nIoU']*100:.2f} test_mIoU={selected_test['test_mIoU']*100:.2f} "
        f"Pd={selected_test['test_Pd']*100:.2f} Fa={selected_test['test_Fa']*1e6:.2f}",
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
                "train_caches": args.train_cache,
                "test_cache": args.test_cache,
                "models": names,
                "selected_by_train_fast_nIoU": selected_test,
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
