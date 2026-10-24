#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))

from eval_cached_prediction_fusion import load_cache  # noqa: E402
from eval_component_fusion_learner import metric_from_maps  # noqa: E402


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def fast_niou(samples: list[dict[str, object]], maps: list[np.ndarray], threshold: float) -> float:
    vals = []
    for sample, pred in zip(samples, maps):
        mask = sample["mask"] > 0
        pred_bin = pred > threshold
        inter = float((pred_bin & mask).sum())
        union = float((pred_bin | mask).sum())
        vals.append(inter / (union + np.spacing(1)))
    return float(np.mean(vals)) if vals else 0.0


def load_many(paths: list[str]) -> tuple[list[dict[str, object]], list[str]]:
    all_samples: list[dict[str, object]] = []
    names_ref: list[str] | None = None
    for raw in paths:
        samples, names = load_cache(Path(raw))
        if names_ref is None:
            names_ref = names
        elif names != names_ref:
            raise ValueError(f"cache model names mismatch in {raw}: {names} != {names_ref}")
        all_samples.extend(samples)
    return all_samples, names_ref or []


def normalize_weights(raw: dict[str, float], names: list[str]) -> dict[str, float]:
    weights = {name: max(float(raw.get(name, 0.0)), 0.0) for name in names}
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("zero weight candidate")
    return {name: val / total for name, val in weights.items()}


def candidate_weights(names: list[str], family: str) -> list[tuple[str, dict[str, float]]]:
    rows: list[tuple[str, dict[str, float]]] = []
    rows.append(("avg_all", {name: 1.0 / len(names) for name in names}))
    for name in names:
        rows.append((f"single_{name}", {n: 1.0 if n == name else 0.0 for n in names}))

    pconv = [name for name in names if name.startswith("pconv")]
    teachers = [name for name in names if name not in pconv]
    if family == "irstd_dual" and len(pconv) >= 2 and teachers:
        for teacher_total, d05_share, p43_share in product(
            [0.20, 0.30, 0.40, 0.50, 0.60],
            [0.20, 0.35, 0.50, 0.65, 0.80],
            [0.20, 0.35, 0.50, 0.65, 0.80],
        ):
            raw = {name: 0.0 for name in names}
            for teacher in teachers:
                raw[teacher] = teacher_total / len(teachers)
            p_total = 1.0 - teacher_total
            raw[pconv[0]] = p_total * d05_share / max(d05_share + p43_share, 1e-6)
            raw[pconv[1]] = p_total * p43_share / max(d05_share + p43_share, 1e-6)
            weights = normalize_weights(raw, names)
            rows.append((f"t{teacher_total:.2f}_p{weights[pconv[0]]:.2f}_{weights[pconv[1]]:.2f}", weights))
    else:
        # Generic teacher-only convex mixtures.
        for vals in product([0.0, 0.25, 0.50, 0.75, 1.0], repeat=len(names)):
            if sum(vals) <= 0:
                continue
            raw = {name: val for name, val in zip(names, vals)}
            weights = normalize_weights(raw, names)
            label = "w_" + "_".join(f"{name}{weights[name]:.2f}" for name in names)
            rows.append((label, weights))

    unique: dict[tuple[tuple[str, float], ...], tuple[str, dict[str, float]]] = {}
    for label, weights in rows:
        key = tuple((name, round(weights[name], 4)) for name in names)
        unique[key] = (label, weights)
    return list(unique.values())


def parse_weight_spec(value: str, names: list[str]) -> tuple[str, dict[str, float]]:
    label, raw = value.split(":", 1)
    weights = {name: 0.0 for name in names}
    for part in raw.split(","):
        key, val = part.split("=")
        key = key.strip()
        if key not in weights:
            raise ValueError(f"unknown weight name {key}; valid names={names}")
        weights[key] = float(val)
    return label, normalize_weights(weights, names)


def render_weighted(samples: list[dict[str, object]], names: list[str], weights: dict[str, float], gamma: float) -> list[np.ndarray]:
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
    parser = argparse.ArgumentParser(description="Search cached weighted fusion on train caches, then evaluate selected rows on test.")
    parser.add_argument("--train_cache", action="append", required=True)
    parser.add_argument("--test_cache", required=True)
    parser.add_argument("--family", default="generic", choices=["generic", "irstd_dual"])
    parser.add_argument("--weight_spec", action="append", default=[], help="label:name=w,name=w")
    parser.add_argument("--manual_only", action="store_true")
    parser.add_argument("--thresholds", default="0.35,0.38,0.40,0.42,0.45,0.48,0.50,0.55,0.60")
    parser.add_argument("--gammas", default="0.85,0.90,0.95,1.00,1.05,1.10,1.15")
    parser.add_argument("--select_metric", default="fast_nIoU", choices=["fast_nIoU", "mIoU"])
    parser.add_argument("--full_train_topk", type=int, default=24)
    parser.add_argument("--test_topk", type=int, default=12)
    parser.add_argument("--skip_full_train_metric", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_samples, names = load_many(args.train_cache)
    test_samples, test_names = load_cache(Path(args.test_cache))
    if test_names != names:
        raise ValueError(f"test cache names mismatch: {test_names} != {names}")

    thresholds = parse_float_list(args.thresholds)
    gammas = parse_float_list(args.gammas)
    candidates = []
    if not args.manual_only:
        candidates.extend(candidate_weights(names, args.family))
    for spec in args.weight_spec:
        candidates.append(parse_weight_spec(spec, names))
    if not candidates:
        raise ValueError("no weight candidates")

    fast_rows = []
    for label, weights in candidates:
        for gamma in gammas:
            maps = render_weighted(train_samples, names, weights, gamma)
            for threshold in thresholds:
                fast_rows.append(
                    {
                        "label": label,
                        "weights": weights,
                        "gamma": gamma,
                        "threshold": threshold,
                        "train_fast_nIoU": fast_niou(train_samples, maps, threshold),
                    }
                )
    fast_rows.sort(key=lambda row: row["train_fast_nIoU"], reverse=True)

    train_rows = []
    for row in fast_rows[: max(int(args.full_train_topk), 1)]:
        if args.skip_full_train_metric:
            train_rows.append(
                {
                    **row,
                    "train_mIoU": None,
                    "train_nIoU": row["train_fast_nIoU"],
                    "train_Pd": None,
                    "train_Fa": None,
                    "train_pixAcc": None,
                }
            )
        else:
            maps = render_weighted(train_samples, names, row["weights"], row["gamma"])
            train_rows.append({**row, **{f"train_{k}": v for k, v in metric_from_maps(train_samples, maps, row["threshold"]).items()}})
    sort_key = "train_fast_nIoU" if args.skip_full_train_metric else f"train_{args.select_metric}"
    train_rows.sort(key=lambda row: row[sort_key], reverse=True)

    test_rows = []
    for row in train_rows[: max(int(args.test_topk), 1)]:
        maps = render_weighted(test_samples, names, row["weights"], row["gamma"])
        test_rows.append({**row, **{f"test_{k}": v for k, v in metric_from_maps(test_samples, maps, row["threshold"]).items()}})
    test_rows.sort(key=lambda row: row["test_mIoU"], reverse=True)
    selected = train_rows[0]
    selected_test = next(
        row
        for row in test_rows
        if row["label"] == selected["label"] and row["gamma"] == selected["gamma"] and row["threshold"] == selected["threshold"]
    )
    best_test_reference = test_rows[0]

    print(
        f"selected label={selected_test['label']} th={selected_test['threshold']} gamma={selected_test['gamma']} "
        f"train_mIoU={selected_test['train_mIoU']*100:.2f} test_mIoU={selected_test['test_mIoU']*100:.2f} "
        f"Pd={selected_test['test_Pd']*100:.2f} Fa={selected_test['test_Fa']*1e6:.2f}",
        flush=True,
    )
    print(
        f"best_test_reference label={best_test_reference['label']} th={best_test_reference['threshold']} "
        f"gamma={best_test_reference['gamma']} test_mIoU={best_test_reference['test_mIoU']*100:.2f} "
        f"Pd={best_test_reference['test_Pd']*100:.2f} Fa={best_test_reference['test_Fa']*1e6:.2f}",
        flush=True,
    )

    Path(args.output).write_text(
        json.dumps(
            {
                "train_caches": args.train_cache,
                "test_cache": args.test_cache,
                "models": names,
                "family": args.family,
                "select_metric": args.select_metric,
                "selected_by_train": selected,
                "selected_test": selected_test,
                "best_test_reference_from_train_topk": best_test_reference,
                "train_rows": train_rows,
                "test_rows": test_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
