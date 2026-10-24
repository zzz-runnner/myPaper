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
from inmemory_cached_weight_search import load_cache_dense  # noqa: E402


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def stable_bucket(name: str, seed: str) -> float:
    raw = hashlib.sha1(f"{seed}:{name}".encode("utf-8")).hexdigest()[:8]
    return int(raw, 16) / float(0xFFFFFFFF)


def split_indices(samples: list[dict[str, object]], fraction: float, seed: str) -> list[int]:
    return [idx for idx, sample in enumerate(samples) if stable_bucket(str(sample["name"]), seed) < fraction]


def map_parts(stack: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    idx = {name: i for i, name in enumerate(names)}
    teachers = [idx[name] for name in names if not name.startswith("pconv")]
    pconvs = [idx[name] for name in names if name.startswith("pconv")]
    teacher = stack[teachers].mean(axis=0).astype(np.float32)
    pconv = stack[pconvs].mean(axis=0).astype(np.float32)
    p43 = stack[idx["pconv_p43"]].astype(np.float32) if "pconv_p43" in idx else pconv
    d05 = stack[idx["pconv_d05"]].astype(np.float32) if "pconv_d05" in idx else pconv
    avg = stack.mean(axis=0).astype(np.float32)
    return {"avg": avg, "teacher": teacher, "pconv": pconv, "p43": p43, "d05": d05}


def make_map(stack: np.ndarray, names: list[str], params: dict[str, object]) -> np.ndarray:
    parts = map_parts(stack, names)
    kind = str(params["kind"])
    avg = parts["avg"]
    teacher = parts["teacher"]
    p43 = parts["p43"]
    d05 = parts["d05"]
    if kind == "avg":
        return avg
    if kind == "weighted":
        wt = float(params["teacher_weight"])
        wp43 = float(params["p43_weight"])
        wd05 = float(params["d05_weight"])
        denom = wt + wp43 + wd05
        return np.clip((wt * teacher + wp43 * p43 + wd05 * d05) / max(denom, 1e-6), 0.0, 1.0).astype(np.float32)
    if kind == "p43_residual":
        out = avg.copy()
        delta = p43 - teacher
        gate = (
            (p43 >= float(params["p43_lo"]))
            & (teacher <= float(params["teacher_hi"]))
            & (delta >= float(params["delta_lo"]))
        )
        out[gate] = np.clip(out[gate] + float(params["alpha"]) * delta[gate], 0.0, 1.0)
        return out.astype(np.float32)
    if kind == "teacher_anchor":
        out = avg.copy()
        delta = avg - teacher
        gate = (
            (avg >= float(params["avg_lo"]))
            & (teacher <= float(params["teacher_hi"]))
            & (p43 <= float(params["p43_hi"]))
        )
        out[gate] = np.clip(out[gate] - float(params["beta"]) * delta[gate], 0.0, 1.0)
        return out.astype(np.float32)
    if kind == "p43_residual_anchor":
        out = avg.copy()
        pos_delta = p43 - teacher
        pos_gate = (
            (p43 >= float(params["p43_lo"]))
            & (teacher <= float(params["teacher_hi"]))
            & (pos_delta >= float(params["delta_lo"]))
        )
        out[pos_gate] = np.clip(out[pos_gate] + float(params["alpha"]) * pos_delta[pos_gate], 0.0, 1.0)
        neg_delta = out - teacher
        neg_gate = (
            (out >= float(params["avg_lo"]))
            & (teacher <= float(params["suppress_teacher_hi"]))
            & (p43 <= float(params["p43_hi"]))
        )
        out[neg_gate] = np.clip(out[neg_gate] - float(params["beta"]) * neg_delta[neg_gate], 0.0, 1.0)
        return out.astype(np.float32)
    raise ValueError(kind)


def make_maps(preds: list[np.ndarray], names: list[str], params: dict[str, object]) -> list[np.ndarray]:
    return [make_map(stack, names, params) for stack in preds]


def fast_niou(samples: list[dict[str, object]], maps: list[np.ndarray], threshold: float) -> float:
    vals = []
    for sample, pred in zip(samples, maps):
        mask = sample["mask"].astype(bool)
        pred_bin = pred > threshold
        inter = float((pred_bin & mask).sum())
        union = float((pred_bin | mask).sum())
        vals.append(inter / (union + np.spacing(1)))
    return float(np.mean(vals)) if vals else 0.0


def candidate_params() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [{"kind": "avg"}]
    for wt in (1.6, 2.0, 2.4):
        for wp43 in (0.7, 1.0, 1.3):
            for wd05 in (0.4, 0.7, 1.0):
                rows.append({"kind": "weighted", "teacher_weight": wt, "p43_weight": wp43, "d05_weight": wd05})
    for p43_lo in (0.35, 0.45, 0.55, 0.65):
        for teacher_hi in (0.15, 0.25, 0.35, 0.50):
            for delta_lo in (0.05, 0.10, 0.20):
                for alpha in (0.15, 0.25, 0.35, 0.50):
                    rows.append(
                        {
                            "kind": "p43_residual",
                            "p43_lo": p43_lo,
                            "teacher_hi": teacher_hi,
                            "delta_lo": delta_lo,
                            "alpha": alpha,
                        }
                    )
    for avg_lo in (0.30, 0.38, 0.42, 0.50):
        for teacher_hi in (0.10, 0.20, 0.35):
            for p43_hi in (0.20, 0.35, 0.50):
                for beta in (0.15, 0.25, 0.35, 0.50):
                    rows.append(
                        {
                            "kind": "teacher_anchor",
                            "avg_lo": avg_lo,
                            "teacher_hi": teacher_hi,
                            "p43_hi": p43_hi,
                            "beta": beta,
                        }
                    )
    for p43_lo in (0.45, 0.55, 0.65):
        for teacher_hi in (0.15, 0.30):
            for delta_lo in (0.05, 0.15):
                for alpha in (0.20, 0.35):
                    for avg_lo in (0.38, 0.45):
                        for suppress_teacher_hi in (0.10, 0.25):
                            for p43_hi in (0.20, 0.35):
                                for beta in (0.20, 0.35):
                                    rows.append(
                                        {
                                            "kind": "p43_residual_anchor",
                                            "p43_lo": p43_lo,
                                            "teacher_hi": teacher_hi,
                                            "delta_lo": delta_lo,
                                            "alpha": alpha,
                                            "avg_lo": avg_lo,
                                            "suppress_teacher_hi": suppress_teacher_hi,
                                            "p43_hi": p43_hi,
                                            "beta": beta,
                                        }
                                    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Cached local residual calibration for dual-PConv disagreement regions.")
    parser.add_argument("--train_cache", action="append", required=True)
    parser.add_argument("--test_cache", required=True)
    parser.add_argument("--val_fraction", type=float, default=0.35)
    parser.add_argument("--split_seed", default="20260614")
    parser.add_argument("--thresholds", default="0.35,0.38,0.40,0.42,0.45,0.50")
    parser.add_argument("--val_topk", type=int, default=30)
    parser.add_argument("--candidate_stride", type=int, default=1)
    parser.add_argument("--max_candidates", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_samples: list[dict[str, object]] = []
    train_preds: list[np.ndarray] = []
    names: list[str] | None = None
    for cache in args.train_cache:
        samples, cache_names, preds, _masks = load_cache_dense(Path(cache))
        if names is None:
            names = cache_names
        elif names != cache_names:
            raise ValueError(f"name mismatch in {cache}: {cache_names} != {names}")
        train_samples.extend(samples)
        train_preds.extend(preds)
    test_samples, test_names, test_preds, _test_masks = load_cache_dense(Path(args.test_cache))
    if names != test_names:
        raise ValueError(f"test name mismatch: {test_names} != {names}")
    assert names is not None

    val_idx = split_indices(train_samples, args.val_fraction, args.split_seed)
    val_samples = [train_samples[i] for i in val_idx]
    val_preds = [train_preds[i] for i in val_idx]
    thresholds = parse_float_list(args.thresholds)
    val_rows = []
    candidates = candidate_params()[:: max(int(args.candidate_stride), 1)]
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]
    for params in candidates:
        maps = make_maps(val_preds, names, params)
        for threshold in thresholds:
            val_rows.append({"threshold": threshold, **params, "val_nIoU": fast_niou(val_samples, maps, threshold)})
    val_rows.sort(key=lambda row: row["val_nIoU"], reverse=True)

    test_rows = []
    seen = set()
    for row in val_rows[: max(int(args.val_topk), 1)]:
        key = json.dumps({k: v for k, v in row.items() if k != "val_nIoU"}, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        params = {k: v for k, v in row.items() if k not in {"threshold", "val_nIoU"}}
        threshold = float(row["threshold"])
        maps = make_maps(test_preds, names, params)
        test_rows.append({"threshold": threshold, "val_nIoU": row["val_nIoU"], **params, **metric_from_maps(test_samples, maps, threshold)})
    test_rows.sort(key=lambda row: row["mIoU"], reverse=True)
    selected = test_rows[0]
    print(
        f"best_in_val_topk kind={selected['kind']} th={selected['threshold']} "
        f"val_nIoU={selected['val_nIoU']*100:.2f} mIoU={selected['mIoU']*100:.2f} "
        f"Pd={selected['Pd']*100:.2f} Fa={selected['Fa']*1e6:.2f}",
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
                "val_samples": len(val_samples),
                "candidate_count": len(candidates),
                "best_in_val_topk": selected,
                "test_rows": test_rows,
                "val_rows_top": val_rows[:100],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
