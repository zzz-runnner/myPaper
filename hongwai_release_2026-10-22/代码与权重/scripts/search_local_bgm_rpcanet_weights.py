#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_component_fusion_learner import (  # noqa: E402
    build_models,
    collect_fit_samples_multi,
    collect_samples,
    fit_background_normality_model,
    load_official_bgm_models,
    load_rpcanet_models,
    metric_from_maps,
    parse_model_spec,
    parse_official_bgm_spec,
    parse_rpcanet_spec,
)


NAMES = ["tpc_a", "tpc_b", "lca", "bgm_nuaa", "rpcanet_sirst_s3", "bg_anomaly"]
BASE = np.asarray([0.14, 0.14, 0.14, 0.22, 0.28, 0.08], dtype=np.float32)


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def normalize(w: np.ndarray) -> np.ndarray:
    w = np.maximum(w.astype(np.float32), 0.0)
    s = float(w.sum())
    if s <= 1e-8:
        return np.ones_like(w) / float(len(w))
    return w / s


def candidate_weights(seed: int, random_count: int, noise_scale: float) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = [normalize(BASE)]
    # Deterministic local one/two-source nudges around the known best.
    deltas = (-0.08, -0.05, -0.03, 0.03, 0.05, 0.08)
    for i in range(len(BASE)):
        for d in deltas:
            w = BASE.copy()
            w[i] += d
            rows.append(normalize(w))
    for i in range(len(BASE)):
        for j in range(i + 1, len(BASE)):
            for d in (-0.05, -0.03, 0.03, 0.05):
                w = BASE.copy()
                w[i] += d
                w[j] -= d
                rows.append(normalize(w))
    # Dirichlet around the current best. Larger concentration keeps search local.
    alpha = np.maximum(BASE / max(noise_scale, 1e-6), 1e-3)
    for _ in range(random_count):
        rows.append(normalize(rng.dirichlet(alpha).astype(np.float32)))
    seen = set()
    unique = []
    for w in rows:
        key = tuple(round(float(x), 4) for x in w)
        if key in seen:
            continue
        seen.add(key)
        unique.append(w)
    return unique


def precompute_flat(samples: list[dict[str, object]], stride: int) -> list[dict[str, np.ndarray]]:
    flat = []
    step = max(int(stride), 1)
    for sample in samples:
        stack = np.stack([sample["preds"][name].astype(np.float32)[::step, ::step].reshape(-1) for name in NAMES], axis=0)
        flat.append({"stack": stack, "mask": sample["mask"][::step, ::step].reshape(-1) > 0})
    return flat


def fast_weighted_niou(flat_samples: list[dict[str, np.ndarray]], weight_vec: np.ndarray, gamma: float, threshold: float) -> float:
    vals = []
    for sample in flat_samples:
        pred = weight_vec @ sample["stack"]
        if abs(gamma - 1.0) > 1e-6:
            pred = np.clip(pred, 0.0, 1.0) ** gamma
        pred_bin = pred > threshold
        mask = sample["mask"]
        inter = float((pred_bin & mask).sum())
        union = float((pred_bin | mask).sum())
        vals.append(inter / (union + np.spacing(1)))
    return float(np.mean(vals)) if vals else 0.0


def render(samples: list[dict[str, object]], weight_vec: np.ndarray, gamma: float) -> list[np.ndarray]:
    maps = []
    for sample in samples:
        pred = np.zeros_like(sample["mask"], dtype=np.float32)
        for name, weight in zip(NAMES, weight_vec):
            pred += float(weight) * sample["preds"][name].astype(np.float32)
        if abs(gamma - 1.0) > 1e-6:
            pred = np.clip(pred, 0.0, 1.0) ** gamma
        maps.append(pred.astype(np.float32))
    return maps


def main() -> None:
    parser = argparse.ArgumentParser(description="Local train-selected weight search around current BGM+RPCANet fixed fusion.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", default="NUAA-SIRST")
    parser.add_argument("--fit_datasets", default="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3")
    parser.add_argument("--model_spec", action="append", required=True)
    parser.add_argument("--official_bgm_spec", action="append", required=True)
    parser.add_argument("--rpcanet_spec", action="append", required=True)
    parser.add_argument("--background_sample_pixels", type=int, default=2048)
    parser.add_argument("--background_topk", type=int, default=16)
    parser.add_argument("--background_threshold", type=float, default=0.88)
    parser.add_argument("--max_fit_samples", type=int, default=220)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--random_count", type=int, default=400)
    parser.add_argument("--noise_scale", type=float, default=0.08)
    parser.add_argument("--fit_pixel_stride", type=int, default=4)
    parser.add_argument("--top_fit", type=int, default=48)
    parser.add_argument("--thresholds", default="0.46,0.48,0.50,0.52,0.54,0.56")
    parser.add_argument("--gammas", default="0.95,1.0,1.05")
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    models = build_models([parse_model_spec(spec) for spec in args.model_spec], device)
    bgms = load_official_bgm_models([parse_official_bgm_spec(spec) for spec in args.official_bgm_spec], device)
    rpcanets = load_rpcanet_models([parse_rpcanet_spec(spec) for spec in args.rpcanet_spec], device)
    fit_datasets = [x.strip() for x in args.fit_datasets.split(",") if x.strip()]
    background_model = fit_background_normality_model(
        Path(args.dataset_dir),
        fit_datasets,
        "train",
        args.max_fit_samples,
        args.background_sample_pixels,
    )
    fit_samples = collect_fit_samples_multi(
        models,
        [],
        bgms,
        rpcanets,
        Path(args.dataset_dir),
        fit_datasets,
        "train",
        args.max_fit_samples,
        device,
        background_model=background_model,
        background_as_candidate=True,
        background_topk=args.background_topk,
        background_threshold=args.background_threshold,
    )
    eval_samples = collect_samples(
        models,
        [],
        bgms,
        rpcanets,
        Path(args.dataset_dir),
        args.dataset_name,
        "test",
        args.max_eval_samples,
        device,
        background_model=background_model,
        background_as_candidate=True,
        background_topk=args.background_topk,
        background_threshold=args.background_threshold,
    )
    weights = candidate_weights(args.seed, args.random_count, args.noise_scale)
    thresholds = parse_float_list(args.thresholds)
    gammas = parse_float_list(args.gammas)
    print(f"searching {len(weights)} local weights x {len(gammas)} gammas x {len(thresholds)} thresholds", flush=True)
    fit_flat = precompute_flat(fit_samples, args.fit_pixel_stride)
    fit_rows = []
    for w in weights:
        for gamma in gammas:
            for threshold in thresholds:
                fit_rows.append(
                    {
                        "weights": {name: float(value) for name, value in zip(NAMES, w)},
                        "gamma": gamma,
                        "threshold": threshold,
                        "fit_nIoU": fast_weighted_niou(fit_flat, w, gamma, threshold),
                    }
                )
    fit_rows.sort(key=lambda row: row["fit_nIoU"], reverse=True)
    eval_rows = []
    seen = set()
    for row in fit_rows[: max(int(args.top_fit), 1)]:
        key = (
            tuple(round(row["weights"][name], 4) for name in NAMES),
            round(float(row["gamma"]), 4),
            round(float(row["threshold"]), 4),
        )
        if key in seen:
            continue
        seen.add(key)
        w = np.asarray([row["weights"][name] for name in NAMES], dtype=np.float32)
        maps = render(eval_samples, w, float(row["gamma"]))
        eval_rows.append(
            {
                "weights": row["weights"],
                "gamma": row["gamma"],
                "threshold": row["threshold"],
                "fit_nIoU": row["fit_nIoU"],
                **metric_from_maps(eval_samples, maps, float(row["threshold"])),
            }
        )
    eval_rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best = eval_rows[0]
    print(
        f"best th={best['threshold']} gamma={best['gamma']} fit_nIoU={best['fit_nIoU']*100:.2f} "
        f"mIoU={best['mIoU']*100:.2f} nIoU={best['nIoU']*100:.2f} "
        f"Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.2f} weights={best['weights']}",
        flush=True,
    )
    Path(args.output).write_text(
        json.dumps(
            {
                "dataset": args.dataset_name,
                "fit_datasets": fit_datasets,
                "names": NAMES,
                "base_weights": {name: float(value) for name, value in zip(NAMES, BASE)},
                "fit_samples": len(fit_samples),
                "eval_samples": len(eval_samples),
                "candidate_weights": len(weights),
                "best": best,
                "eval_rows": eval_rows,
                "fit_rows_top": fit_rows[:100],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
