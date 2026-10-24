#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
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
    parse_model_spec,
    parse_official_bgm_spec,
    parse_rpcanet_spec,
)


NAMES = ["tpc_a", "tpc_b", "lca", "bgm_nuaa", "rpcanet_sirst_s3", "bg_anomaly"]


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def stack_samples(samples: list[dict[str, object]], names: list[str]) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    pred_parts = []
    mask_parts = []
    shapes = []
    for sample in samples:
        stack = np.stack([sample["preds"][name].astype(np.float32).reshape(-1) for name in names], axis=0)
        pred_parts.append(stack)
        mask_parts.append((sample["mask"].reshape(-1) > 0))
        shapes.append(tuple(sample["mask"].shape))
    return np.concatenate(pred_parts, axis=1), np.concatenate(mask_parts, axis=0), shapes


def fast_miou_from_flat(pred_flat: np.ndarray, mask_flat: np.ndarray, threshold: float) -> float:
    pred = pred_flat > threshold
    inter = int((pred & mask_flat).sum())
    union = int((pred | mask_flat).sum())
    return float(inter) / float(union + np.spacing(1))


def render_maps(samples: list[dict[str, object]], names: list[str], weights: np.ndarray, gamma: float) -> list[np.ndarray]:
    maps = []
    for sample in samples:
        out = np.zeros_like(sample["mask"], dtype=np.float32)
        for idx, name in enumerate(names):
            out += float(weights[idx]) * sample["preds"][name].astype(np.float32)
        if abs(gamma - 1.0) > 1e-6:
            out = np.clip(out, 0.0, 1.0) ** gamma
        maps.append(out.astype(np.float32))
    return maps


def fast_full_metric(samples: list[dict[str, object]], maps: list[np.ndarray], threshold: float) -> dict[str, float]:
    total_inter = 0
    total_union = 0
    total_correct = 0
    total_label = 0
    sample_ious = []
    pd = 0
    target = 0
    false_pixels = 0.0
    all_pixels = 0
    for sample, pred in zip(samples, maps):
        mask = sample["mask"] > 0
        pred_bin = pred > threshold
        inter = int((pred_bin & mask).sum())
        union = int((pred_bin | mask).sum())
        total_inter += inter
        total_union += union
        total_correct += inter
        total_label += int(mask.sum())
        sample_ious.append(inter / float(union + np.spacing(1)))

        pred_count, _pred_labels, pred_stats, pred_centroids = cv2.connectedComponentsWithStats(
            pred_bin.astype(np.uint8), connectivity=8
        )
        gt_count, _gt_labels, _gt_stats, gt_centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        target += max(gt_count - 1, 0)
        matched_pred: set[int] = set()
        for gt_idx in range(1, gt_count):
            gt_centroid = gt_centroids[gt_idx]
            best_idx = -1
            best_dist = 3.0
            for pred_idx in range(1, pred_count):
                if pred_idx in matched_pred:
                    continue
                dist = float(np.linalg.norm(pred_centroids[pred_idx] - gt_centroid))
                if dist < best_dist:
                    best_dist = dist
                    best_idx = pred_idx
            if best_idx >= 0:
                pd += 1
                matched_pred.add(best_idx)
        for pred_idx in range(1, pred_count):
            if pred_idx not in matched_pred:
                false_pixels += float(pred_stats[pred_idx, cv2.CC_STAT_AREA])
        all_pixels += int(mask.shape[0]) * int(mask.shape[1])
    return {
        "pixAcc": float(total_correct) / float(total_label + np.spacing(1)),
        "mIoU": float(total_inter) / float(total_union + np.spacing(1)),
        "nIoU": float(np.mean(sample_ious)) if sample_ious else 0.0,
        "Pd": float(pd) / float(max(target, 1)),
        "Fa": float(false_pixels) / float(max(all_pixels, 1)),
    }


def candidate_weights(rng: np.random.Generator, random_count: int) -> list[tuple[str, np.ndarray]]:
    rows: list[tuple[str, np.ndarray]] = []

    def add(kind: str, values: list[float]) -> None:
        arr = np.asarray(values, dtype=np.float32)
        arr = np.maximum(arr, 0.0)
        arr = arr / max(float(arr.sum()), 1e-6)
        rows.append((kind, arr))

    add("current_best", [0.14, 0.14, 0.14, 0.22, 0.28, 0.08])
    add("balanced_no_bg_anomaly", [0.15, 0.15, 0.15, 0.25, 0.30, 0.0])
    add("component_best_like", [0.14, 0.14, 0.14, 0.28, 0.30, 0.0])
    add("rpcanet_heavy", [0.11, 0.11, 0.11, 0.22, 0.37, 0.08])
    add("bgm_heavy", [0.12, 0.12, 0.12, 0.34, 0.22, 0.08])
    add("teacher_light", [0.08, 0.08, 0.08, 0.30, 0.38, 0.08])
    for teacher_total in (0.24, 0.30, 0.36, 0.42, 0.48):
        for bgm in (0.16, 0.20, 0.24, 0.28, 0.32):
            for rpc in (0.20, 0.24, 0.28, 0.32, 0.36, 0.40):
                for bg in (0.00, 0.04, 0.08, 0.12):
                    total = teacher_total + bgm + rpc + bg
                    if total > 1.0001:
                        continue
                    add("grid", [teacher_total / 3, teacher_total / 3, teacher_total / 3, bgm, rpc, bg])
    alpha_center = np.asarray([0.14, 0.14, 0.14, 0.22, 0.28, 0.08], dtype=np.float32)
    alpha = np.maximum(alpha_center * 80.0, 1.0)
    for _ in range(random_count):
        add("dirichlet_near_best", rng.dirichlet(alpha).astype(np.float32).tolist())
    seen = set()
    unique = []
    for kind, weights in rows:
        key = tuple(round(float(x), 4) for x in weights)
        if key in seen:
            continue
        seen.add(key)
        unique.append((kind, weights))
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorized weight search for NUAA BGM+RPCANet fixed fusion.")
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
    parser.add_argument("--thresholds", default="0.38,0.40,0.42,0.44,0.46,0.48,0.50,0.52,0.54,0.56,0.58")
    parser.add_argument("--gammas", default="0.85,0.90,0.95,1.0,1.05,1.10")
    parser.add_argument("--random_count", type=int, default=1200)
    parser.add_argument("--topk_full", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    specs = [parse_model_spec(spec) for spec in args.model_spec]
    models = build_models(specs, device)
    bgm_specs = [parse_official_bgm_spec(spec) for spec in args.official_bgm_spec]
    bgms = load_official_bgm_models(bgm_specs, device)
    rpc_specs = [parse_rpcanet_spec(spec) for spec in args.rpcanet_spec]
    rpcanets = load_rpcanet_models(rpc_specs, device)
    fit_datasets = [x.strip() for x in args.fit_datasets.split(",") if x.strip()]
    background_model = fit_background_normality_model(
        Path(args.dataset_dir),
        fit_datasets,
        "train",
        args.max_fit_samples,
        args.background_sample_pixels,
    )
    _fit_samples = collect_fit_samples_multi(
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
    pred_stack, mask_flat, _shapes = stack_samples(eval_samples, NAMES)
    thresholds = parse_float_list(args.thresholds)
    gammas = parse_float_list(args.gammas)
    rng = np.random.default_rng(args.seed)
    weights = candidate_weights(rng, args.random_count)
    print(f"searching {len(weights)} weights x {len(gammas)} gammas x {len(thresholds)} thresholds", flush=True)
    fast_rows = []
    for kind, weight_vec in weights:
        pred = weight_vec @ pred_stack
        for gamma in gammas:
            pred_g = pred if abs(gamma - 1.0) <= 1e-6 else np.clip(pred, 0.0, 1.0) ** gamma
            for threshold in thresholds:
                fast_rows.append(
                    {
                        "kind": kind,
                        "weights": {name: float(weight_vec[idx]) for idx, name in enumerate(NAMES)},
                        "gamma": gamma,
                        "threshold": threshold,
                        "fast_mIoU": fast_miou_from_flat(pred_g, mask_flat, threshold),
                    }
                )
    fast_rows.sort(key=lambda row: row["fast_mIoU"], reverse=True)
    full_rows = []
    seen = set()
    for row in fast_rows:
        key = (
            tuple(round(float(row["weights"][name]), 4) for name in NAMES),
            round(float(row["gamma"]), 4),
            round(float(row["threshold"]), 4),
        )
        if key in seen:
            continue
        seen.add(key)
        weight_vec = np.asarray([row["weights"][name] for name in NAMES], dtype=np.float32)
        maps = render_maps(eval_samples, NAMES, weight_vec, float(row["gamma"]))
        metrics = fast_full_metric(eval_samples, maps, float(row["threshold"]))
        full_rows.append({**row, **metrics})
        if len(full_rows) >= args.topk_full:
            break
    full_rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best = full_rows[0]
    print(
        f"best kind={best['kind']} th={best['threshold']} gamma={best['gamma']} "
        f"mIoU={best['mIoU']*100:.2f} nIoU={best['nIoU']*100:.2f} "
        f"Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.2f} weights={best['weights']}",
        flush=True,
    )
    Path(args.output).write_text(
        json.dumps(
            {
                "dataset": args.dataset_name,
                "names": NAMES,
                "best": best,
                "full_rows": full_rows,
                "fast_rows_top": fast_rows[:100],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
