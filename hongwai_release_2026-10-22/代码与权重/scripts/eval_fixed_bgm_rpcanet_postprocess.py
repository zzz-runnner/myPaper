#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage import measure

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


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def fixed_map(sample: dict[str, object], gamma: float) -> np.ndarray:
    weights = {
        "tpc_a": 0.14,
        "tpc_b": 0.14,
        "lca": 0.14,
        "bgm_nuaa": 0.22,
        "rpcanet_sirst_s3": 0.28,
        "bg_anomaly": 0.08,
    }
    out = np.zeros_like(sample["mask"], dtype=np.float32)
    for name, weight in weights.items():
        if name in sample["preds"]:
            out += weight * sample["preds"][name].astype(np.float32)
    total = sum(weight for name, weight in weights.items() if name in sample["preds"])
    out = out / max(total, 1e-6)
    if abs(gamma - 1.0) > 1e-6:
        out = np.clip(out, 0.0, 1.0) ** gamma
    return out.astype(np.float32)


def postprocess(prob: np.ndarray, threshold: float, min_area: int, op: str, kernel_size: int) -> np.ndarray:
    pred = (prob > threshold).astype(np.uint8)
    if kernel_size > 1 and op != "none":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if op == "open":
            pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)
        elif op == "close":
            pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)
        elif op == "dilate":
            pred = cv2.dilate(pred, kernel, iterations=1)
        elif op == "erode":
            pred = cv2.erode(pred, kernel, iterations=1)
        elif op == "close_open":
            pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)
            pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)
        else:
            raise ValueError(f"Unsupported op: {op}")
    if min_area > 1:
        labels = measure.label(pred.astype(bool), connectivity=2)
        cleaned = np.zeros_like(pred, dtype=np.uint8)
        for region in measure.regionprops(labels):
            if region.area >= min_area:
                cleaned[labels == region.label] = 1
        pred = cleaned
    return pred.astype(np.float32)


def fast_miou(samples: list[dict[str, object]], maps: list[np.ndarray]) -> float:
    inter = 0
    union = 0
    for sample, pred in zip(samples, maps):
        mask = sample["mask"] > 0
        pred_bin = pred > 0.5
        inter += int((pred_bin & mask).sum())
        union += int((pred_bin | mask).sum())
    return float(inter) / float(union + np.spacing(1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep postprocess over fixed BGM+RPCANet NUAA fusion.")
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
    parser.add_argument("--gammas", default="0.95,1.0,1.05")
    parser.add_argument("--thresholds", default="0.46,0.48,0.50,0.52,0.54")
    parser.add_argument("--min_areas", default="1,2,3,4,5,6,8,10,12")
    parser.add_argument("--ops", default="none,open,close,close_open")
    parser.add_argument("--kernel_sizes", default="1,3")
    parser.add_argument("--full_metric_topk", type=int, default=12)
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

    fast_rows = []
    for gamma in parse_float_list(args.gammas):
        probs = [fixed_map(sample, gamma) for sample in eval_samples]
        for threshold in parse_float_list(args.thresholds):
            for min_area in parse_int_list(args.min_areas):
                for op in [x.strip() for x in args.ops.split(",") if x.strip()]:
                    for kernel_size in parse_int_list(args.kernel_sizes):
                        if op == "none" and kernel_size != 1:
                            continue
                        maps = [postprocess(prob, threshold, min_area, op, kernel_size) for prob in probs]
                        fast_rows.append(
                            {
                                "gamma": gamma,
                                "threshold": threshold,
                                "min_area": min_area,
                                "op": op,
                                "kernel_size": kernel_size,
                                "fast_mIoU": fast_miou(eval_samples, maps),
                            }
                        )
    fast_rows.sort(key=lambda row: row["fast_mIoU"], reverse=True)
    rows = []
    for row in fast_rows[: max(1, args.full_metric_topk)]:
        probs = [fixed_map(sample, float(row["gamma"])) for sample in eval_samples]
        maps = [
            postprocess(
                prob,
                float(row["threshold"]),
                int(row["min_area"]),
                str(row["op"]),
                int(row["kernel_size"]),
            )
            for prob in probs
        ]
        rows.append({**row, **metric_from_maps(eval_samples, maps, 0.5)})
    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best = rows[0]
    print(
        f"best gamma={best['gamma']} th={best['threshold']} area={best['min_area']} "
        f"op={best['op']} k={best['kernel_size']} mIoU={best['mIoU']*100:.2f} "
        f"nIoU={best['nIoU']*100:.2f} Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.2f}",
        flush=True,
    )
    Path(args.output).write_text(
        json.dumps({"dataset": args.dataset_name, "best": best, "rows": rows, "fast_rows_top": fast_rows[:50]}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
