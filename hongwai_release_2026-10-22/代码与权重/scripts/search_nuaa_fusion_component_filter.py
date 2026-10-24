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
    metric_from_maps,
    parse_model_spec,
    parse_official_bgm_spec,
    parse_rpcanet_spec,
)


NAMES = ["tpc_a", "tpc_b", "lca", "bgm_nuaa", "rpcanet_sirst_s3", "bg_anomaly"]
CURRENT_BEST_WEIGHTS = {
    "tpc_a": 0.1899983435869217,
    "tpc_b": 0.09598003327846527,
    "lca": 0.10436037927865982,
    "bgm_nuaa": 0.4004937410354614,
    "rpcanet_sirst_s3": 0.17499732971191406,
    "bg_anomaly": 0.034170228987932205,
}


def parse_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def render_weighted(samples: list[dict[str, object]], gamma: float) -> list[np.ndarray]:
    maps: list[np.ndarray] = []
    for sample in samples:
        pred = np.zeros_like(sample["mask"], dtype=np.float32)
        for name in NAMES:
            pred += float(CURRENT_BEST_WEIGHTS[name]) * sample["preds"][name].astype(np.float32)
        if abs(gamma - 1.0) > 1e-6:
            pred = np.clip(pred, 0.0, 1.0) ** gamma
        maps.append(pred.astype(np.float32))
    return maps


def filter_one(
    prob: np.ndarray,
    threshold: float,
    min_area: int,
    max_area: int,
    max_side: int,
    min_fill: float,
    max_fill: float,
) -> np.ndarray:
    binary = (prob > threshold).astype(np.uint8)
    if min_area <= 1 and max_area <= 0 and max_side <= 0 and min_fill <= 0 and max_fill >= 1:
        return binary.astype(np.float32)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros_like(binary)
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        fill = float(area) / max(float(width * height), 1.0)
        if area < min_area:
            continue
        if max_area > 0 and area > max_area:
            continue
        if max_side > 0 and max(width, height) > max_side:
            continue
        if fill < min_fill or fill > max_fill:
            continue
        out[labels == label_id] = 1
    return out.astype(np.float32)


def filter_maps(
    maps: list[np.ndarray],
    threshold: float,
    min_area: int,
    max_area: int,
    max_side: int,
    min_fill: float,
    max_fill: float,
) -> list[np.ndarray]:
    return [filter_one(m, threshold, min_area, max_area, max_side, min_fill, max_fill) for m in maps]


def score_row(row: dict[str, float]) -> float:
    # Select on train/fit with a small false-alarm penalty, to avoid simply maximizing recall.
    return float(row["mIoU"]) - 0.0000015 * float(row["Fa"]) * 1e6


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-selected component filtering for current best NUAA BGM/RPCANet fusion.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", default="NUAA-SIRST")
    parser.add_argument("--fit_datasets", default="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3")
    parser.add_argument("--model_spec", action="append", required=True)
    parser.add_argument("--official_bgm_spec", action="append", required=True)
    parser.add_argument("--rpcanet_spec", action="append", required=True)
    parser.add_argument("--max_fit_samples", type=int, default=220)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--background_sample_pixels", type=int, default=2048)
    parser.add_argument("--background_topk", type=int, default=16)
    parser.add_argument("--background_threshold", type=float, default=0.88)
    parser.add_argument("--gammas", default="0.95")
    parser.add_argument("--thresholds", default="0.48,0.50,0.52,0.54,0.56")
    parser.add_argument("--min_areas", default="1,2,3,4,5,6,8")
    parser.add_argument("--max_areas", default="0,64,128,256,512")
    parser.add_argument("--max_sides", default="0,16,24,32,48")
    parser.add_argument("--min_fills", default="0.0,0.15,0.25,0.35")
    parser.add_argument("--max_fills", default="1.0")
    parser.add_argument("--top_fit", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    fit_datasets = [x.strip() for x in args.fit_datasets.split(",") if x.strip()]
    models = build_models([parse_model_spec(spec) for spec in args.model_spec], device)
    bgms = load_official_bgm_models([parse_official_bgm_spec(spec) for spec in args.official_bgm_spec], device)
    rpcanets = load_rpcanet_models([parse_rpcanet_spec(spec) for spec in args.rpcanet_spec], device)
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

    fit_maps_by_gamma = {gamma: render_weighted(fit_samples, gamma) for gamma in parse_floats(args.gammas)}
    fit_rows = []
    for gamma, base_maps in fit_maps_by_gamma.items():
        for threshold in parse_floats(args.thresholds):
            raw = metric_from_maps(fit_samples, base_maps, threshold)
            fit_rows.append(
                {
                    "kind": "raw",
                    "gamma": gamma,
                    "threshold": threshold,
                    "min_area": 1,
                    "max_area": 0,
                    "max_side": 0,
                    "min_fill": 0.0,
                    "max_fill": 1.0,
                    **raw,
                }
            )
            for min_area in parse_ints(args.min_areas):
                for max_area in parse_ints(args.max_areas):
                    for max_side in parse_ints(args.max_sides):
                        for min_fill in parse_floats(args.min_fills):
                            for max_fill in parse_floats(args.max_fills):
                                if min_area <= 1 and max_area <= 0 and max_side <= 0 and min_fill <= 0 and max_fill >= 1:
                                    continue
                                maps = filter_maps(base_maps, threshold, min_area, max_area, max_side, min_fill, max_fill)
                                fit_rows.append(
                                    {
                                        "kind": "component_filter",
                                        "gamma": gamma,
                                        "threshold": threshold,
                                        "min_area": min_area,
                                        "max_area": max_area,
                                        "max_side": max_side,
                                        "min_fill": min_fill,
                                        "max_fill": max_fill,
                                        **metric_from_maps(fit_samples, maps, 0.5),
                                    }
                                )
    fit_rows.sort(key=score_row, reverse=True)

    eval_maps_by_gamma = {gamma: render_weighted(eval_samples, gamma) for gamma in parse_floats(args.gammas)}
    eval_rows = []
    selected_key = None
    seen = set()
    for row in fit_rows[: max(1, args.top_fit)]:
        key = (
            row["kind"],
            row["gamma"],
            row["threshold"],
            row["min_area"],
            row["max_area"],
            row["max_side"],
            row["min_fill"],
            row["max_fill"],
        )
        if key in seen:
            continue
        seen.add(key)
        if selected_key is None:
            selected_key = key
        base_maps = eval_maps_by_gamma[float(row["gamma"])]
        if row["kind"] == "raw":
            metrics = metric_from_maps(eval_samples, base_maps, float(row["threshold"]))
        else:
            maps = filter_maps(
                base_maps,
                float(row["threshold"]),
                int(row["min_area"]),
                int(row["max_area"]),
                int(row["max_side"]),
                float(row["min_fill"]),
                float(row["max_fill"]),
            )
            metrics = metric_from_maps(eval_samples, maps, 0.5)
        eval_rows.append({**row, **{f"fit_{k}": row[k] for k in ("mIoU", "nIoU", "Pd", "Fa")}, **metrics})
    eval_rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best_eval = eval_rows[0]
    selected = next(
        row
        for row in eval_rows
        if (
            row["kind"],
            row["gamma"],
            row["threshold"],
            row["min_area"],
            row["max_area"],
            row["max_side"],
            row["min_fill"],
            row["max_fill"],
        )
        == selected_key
    )
    print(
        f"best_eval kind={best_eval['kind']} gamma={best_eval['gamma']} th={best_eval['threshold']} "
        f"min_area={best_eval['min_area']} max_area={best_eval['max_area']} max_side={best_eval['max_side']} "
        f"min_fill={best_eval['min_fill']} mIoU={best_eval['mIoU']*100:.4f} "
        f"nIoU={best_eval['nIoU']*100:.4f} Pd={best_eval['Pd']*100:.2f} Fa={best_eval['Fa']*1e6:.4f}",
        flush=True,
    )
    Path(args.output).write_text(
        json.dumps(
            {
                "dataset": args.dataset_name,
                "fit_datasets": fit_datasets,
                "weights": CURRENT_BEST_WEIGHTS,
                "fit_samples": len(fit_samples),
                "eval_samples": len(eval_samples),
                "selected_by_fit": selected,
                "best_eval_among_fit_top": best_eval,
                "eval_rows": eval_rows,
                "fit_rows_top": fit_rows[:100],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
