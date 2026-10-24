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


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def render(samples: list[dict[str, object]], weights: dict[str, float], gamma: float) -> list[np.ndarray]:
    maps: list[np.ndarray] = []
    for sample in samples:
        pred = np.zeros_like(sample["mask"], dtype=np.float32)
        for name, weight in weights.items():
            pred += float(weight) * sample["preds"][name].astype(np.float32)
        if abs(gamma - 1.0) > 1e-6:
            pred = np.clip(pred, 0.0, 1.0) ** gamma
        maps.append(pred.astype(np.float32))
    return maps


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fixed BGM+RPCANet weighted fusion on full NUAA.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", default="NUAA-SIRST")
    parser.add_argument("--fit_datasets", default="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3")
    parser.add_argument("--model_spec", action="append", required=True)
    parser.add_argument("--official_bgm_spec", action="append", required=True)
    parser.add_argument("--rpcanet_spec", action="append", required=True)
    parser.add_argument("--background_normality_candidates", action="store_true")
    parser.add_argument("--background_as_candidate", action="store_true")
    parser.add_argument("--background_topk", type=int, default=16)
    parser.add_argument("--background_threshold", type=float, default=0.88)
    parser.add_argument("--background_sample_pixels", type=int, default=2048)
    parser.add_argument("--max_fit_samples", type=int, default=220)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--gammas", default="0.85,0.90,0.95,1.0,1.05")
    parser.add_argument("--thresholds", default="0.35,0.38,0.40,0.42,0.45,0.48,0.50")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    specs = [parse_model_spec(spec) for spec in args.model_spec]
    names = [spec[0] for spec in specs]
    models = build_models(specs, device)
    bgm_specs = [parse_official_bgm_spec(spec) for spec in args.official_bgm_spec]
    bgms = load_official_bgm_models(bgm_specs, device)
    rpc_specs = [parse_rpcanet_spec(spec) for spec in args.rpcanet_spec]
    rpcanets = load_rpcanet_models(rpc_specs, device)
    names.extend(str(spec["name"]) for spec in bgm_specs)
    names.extend(str(spec["name"]) for spec in rpc_specs)

    fit_datasets = [x.strip() for x in args.fit_datasets.split(",") if x.strip()]
    background_model = None
    if args.background_normality_candidates:
        background_model = fit_background_normality_model(
            Path(args.dataset_dir),
            fit_datasets or [args.dataset_name],
            "train",
            args.max_fit_samples,
            args.background_sample_pixels,
        )
        if args.background_as_candidate:
            names.append("bg_anomaly")
    # Fitting is used only to keep the same background-normality model as the component screen.
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
        background_as_candidate=args.background_as_candidate,
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
        background_as_candidate=args.background_as_candidate,
        background_topk=args.background_topk,
        background_threshold=args.background_threshold,
    )

    rows = []
    fixed_rows = [
        {
            "kind": "smoke_best",
            "weights": {
                "tpc_a": 1.0 / 6.0,
                "tpc_b": 1.0 / 6.0,
                "lca": 1.0 / 6.0,
                "bgm_nuaa": 1.0 / 6.0,
                "rpcanet_sirst_s3": 1.0 / 3.0,
            },
        },
        {
            "kind": "component_best_like",
            "weights": {
                "tpc_a": 0.14,
                "tpc_b": 0.14,
                "lca": 0.14,
                "bgm_nuaa": 0.28,
                "rpcanet_sirst_s3": 0.30,
            },
        },
        {
            "kind": "balanced_no_bg_anomaly",
            "weights": {
                "tpc_a": 0.15,
                "tpc_b": 0.15,
                "lca": 0.15,
                "bgm_nuaa": 0.25,
                "rpcanet_sirst_s3": 0.30,
            },
        },
    ]
    if "bg_anomaly" in names:
        fixed_rows.append(
            {
                "kind": "with_bg_anomaly",
                "weights": {
                    "tpc_a": 0.14,
                    "tpc_b": 0.14,
                    "lca": 0.14,
                    "bgm_nuaa": 0.22,
                    "rpcanet_sirst_s3": 0.28,
                    "bg_anomaly": 0.08,
                },
            }
        )
    for row in fixed_rows:
        weights = {name: 0.0 for name in names}
        weights.update({name: value for name, value in row["weights"].items() if name in names})
        total = sum(weights.values())
        weights = {name: value / total for name, value in weights.items()}
        for gamma in parse_float_list(args.gammas):
            maps = render(eval_samples, weights, gamma)
            for threshold in parse_float_list(args.thresholds):
                rows.append(
                    {
                        "kind": row["kind"],
                        "weights": weights,
                        "gamma": gamma,
                        "output_threshold": threshold,
                        **metric_from_maps(eval_samples, maps, threshold),
                    }
                )
    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best = rows[0]
    print(
        f"best kind={best['kind']} out={best['output_threshold']} gamma={best['gamma']} "
        f"mIoU={best['mIoU']*100:.2f} nIoU={best['nIoU']*100:.2f} "
        f"Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.2f}",
        flush=True,
    )
    Path(args.output).write_text(
        json.dumps(
            {
                "dataset": args.dataset_name,
                "models": names,
                "best": best,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
