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

from build_prediction_cache import save_samples_npz  # noqa: E402
from eval_component_fusion_learner import (  # noqa: E402
    build_models,
    collect_samples,
    fit_background_normality_model,
    load_official_bgm_models,
    load_rpcanet_models,
    parse_model_spec,
    parse_official_bgm_spec,
    parse_rpcanet_spec,
)


NAMES = ["tpc_a", "tpc_b", "lca", "bgm_nuaa", "rpcanet_sirst_s3", "bg_anomaly"]
BEST_WEIGHTS = {
    "tpc_a": 0.1899983435869217,
    "tpc_b": 0.09598003327846527,
    "lca": 0.10436037927865982,
    "bgm_nuaa": 0.4004937410354614,
    "rpcanet_sirst_s3": 0.17499732971191406,
    "bg_anomaly": 0.034170228987932205,
}


def parse_dataset_split(value: str) -> tuple[str, str]:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError("dataset split must be DATASET:SPLIT")
    return parts[0], parts[1]


def add_weighted_fusion(samples: list[dict[str, object]], gamma: float) -> None:
    for sample in samples:
        preds = sample["preds"]
        missing = [name for name in NAMES if name not in preds]
        if missing:
            raise KeyError(f"missing candidate maps: {missing}")
        weighted = None
        for name in NAMES:
            arr = preds[name].astype(np.float32)
            weighted = arr * BEST_WEIGHTS[name] if weighted is None else weighted + arr * BEST_WEIGHTS[name]
        weighted = np.clip(weighted.astype(np.float32), 0.0, 1.0)
        preds["weighted_raw"] = weighted
        preds["strong_fusion"] = (weighted**gamma).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NUAA strong BGM/RPCANet fusion caches for anchor-refiner training.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_split", action="append", required=True, help="DATASET:SPLIT")
    parser.add_argument("--fit_datasets", default="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3")
    parser.add_argument("--model_spec", action="append", required=True)
    parser.add_argument("--official_bgm_spec", action="append", required=True)
    parser.add_argument("--rpcanet_spec", action="append", required=True)
    parser.add_argument("--background_sample_pixels", type=int, default=2048)
    parser.add_argument("--background_topk", type=int, default=16)
    parser.add_argument("--background_threshold", type=float, default=0.88)
    parser.add_argument("--background_fit_samples", type=int, default=220)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    models = build_models([parse_model_spec(spec) for spec in args.model_spec], device)
    bgms = load_official_bgm_models([parse_official_bgm_spec(spec) for spec in args.official_bgm_spec], device)
    rpcanets = load_rpcanet_models([parse_rpcanet_spec(spec) for spec in args.rpcanet_spec], device)
    fit_datasets = [item.strip() for item in args.fit_datasets.split(",") if item.strip()]
    background_model = fit_background_normality_model(
        Path(args.dataset_dir),
        fit_datasets,
        "train",
        args.background_fit_samples,
        args.background_sample_pixels,
    )

    output_dir = Path(args.output_dir)
    output_names = ["strong_fusion", "weighted_raw", *NAMES]
    manifest = {
        "dataset_dir": str(args.dataset_dir),
        "model_specs": args.model_spec,
        "official_bgm_specs": args.official_bgm_spec,
        "rpcanet_specs": args.rpcanet_spec,
        "background_fit_datasets": fit_datasets,
        "weights": BEST_WEIGHTS,
        "gamma": args.gamma,
        "names": output_names,
        "splits": [],
    }

    for raw in args.dataset_split:
        dataset_name, split = parse_dataset_split(raw)
        print(f"collecting {dataset_name}:{split}", flush=True)
        samples = collect_samples(
            models,
            [],
            bgms,
            rpcanets,
            Path(args.dataset_dir),
            dataset_name,
            split,
            args.max_samples,
            device,
            background_model=background_model,
            background_as_candidate=True,
            background_topk=args.background_topk,
            background_threshold=args.background_threshold,
        )
        add_weighted_fusion(samples, args.gamma)
        out = output_dir / f"{dataset_name}__{split}.npz"
        save_samples_npz(samples, output_names, out)
        manifest["splits"].append({"dataset": dataset_name, "split": split, "path": str(out), "samples": len(samples)})
        print(f"wrote {out} samples={len(samples)} size_mb={out.stat().st_size / 1024 / 1024:.2f}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest {output_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
