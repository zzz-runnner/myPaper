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
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))

from eval_component_fusion_learner import (  # noqa: E402
    build_models,
    collect_samples,
    load_official_pconv_models,
    parse_model_spec,
    parse_official_pconv_spec,
)


def parse_dataset_split(value: str) -> tuple[str, str]:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError("dataset split must be DATASET:SPLIT")
    return parts[0], parts[1]


def save_samples_npz(samples: list[dict[str, object]], names: list[str], output: Path) -> None:
    arrays: dict[str, np.ndarray] = {}
    meta = []
    for idx, sample in enumerate(samples):
        height, width = [int(x) for x in sample["size"]]
        meta.append({"name": str(sample["name"]), "size": [height, width]})
        arrays[f"mask_{idx}"] = (sample["mask"][:height, :width] > 0).astype(np.uint8)
        for pred_name in names:
            arrays[f"pred_{idx}_{pred_name}"] = np.asarray(sample["preds"][pred_name][:height, :width], dtype=np.float16)
    arrays["meta_json"] = np.asarray(json.dumps(meta), dtype=np.str_)
    arrays["names_json"] = np.asarray(json.dumps(names), dtype=np.str_)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact per-split prediction caches for teacher/PConv candidates.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_split", action="append", required=True, help="DATASET:SPLIT, e.g. IRSTD-1K:test")
    parser.add_argument("--model_spec", action="append", required=True)
    parser.add_argument("--official_pconv_mshnet_spec", action="append", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    specs = [parse_model_spec(spec) for spec in args.model_spec]
    names = [spec[0] for spec in specs]
    models = build_models(specs, device)
    official_specs = [parse_official_pconv_spec(spec) for spec in args.official_pconv_mshnet_spec]
    officials = load_official_pconv_models(official_specs, device)
    names.extend(str(spec["name"]) for spec in official_specs)

    output_dir = Path(args.output_dir)
    manifest = {
        "dataset_dir": str(args.dataset_dir),
        "model_specs": args.model_spec,
        "official_pconv_mshnet_specs": args.official_pconv_mshnet_spec,
        "names": names,
        "splits": [],
    }
    for raw in args.dataset_split:
        dataset_name, split = parse_dataset_split(raw)
        print(f"collecting {dataset_name}:{split}", flush=True)
        samples = collect_samples(models, officials, Path(args.dataset_dir), dataset_name, split, args.max_samples, device)
        out = output_dir / f"{dataset_name}__{split}.npz"
        save_samples_npz(samples, names, out)
        manifest["splits"].append(
            {
                "dataset": dataset_name,
                "split": split,
                "path": str(out),
                "samples": len(samples),
            }
        )
        print(f"wrote {out} samples={len(samples)} size_mb={out.stat().st_size / 1024 / 1024:.2f}", flush=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest {output_dir / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
