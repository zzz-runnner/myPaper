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
    collect_fit_samples_multi,
    collect_samples,
    load_official_pconv_models,
    metric_from_maps,
    parse_model_spec,
    parse_official_pconv_spec,
)


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def avg_maps(samples: list[dict[str, object]], names: list[str]) -> list[np.ndarray]:
    maps = []
    for sample in samples:
        stack = np.stack([sample["preds"][name] for name in names], axis=0)
        maps.append(stack.mean(axis=0).astype(np.float32))
    return maps


def main() -> None:
    parser = argparse.ArgumentParser(description="Select avg-fusion output threshold on train data and evaluate test data.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--fit_datasets", default="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3")
    parser.add_argument("--model_spec", action="append", required=True)
    parser.add_argument("--official_pconv_mshnet_spec", action="append", default=[])
    parser.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60")
    parser.add_argument("--max_fit_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    specs = [parse_model_spec(spec) for spec in args.model_spec]
    names = [spec[0] for spec in specs]
    models = build_models(specs, device)
    official_specs = [parse_official_pconv_spec(spec) for spec in args.official_pconv_mshnet_spec]
    officials = load_official_pconv_models(official_specs, device)
    names.extend(str(spec["name"]) for spec in official_specs)
    fit_datasets = [x.strip() for x in args.fit_datasets.split(",") if x.strip()]

    fit_samples = collect_fit_samples_multi(models, officials, Path(args.dataset_dir), fit_datasets, "train", args.max_fit_samples, device)
    eval_samples = collect_samples(models, officials, Path(args.dataset_dir), args.dataset_name, "test", args.max_eval_samples, device)
    fit_maps = avg_maps(fit_samples, names)
    eval_maps = avg_maps(eval_samples, names)

    thresholds = parse_float_list(args.thresholds)
    fit_rows = []
    eval_by_threshold = {}
    for th in thresholds:
        fit_rows.append({"threshold": th, **metric_from_maps(fit_samples, fit_maps, th)})
        eval_by_threshold[str(th)] = metric_from_maps(eval_samples, eval_maps, th)
    fit_rows.sort(key=lambda row: row["mIoU"], reverse=True)
    selected = fit_rows[0]
    selected_th = float(selected["threshold"])
    selected_eval = eval_by_threshold[str(selected_th)]
    eval_rows = [{"threshold": float(k), **v} for k, v in eval_by_threshold.items()]
    eval_rows.sort(key=lambda row: row["mIoU"], reverse=True)

    print(
        f"selected threshold={selected_th} fit_mIoU={selected['mIoU']*100:.2f} "
        f"eval_mIoU={selected_eval['mIoU']*100:.2f} eval_Pd={selected_eval['Pd']*100:.2f} "
        f"eval_Fa={selected_eval['Fa']*1e6:.2f}",
        flush=True,
    )
    print(
        f"test_best threshold={eval_rows[0]['threshold']} mIoU={eval_rows[0]['mIoU']*100:.2f} "
        f"Pd={eval_rows[0]['Pd']*100:.2f} Fa={eval_rows[0]['Fa']*1e6:.2f}",
        flush=True,
    )
    Path(args.output).write_text(
        json.dumps(
            {
                "dataset": args.dataset_name,
                "fit_datasets": fit_datasets,
                "models": names,
                "selected_by_fit": selected,
                "selected_eval": {"threshold": selected_th, **selected_eval},
                "test_best": eval_rows[0],
                "fit_rows": fit_rows,
                "eval_rows": eval_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
