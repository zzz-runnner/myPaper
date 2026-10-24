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
    metric_from_maps,
    parse_model_spec,
    parse_official_pconv_spec,
)


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def parse_weight_spec(value: str, names: list[str]) -> tuple[str, dict[str, float]]:
    label, raw = value.split(":", 1)
    weights = {name: 0.0 for name in names}
    for part in raw.split(","):
        key, val = part.split("=")
        weights[key.strip()] = float(val)
    total = sum(weights.values())
    if total <= 0:
        raise ValueError(value)
    return label, {k: v / total for k, v in weights.items()}


def default_weights(names: list[str]) -> list[tuple[str, dict[str, float]]]:
    rows = []
    rows.append(("avg_all", {name: 1.0 / len(names) for name in names}))
    for name in names:
        rows.append((f"single_{name}", {n: 1.0 if n == name else 0.0 for n in names}))
    pconv = [name for name in names if name.startswith("pconv")]
    teachers = [name for name in names if name not in pconv]
    if teachers and pconv:
        for teacher_total, split in [
            (0.50, 0.50),
            (0.55, 0.50),
            (0.60, 0.50),
            (0.65, 0.50),
            (0.70, 0.50),
            (0.60, 0.35),
            (0.60, 0.65),
            (0.70, 0.35),
            (0.70, 0.65),
        ]:
            raw = {name: 0.0 for name in names}
            for name in teachers:
                raw[name] = teacher_total / len(teachers)
            p_total = 1.0 - teacher_total
            if len(pconv) == 1:
                raw[pconv[0]] = p_total
            else:
                raw[pconv[0]] = p_total * split
                for name in pconv[1:]:
                    raw[name] = p_total * (1.0 - split) / (len(pconv) - 1)
            rows.append((f"teacher{teacher_total:.2f}_pconv_split{split:.2f}", raw))
    return rows


def render(samples: list[dict[str, object]], names: list[str], weights: dict[str, float], gamma: float) -> list[np.ndarray]:
    out = []
    for sample in samples:
        pred = np.zeros_like(sample["mask"], dtype=np.float32)
        for name in names:
            pred += float(weights[name]) * sample["preds"][name].astype(np.float32)
        pred = np.clip(pred, 0.0, 1.0)
        if abs(gamma - 1.0) > 1e-6:
            pred = pred ** gamma
        out.append(pred.astype(np.float32))
    return out


def fast_niou(samples: list[dict[str, object]], maps: list[np.ndarray], threshold: float) -> float:
    vals = []
    for sample, pred in zip(samples, maps):
        mask = sample["mask"] > 0
        pred_bin = pred > threshold
        inter = float((pred_bin & mask).sum())
        union = float((pred_bin | mask).sum())
        vals.append(inter / (union + np.spacing(1)))
    return float(np.mean(vals)) if vals else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a small fixed set of weighted fusion candidates on the full test split.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--model_spec", action="append", required=True)
    parser.add_argument("--official_pconv_mshnet_spec", action="append", default=[])
    parser.add_argument("--weight_spec", action="append", default=[])
    parser.add_argument("--output_thresholds", default="0.40,0.45,0.50,0.55")
    parser.add_argument("--gammas", default="0.9,1.0,1.1")
    parser.add_argument("--full_metric_topk", type=int, default=12)
    parser.add_argument("--fast_only", action="store_true")
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
    samples = collect_samples(models, officials, Path(args.dataset_dir), args.dataset_name, "test", args.max_eval_samples, device)

    candidates = default_weights(names)
    for spec in args.weight_spec:
        candidates.append(parse_weight_spec(spec, names))
    gammas = parse_float_list(args.gammas)
    thresholds = parse_float_list(args.output_thresholds)
    fast_rows = []
    for label, weights in candidates:
        for gamma in gammas:
            maps = render(samples, names, weights, gamma)
            for threshold in thresholds:
                fast_rows.append(
                    {
                        "label": label,
                        "weights": weights,
                        "gamma": gamma,
                        "output_threshold": threshold,
                        "fast_nIoU": fast_niou(samples, maps, threshold),
                    }
                )
    fast_rows.sort(key=lambda row: row["fast_nIoU"], reverse=True)

    if args.fast_only:
        best = fast_rows[0]
        print(
            f"best-fast label={best['label']} out={best['output_threshold']} gamma={best['gamma']} "
            f"fast_nIoU={best['fast_nIoU']*100:.2f} weights={best['weights']}",
            flush=True,
        )
        for row in fast_rows[:20]:
            print(
                f"{row['label']} out={row['output_threshold']} gamma={row['gamma']} "
                f"fast_nIoU={row['fast_nIoU']*100:.2f}",
                flush=True,
            )
        Path(args.output).write_text(
            json.dumps(
                {
                    "dataset": args.dataset_name,
                    "models": names,
                    "official_pconv_mshnet_specs": args.official_pconv_mshnet_spec,
                    "metric_mode": "fast_nIoU_only",
                    "best": best,
                    "rows": fast_rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return

    rows = []
    seen = set()
    for row in fast_rows[: max(int(args.full_metric_topk), 1)]:
        key = (row["label"], row["gamma"], row["output_threshold"])
        if key in seen:
            continue
        seen.add(key)
        maps = render(samples, names, row["weights"], row["gamma"])
        rows.append({**row, **metric_from_maps(samples, maps, row["output_threshold"])})
    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best = rows[0]
    print(
        f"best label={best['label']} out={best['output_threshold']} gamma={best['gamma']} "
        f"mIoU={best['mIoU']*100:.2f} Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.2f} weights={best['weights']}",
        flush=True,
    )
    for row in rows[:20]:
        print(
            f"{row['label']} out={row['output_threshold']} gamma={row['gamma']} "
            f"mIoU={row['mIoU']*100:.2f} Pd={row['Pd']*100:.2f} Fa={row['Fa']*1e6:.2f}",
            flush=True,
        )
    Path(args.output).write_text(
        json.dumps(
            {
                "dataset": args.dataset_name,
                "models": names,
                "official_pconv_mshnet_specs": args.official_pconv_mshnet_spec,
                "best": best,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
