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
    fit_background_normality_model,
    load_official_pconv_models,
    load_official_bgm_models,
    load_rpcanet_models,
    metric_from_maps,
    parse_model_spec,
    parse_official_pconv_spec,
    parse_official_bgm_spec,
    parse_rpcanet_spec,
)


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def softmax_weights(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total <= 1e-8:
        return [1.0 / len(values)] * len(values)
    return (arr / total).astype(float).tolist()


def candidate_weight_rows(names: list[str], mode: str) -> list[dict[str, object]]:
    pconv = [name for name in names if name.startswith("pconv")]
    external = [name for name in names if name.startswith(("pconv", "bgm", "rpcanet", "bg_"))]
    teachers = [name for name in names if name not in external]
    bgm = [name for name in names if name.startswith("bgm")]
    rpcanet = [name for name in names if name.startswith("rpcanet")]
    bg = [name for name in names if name.startswith("bg_")]
    rows: list[dict[str, object]] = []

    def add(kind: str, raw: dict[str, float]) -> None:
        weights = {name: 0.0 for name in names}
        for name, value in raw.items():
            weights[name] = float(value)
        norm = softmax_weights([weights[name] for name in names])
        rows.append({"kind": kind, "weights": {name: norm[i] for i, name in enumerate(names)}})

    add("equal_avg", {name: 1.0 for name in names})
    for name in names:
        add(f"single_{name}", {name: 1.0})

    if teachers and external:
        if mode == "coarse":
            teacher_totals = (0.30, 0.40, 0.50, 0.60)
            bgm_totals = (0.10, 0.20, 0.30, 0.40)
            rpc_totals = (0.05, 0.10, 0.15, 0.20)
            bg_totals = (0.00, 0.05, 0.10)
        else:
            teacher_totals = (0.25, 0.32, 0.39, 0.46, 0.53, 0.60)
            bgm_totals = (0.08, 0.16, 0.24, 0.32, 0.40)
            rpc_totals = (0.04, 0.08, 0.12, 0.16, 0.20, 0.24)
            bg_totals = (0.00, 0.04, 0.08, 0.12)
        for teacher_total in teacher_totals:
            for bgm_total in bgm_totals:
                for rpc_total in rpc_totals:
                    for bg_total in bg_totals:
                        ext_total = bgm_total + rpc_total + bg_total
                        if ext_total <= 0 or teacher_total + ext_total > 1.0001:
                            continue
                        raw: dict[str, float] = {}
                        for name in teachers:
                            raw[name] = teacher_total / len(teachers)
                        for name in bgm:
                            raw[name] = bgm_total / len(bgm)
                        for name in rpcanet:
                            raw[name] = rpc_total / len(rpcanet)
                        for name in bg:
                            raw[name] = bg_total / len(bg)
                        add("teacher_external_weighted", raw)

    if mode == "coarse":
        teacher_totals = (0.30, 0.40, 0.50, 0.60, 0.70)
        pconv_splits = (0.25, 0.40, 0.50, 0.60, 0.75)
    else:
        teacher_totals = (0.34, 0.40, 0.46, 0.52, 0.58, 0.64)
        pconv_splits = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80)

    if teachers and pconv:
        for teacher_total in teacher_totals:
            pconv_total = 1.0 - teacher_total
            for split in pconv_splits:
                raw = {name: teacher_total / len(teachers) for name in teachers}
                if len(pconv) == 1:
                    raw[pconv[0]] = pconv_total
                else:
                    raw[pconv[0]] = pconv_total * split
                    rest = pconv_total * (1.0 - split)
                    for name in pconv[1:]:
                        raw[name] = rest / max(len(pconv) - 1, 1)
                add("teacher_pconv_weighted", raw)

    if len(teachers) >= 2 and pconv:
        for first_teacher_share in (0.35, 0.50, 0.65):
            for teacher_total in teacher_totals:
                pconv_total = 1.0 - teacher_total
                raw = {name: 0.0 for name in names}
                raw[teachers[0]] = teacher_total * first_teacher_share
                remain_teacher = teacher_total * (1.0 - first_teacher_share)
                for name in teachers[1:]:
                    raw[name] = remain_teacher / (len(teachers) - 1)
                for name in pconv:
                    raw[name] = pconv_total / len(pconv)
                add("teacher_split_weighted", raw)

    # Deduplicate near-identical normalized weights.
    seen = set()
    unique = []
    for row in rows:
        key = tuple(round(float(row["weights"][name]), 4) for name in names)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def render_weighted(samples: list[dict[str, object]], names: list[str], weights: dict[str, float], gamma: float) -> list[np.ndarray]:
    maps = []
    for sample in samples:
        out = np.zeros_like(sample["mask"], dtype=np.float32)
        for name in names:
            out += float(weights[name]) * sample["preds"][name].astype(np.float32)
        if abs(gamma - 1.0) > 1e-6:
            out = np.clip(out, 0.0, 1.0) ** gamma
        maps.append(out.astype(np.float32))
    return maps


def fast_niou(samples: list[dict[str, object]], maps: list[np.ndarray], threshold: float) -> float:
    vals = []
    for sample, pred in zip(samples, maps):
        mask = sample["mask"] > 0
        pred_bin = pred > threshold
        inter = float((pred_bin & mask).sum())
        union = float((pred_bin | mask).sum())
        vals.append(inter / (union + np.spacing(1)))
    return float(np.mean(vals)) if vals else 0.0


def precompute_flat(samples: list[dict[str, object]], names: list[str], stride: int) -> list[dict[str, np.ndarray]]:
    flat = []
    for sample in samples:
        step = max(int(stride), 1)
        stack = np.stack([sample["preds"][name].astype(np.float32)[::step, ::step].reshape(-1) for name in names], axis=0)
        flat.append({"stack": stack, "mask": (sample["mask"][::step, ::step].reshape(-1) > 0)})
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-fit weighted fusion of teacher and external candidate maps.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--fit_datasets", default="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3")
    parser.add_argument("--model_spec", action="append", required=True)
    parser.add_argument("--official_pconv_mshnet_spec", action="append", default=[])
    parser.add_argument("--official_bgm_spec", action="append", default=[])
    parser.add_argument("--rpcanet_spec", action="append", default=[])
    parser.add_argument("--background_normality_candidates", action="store_true")
    parser.add_argument("--background_as_candidate", action="store_true")
    parser.add_argument("--background_topk", type=int, default=16)
    parser.add_argument("--background_threshold", type=float, default=0.88)
    parser.add_argument("--background_sample_pixels", type=int, default=2048)
    parser.add_argument("--output_thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60")
    parser.add_argument("--gammas", default="0.85,1.0,1.15")
    parser.add_argument("--weight_mode", default="fine", choices=["coarse", "fine"])
    parser.add_argument("--top_fit", type=int, default=32)
    parser.add_argument("--fit_pixel_stride", type=int, default=4)
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
    official_bgm_specs = [parse_official_bgm_spec(spec) for spec in args.official_bgm_spec]
    official_bgms = load_official_bgm_models(official_bgm_specs, device)
    rpcanet_specs = [parse_rpcanet_spec(spec) for spec in args.rpcanet_spec]
    rpcanets = load_rpcanet_models(rpcanet_specs, device)
    names.extend(str(spec["name"]) for spec in official_specs)
    names.extend(str(spec["name"]) for spec in official_bgm_specs)
    names.extend(str(spec["name"]) for spec in rpcanet_specs)
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
    fit_samples = collect_fit_samples_multi(
        models,
        officials,
        official_bgms,
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
        officials,
        official_bgms,
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

    out_ths = parse_float_list(args.output_thresholds)
    gammas = parse_float_list(args.gammas)
    weight_rows = candidate_weight_rows(names, args.weight_mode)
    print(f"searching {len(weight_rows)} weights x {len(gammas)} gammas x {len(out_ths)} thresholds", flush=True)

    fit_rows = []
    fit_flat = precompute_flat(fit_samples, names, args.fit_pixel_stride)
    for row in weight_rows:
        weight_vec = np.asarray([row["weights"][name] for name in names], dtype=np.float32)
        for gamma in gammas:
            for out_th in out_ths:
                fit_rows.append(
                    {
                        "stage": "fit",
                        **row,
                        "gamma": gamma,
                        "output_threshold": out_th,
                        "fast_nIoU": fast_weighted_niou(fit_flat, weight_vec, gamma, out_th),
                    }
                )
    fit_rows.sort(key=lambda row: row["fast_nIoU"], reverse=True)

    eval_rows = []
    seen = set()
    for row in fit_rows[: args.top_fit]:
        key = (
            tuple(round(float(row["weights"][name]), 4) for name in names),
            round(float(row["gamma"]), 4),
            round(float(row["output_threshold"]), 4),
        )
        if key in seen:
            continue
        seen.add(key)
        maps = render_weighted(eval_samples, names, row["weights"], float(row["gamma"]))
        metrics = metric_from_maps(eval_samples, maps, float(row["output_threshold"]))
        eval_rows.append(
            {
                "stage": "eval",
                "fit_fast_nIoU": row["fast_nIoU"],
                "kind": row["kind"],
                "weights": row["weights"],
                "gamma": row["gamma"],
                "output_threshold": row["output_threshold"],
                **metrics,
            }
        )
    eval_rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best = eval_rows[0]
    print(
        f"best kind={best['kind']} out={best['output_threshold']} gamma={best['gamma']} "
        f"mIoU={best['mIoU']*100:.2f} Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.2f} weights={best['weights']}",
        flush=True,
    )
    for row in eval_rows[:20]:
        print(
            f"{row['kind']} out={row['output_threshold']} gamma={row['gamma']} "
            f"fit_fast={row['fit_fast_nIoU']*100:.2f} mIoU={row['mIoU']*100:.2f} "
            f"Pd={row['Pd']*100:.2f} Fa={row['Fa']*1e6:.2f} weights={row['weights']}",
            flush=True,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "dataset": args.dataset_name,
                "fit_datasets": fit_datasets,
                "models": names,
                "official_pconv_mshnet_specs": args.official_pconv_mshnet_spec,
                "official_bgm_specs": args.official_bgm_spec,
                "rpcanet_specs": args.rpcanet_spec,
                "background_normality_candidates": bool(args.background_normality_candidates),
                "background_as_candidate": bool(args.background_as_candidate),
                "best_eval": best,
                "eval_rows": eval_rows,
                "fit_rows_top": fit_rows[:50],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
