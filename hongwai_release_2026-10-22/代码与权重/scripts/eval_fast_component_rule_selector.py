#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from eval_component_fusion_learner import (
    build_component_cache,
    build_models,
    collect_fit_samples_multi,
    collect_samples,
    load_official_pconv_models,
    metric_from_maps,
    parse_model_spec,
    parse_official_pconv_spec,
    render_components,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def feature_names(names: list[str], extra_dim: int) -> list[str]:
    out = [
        "log_area",
        "log_h",
        "log_w",
        "fill",
        "mean_prob",
        "max_prob",
        "min_prob",
        "std_prob",
        "vote_frac",
        "fg_mean",
        "local_delta",
        "local_z",
        "lap_mean",
        "center_y",
        "center_x",
    ]
    out.extend([f"{name}_mean" for name in names])
    out.extend([f"{name}_max" for name in names])
    out.extend([f"{name}_vote" for name in names])
    out.extend([f"extra_{idx}" for idx in range(max(0, extra_dim))])
    return out


def standardize(train_x: np.ndarray, eval_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    return ((train_x - mean) / std).astype(np.float32), ((eval_x - mean) / std).astype(np.float32)


def make_weight_vector(kind: str, feats: list[str]) -> np.ndarray:
    idx = {name: i for i, name in enumerate(feats)}
    w = np.zeros((len(feats),), dtype=np.float32)

    def add(name: str, value: float) -> None:
        if name in idx:
            w[idx[name]] += value

    # Conservative target-likeness: confidence, local contrast, compact shape.
    if kind == "target_compact":
        add("max_prob", 1.0)
        add("mean_prob", 0.55)
        add("local_z", 0.35)
        add("local_delta", 0.20)
        add("fill", 0.15)
        add("log_area", -0.25)
        add("log_h", -0.08)
        add("log_w", -0.08)
        add("lap_mean", 0.08)
    elif kind == "consensus_clean":
        add("vote_frac", 0.90)
        add("mean_prob", 0.80)
        add("max_prob", 0.30)
        add("local_z", 0.18)
        add("std_prob", -0.25)
        add("log_area", -0.12)
    elif kind == "pconv_rescue":
        add("pconv_p43_max", 1.20)
        add("pconv_p43_mean", 0.55)
        add("local_z", 0.35)
        add("local_delta", 0.18)
        add("fill", 0.10)
        add("baseq_mean", -0.18)
        add("refiner_mean", -0.18)
        add("lca_mean", -0.18)
        add("tpc_a_mean", -0.18)
        add("tcd_a_mean", -0.18)
        add("log_area", -0.18)
    elif kind == "pconv_agree":
        add("pconv_p43_max", 0.85)
        add("pconv_p43_mean", 0.45)
        add("mean_prob", 0.65)
        add("vote_frac", 0.35)
        add("local_z", 0.22)
        add("log_area", -0.15)
    elif kind == "fa_suppress":
        add("max_prob", 0.90)
        add("local_z", 0.45)
        add("fill", 0.20)
        add("std_prob", -0.35)
        add("center_y", -0.05)
        add("center_x", -0.05)
        add("log_area", -0.20)
    else:
        raise ValueError(kind)
    return w


def score_rule(kind: str, train_x: np.ndarray, eval_x: np.ndarray, feats: list[str]) -> tuple[np.ndarray, np.ndarray]:
    train_z, eval_z = standardize(train_x, eval_x)
    w = make_weight_vector(kind, feats)
    return train_z @ w, eval_z @ w


def render_with_scores(cache, names: list[str], scores: np.ndarray, keep_threshold: float, fill_mode: str) -> list[np.ndarray]:
    maps = []
    offset = 0
    for sample, comps, _x, _y in cache:
        n = len(comps)
        maps.append(render_components(sample, names, comps, scores[offset : offset + n], keep_threshold, fill_mode))
        offset += n
    return maps


def baseline_maps(samples: list[dict[str, object]], names: list[str], mode: str) -> list[np.ndarray]:
    maps = []
    for sample in samples:
        stack = np.stack([sample["preds"][name] for name in names], axis=0)
        if mode == "max":
            maps.append(stack.max(axis=0).astype(np.float32))
        elif mode == "avg":
            maps.append(stack.mean(axis=0).astype(np.float32))
        elif mode == "min":
            maps.append(stack.min(axis=0).astype(np.float32))
        else:
            maps.append(sample["preds"][mode].astype(np.float32))
    return maps


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast rule search over component features.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--fit_datasets", default="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3")
    parser.add_argument("--model_spec", action="append", required=True)
    parser.add_argument("--official_pconv_mshnet_spec", action="append", default=[])
    parser.add_argument("--seed_thresholds", default="0.45")
    parser.add_argument("--output_thresholds", default="0.45,0.50")
    parser.add_argument("--rule_kinds", default="target_compact,consensus_clean,pconv_rescue,pconv_agree,fa_suppress")
    parser.add_argument("--keep_percentiles", default="35,45,55,65,75,85")
    parser.add_argument("--max_components_per_image", type=int, default=160)
    parser.add_argument("--max_fit_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    specs = [parse_model_spec(spec) for spec in args.model_spec]
    names = [spec[0] for spec in specs]
    models = build_models(specs, device)
    pconv_specs = [parse_official_pconv_spec(spec) for spec in args.official_pconv_mshnet_spec]
    pconv_models = load_official_pconv_models(pconv_specs, device)
    names.extend(str(spec["name"]) for spec in pconv_specs)

    fit_datasets = [x.strip() for x in args.fit_datasets.split(",") if x.strip()]
    fit_samples = collect_fit_samples_multi(
        models,
        pconv_models,
        Path(args.dataset_dir),
        fit_datasets,
        "train",
        args.max_fit_samples,
        device,
    )
    eval_samples = collect_samples(
        models,
        pconv_models,
        Path(args.dataset_dir),
        args.dataset_name,
        "test",
        args.max_eval_samples,
        device,
    )

    seed_thresholds = parse_float_list(args.seed_thresholds)
    output_thresholds = parse_float_list(args.output_thresholds)
    keep_percentiles = parse_float_list(args.keep_percentiles)
    rule_kinds = [x.strip() for x in args.rule_kinds.split(",") if x.strip()]

    rows: list[dict[str, object]] = []
    for mode in [*names, "avg", "max", "min"]:
        for out_th in output_thresholds:
            rows.append({"kind": "baseline", "mode": mode, "output_threshold": out_th, **metric_from_maps(eval_samples, baseline_maps(eval_samples, names, mode), out_th)})

    for seed in seed_thresholds:
        fit_cache, train_x, train_y = build_component_cache(fit_samples, names, seed, 1, 0.85, args.max_components_per_image)
        eval_cache, eval_x, eval_y = build_component_cache(eval_samples, names, seed, 1, 0.85, args.max_components_per_image)
        if train_x.shape[0] == 0 or eval_x.shape[0] == 0:
            continue
        feats = feature_names(names, train_x.shape[1] - (15 + 3 * len(names)))
        for rule in rule_kinds:
            train_scores, eval_scores = score_rule(rule, train_x, eval_x, feats)
            thresholds = sorted(set(float(np.percentile(train_scores, p)) for p in keep_percentiles))
            candidates = []
            for keep_th in thresholds:
                train_maps = render_with_scores(fit_cache, names, train_scores, keep_th, "max")
                for out_th in output_thresholds:
                    m = metric_from_maps(fit_samples, train_maps, out_th)
                    candidates.append((m["mIoU"], keep_th, out_th, m))
            candidates.sort(reverse=True, key=lambda item: item[0])
            for rank, (_score, keep_th, out_th, train_metrics) in enumerate(candidates[:3]):
                eval_maps = render_with_scores(eval_cache, names, eval_scores, keep_th, "max")
                eval_metrics = metric_from_maps(eval_samples, eval_maps, out_th)
                rows.append(
                    {
                        "kind": "rule_eval",
                        "rule": rule,
                        "rank": rank,
                        "seed_threshold": seed,
                        "keep_threshold": keep_th,
                        "output_threshold": out_th,
                        "train_mIoU": train_metrics["mIoU"],
                        "train_Pd": train_metrics["Pd"],
                        "train_Fa": train_metrics["Fa"],
                        "train_components": int(train_x.shape[0]),
                        "eval_components": int(eval_x.shape[0]),
                        "eval_positive_components": int(eval_y.sum()),
                        **eval_metrics,
                    }
                )

    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best_valid = rows[0]
    print(
        f"best kind={best_valid['kind']} mode={best_valid.get('mode','')} rule={best_valid.get('rule','')} "
        f"mIoU={best_valid['mIoU']*100:.2f} Pd={best_valid['Pd']*100:.2f} Fa={best_valid['Fa']*1e6:.2f}",
        flush=True,
    )
    for row in rows[:20]:
        print(
            f"{row['kind']} mode={row.get('mode','')} rule={row.get('rule','')} out={row.get('output_threshold','')} "
            f"mIoU={row['mIoU']*100:.2f} Pd={row['Pd']*100:.2f} Fa={row['Fa']*1e6:.2f}",
            flush=True,
        )
    Path(args.output).write_text(
        json.dumps(
            {
                "dataset": args.dataset_name,
                "fit_datasets": fit_datasets,
                "models": names,
                "official_pconv_mshnet_specs": args.official_pconv_mshnet_spec,
                "best_valid": best_valid,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
