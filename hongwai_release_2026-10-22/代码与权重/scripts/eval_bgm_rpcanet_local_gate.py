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
BASE_WEIGHTS = {
    "tpc_a": 0.14,
    "tpc_b": 0.14,
    "lca": 0.14,
    "bgm_nuaa": 0.22,
    "rpcanet_sirst_s3": 0.28,
    "bg_anomaly": 0.08,
}


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def robust01(x: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(x.astype(np.float32), [low, high])
    if hi <= lo + 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def local_cues(image: np.ndarray) -> dict[str, np.ndarray]:
    src = robust01(image)
    mean = cv2.blur(src, (9, 9))
    sq_mean = cv2.blur(src * src, (9, 9))
    std = np.sqrt(np.maximum(sq_mean - mean * mean, 1e-6))
    z = np.maximum((src - mean) / (std + 1e-3), 0.0)
    z01 = np.clip(z / 6.0, 0.0, 1.0).astype(np.float32)
    dog = np.maximum(cv2.GaussianBlur(src, (5, 5), 0) - cv2.GaussianBlur(src, (13, 13), 0), 0.0)
    top_maps = []
    for ksize in (5, 7, 9):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        top_maps.append(robust01(cv2.morphologyEx(src, cv2.MORPH_TOPHAT, kernel), 50.0, 99.8))
    top = np.stack(top_maps, axis=0).max(axis=0).astype(np.float32)
    return {"z": z01, "dog": robust01(dog, 50.0, 99.8), "top": top}


def base_map(sample: dict[str, object]) -> np.ndarray:
    pred = np.zeros_like(sample["mask"], dtype=np.float32)
    for name, weight in BASE_WEIGHTS.items():
        pred += float(weight) * sample["preds"][name].astype(np.float32)
    return np.clip(pred, 0.0, 1.0).astype(np.float32)


def source_parts(sample: dict[str, object]) -> dict[str, np.ndarray]:
    preds = sample["preds"]
    tpc_lca = np.stack([preds["tpc_a"], preds["tpc_b"], preds["lca"]], axis=0).astype(np.float32)
    all_maps = np.stack([preds[name] for name in NAMES], axis=0).astype(np.float32)
    return {
        "base": base_map(sample),
        "core": tpc_lca.mean(axis=0),
        "core_max": tpc_lca.max(axis=0),
        "bgm": preds["bgm_nuaa"].astype(np.float32),
        "rpc": preds["rpcanet_sirst_s3"].astype(np.float32),
        "bg": preds["bg_anomaly"].astype(np.float32),
        "agree": (all_maps >= 0.5).mean(axis=0).astype(np.float32),
    }


def prepare_samples(samples: list[dict[str, object]]) -> None:
    for idx, sample in enumerate(samples):
        sample["gate_parts"] = source_parts(sample)
        sample["gate_cues"] = local_cues(sample["image"].astype(np.float32))
        if (idx + 1) % 100 == 0:
            print(f"prepared local-gate features {idx+1}/{len(samples)}", flush=True)


def make_map(sample: dict[str, object], params: dict[str, object]) -> np.ndarray:
    parts = sample.get("gate_parts")
    cues = sample.get("gate_cues")
    if parts is None or cues is None:
        parts = source_parts(sample)
        cues = local_cues(sample["image"].astype(np.float32))
    out = parts["base"].copy()
    kind = str(params["kind"])
    sal = np.maximum.reduce([cues["z"], cues["dog"], cues["top"]])
    support = np.maximum.reduce([parts["core_max"], parts["bgm"], parts["rpc"]])

    if kind == "base":
        return out
    if kind == "suppress_low_saliency":
        bad = (
            (out >= float(params["base_lo"]))
            & (support <= float(params["support_hi"]))
            & (sal <= float(params["sal_hi"]))
            & (parts["agree"] <= float(params["agree_hi"]))
        )
        out[bad] *= float(params["factor"])
    elif kind == "suppress_bg_only":
        bad = (
            (out >= float(params["base_lo"]))
            & (parts["bg"] >= float(params["bg_lo"]))
            & (parts["core_max"] <= float(params["core_hi"]))
            & (parts["rpc"] <= float(params["rpc_hi"]))
        )
        out[bad] *= float(params["factor"])
    elif kind == "rpc_rescue":
        gate = (
            (parts["rpc"] >= float(params["rpc_lo"]))
            & (out <= float(params["base_hi"]))
            & (sal >= float(params["sal_lo"]))
            & (parts["bg"] <= float(params["bg_hi"]))
        )
        out[gate] = np.maximum(out[gate], float(params["mix"]) * parts["rpc"][gate])
    elif kind == "bgm_rpc_rescue":
        rescue = np.maximum(parts["bgm"], parts["rpc"])
        gate = (
            (rescue >= float(params["rescue_lo"]))
            & (out <= float(params["base_hi"]))
            & (sal >= float(params["sal_lo"]))
            & (parts["agree"] >= float(params["agree_lo"]))
        )
        out[gate] = np.maximum(out[gate], float(params["mix"]) * rescue[gate])
    elif kind == "rescue_then_suppress":
        rescue = np.maximum(parts["bgm"], parts["rpc"])
        gate = (
            (rescue >= float(params["rescue_lo"]))
            & (out <= float(params["base_hi"]))
            & (sal >= float(params["sal_lo"]))
            & (parts["agree"] >= float(params["agree_lo"]))
        )
        out[gate] = np.maximum(out[gate], float(params["mix"]) * rescue[gate])
        bad = (
            (out >= float(params["s_base_lo"]))
            & (support <= float(params["support_hi"]))
            & (sal <= float(params["sal_hi"]))
            & (parts["agree"] <= float(params["agree_hi"]))
        )
        out[bad] *= float(params["factor"])
    else:
        raise ValueError(kind)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def make_maps(samples: list[dict[str, object]], params: dict[str, object]) -> list[np.ndarray]:
    return [make_map(sample, params) for sample in samples]


def fast_niou(samples: list[dict[str, object]], maps: list[np.ndarray], threshold: float) -> float:
    vals = []
    for sample, pred in zip(samples, maps):
        mask = sample["mask"].astype(bool)
        pred_bin = pred > threshold
        inter = float((pred_bin & mask).sum())
        union = float((pred_bin | mask).sum())
        vals.append(inter / (union + np.spacing(1)))
    return float(np.mean(vals)) if vals else 0.0


def candidate_params() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [{"kind": "base"}]
    for base_lo in (0.36, 0.42, 0.48, 0.54):
        for support_hi in (0.25, 0.35, 0.45):
            for sal_hi in (0.10, 0.16, 0.24):
                for agree_hi in (0.17, 0.34, 0.50):
                    for factor in (0.35, 0.50, 0.70):
                        rows.append(
                            {
                                "kind": "suppress_low_saliency",
                                "base_lo": base_lo,
                                "support_hi": support_hi,
                                "sal_hi": sal_hi,
                                "agree_hi": agree_hi,
                                "factor": factor,
                            }
                        )
    for base_lo in (0.36, 0.44, 0.52):
        for bg_lo in (0.30, 0.45, 0.60):
            for core_hi in (0.15, 0.25, 0.35):
                for rpc_hi in (0.20, 0.35, 0.50):
                    for factor in (0.35, 0.50, 0.70):
                        rows.append(
                            {
                                "kind": "suppress_bg_only",
                                "base_lo": base_lo,
                                "bg_lo": bg_lo,
                                "core_hi": core_hi,
                                "rpc_hi": rpc_hi,
                                "factor": factor,
                            }
                        )
    for rpc_lo in (0.45, 0.55, 0.65, 0.75):
        for base_hi in (0.28, 0.35, 0.42, 0.50):
            for sal_lo in (0.10, 0.16, 0.24, 0.34):
                for bg_hi in (0.30, 0.45, 0.60, 1.01):
                    for mix in (0.75, 0.90, 1.0):
                        rows.append(
                            {
                                "kind": "rpc_rescue",
                                "rpc_lo": rpc_lo,
                                "base_hi": base_hi,
                                "sal_lo": sal_lo,
                                "bg_hi": bg_hi,
                                "mix": mix,
                            }
                        )
    for rescue_lo in (0.45, 0.55, 0.65):
        for base_hi in (0.28, 0.35, 0.42):
            for sal_lo in (0.10, 0.16, 0.24):
                for agree_lo in (0.17, 0.34):
                    for mix in (0.75, 0.90, 1.0):
                        rows.append(
                            {
                                "kind": "bgm_rpc_rescue",
                                "rescue_lo": rescue_lo,
                                "base_hi": base_hi,
                                "sal_lo": sal_lo,
                                "agree_lo": agree_lo,
                                "mix": mix,
                            }
                        )
    for rescue_lo in (0.50, 0.60):
        for base_hi in (0.32, 0.40):
            for sal_lo in (0.12, 0.20):
                for agree_lo in (0.17, 0.34):
                    for s_base_lo in (0.42, 0.50):
                        for support_hi in (0.30, 0.42):
                            for sal_hi in (0.12, 0.20):
                                for agree_hi in (0.17, 0.34):
                                    rows.append(
                                        {
                                            "kind": "rescue_then_suppress",
                                            "rescue_lo": rescue_lo,
                                            "base_hi": base_hi,
                                            "sal_lo": sal_lo,
                                            "agree_lo": agree_lo,
                                            "mix": 0.90,
                                            "s_base_lo": s_base_lo,
                                            "support_hi": support_hi,
                                            "sal_hi": sal_hi,
                                            "agree_hi": agree_hi,
                                            "factor": 0.50,
                                        }
                                    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-selected local gate over fixed BGM+RPCANet NUAA fusion.")
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
    parser.add_argument("--thresholds", default="0.46,0.48,0.50,0.52,0.54,0.56")
    parser.add_argument("--fit_topk", type=int, default=60)
    parser.add_argument("--candidate_stride", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    models = build_models([parse_model_spec(spec) for spec in args.model_spec], device)
    bgms = load_official_bgm_models([parse_official_bgm_spec(spec) for spec in args.official_bgm_spec], device)
    rpcanets = load_rpcanet_models([parse_rpcanet_spec(spec) for spec in args.rpcanet_spec], device)
    fit_datasets = [x.strip() for x in args.fit_datasets.split(",") if x.strip()]
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
    prepare_samples(fit_samples)
    prepare_samples(eval_samples)

    thresholds = parse_float_list(args.thresholds)
    candidates = candidate_params()[:: max(int(args.candidate_stride), 1)]
    fit_rows = []
    for idx, params in enumerate(candidates):
        maps = make_maps(fit_samples, params)
        for threshold in thresholds:
            fit_rows.append({"threshold": threshold, **params, "fit_nIoU": fast_niou(fit_samples, maps, threshold)})
        if (idx + 1) % 250 == 0:
            print(f"screened {idx+1}/{len(candidates)} candidates", flush=True)
    fit_rows.sort(key=lambda row: row["fit_nIoU"], reverse=True)

    eval_rows = []
    seen = set()
    for row in fit_rows[: max(int(args.fit_topk), 1)]:
        params = {k: v for k, v in row.items() if k not in {"threshold", "fit_nIoU"}}
        key = json.dumps({"threshold": row["threshold"], **params}, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        maps = make_maps(eval_samples, params)
        eval_rows.append(
            {
                "threshold": float(row["threshold"]),
                "fit_nIoU": row["fit_nIoU"],
                **params,
                **metric_from_maps(eval_samples, maps, float(row["threshold"])),
            }
        )
    eval_rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best = eval_rows[0]
    print(
        f"best kind={best['kind']} th={best['threshold']} fit_nIoU={best['fit_nIoU']*100:.2f} "
        f"mIoU={best['mIoU']*100:.2f} nIoU={best['nIoU']*100:.2f} "
        f"Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.2f}",
        flush=True,
    )
    Path(args.output).write_text(
        json.dumps(
            {
                "dataset": args.dataset_name,
                "fit_datasets": fit_datasets,
                "base_weights": BASE_WEIGHTS,
                "fit_samples": len(fit_samples),
                "eval_samples": len(eval_samples),
                "candidate_count": len(candidates),
                "best": best,
                "eval_rows": eval_rows,
                "fit_rows_top": fit_rows[:100],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
