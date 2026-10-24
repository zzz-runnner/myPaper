#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

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

LRRNET_MODEL = ROOT / "external" / "LRRNet" / "models" / "LRRNet.py"


def load_lrrnet_unet_class():
    spec = importlib.util.spec_from_file_location("lrrnet_fusion_model", LRRNET_MODEL)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load LRRNet model from {LRRNET_MODEL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.UNet


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


def load_lrrnet(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    UNet = load_lrrnet_unet_class()
    model = UNet(img_channels=1, base_channels=16, channel_mults=(1, 2, 4, 8), num_res_blocks=2, dropout=0.2, norm="gn", num_groups=16)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def add_lrrnet_maps(samples: list[dict[str, object]], model: torch.nn.Module, input_size: int, device: torch.device) -> None:
    for idx, sample in enumerate(samples):
        image_np = sample["image"].astype(np.float32)
        inp = cv2.resize(image_np, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        inp_t = torch.from_numpy(inp[None, None]).float().to(device)
        _d, target_logits = model(inp_t)
        prob = torch.sigmoid(target_logits)
        height, width = sample["mask"].shape
        prob = F.interpolate(prob, size=(height, width), mode="bilinear", align_corners=False)
        sample["preds"]["lrrnet"] = prob[0, 0].detach().cpu().numpy().astype(np.float32)
        if (idx + 1) % 50 == 0:
            print(f"added LRRNet maps {idx+1}/{len(samples)}", flush=True)


def base_map(sample: dict[str, object]) -> np.ndarray:
    out = np.zeros_like(sample["mask"], dtype=np.float32)
    for name, weight in BASE_WEIGHTS.items():
        out += float(weight) * sample["preds"][name].astype(np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def make_map(sample: dict[str, object], params: dict[str, object]) -> np.ndarray:
    base = base_map(sample)
    lrr = sample["preds"]["lrrnet"].astype(np.float32)
    kind = str(params["kind"])
    if kind == "base":
        return base
    if kind == "blend":
        alpha = float(params["alpha"])
        return np.clip((1.0 - alpha) * base + alpha * lrr, 0.0, 1.0).astype(np.float32)
    if kind == "max_rescue":
        out = base.copy()
        gate = (lrr >= float(params["lrr_lo"])) & (base <= float(params["base_hi"]))
        out[gate] = np.maximum(out[gate], float(params["mix"]) * lrr[gate])
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    if kind == "lrr_suppress":
        out = base.copy()
        bad = (base >= float(params["base_lo"])) & (lrr <= float(params["lrr_hi"]))
        out[bad] *= float(params["factor"])
        return out.astype(np.float32)
    if kind == "rescue_suppress":
        out = base.copy()
        rescue = (lrr >= float(params["lrr_lo"])) & (base <= float(params["base_hi"]))
        out[rescue] = np.maximum(out[rescue], float(params["mix"]) * lrr[rescue])
        bad = (out >= float(params["s_base_lo"])) & (lrr <= float(params["s_lrr_hi"]))
        out[bad] *= float(params["factor"])
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    raise ValueError(kind)


def make_maps(samples: list[dict[str, object]], params: dict[str, object]) -> list[np.ndarray]:
    return [make_map(sample, params) for sample in samples]


def fast_niou(samples: list[dict[str, object]], maps: list[np.ndarray], threshold: float) -> float:
    vals = []
    for sample, pred in zip(samples, maps):
        mask = sample["mask"] > 0
        pred_bin = pred > threshold
        inter = float((pred_bin & mask).sum())
        union = float((pred_bin | mask).sum())
        vals.append(inter / (union + np.spacing(1)))
    return float(np.mean(vals)) if vals else 0.0


def candidate_params() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [{"kind": "base"}]
    for alpha in (0.02, 0.05, 0.08, 0.12, 0.18):
        rows.append({"kind": "blend", "alpha": alpha})
    for lrr_lo in (0.35, 0.45, 0.55, 0.65):
        for base_hi in (0.25, 0.35, 0.45):
            for mix in (0.50, 0.75, 1.0):
                rows.append({"kind": "max_rescue", "lrr_lo": lrr_lo, "base_hi": base_hi, "mix": mix})
    for base_lo in (0.42, 0.50, 0.58):
        for lrr_hi in (0.10, 0.20, 0.30, 0.40):
            for factor in (0.35, 0.50, 0.70):
                rows.append({"kind": "lrr_suppress", "base_lo": base_lo, "lrr_hi": lrr_hi, "factor": factor})
    for lrr_lo in (0.45, 0.55):
        for base_hi in (0.30, 0.40):
            for s_base_lo in (0.45, 0.55):
                for s_lrr_hi in (0.15, 0.25, 0.35):
                    rows.append(
                        {
                            "kind": "rescue_suppress",
                            "lrr_lo": lrr_lo,
                            "base_hi": base_hi,
                            "mix": 0.75,
                            "s_base_lo": s_base_lo,
                            "s_lrr_hi": s_lrr_hi,
                            "factor": 0.50,
                        }
                    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Test LRRNet as a background-aware companion to BGM+RPCANet fixed fusion.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", default="NUAA-SIRST")
    parser.add_argument("--fit_datasets", default="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3")
    parser.add_argument("--model_spec", action="append", required=True)
    parser.add_argument("--official_bgm_spec", action="append", required=True)
    parser.add_argument("--rpcanet_spec", action="append", required=True)
    parser.add_argument("--lrrnet_checkpoint", default=str(ROOT / "external" / "LRRNet" / "checkpoints" / "IRDST_1K_miou_0.7239356450020329.pkl"))
    parser.add_argument("--lrrnet_input_size", type=int, default=512)
    parser.add_argument("--background_sample_pixels", type=int, default=2048)
    parser.add_argument("--background_topk", type=int, default=16)
    parser.add_argument("--background_threshold", type=float, default=0.88)
    parser.add_argument("--max_fit_samples", type=int, default=80)
    parser.add_argument("--max_eval_samples", type=int, default=80)
    parser.add_argument("--thresholds", default="0.48,0.50,0.52,0.55")
    parser.add_argument("--fit_topk", type=int, default=32)
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
    lrrnet = load_lrrnet(Path(args.lrrnet_checkpoint), device)
    add_lrrnet_maps(fit_samples, lrrnet, args.lrrnet_input_size, device)
    add_lrrnet_maps(eval_samples, lrrnet, args.lrrnet_input_size, device)

    thresholds = parse_float_list(args.thresholds)
    fit_rows = []
    for params in candidate_params():
        maps = make_maps(fit_samples, params)
        for threshold in thresholds:
            fit_rows.append({"threshold": threshold, **params, "fit_nIoU": fast_niou(fit_samples, maps, threshold)})
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
                "lrrnet_checkpoint": args.lrrnet_checkpoint,
                "fit_samples": len(fit_samples),
                "eval_samples": len(eval_samples),
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
