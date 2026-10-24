#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))

from eval_component_fusion_learner import (  # noqa: E402
    FullSplitSet,
    build_models,
    load_official_pconv_models,
    metric_from_maps,
    parse_model_spec,
    parse_official_pconv_spec,
)
from eval_pconv_candidate_gate import pconv_prob_map  # noqa: E402


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def robust01(x: np.ndarray, low: float = 1.0, high: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(x, [low, high])
    if hi <= lo + 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def local_z_score(image: np.ndarray, k: int = 9) -> np.ndarray:
    src = robust01(image.astype(np.float32), 0.5, 99.5)
    mean = cv2.blur(src, (k, k))
    sq_mean = cv2.blur(src * src, (k, k))
    std = np.sqrt(np.maximum(sq_mean - mean * mean, 0.0))
    return ((src - mean) / (std + 1e-3)).astype(np.float32)


def contrast_score(image: np.ndarray) -> np.ndarray:
    src = robust01(image.astype(np.float32), 0.5, 99.5)
    maps = []
    for k in (5, 9, 13):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        maps.append(robust01(cv2.morphologyEx(src, cv2.MORPH_TOPHAT, kernel), 50.0, 99.8))
        z = local_z_score(src, k)
        maps.append(robust01(z, 50.0, 99.8))
    for sigma in (1.0, 1.6):
        blur = cv2.GaussianBlur(src, (0, 0), sigmaX=sigma, sigmaY=sigma)
        maps.append(robust01(cv2.Laplacian(blur, cv2.CV_32F, ksize=3), 50.0, 99.8))
    return np.stack(maps, axis=0).max(axis=0).astype(np.float32)


@torch.no_grad()
def collect_samples(
    models,
    officials,
    dataset_dir: Path,
    dataset_name: str,
    split: str,
    max_samples: int,
    device: torch.device,
) -> list[dict[str, object]]:
    dataset = FullSplitSet(dataset_dir, dataset_name, dataset_name, split, max_samples=max_samples)
    loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
    samples: list[dict[str, object]] = []
    for idx, (image, mask, size, name) in enumerate(loader):
        image = image.to(device)
        height = int(size[0])
        width = int(size[1])
        teacher_maps = []
        pred_dict = {}
        for model_name, model in models:
            pred = model(image, epoch=10**9)
            if isinstance(pred, (tuple, list)):
                pred = pred[-1]
            arr = pred[0, 0, :height, :width].detach().cpu().numpy().astype(np.float32)
            teacher_maps.append(arr)
            pred_dict[model_name] = arr
        teacher_max = np.stack(teacher_maps, axis=0).max(axis=0)
        pconv_maps = {}
        for official in officials:
            pconv_maps[str(official["name"])] = pconv_prob_map(
                official, dataset_dir, dataset_name, name[0], image, height, width, device
            )
        image_np = image[0, 0, :height, :width].detach().cpu().numpy().astype(np.float32)
        mask_np = mask[0, 0, :height, :width].numpy().astype(np.float32)
        samples.append(
            {
                "name": name[0],
                "image": image_np,
                "mask": mask_np,
                "size": [height, width],
                "teacher": teacher_max,
                "preds": pred_dict,
                "pconv": pconv_maps,
                "contrast": contrast_score(image_np),
                "local_z": local_z_score(image_np),
            }
        )
        if (idx + 1) % 100 == 0:
            print(f"collected {dataset_name} {split} {idx + 1}/{len(dataset)}", flush=True)
    return samples


def collect_fit_samples(models, officials, dataset_dir: Path, datasets: list[str], max_samples: int, device: torch.device):
    out = []
    for dataset in datasets:
        out.extend(collect_samples(models, officials, dataset_dir, dataset, "train", max_samples, device))
    return out


def metric(samples: list[dict[str, object]], maps: list[np.ndarray], threshold: float) -> dict[str, float]:
    wrapped = [{"mask": sample["mask"], "size": sample["size"]} for sample in samples]
    return metric_from_maps(wrapped, maps, threshold)


def fast_niou(samples: list[dict[str, object]], maps: list[np.ndarray], threshold: float) -> float:
    vals = []
    for sample, pred in zip(samples, maps):
        mask = sample["mask"] > 0
        pred_bin = pred > threshold
        inter = float((pred_bin & mask).sum())
        union = float((pred_bin | mask).sum())
        vals.append(inter / (union + np.spacing(1)))
    return float(np.mean(vals)) if vals else 0.0


def make_maps(samples: list[dict[str, object]], params: dict[str, object]) -> list[np.ndarray]:
    kind = str(params["kind"])
    out = []
    for sample in samples:
        teacher = sample["teacher"]
        contrast = sample["contrast"]
        z = sample["local_z"]
        result = teacher.copy()
        if kind == "teacher":
            result = teacher.copy()
        elif kind == "pconv_max":
            p = sample["pconv"][str(params["pconv_name"])]
            result = np.maximum(teacher, p)
        elif kind == "pconv_rescue":
            p = sample["pconv"][str(params["pconv_name"])]
            gate = (p >= float(params["p_th"])) & (z >= float(params["z_th"])) & (teacher <= float(params["teacher_hi"]))
            result = teacher.copy()
            result[gate] = np.maximum(result[gate], p[gate])
        elif kind == "contrast_rescue":
            gate = (contrast >= float(params["c_th"])) & (z >= float(params["z_th"])) & (teacher <= float(params["teacher_hi"]))
            result = teacher.copy()
            result[gate] = np.maximum(result[gate], contrast[gate] * float(params["scale"]))
        elif kind == "pconv_contrast_rescue":
            p = sample["pconv"][str(params["pconv_name"])]
            rescue = np.maximum(p, contrast * float(params["scale"]))
            gate = (
                (rescue >= float(params["r_th"]))
                & (z >= float(params["z_th"]))
                & (teacher <= float(params["teacher_hi"]))
            )
            result = teacher.copy()
            result[gate] = np.maximum(result[gate], rescue[gate])
        elif kind == "suppress_low_contrast":
            result = teacher.copy()
            bad = (teacher >= float(params["teacher_lo"])) & (contrast <= float(params["c_low"])) & (z <= float(params["z_low"]))
            result[bad] *= float(params["factor"])
        else:
            raise ValueError(kind)
        out.append(result.astype(np.float32))
    return out


def candidate_params(pconv_names: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = [{"kind": "teacher"}]
    for name in pconv_names:
        rows.append({"kind": "pconv_max", "pconv_name": name})
        for p_th in (0.45, 0.55, 0.65):
            for z_th in (-0.5, 0.0, 0.5):
                for teacher_hi in (0.15, 0.30, 0.50, 1.01):
                    rows.append({"kind": "pconv_rescue", "pconv_name": name, "p_th": p_th, "z_th": z_th, "teacher_hi": teacher_hi})
        for r_th in (0.55, 0.65, 0.75):
            for z_th in (-0.5, 0.0, 0.5):
                for teacher_hi in (0.15, 0.30, 0.50):
                    rows.append(
                        {
                            "kind": "pconv_contrast_rescue",
                            "pconv_name": name,
                            "r_th": r_th,
                            "z_th": z_th,
                            "teacher_hi": teacher_hi,
                            "scale": 0.85,
                        }
                    )
    for c_th in (0.70, 0.80, 0.90):
        for z_th in (0.0, 0.5, 1.0):
            for teacher_hi in (0.15, 0.30, 0.50):
                rows.append({"kind": "contrast_rescue", "c_th": c_th, "z_th": z_th, "teacher_hi": teacher_hi, "scale": 0.85})
    for teacher_lo in (0.40, 0.50):
        for c_low in (0.20, 0.35):
            for z_low in (-0.5, 0.0):
                rows.append({"kind": "suppress_low_contrast", "teacher_lo": teacher_lo, "c_low": c_low, "z_low": z_low, "factor": 0.35})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast vectorized pixel-rule fusion diagnostic.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--fit_datasets", default="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3")
    parser.add_argument("--model_spec", action="append", required=True)
    parser.add_argument("--official_pconv_mshnet_spec", action="append", default=[])
    parser.add_argument("--output_thresholds", default="0.45,0.50")
    parser.add_argument("--max_fit_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    models = build_models([parse_model_spec(spec) for spec in args.model_spec], device)
    pconv_specs = [parse_official_pconv_spec(spec) for spec in args.official_pconv_mshnet_spec]
    officials = load_official_pconv_models(pconv_specs, device)
    pconv_names = [str(spec["name"]) for spec in pconv_specs]
    fit_datasets = [x.strip() for x in args.fit_datasets.split(",") if x.strip()]
    out_ths = parse_float_list(args.output_thresholds)
    fit_samples = collect_fit_samples(models, officials, Path(args.dataset_dir), fit_datasets, args.max_fit_samples, device)
    eval_samples = collect_samples(models, officials, Path(args.dataset_dir), args.dataset_name, "test", args.max_eval_samples, device)

    rows = []
    for params in candidate_params(pconv_names):
        fit_maps = make_maps(fit_samples, params)
        for out_th in out_ths:
            rows.append({"stage": "fit", "output_threshold": out_th, **params, "fast_nIoU": fast_niou(fit_samples, fit_maps, out_th)})
    fit_rank = sorted(rows, key=lambda row: row["fast_nIoU"], reverse=True)
    eval_rows = []
    seen = set()
    for row in fit_rank[:30]:
        key = json.dumps({k: v for k, v in row.items() if k not in {"stage", "fast_nIoU"}}, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        params = {k: v for k, v in row.items() if k not in {"stage", "output_threshold", "fast_nIoU"}}
        maps = make_maps(eval_samples, params)
        out_th = float(row["output_threshold"])
        eval_rows.append({"stage": "eval", "fit_fast_nIoU": row["fast_nIoU"], "output_threshold": out_th, **params, **metric(eval_samples, maps, out_th)})
    eval_rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best = eval_rows[0]
    print(
        f"best kind={best['kind']} out={best['output_threshold']} mIoU={best['mIoU']*100:.2f} "
        f"Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.2f}",
        flush=True,
    )
    for row in eval_rows[:20]:
        print(
            f"{row['kind']} out={row['output_threshold']} fit={row['fit_fast_nIoU']*100:.2f} "
            f"mIoU={row['mIoU']*100:.2f} Pd={row['Pd']*100:.2f} Fa={row['Fa']*1e6:.2f}",
            flush=True,
        )
    Path(args.output).write_text(
        json.dumps(
            {
                "dataset": args.dataset_name,
                "fit_datasets": fit_datasets,
                "official_pconv_mshnet_specs": args.official_pconv_mshnet_spec,
                "best_eval": best,
                "eval_rows": eval_rows,
                "fit_rows_top": fit_rank[:50],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
