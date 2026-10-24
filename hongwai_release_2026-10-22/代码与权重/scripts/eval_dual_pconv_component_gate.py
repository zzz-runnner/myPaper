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


def parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def fast_niou(samples: list[dict[str, object]], maps: list[np.ndarray], threshold: float) -> float:
    vals = []
    for sample, pred in zip(samples, maps):
        mask = sample["mask"] > 0
        pred_bin = pred > threshold
        inter = float((pred_bin & mask).sum())
        union = float((pred_bin | mask).sum())
        vals.append(inter / (union + np.spacing(1)))
    return float(np.mean(vals)) if vals else 0.0


def local_z_map(image: np.ndarray) -> np.ndarray:
    src = image.astype(np.float32)
    mean = cv2.blur(src, (9, 9))
    sq_mean = cv2.blur(src * src, (9, 9))
    std = np.sqrt(np.maximum(sq_mean - mean * mean, 0.0))
    return ((src - mean) / (std + 1e-6)).astype(np.float32)


def local_delta(image: np.ndarray, comp: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    y0, x0, y1, x1 = bbox
    pad = 5
    yy0 = max(0, y0 - pad)
    xx0 = max(0, x0 - pad)
    yy1 = min(image.shape[0], y1 + pad)
    xx1 = min(image.shape[1], x1 + pad)
    roi = image[yy0:yy1, xx0:xx1]
    ring = np.ones_like(roi, dtype=bool)
    ring[y0 - yy0 : y1 - yy0, x0 - xx0 : x1 - xx0] = False
    fg = image[y0:y1, x0:x1][comp[y0:y1, x0:x1]]
    bg = roi[ring]
    if fg.size == 0 or bg.size == 0:
        return 0.0
    return float(fg.mean() - bg.mean())


def base_map(sample: dict[str, object], teacher_names: list[str], pconv_names: list[str], mode: str) -> np.ndarray:
    preds = sample["preds"]
    if mode == "teacher_max":
        return np.stack([preds[name] for name in teacher_names], axis=0).max(axis=0).astype(np.float32)
    if mode == "avg_all":
        return np.stack([preds[name] for name in [*teacher_names, *pconv_names]], axis=0).mean(axis=0).astype(np.float32)
    if mode == "max_all":
        return np.stack([preds[name] for name in [*teacher_names, *pconv_names]], axis=0).max(axis=0).astype(np.float32)
    raise ValueError(mode)


def accepted_mask(sample: dict[str, object], teacher_names: list[str], pconv_names: list[str], params: dict[str, object]) -> np.ndarray:
    preds = sample["preds"]
    image = sample["image"]
    teacher = np.stack([preds[name] for name in teacher_names], axis=0).max(axis=0)
    pconv = np.stack([preds[name] for name in pconv_names], axis=0).max(axis=0)
    zmap = local_z_map(image)
    seed = pconv >= float(params["seed"])
    num_labels, label_img, stats, _centroids = cv2.connectedComponentsWithStats(seed.astype(np.uint8), 8)
    keep = np.zeros_like(seed, dtype=bool)
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        if area < int(params["min_area"]) or area > int(params["max_area"]):
            continue
        if max(width, height) > int(params["max_side"]):
            continue
        fill = area / float(max(width * height, 1))
        if fill < float(params["min_fill"]):
            continue
        x0 = int(stats[label_id, cv2.CC_STAT_LEFT])
        y0 = int(stats[label_id, cv2.CC_STAT_TOP])
        comp = label_img == label_id
        y1 = y0 + height
        x1 = x0 + width
        if float(zmap[comp].mean()) < float(params["min_z"]):
            continue
        if local_delta(image, comp, (y0, x0, y1, x1)) < float(params["min_delta"]):
            continue
        teacher_mean = float(teacher[comp].mean())
        if teacher_mean > float(params["max_teacher_mean"]):
            continue
        keep[comp] = True
    return keep


def render(samples: list[dict[str, object]], teacher_names: list[str], pconv_names: list[str], params: dict[str, object]) -> list[np.ndarray]:
    maps = []
    for sample in samples:
        preds = sample["preds"]
        pconv = np.stack([preds[name] for name in pconv_names], axis=0).max(axis=0)
        out = base_map(sample, teacher_names, pconv_names, str(params["base_mode"]))
        keep = accepted_mask(sample, teacher_names, pconv_names, params)
        if str(params["fill"]) == "binary":
            out[keep] = 1.0
        elif str(params["fill"]) == "prob":
            out[keep] = np.maximum(out[keep], pconv[keep])
        elif str(params["fill"]) == "boost":
            out[keep] = np.maximum(out[keep], np.clip(pconv[keep] * float(params["boost"]), 0.0, 1.0))
        else:
            raise ValueError(str(params["fill"]))
        maps.append(out.astype(np.float32))
    return maps


def candidate_params(args: argparse.Namespace) -> list[dict[str, object]]:
    rows = []
    for base_mode in args.base_modes.split(","):
        for seed in parse_float_list(args.seed_thresholds):
            for max_area in parse_int_list(args.max_areas):
                for max_side in parse_int_list(args.max_sides):
                    for min_fill in parse_float_list(args.min_fills):
                        for min_z in parse_float_list(args.min_zs):
                            for min_delta in parse_float_list(args.min_deltas):
                                for max_teacher_mean in parse_float_list(args.max_teacher_means):
                                    for fill in args.fills.split(","):
                                        rows.append(
                                            {
                                                "base_mode": base_mode,
                                                "seed": seed,
                                                "min_area": 1,
                                                "max_area": max_area,
                                                "max_side": max_side,
                                                "min_fill": min_fill,
                                                "min_z": min_z,
                                                "min_delta": min_delta,
                                                "max_teacher_mean": max_teacher_mean,
                                                "fill": fill,
                                                "boost": args.boost,
                                            }
                                        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-PConv local component gate with train fast-nIoU selection.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--fit_datasets", default="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3")
    parser.add_argument("--model_spec", action="append", required=True)
    parser.add_argument("--official_pconv_mshnet_spec", action="append", required=True)
    parser.add_argument("--seed_thresholds", default="0.35,0.45,0.55,0.65")
    parser.add_argument("--max_areas", default="16,32,64,128")
    parser.add_argument("--max_sides", default="8,12,18,28")
    parser.add_argument("--min_fills", default="0.10,0.20,0.35")
    parser.add_argument("--min_zs", default="-1.0,-0.25,0.25,0.75")
    parser.add_argument("--min_deltas", default="-1.0,0.0,0.01")
    parser.add_argument("--max_teacher_means", default="1.1,0.50,0.30,0.15")
    parser.add_argument("--base_modes", default="avg_all,teacher_max")
    parser.add_argument("--fills", default="prob,binary")
    parser.add_argument("--boost", type=float, default=1.15)
    parser.add_argument("--output_thresholds", default="0.40,0.45,0.50,0.55")
    parser.add_argument("--topk_eval", type=int, default=12)
    parser.add_argument("--fast_only", action="store_true")
    parser.add_argument("--max_fit_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    specs = [parse_model_spec(spec) for spec in args.model_spec]
    teacher_names = [spec[0] for spec in specs]
    models = build_models(specs, device)
    pconv_specs = [parse_official_pconv_spec(spec) for spec in args.official_pconv_mshnet_spec]
    pconv_models = load_official_pconv_models(pconv_specs, device)
    pconv_names = [str(spec["name"]) for spec in pconv_specs]
    fit_datasets = [x.strip() for x in args.fit_datasets.split(",") if x.strip()]

    fit_samples = collect_fit_samples_multi(models, pconv_models, Path(args.dataset_dir), fit_datasets, "train", args.max_fit_samples, device)
    eval_samples = collect_samples(models, pconv_models, Path(args.dataset_dir), args.dataset_name, "test", args.max_eval_samples, device)
    out_ths = parse_float_list(args.output_thresholds)

    rows = []
    for mode in ("teacher_max", "avg_all", "max_all"):
        maps = [base_map(sample, teacher_names, pconv_names, mode) for sample in eval_samples]
        for out_th in out_ths:
            rows.append({"kind": "baseline", "mode": mode, "output_threshold": out_th, **metric_from_maps(eval_samples, maps, out_th)})

    fit_rank = []
    for params in candidate_params(args):
        fit_maps = render(fit_samples, teacher_names, pconv_names, params)
        for out_th in out_ths:
            fit_rank.append({"params": params, "output_threshold": out_th, "fit_fast_nIoU": fast_niou(fit_samples, fit_maps, out_th)})
    fit_rank.sort(key=lambda row: row["fit_fast_nIoU"], reverse=True)

    if args.fast_only:
        best = fit_rank[0]
        print(
            f"best-fast out={best['output_threshold']} fit_fast_nIoU={best['fit_fast_nIoU']*100:.2f} "
            f"params={best['params']}",
            flush=True,
        )
        Path(args.output).write_text(
            json.dumps(
                {
                    "dataset": args.dataset_name,
                    "fit_datasets": fit_datasets,
                    "teacher_names": teacher_names,
                    "pconv_names": pconv_names,
                    "metric_mode": "fit_fast_nIoU_only",
                    "best": best,
                    "fit_rank": fit_rank[:100],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return

    seen = set()
    for row in fit_rank[: max(int(args.topk_eval), 1)]:
        key = json.dumps({"params": row["params"], "output_threshold": row["output_threshold"]}, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        maps = render(eval_samples, teacher_names, pconv_names, row["params"])
        rows.append(
            {
                "kind": "dual_pconv_gate",
                "fit_fast_nIoU": row["fit_fast_nIoU"],
                "output_threshold": row["output_threshold"],
                **row["params"],
                **metric_from_maps(eval_samples, maps, row["output_threshold"]),
            }
        )

    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best = rows[0]
    print(
        f"best kind={best['kind']} mode={best.get('mode', best.get('base_mode', ''))} out={best['output_threshold']} "
        f"mIoU={best['mIoU']*100:.2f} Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.2f}",
        flush=True,
    )
    for row in rows[:20]:
        print(
            f"{row['kind']} mode={row.get('mode', row.get('base_mode', ''))} out={row['output_threshold']} "
            f"mIoU={row['mIoU']*100:.2f} Pd={row['Pd']*100:.2f} Fa={row['Fa']*1e6:.2f}",
            flush=True,
        )
    Path(args.output).write_text(
        json.dumps(
            {
                "dataset": args.dataset_name,
                "fit_datasets": fit_datasets,
                "teacher_names": teacher_names,
                "pconv_names": pconv_names,
                "best": best,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
