#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))

from eval_component_fusion_learner import metric_from_maps  # noqa: E402
from generate_sam_refined_pseudo_labels import (  # noqa: E402
    find_image_path,
    gray_to_sam_rgb,
    read_gray,
    refine_with_sam,
)
from inmemory_cached_weight_search import load_cache_dense  # noqa: E402
from segment_anything import SamPredictor, sam_model_registry  # noqa: E402


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def cached_maps(stack: np.ndarray, names: list[str], mode: str) -> np.ndarray:
    if mode == "avg":
        return stack.mean(axis=0).astype(np.float32)
    if mode == "max":
        return stack.max(axis=0).astype(np.float32)
    if mode == "min":
        return stack.min(axis=0).astype(np.float32)
    if mode in names:
        return stack[names.index(mode)].astype(np.float32)
    raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate test-time SAM refinement from cached predictions.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--mode", default="avg")
    parser.add_argument("--thresholds", default="0.35,0.40,0.42,0.45,0.50")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sam_checkpoint", default=str(ROOT / "pretrained" / "sam" / "sam_vit_b_01ec64.pth"))
    parser.add_argument("--sam_model_type", default="vit_b")
    parser.add_argument("--teacher_thresholds", default="0.30,0.35,0.40,0.45")
    parser.add_argument("--fusions", default="soft_rim,max")
    parser.add_argument("--max_prompts", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    samples, names, preds, _masks = load_cache_dense(Path(args.cache))
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
        preds = preds[: args.max_samples]
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint).to(device)
    sam.eval()
    predictor = SamPredictor(sam)

    base_maps = [cached_maps(stack, names, args.mode) for stack in preds]
    rows: list[dict[str, object]] = []
    for th in parse_float_list(args.thresholds):
        rows.append({"kind": "cached_baseline", "mode": args.mode, "threshold": th, **metric_from_maps(samples, base_maps, th)})

    refine_common = {
        "fallback_peak_threshold": 0.22,
        "fallback_peak_ratio": 0.70,
        "min_component_area": 1,
        "max_component_area": 600,
        "max_component_area_ratio": 0.025,
        "max_prompts": args.max_prompts,
        "box_pad": 8,
        "negative_points": 4,
        "sam_guard_dilate": 13,
        "min_sam_area": 2,
        "max_sam_area_mult": 14.0,
        "max_sam_area_ratio": 0.020,
        "min_component_cover": 0.10,
        "min_mask_teacher_mean": 0.06,
        "sam_conf_base": 0.35,
        "sam_conf_score_weight": 0.22,
        "sam_conf_cover_weight": 0.18,
        "sam_conf_teacher_weight": 0.20,
        "sam_conf_min": 0.55,
        "sam_conf_max": 0.92,
        "teacher_keep": 0.92,
        "teacher_core_threshold": 0.50,
        "sam_rim_value": 0.18,
        "sam_rim_teacher_scale": 0.55,
        "sam_rim_min": 0.20,
        "sam_rim_max": 0.42,
    }

    for fusion in [x.strip() for x in args.fusions.split(",") if x.strip()]:
        for teacher_threshold in parse_float_list(args.teacher_thresholds):
            refined_maps = []
            accepted = 0
            components = 0
            for sample, teacher_prob in zip(samples, base_maps):
                image_path = find_image_path(Path(args.dataset_dir) / args.dataset_name, str(sample["name"]))
                gray = read_gray(image_path)
                ns = SimpleNamespace(**refine_common, fusion=fusion, teacher_threshold=teacher_threshold)
                refined, meta = refine_with_sam(predictor, gray_to_sam_rgb(gray), teacher_prob, ns)
                refined_maps.append(refined.astype(np.float32))
                accepted += int(meta["accepted"])
                components += int(meta["components"])
            for th in parse_float_list(args.thresholds):
                rows.append(
                    {
                        "kind": "sam_refined",
                        "mode": args.mode,
                        "fusion": fusion,
                        "teacher_threshold": teacher_threshold,
                        "threshold": th,
                        "accepted": accepted,
                        "components": components,
                        **metric_from_maps(samples, refined_maps, th),
                    }
                )

    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best = rows[0]
    print(
        f"best kind={best['kind']} fusion={best.get('fusion','')} teacher_th={best.get('teacher_threshold','')} "
        f"th={best['threshold']} mIoU={best['mIoU']*100:.2f} Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.2f}",
        flush=True,
    )
    Path(args.output).write_text(
        json.dumps(
            {
                "cache": args.cache,
                "dataset": args.dataset_name,
                "split": args.split,
                "mode": args.mode,
                "models": names,
                "max_samples": args.max_samples,
                "best": best,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
