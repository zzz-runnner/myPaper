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

from eval_component_fusion_learner import metric_from_maps  # noqa: E402
from generate_sam_refined_pseudo_labels import (  # noqa: E402
    find_image_path,
    gray_to_sam_rgb,
    read_gray,
    refine_with_sam,
)
from segment_anything import SamPredictor, sam_model_registry  # noqa: E402


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SAM refinement for an exported probability-map npz.")
    parser.add_argument("--maps_npz", required=True)
    parser.add_argument("--map_key", default="maps_no_tta")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--thresholds", default="0.45,0.47,0.49,0.51")
    parser.add_argument("--teacher_thresholds", default="0.46,0.48,0.50")
    parser.add_argument("--fusions", default="soft_rim,max")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_prompts", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sam_checkpoint", default=str(ROOT / "pretrained" / "sam" / "sam_vit_b_01ec64.pth"))
    parser.add_argument("--sam_model_type", default="vit_b")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = np.load(args.maps_npz, allow_pickle=True)
    names = [str(x) for x in list(data["names"])]
    maps = [np.asarray(x, dtype=np.float32) for x in list(data[args.map_key])]
    masks = [np.asarray(x, dtype=np.float32) for x in list(data["masks"])]
    sizes = [list(map(int, x)) for x in list(data["sizes"])]
    if args.max_samples > 0:
        names = names[: args.max_samples]
        maps = maps[: args.max_samples]
        masks = masks[: args.max_samples]
        sizes = sizes[: args.max_samples]
    samples = [{"name": name, "mask": mask, "size": size} for name, mask, size in zip(names, masks, sizes)]

    rows: list[dict[str, object]] = []
    for th in parse_float_list(args.thresholds):
        rows.append({"kind": "baseline", "threshold": th, **metric_from_maps(samples, maps, th)})

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint).to(device)
    sam.eval()
    predictor = SamPredictor(sam)

    refine_common = {
        "fallback_peak_threshold": 0.22,
        "fallback_peak_ratio": 0.70,
        "min_component_area": 1,
        "max_component_area": 500,
        "max_component_area_ratio": 0.020,
        "max_prompts": args.max_prompts,
        "box_pad": 6,
        "negative_points": 4,
        "sam_guard_dilate": 9,
        "min_sam_area": 2,
        "max_sam_area_mult": 8.0,
        "max_sam_area_ratio": 0.012,
        "min_component_cover": 0.15,
        "min_mask_teacher_mean": 0.12,
        "sam_conf_base": 0.45,
        "sam_conf_score_weight": 0.20,
        "sam_conf_cover_weight": 0.20,
        "sam_conf_teacher_weight": 0.22,
        "sam_conf_min": 0.60,
        "sam_conf_max": 0.88,
        "teacher_keep": 0.95,
        "teacher_core_threshold": 0.55,
        "sam_rim_value": 0.12,
        "sam_rim_teacher_scale": 0.45,
        "sam_rim_min": 0.16,
        "sam_rim_max": 0.34,
    }

    for fusion in [x.strip() for x in args.fusions.split(",") if x.strip()]:
        for teacher_threshold in parse_float_list(args.teacher_thresholds):
            refined_maps = []
            accepted = 0
            components = 0
            for name, teacher_prob in zip(names, maps):
                image_path = find_image_path(Path(args.dataset_dir) / args.dataset_name, name)
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
        f"th={best['threshold']} mIoU={best['mIoU']*100:.4f} nIoU={best['nIoU']*100:.4f} "
        f"Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.4f}",
        flush=True,
    )
    Path(args.output).write_text(json.dumps({"args": vars(args), "best": best, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
