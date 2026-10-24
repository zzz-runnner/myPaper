#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage import measure
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_hybrid_methods as th  # noqa: E402


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


@torch.no_grad()
def collect_samples(args: argparse.Namespace) -> list[dict[str, np.ndarray]]:
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model = th.make_model(args.model, warm_epoch=1, base_checkpoint=args.base_checkpoint).to(device)
    if isinstance(model, th.SCTRefiner):
        model.initialize_refiner()
    else:
        model.apply(th.weights_init_kaiming)
    if args.checkpoint:
        th.load_model_checkpoint(model, Path(args.checkpoint))
    model.eval()

    dataset = th.TestSetLoader(str(args.dataset_dir), args.dataset_name, args.dataset_name)
    loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
    samples: list[dict[str, np.ndarray]] = []
    for image, mask, size, _ in loader:
        image = image.to(device)
        pred = model(image, epoch=10**9)
        if isinstance(pred, (tuple, list)):
            pred = pred[-1]
        height = int(size[0])
        width = int(size[1])
        pred = pred[:, :, :height, :width]
        image = image[:, :, :height, :width]
        mask = mask[:, :, :height, :width]
        luma = th.image_to_luma(image)
        local = F.avg_pool2d(luma, 9, stride=1, padding=4)
        contrast = F.relu(luma - local)
        contrast = contrast / (contrast.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        lap = F.conv2d(
            luma,
            luma.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3),
            padding=1,
        ).abs()
        lap = lap / (lap.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        samples.append(
            {
                "prob": pred[0, 0].detach().cpu().numpy().astype(np.float32),
                "mask": (mask[0, 0].detach().cpu().numpy() > 0).astype(np.uint8),
                "contrast": contrast[0, 0].detach().cpu().numpy().astype(np.float32),
                "lap": lap[0, 0].detach().cpu().numpy().astype(np.float32),
            }
        )
    return samples


def filter_components(
    prob: np.ndarray,
    pred: np.ndarray,
    contrast: np.ndarray,
    lap: np.ndarray,
    min_area: int,
    max_area: int,
    min_contrast: float,
    min_lap: float,
) -> np.ndarray:
    if min_area <= 1 and max_area <= 0 and min_contrast <= 0 and min_lap <= 0:
        return pred
    labels = measure.label(pred.astype(np.uint8), connectivity=2)
    out = np.zeros_like(pred, dtype=bool)
    for region in measure.regionprops(labels):
        if region.area < min_area:
            continue
        if max_area > 0 and region.area > max_area:
            continue
        coords = region.coords
        rr = coords[:, 0]
        cc = coords[:, 1]
        score = float(prob[rr, cc].mean())
        c_score = float(contrast[rr, cc].mean())
        l_score = float(lap[rr, cc].mean())
        if c_score < min_contrast or l_score < min_lap:
            continue
        if score <= 0:
            continue
        out[rr, cc] = True
    return out


def evaluate_samples(samples: list[dict[str, np.ndarray]], args: argparse.Namespace) -> dict[str, object]:
    thresholds = parse_floats(args.thresholds)
    min_areas = parse_ints(args.min_areas)
    max_areas = parse_ints(args.max_areas)
    contrast_thresholds = parse_floats(args.contrast_thresholds)
    lap_thresholds = parse_floats(args.lap_thresholds)
    rows = []
    for threshold in thresholds:
        for min_area in min_areas:
            for max_area in max_areas:
                for min_contrast in contrast_thresholds:
                    for min_lap in lap_thresholds:
                        inter = 0
                        union = 0
                        pred_pixels = 0
                        gt_pixels = 0
                        for sample in samples:
                            pred = sample["prob"] > threshold
                            pred = filter_components(
                                sample["prob"],
                                pred,
                                sample["contrast"],
                                sample["lap"],
                                min_area,
                                max_area,
                                min_contrast,
                                min_lap,
                            )
                            gt = sample["mask"].astype(bool)
                            inter += int(np.logical_and(pred, gt).sum())
                            union += int(np.logical_or(pred, gt).sum())
                            pred_pixels += int(pred.sum())
                            gt_pixels += int(gt.sum())
                        iou = inter / union if union else 0.0
                        precision = inter / pred_pixels if pred_pixels else 0.0
                        recall = inter / gt_pixels if gt_pixels else 0.0
                        rows.append(
                            {
                                "mIoU": iou,
                                "precision": precision,
                                "recall": recall,
                                "threshold": threshold,
                                "min_area": min_area,
                                "max_area": max_area,
                                "min_contrast": min_contrast,
                                "min_lap": min_lap,
                                "intersection": inter,
                                "union": union,
                                "pred_pixels": pred_pixels,
                                "gt_pixels": gt_pixels,
                            }
                        )
    rows.sort(key=lambda item: item["mIoU"], reverse=True)
    return {
        "dataset": args.dataset_name,
        "model": args.model,
        "checkpoint": args.checkpoint,
        "best": rows[0] if rows else None,
        "top": rows[: args.topk],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep selective component/contrast postprocessing for IRST predictions.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60,0.65")
    parser.add_argument("--min_areas", default="1,2,3,4,5")
    parser.add_argument("--max_areas", default="0,30,60,120,240")
    parser.add_argument("--contrast_thresholds", default="0.0,0.15,0.25,0.35,0.50")
    parser.add_argument("--lap_thresholds", default="0.0,0.15,0.25,0.35,0.50")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    samples = collect_samples(args)
    result = evaluate_samples(samples, args)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
