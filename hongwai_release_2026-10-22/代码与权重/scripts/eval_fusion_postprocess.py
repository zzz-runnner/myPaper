#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from skimage import measure
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))
sys.path.insert(0, str(ROOT / "scripts"))

from dataset import TestSetLoader  # noqa: E402
from metrics import PD_FA, mIoU  # noqa: E402
from train_hybrid_methods import load_model_checkpoint, make_model  # noqa: E402


def parse_floats(spec: str) -> list[float]:
    values: list[float] = []
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" in item:
            start, stop, step = [float(x) for x in item.split(":")]
            cur = start
            while cur <= stop + 1e-12:
                values.append(round(cur, 6))
                cur += step
        else:
            values.append(float(item))
    return sorted(set(values))


def parse_ints(spec: str) -> list[int]:
    values: list[int] = []
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" in item:
            start, stop, step = [int(x) for x in item.split(":")]
            values.extend(range(start, stop + 1, step))
        else:
            values.append(int(item))
    return sorted(set(values))


def parse_model_spec(spec: str) -> tuple[str, str, str, str]:
    parts = spec.split("|")
    if len(parts) != 4:
        raise ValueError("Model spec must be name|model|base_checkpoint|checkpoint")
    return parts[0], parts[1], parts[2], parts[3]


def filter_components(
    pred_bin: torch.Tensor,
    min_area: int,
    max_area: int,
    max_height: int,
    max_width: int,
) -> torch.Tensor:
    if min_area <= 1 and max_area <= 0 and max_height <= 0 and max_width <= 0:
        return pred_bin
    pred_np = pred_bin.cpu().numpy().astype(np.uint8)
    filtered = np.zeros_like(pred_np)
    for batch_idx in range(pred_np.shape[0]):
        for channel_idx in range(pred_np.shape[1]):
            label_img = measure.label(pred_np[batch_idx, channel_idx] > 0, connectivity=2)
            for region in measure.regionprops(label_img):
                height = region.bbox[2] - region.bbox[0]
                width = region.bbox[3] - region.bbox[1]
                if region.area < min_area:
                    continue
                if max_area > 0 and region.area > max_area:
                    continue
                if max_height > 0 and height > max_height:
                    continue
                if max_width > 0 and width > max_width:
                    continue
                filtered[batch_idx, channel_idx][label_img == region.label] = 1
    return torch.from_numpy(filtered.astype(bool))


@torch.no_grad()
def build_models(
    specs: list[tuple[str, str, str, str]],
    device: torch.device,
) -> list[tuple[str, torch.nn.Module]]:
    models: list[tuple[str, torch.nn.Module]] = []
    for name, model_name, base_checkpoint, checkpoint in specs:
        model = make_model(model_name, warm_epoch=1, base_checkpoint=base_checkpoint).to(device)
        if hasattr(model, "initialize_refiner"):
            model.initialize_refiner()
        load_model_checkpoint(model, Path(checkpoint))
        if hasattr(model, "freeze_inherited_refiner"):
            model.freeze_inherited_refiner()
        model.eval()
        models.append((name, model))
    return models


@torch.no_grad()
def collect_samples(
    models: list[tuple[str, torch.nn.Module]],
    dataset_dir: Path,
    dataset_name: str,
    device: torch.device,
) -> list[dict[str, object]]:
    dataset = TestSetLoader(str(dataset_dir), dataset_name, dataset_name)
    loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
    samples: list[dict[str, object]] = []
    for image, mask, size, sample_name in loader:
        image = image.to(device)
        height = int(size[0])
        width = int(size[1])
        preds: dict[str, torch.Tensor] = {}
        for name, model in models:
            pred = model(image, epoch=10**9)
            if isinstance(pred, (tuple, list)):
                pred = pred[-1]
            preds[name] = pred[:, :, :height, :width].cpu()
        samples.append(
            {
                "name": sample_name[0],
                "mask": mask[:, :, :height, :width].cpu(),
                "size": [height, width],
                "preds": preds,
            }
        )
    return samples


def sample_iou(pred_bin: torch.Tensor, mask: torch.Tensor) -> float:
    pred = pred_bin.bool()
    gt = mask > 0
    inter = (pred & gt).sum().item()
    union = (pred | gt).sum().item()
    return float(inter) / float(union + np.spacing(1))


def filters_disabled(min_area: int, max_area: int, max_height: int, max_width: int) -> bool:
    return min_area <= 1 and max_area <= 0 and max_height <= 0 and max_width <= 0


def flat_gt(samples: list[dict[str, object]]) -> np.ndarray:
    return np.concatenate([(sample["mask"].numpy() > 0).reshape(-1) for sample in samples])


def fast_global_metrics(flat_pred: np.ndarray, gt: np.ndarray, threshold: float) -> dict[str, float]:
    pred_bin = flat_pred > threshold
    inter = int(np.logical_and(pred_bin, gt).sum())
    pred_pixels = int(pred_bin.sum())
    gt_pixels = int(gt.sum())
    union = pred_pixels + gt_pixels - inter
    pix_acc = inter / float(gt_pixels + np.spacing(1))
    miou = inter / float(union + np.spacing(1))
    return {"pixAcc": float(pix_acc), "mIoU": float(miou)}


def exact_metrics(
    samples: list[dict[str, object]],
    maps: list[torch.Tensor],
    threshold: float,
    min_area: int,
    max_area: int,
    max_height: int,
    max_width: int,
) -> dict[str, float]:
    iou_metric = mIoU()
    pdfa_metric = PD_FA()
    sample_ious = []
    for sample, pred in zip(samples, maps):
        mask = sample["mask"]
        pred_bin = filter_components(pred > threshold, min_area, max_area, max_height, max_width)
        iou_metric.update(pred_bin, mask)
        pdfa_metric.update(pred_bin[0, 0], mask[0, 0], sample["size"])
        sample_ious.append(sample_iou(pred_bin, mask))
    pix_acc, miou = iou_metric.get()
    pd = float(pdfa_metric.PD) / float(max(pdfa_metric.target, 1))
    fa = float(pdfa_metric.dismatch_pixel) / float(max(pdfa_metric.all_pixel, 1))
    return {
        "pixAcc": float(pix_acc),
        "mIoU": float(miou),
        "nIoU": float(np.mean(sample_ious)),
        "Pd": float(pd),
        "Fa": float(fa),
    }


def evaluate_binary_maps(
    samples: list[dict[str, object]],
    maps: list[torch.Tensor],
    threshold: float,
    min_area: int,
    max_area: int,
    max_height: int,
    max_width: int,
) -> dict[str, float]:
    iou_metric = mIoU()
    pdfa_metric = PD_FA()
    sample_ious = []
    for sample, pred in zip(samples, maps):
        mask = sample["mask"]
        pred_bin = filter_components(pred > threshold, min_area, max_area, max_height, max_width)
        iou_metric.update(pred_bin, mask)
        pdfa_metric.update(pred_bin[0, 0], mask[0, 0], sample["size"])
        sample_ious.append(sample_iou(pred_bin, mask))
    pix_acc, miou = iou_metric.get()
    pd = float(pdfa_metric.PD) / float(max(pdfa_metric.target, 1))
    fa = float(pdfa_metric.dismatch_pixel) / float(max(pdfa_metric.all_pixel, 1))
    return {
        "threshold": float(threshold),
        "min_area": int(min_area),
        "max_area": int(max_area),
        "max_height": int(max_height),
        "max_width": int(max_width),
        "pixAcc": float(pix_acc),
        "mIoU": float(miou),
        "nIoU": float(np.mean(sample_ious)),
        "Pd": float(pd),
        "Fa": float(fa),
    }


def flatten_maps(maps: list[torch.Tensor]) -> np.ndarray:
    return np.concatenate([pred.numpy().astype(np.float32, copy=False).reshape(-1) for pred in maps])


def fused_maps_for_mode(
    samples: list[dict[str, object]],
    names: list[str],
    mode: str,
    vote_threshold: float,
) -> list[torch.Tensor]:
    maps: list[torch.Tensor] = []
    for sample in samples:
        preds = [sample["preds"][name] for name in names]
        stack = torch.stack(preds, dim=0)
        if mode in names:
            maps.append(sample["preds"][mode])
        elif mode == "avg_prob":
            maps.append(stack.mean(dim=0))
        elif mode == "max_prob":
            maps.append(stack.max(dim=0).values)
        elif mode == "min_prob":
            maps.append(stack.min(dim=0).values)
        elif mode == "vote_prob":
            maps.append((stack > vote_threshold).float().mean(dim=0))
        else:
            raise ValueError(mode)
    return maps


def sample_oracle_maps(
    samples: list[dict[str, object]],
    names: list[str],
    threshold: float,
    min_area: int,
    max_area: int,
    max_height: int,
    max_width: int,
) -> tuple[list[torch.Tensor], dict[str, int]]:
    maps: list[torch.Tensor] = []
    choices = {name: 0 for name in names}
    for sample in samples:
        mask = sample["mask"]
        best_name = names[0]
        best_iou = -1.0
        for name in names:
            pred_bin = filter_components(
                sample["preds"][name] > threshold,
                min_area,
                max_area,
                max_height,
                max_width,
            )
            iou = sample_iou(pred_bin, mask)
            if iou > best_iou:
                best_iou = iou
                best_name = name
        choices[best_name] += 1
        maps.append(sample["preds"][best_name])
    return maps, choices


def evaluate_sample_oracle(
    samples: list[dict[str, object]],
    names: list[str],
    threshold: float,
    min_area: int,
    max_area: int,
    max_height: int,
    max_width: int,
) -> tuple[dict[str, float], dict[str, int]]:
    maps, choices = sample_oracle_maps(samples, names, threshold, min_area, max_area, max_height, max_width)
    return (
        evaluate_binary_maps(samples, maps, threshold, min_area, max_area, max_height, max_width),
        choices,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint fusion and component postprocessing with official metrics.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--model_spec", action="append", required=True, help="name|model|base_checkpoint|checkpoint")
    parser.add_argument("--thresholds", default="0.35:0.75:0.05")
    parser.add_argument("--min_areas", default="1,2,3")
    parser.add_argument("--max_areas", default="0")
    parser.add_argument("--max_heights", default="0")
    parser.add_argument("--max_widths", default="0")
    parser.add_argument("--modes", default="individual,avg_prob,max_prob,min_prob,vote_prob,sample_oracle")
    parser.add_argument("--vote_threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    specs = [parse_model_spec(spec) for spec in args.model_spec]
    names = [spec[0] for spec in specs]
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    expanded_modes: list[str] = []
    for mode in modes:
        if mode == "individual":
            expanded_modes.extend(names)
        else:
            expanded_modes.append(mode)
    models = build_models(specs, device)
    samples = collect_samples(models, Path(args.dataset_dir), args.dataset_name, device)
    gt_flat = flat_gt(samples)

    rows = []
    oracle_choices: dict[str, dict[str, int]] = {}
    for threshold in parse_floats(args.thresholds):
        for min_area in parse_ints(args.min_areas):
            for max_area in parse_ints(args.max_areas):
                for max_height in parse_ints(args.max_heights):
                    for max_width in parse_ints(args.max_widths):
                        for mode in expanded_modes:
                            if filters_disabled(min_area, max_area, max_height, max_width):
                                if mode == "sample_oracle":
                                    maps, choices = sample_oracle_maps(
                                        samples,
                                        names,
                                        threshold,
                                        min_area,
                                        max_area,
                                        max_height,
                                        max_width,
                                    )
                                    choice_key = f"{threshold}|{min_area}|{max_area}|{max_height}|{max_width}"
                                    oracle_choices[choice_key] = choices
                                else:
                                    maps = fused_maps_for_mode(samples, names, mode, args.vote_threshold)
                                metrics = fast_global_metrics(flatten_maps(maps), gt_flat, threshold)
                                metrics.update({"nIoU": None, "Pd": None, "Fa": None})
                            else:
                                if mode == "sample_oracle":
                                    metrics, choices = evaluate_sample_oracle(
                                        samples,
                                        names,
                                        threshold,
                                        min_area,
                                        max_area,
                                        max_height,
                                        max_width,
                                    )
                                    choice_key = f"{threshold}|{min_area}|{max_area}|{max_height}|{max_width}"
                                    oracle_choices[choice_key] = choices
                                else:
                                    maps = fused_maps_for_mode(samples, names, mode, args.vote_threshold)
                                    metrics = evaluate_binary_maps(
                                        samples,
                                        maps,
                                        threshold,
                                        min_area,
                                        max_area,
                                        max_height,
                                        max_width,
                                    )
                            row = {"mode": mode, **metrics}
                            row.update(
                                {
                                    "threshold": float(threshold),
                                    "min_area": int(min_area),
                                    "max_area": int(max_area),
                                    "max_height": int(max_height),
                                    "max_width": int(max_width),
                                }
                            )
                            rows.append(row)

    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    for row in rows[: args.topk]:
        if row["mode"] == "sample_oracle":
            maps, choices = sample_oracle_maps(
                samples,
                names,
                row["threshold"],
                row["min_area"],
                row["max_area"],
                row["max_height"],
                row["max_width"],
            )
            choice_key = (
                f"{row['threshold']}|{row['min_area']}|{row['max_area']}|"
                f"{row['max_height']}|{row['max_width']}"
            )
            oracle_choices[choice_key] = choices
        else:
            maps = fused_maps_for_mode(samples, names, row["mode"], args.vote_threshold)
        row.update(
            exact_metrics(
                samples,
                maps,
                row["threshold"],
                row["min_area"],
                row["max_area"],
                row["max_height"],
                row["max_width"],
            )
        )
    best = rows[0]
    print(
        "best "
        f"mode={best['mode']} threshold={best['threshold']:.3f} min_area={best['min_area']} "
        f"max_area={best['max_area']} max_h={best['max_height']} max_w={best['max_width']} "
        f"mIoU={best['mIoU'] * 100:.2f} nIoU={best['nIoU'] * 100:.2f} "
        f"Pd={best['Pd'] * 100:.2f} Fa={best['Fa'] * 1e6:.2f} pixAcc={best['pixAcc'] * 100:.2f}",
        flush=True,
    )
    for row in rows[: args.topk]:
        print(
            f"{row['mode']}: th={row['threshold']:.3f} min={row['min_area']} max={row['max_area']} "
            f"mh={row['max_height']} mw={row['max_width']} mIoU={row['mIoU'] * 100:.2f} "
            f"nIoU={row['nIoU'] * 100:.2f} Pd={row['Pd'] * 100:.2f} Fa={row['Fa'] * 1e6:.2f}",
            flush=True,
        )

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "dataset": args.dataset_name,
                    "models": names,
                    "model_specs": args.model_spec,
                    "best": best,
                    "top": rows[: args.topk],
                    "rows": rows,
                    "oracle_choices": oracle_choices,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
