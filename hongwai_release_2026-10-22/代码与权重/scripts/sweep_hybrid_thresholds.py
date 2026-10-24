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


def filter_components(pred_bin: torch.Tensor, min_area: int, max_area: int, max_h: int, max_w: int) -> torch.Tensor:
    if min_area <= 1 and max_area <= 0 and max_h <= 0 and max_w <= 0:
        return pred_bin
    pred_np = pred_bin.cpu().numpy().astype(np.uint8)
    filtered = np.zeros_like(pred_np)
    for batch_idx in range(pred_np.shape[0]):
        for channel_idx in range(pred_np.shape[1]):
            label_img = measure.label(pred_np[batch_idx, channel_idx] > 0, connectivity=2)
            for region in measure.regionprops(label_img):
                h = region.bbox[2] - region.bbox[0]
                w = region.bbox[3] - region.bbox[1]
                if region.area < min_area:
                    continue
                if max_area > 0 and region.area > max_area:
                    continue
                if max_h > 0 and h > max_h:
                    continue
                if max_w > 0 and w > max_w:
                    continue
                filtered[batch_idx, channel_idx][label_img == region.label] = 1
    return torch.from_numpy(filtered.astype(bool))


@torch.no_grad()
def collect_predictions(model: torch.nn.Module, dataset_dir: Path, dataset_name: str, device: torch.device):
    test_set = TestSetLoader(str(dataset_dir), dataset_name, dataset_name)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=1, shuffle=False)
    model.eval()
    samples = []
    for image, mask, size, _name in test_loader:
        image = image.to(device)
        pred = model(image, epoch=10**9)
        if isinstance(pred, (tuple, list)):
            pred = pred[-1]
        h, w = int(size[0]), int(size[1])
        samples.append((pred[:, :, :h, :w].cpu(), mask[:, :, :h, :w].cpu(), [h, w]))
    return samples


def evaluate(samples, threshold: float, min_area: int, max_area: int, max_h: int, max_w: int) -> dict[str, float]:
    metric_iou = mIoU()
    metric_pdfa = PD_FA()
    for pred, mask, size in samples:
        pred_bin = filter_components(pred > threshold, min_area, max_area, max_h, max_w)
        metric_iou.update(pred_bin, mask)
        metric_pdfa.update(pred_bin[0, 0], mask[0, 0], size)
    pix_acc, miou = metric_iou.get()
    pd = float(metric_pdfa.PD) / float(max(metric_pdfa.target, 1))
    fa = float(metric_pdfa.dismatch_pixel) / float(max(metric_pdfa.all_pixel, 1))
    return {
        "threshold": float(threshold),
        "min_area": int(min_area),
        "max_area": int(max_area),
        "max_height": int(max_h),
        "max_width": int(max_w),
        "pixAcc": float(pix_acc),
        "mIoU": float(miou),
        "Pd": float(pd),
        "Fa": float(fa),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep thresholds/component filters for hybrid IRSTD models.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--thresholds", default="0.05:0.95:0.05")
    parser.add_argument("--min_areas", default="1")
    parser.add_argument("--max_areas", default="0")
    parser.add_argument("--max_heights", default="0")
    parser.add_argument("--max_widths", default="0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = make_model(args.model, warm_epoch=1, base_checkpoint=args.base_checkpoint).to(device)
    if hasattr(model, "initialize_refiner"):
        model.initialize_refiner()
    load_model_checkpoint(model, Path(args.checkpoint))
    if hasattr(model, "freeze_inherited_refiner"):
        model.freeze_inherited_refiner()

    samples = collect_predictions(model, Path(args.dataset_dir), args.dataset_name, device)
    rows = [
        evaluate(samples, threshold, min_area, max_area, max_h, max_w)
        for threshold in parse_floats(args.thresholds)
        for min_area in parse_ints(args.min_areas)
        for max_area in parse_ints(args.max_areas)
        for max_h in parse_ints(args.max_heights)
        for max_w in parse_ints(args.max_widths)
    ]
    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best = rows[0]
    print(
        "best "
        f"threshold={best['threshold']:.3f} min_area={best['min_area']} "
        f"max_area={best['max_area']} max_h={best['max_height']} max_w={best['max_width']} "
        f"mIoU={best['mIoU'] * 100:.2f} Pd={best['Pd'] * 100:.2f} Fa={best['Fa'] * 1e6:.2f} "
        f"pixAcc={best['pixAcc'] * 100:.2f}",
        flush=True,
    )
    if args.output:
        Path(args.output).write_text(json.dumps({"best": best, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
