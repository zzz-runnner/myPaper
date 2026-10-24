#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MTML_ROOT = PROJECT_ROOT / "external" / "MTMLNet"
if str(MTML_ROOT) not in sys.path:
    sys.path.insert(0, str(MTML_ROOT))

from models.experimental import attempt_load  # noqa: E402


def read_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def resize_long_side(image: Image.Image, base_size: int) -> tuple[Image.Image, tuple[int, int]]:
    width, height = image.size
    if width >= height:
        new_width = base_size
        new_height = max(32, int(round(height * base_size / width / 32)) * 32)
    else:
        new_height = base_size
        new_width = max(32, int(round(width * base_size / height / 32)) * 32)
    return image.resize((new_width, new_height), Image.Resampling.BILINEAR), (width, height)


def image_tensor(path: Path, base_size: int, device: torch.device, half: bool) -> tuple[torch.Tensor, tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    resized, original_size = resize_long_side(image, base_size)
    arr = np.asarray(resized).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return (tensor.half() if half else tensor.float()), original_size


def mask_array(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    model = attempt_load(args.weights, device=device)
    model.eval()
    half = args.half and device.type != "cpu"
    if half:
        model.half()

    dataset_root = Path(args.dataset_root) / args.dataset_name
    split_path = dataset_root / "img_idx" / f"test_{args.dataset_name}.txt"
    names = read_names(split_path)
    thresholds = [float(x) for x in args.thresholds.split(",")]
    totals = {
        thr: {"inter": 0, "union": 0, "tp": 0, "pred": 0, "gt": 0}
        for thr in thresholds
    }

    with torch.no_grad():
        for name in names:
            image_path = dataset_root / "images" / f"{name}.png"
            mask_path = dataset_root / "masks" / f"{name}.png"
            image, original_size = image_tensor(image_path, args.base_size, device, half)
            outputs = model(image)
            logits = outputs[1]
            logits = F.interpolate(logits, (original_size[1], original_size[0]), mode="bilinear", align_corners=True)
            prob = torch.softmax(logits.float(), dim=1)[:, 1].squeeze(0).cpu().numpy()
            gt = mask_array(mask_path)
            for thr in thresholds:
                pred = prob >= thr
                inter = np.logical_and(pred, gt).sum()
                union = np.logical_or(pred, gt).sum()
                totals[thr]["inter"] += int(inter)
                totals[thr]["union"] += int(union)
                totals[thr]["tp"] += int(inter)
                totals[thr]["pred"] += int(pred.sum())
                totals[thr]["gt"] += int(gt.sum())

    by_threshold = {}
    for thr, values in totals.items():
        inter = values["inter"]
        union = values["union"]
        pred = values["pred"]
        gt = values["gt"]
        precision = inter / pred if pred else 0.0
        recall = inter / gt if gt else 0.0
        iou = inter / union if union else 0.0
        by_threshold[f"{thr:.3f}"] = {
            "mIoU": iou * 100,
            "precision": precision * 100,
            "recall": recall * 100,
            "intersection": inter,
            "union": union,
            "pred_pixels": pred,
            "gt_pixels": gt,
        }

    best_thr, best_metrics = max(by_threshold.items(), key=lambda item: item[1]["mIoU"])
    result = {
        "weights": str(args.weights),
        "dataset": args.dataset_name,
        "base_size": args.base_size,
        "num_images": len(names),
        "best_threshold": best_thr,
        "best": best_metrics,
        "by_threshold": by_threshold,
    }
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MTMLNet segmentation head on IRSTD-style masks.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset_root", default=str(PROJECT_ROOT / "datasets"))
    parser.add_argument("--dataset_name", required=True, choices=["IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST", "SIRST3"])
    parser.add_argument("--base_size", type=int, default=512)
    parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(evaluate(parse_args()), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
