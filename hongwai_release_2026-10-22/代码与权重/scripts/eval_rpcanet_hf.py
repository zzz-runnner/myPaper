#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import measure

ROOT = Path(__file__).resolve().parents[1]
RPCANET = ROOT / "external" / "RPCANet_hf"
sys.path.insert(0, str(RPCANET))

from models import get_model  # noqa: E402


def load_model(model_name: str, weight: Path, device: torch.device) -> torch.nn.Module:
    model = get_model(model_name).to(device).eval()
    state = torch.load(weight, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    return model


def names_for_split(dataset_root: Path, dataset_name: str, split: str) -> list[str]:
    index = dataset_root / dataset_name / "img_idx" / f"{split}_{dataset_name}.txt"
    return [line.strip() for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]


def open_pair(dataset_root: Path, dataset_name: str, name: str) -> tuple[np.ndarray, np.ndarray]:
    root = dataset_root / dataset_name
    for ext in (".png", ".bmp", ".jpg"):
        image_path = root / "images" / f"{name}{ext}"
        mask_path = root / "masks" / f"{name}{ext}"
        if image_path.exists() and mask_path.exists():
            image = np.array(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
            mask = np.array(Image.open(mask_path), dtype=np.float32) / 255.0
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            return image, mask
    raise FileNotFoundError(name)


def metric_from_maps(samples: list[dict[str, object]], maps: list[np.ndarray], threshold: float) -> dict[str, float]:
    total_inter = 0
    total_union = 0
    total_correct = 0
    total_label = 0
    sample_ious = []
    pd = 0
    target = 0
    false_pixels = 0.0
    all_pixels = 0

    for sample, pred in zip(samples, maps):
        mask = sample["mask"] > 0
        pred_bin = pred > threshold
        inter = int((pred_bin & mask).sum())
        union = int((pred_bin | mask).sum())
        total_inter += inter
        total_union += union
        total_correct += inter
        total_label += int(mask.sum())
        sample_ious.append(inter / float(union + np.spacing(1)))

        pred_regions = list(measure.regionprops(measure.label(pred_bin.astype(np.int64), connectivity=2)))
        gt_regions = list(measure.regionprops(measure.label(mask.astype(np.int64), connectivity=2)))
        target += len(gt_regions)
        pred_areas = [float(region.area) for region in pred_regions]
        matched_areas: list[float] = []
        remaining = pred_regions
        for gt_region in gt_regions:
            gt_centroid = np.array(gt_region.centroid)
            for idx, pred_region in enumerate(remaining):
                if np.linalg.norm(np.array(pred_region.centroid) - gt_centroid) < 3:
                    pd += 1
                    matched_areas.append(float(pred_region.area))
                    del remaining[idx]
                    break
        for area in pred_areas:
            if area in matched_areas:
                matched_areas.remove(area)
            else:
                false_pixels += area
        all_pixels += int(sample["size"][0]) * int(sample["size"][1])

    return {
        "pixAcc": float(total_correct) / float(total_label + np.spacing(1)),
        "mIoU": float(total_inter) / float(total_union + np.spacing(1)),
        "nIoU": float(np.mean(sample_ious)),
        "Pd": float(pd) / float(max(target, 1)),
        "Fa": float(false_pixels) / float(max(all_pixels, 1)),
    }


@torch.no_grad()
def predict_dataset(
    model: torch.nn.Module,
    dataset_root: Path,
    dataset_name: str,
    split: str,
    base_size: int,
    max_samples: int,
    device: torch.device,
) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    names = names_for_split(dataset_root, dataset_name, split)
    if max_samples > 0:
        names = names[:max_samples]
    samples = []
    maps = []
    for idx, name in enumerate(names):
        image, mask = open_pair(dataset_root, dataset_name, name)
        height, width = image.shape
        inp = cv2.resize(image, (base_size, base_size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(inp[None, None].astype(np.float32)).to(device)
        out = model(tensor)
        if isinstance(out, (tuple, list)):
            out = out[-1]
        prob = torch.sigmoid(out)
        prob = F.interpolate(prob, size=(height, width), mode="bilinear", align_corners=False)
        prob_np = prob[0, 0].detach().cpu().numpy().astype(np.float32)
        samples.append({"name": name, "size": [height, width], "mask": mask})
        maps.append(prob_np)
        if (idx + 1) % 50 == 0:
            print(f"{dataset_name}: {idx + 1}/{len(names)}", flush=True)
    return samples, maps


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HuggingFace RPCANet/RPCANet++ weights on local IRSTD datasets.")
    parser.add_argument("--dataset_root", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default="rpcanet_pp", choices=["rpcanet", "rpcanet_pp", "rpcanet_pp_s3", "rpcanet_pp_s9"])
    parser.add_argument("--weight", required=True)
    parser.add_argument("--base_size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = load_model(args.model, Path(args.weight), device)
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    samples, maps = predict_dataset(model, Path(args.dataset_root), args.dataset, args.split, args.base_size, args.max_samples, device)
    rows = [{"threshold": th, **metric_from_maps(samples, maps, th)} for th in thresholds]
    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best = rows[0]
    print(
        f"{args.dataset} {Path(args.weight).parent.name}/{Path(args.weight).name}: "
        f"th={best['threshold']} mIoU={best['mIoU']*100:.2f} nIoU={best['nIoU']*100:.2f} "
        f"Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.2f}",
        flush=True,
    )
    Path(args.output).write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "split": args.split,
                "model": args.model,
                "weight": str(Path(args.weight)),
                "base_size": args.base_size,
                "best": best,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
