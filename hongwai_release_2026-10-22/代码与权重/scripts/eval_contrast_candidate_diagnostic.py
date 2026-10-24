#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_component_fusion_learner import contrast_candidate_maps  # noqa: E402
from metrics import PD_FA, mIoU  # noqa: E402
from utils import Normalized, PadImg, get_img_norm_cfg  # noqa: E402


def open_pair(dataset_dir: Path, dataset_name: str, name: str) -> tuple[Image.Image, Image.Image]:
    root = dataset_dir / dataset_name
    for ext in (".png", ".bmp", ".jpg"):
        image_path = root / "images" / f"{name}{ext}"
        mask_path = root / "masks" / f"{name}{ext}"
        if image_path.exists() and mask_path.exists():
            return Image.open(image_path).convert("I"), Image.open(mask_path)
    raise FileNotFoundError(name)


def metric_from_binary(preds: list[np.ndarray], masks: list[np.ndarray], sizes: list[list[int]]) -> dict[str, float]:
    iou_metric = mIoU()
    pdfa_metric = PD_FA()
    sample_ious: list[float] = []
    for pred, mask, size in zip(preds, masks, sizes):
        pred_bin = pred > 0
        mask_bin = mask > 0
        inter = int((pred_bin & mask_bin).sum())
        union = int((pred_bin | mask_bin).sum())
        sample_ious.append(inter / float(union + np.spacing(1)))
        pred_t = torch.from_numpy(pred_bin.astype(np.float32)).view(1, 1, *pred_bin.shape)
        mask_t = torch.from_numpy(mask_bin.astype(np.float32)).view(1, 1, *mask_bin.shape)
        iou_metric.update(pred_t, mask_t)
        pdfa_metric.update(pred_t[0, 0], mask_t[0, 0], size)
    pix_acc, miou = iou_metric.get()
    return {
        "pixAcc": float(pix_acc),
        "mIoU": float(miou),
        "nIoU": float(np.mean(sample_ious)),
        "Pd": float(pdfa_metric.PD) / float(max(pdfa_metric.target, 1)),
        "Fa": float(pdfa_metric.dismatch_pixel) / float(max(pdfa_metric.all_pixel, 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--topks", default="8,16,24,32")
    parser.add_argument("--thresholds", default="0.72,0.76,0.80,0.84,0.88")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    index = dataset_dir / args.dataset_name / "img_idx" / f"{args.split}_{args.dataset_name}.txt"
    names = index.read_text(encoding="utf-8").splitlines()
    if args.max_samples > 0:
        names = names[: args.max_samples]
    norm_cfg = get_img_norm_cfg(args.dataset_name, str(dataset_dir))
    masks: list[np.ndarray] = []
    sizes: list[list[int]] = []
    raw_images: list[np.ndarray] = []
    for name in names:
        image_pil, mask_pil = open_pair(dataset_dir, args.dataset_name, name)
        image = Normalized(np.array(image_pil, dtype=np.float32), norm_cfg)
        mask = np.array(mask_pil, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        height, width = image.shape
        raw_images.append(PadImg(image)[:height, :width].astype(np.float32))
        masks.append((PadImg(mask)[:height, :width] > 0).astype(np.uint8))
        sizes.append([height, width])

    precomputed: list[tuple[np.ndarray, np.ndarray]] = []
    min_threshold = min(float(x) for x in args.thresholds.split(",") if x.strip())
    max_topk = max(int(x) for x in args.topks.split(",") if x.strip())
    for image in raw_images:
        maps = contrast_candidate_maps(image, topk=max_topk, threshold=min_threshold)
        precomputed.append((maps["contrast_score"], maps["contrast_sparse"]))

    rows: list[dict[str, object]] = []
    for topk in [int(x) for x in args.topks.split(",") if x.strip()]:
        for threshold in [float(x) for x in args.thresholds.split(",") if x.strip()]:
            preds: list[np.ndarray] = []
            hit_images = 0
            pred_pixels = 0
            for (score, _sparse), mask in zip(precomputed, masks):
                seed = score > threshold
                num_labels, label_img, stats, _centroids = cv2.connectedComponentsWithStats(seed.astype(np.uint8), 8)
                candidates: list[tuple[float, int]] = []
                for label_id in range(1, num_labels):
                    area = int(stats[label_id, cv2.CC_STAT_AREA])
                    width = int(stats[label_id, cv2.CC_STAT_WIDTH])
                    height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
                    if not (1 <= area <= 380 and width <= 42 and height <= 42):
                        continue
                    comp = label_img == label_id
                    fill = area / float(max(width * height, 1))
                    comp_score = float(score[comp].mean() + 0.9 * score[comp].max() + 0.08 * fill)
                    candidates.append((comp_score, label_id))
                cand = np.zeros_like(seed, dtype=np.uint8)
                for _score, label_id in sorted(candidates, reverse=True)[: max(0, topk)]:
                    cand[label_img == label_id] = 1
                cand = cv2.dilate(cand, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1) > 0
                preds.append(cand.astype(np.uint8))
                hit_images += int((cand & (mask > 0)).any())
                pred_pixels += int(cand.sum())
            metrics = metric_from_binary(preds, masks, sizes)
            rows.append(
                {
                    "topk": topk,
                    "threshold": threshold,
                    "hit_images": hit_images,
                    "num_images": len(names),
                    "pred_pixels": pred_pixels,
                    **metrics,
                }
            )
    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    out = {
        "dataset": args.dataset_name,
        "split": args.split,
        "num_images": len(names),
        "best": rows[0] if rows else None,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if rows:
        b = rows[0]
        print(
            f"best topk={b['topk']} th={b['threshold']} mIoU={b['mIoU']*100:.2f} "
            f"Pd={b['Pd']*100:.2f} Fa={b['Fa']*1e6:.2f} hit={b['hit_images']}/{b['num_images']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
