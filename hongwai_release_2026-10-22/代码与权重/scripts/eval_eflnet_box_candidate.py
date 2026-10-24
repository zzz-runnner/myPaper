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
from skimage import measure

ROOT = Path(__file__).resolve().parents[1]
EFLNET = ROOT / "external" / "EFLNet"
sys.path.insert(0, str(EFLNET))

from utils.datasets import letterbox  # noqa: E402
from utils.general import non_max_suppression, scale_coords  # noqa: E402


class SplitSet:
    def __init__(self, dataset_dir: Path, dataset_name: str, split: str, max_samples: int) -> None:
        self.root = dataset_dir / dataset_name
        index = self.root / "img_idx" / f"{split}_{dataset_name}.txt"
        self.names = [x.strip() for x in index.read_text(encoding="utf-8").splitlines() if x.strip()]
        if max_samples > 0:
            self.names = self.names[:max_samples]

    def __len__(self) -> int:
        return len(self.names)

    def mask(self, name: str) -> tuple[np.ndarray, list[int]]:
        for ext in (".png", ".bmp", ".jpg"):
            path = self.root / "masks" / f"{name}{ext}"
            if path.exists():
                arr = np.array(Image.open(path), dtype=np.float32) / 255.0
                if arr.ndim == 3:
                    arr = arr[:, :, 0]
                h, w = arr.shape
                return arr, [h, w]
        raise FileNotFoundError(name)


def metric_from_maps(samples: list[dict[str, object]], maps: list[np.ndarray], threshold: float) -> dict[str, float]:
    total_inter = 0
    total_union = 0
    total_correct = 0
    total_label = 0
    sample_ious: list[float] = []
    pd = 0
    target = 0
    false_pixels = 0
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

        pred_labels = measure.label(pred_bin.astype(np.int64), connectivity=2)
        pred_regions = list(measure.regionprops(pred_labels))
        gt_labels = measure.label(mask.astype(np.int64), connectivity=2)
        gt_regions = list(measure.regionprops(gt_labels))
        target += len(gt_regions)
        pred_areas = [float(region.area) for region in pred_regions]
        matched_areas: list[float] = []
        remaining = pred_regions
        for gt_region in gt_regions:
            gt_centroid = np.array(gt_region.centroid)
            for idx, pred_region in enumerate(remaining):
                pred_centroid = np.array(pred_region.centroid)
                if np.linalg.norm(pred_centroid - gt_centroid) < 3:
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

    pix_acc = total_correct / float(total_label + np.spacing(1))
    miou = total_inter / float(total_union + np.spacing(1))
    return {
        "pixAcc": float(pix_acc),
        "mIoU": float(miou),
        "nIoU": float(np.mean(sample_ious)),
        "Pd": float(pd) / float(max(target, 1)),
        "Fa": float(false_pixels) / float(max(all_pixels, 1)),
    }


def load_eflnet(weight: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(weight, map_location=device)
    model = ckpt["ema" if ckpt.get("ema") else "model"].float()
    model = model.fuse().eval().to(device)
    return model


def image_path(dataset_dir: Path, dataset_name: str, name: str) -> Path:
    for ext in (".png", ".bmp", ".jpg"):
        path = dataset_dir / dataset_name / "images" / f"{name}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(name)


def add_box(mask: np.ndarray, box: np.ndarray, score: float, mode: str) -> None:
    h, w = mask.shape
    x0, y0, x1, y1 = box.astype(np.float32)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w - 1, x1), min(h - 1, y1)
    bw, bh = max(1.0, x1 - x0 + 1.0), max(1.0, y1 - y0 + 1.0)
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5

    if mode == "box":
        xx0, yy0, xx1, yy1 = int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))
        mask[yy0 : yy1 + 1, xx0 : xx1 + 1] = np.maximum(mask[yy0 : yy1 + 1, xx0 : xx1 + 1], score)
        return

    if mode.startswith("center"):
        frac = float(mode.replace("center", ""))
        hw, hh = max(1.0, bw * frac * 0.5), max(1.0, bh * frac * 0.5)
        xx0, yy0 = int(round(max(0, cx - hw))), int(round(max(0, cy - hh)))
        xx1, yy1 = int(round(min(w - 1, cx + hw))), int(round(min(h - 1, cy + hh)))
        mask[yy0 : yy1 + 1, xx0 : xx1 + 1] = np.maximum(mask[yy0 : yy1 + 1, xx0 : xx1 + 1], score)
        return

    if mode.startswith("ellipse"):
        frac = float(mode.replace("ellipse", ""))
        rx, ry = max(1, int(round(bw * frac * 0.5))), max(1, int(round(bh * frac * 0.5)))
        tmp = np.zeros_like(mask, dtype=np.float32)
        cv2.ellipse(tmp, (int(round(cx)), int(round(cy))), (rx, ry), 0, 0, 360, float(score), -1)
        np.maximum(mask, tmp, out=mask)
        return

    if mode.startswith("dot"):
        radius = int(mode.replace("dot", ""))
        tmp = np.zeros_like(mask, dtype=np.float32)
        cv2.circle(tmp, (int(round(cx)), int(round(cy))), radius, float(score), -1)
        np.maximum(mask, tmp, out=mask)
        return

    if mode.startswith("gauss"):
        frac = float(mode.replace("gauss", ""))
        sigma_x = max(0.75, bw * frac * 0.5)
        sigma_y = max(0.75, bh * frac * 0.5)
        radius_x = max(1, int(round(3.0 * sigma_x)))
        radius_y = max(1, int(round(3.0 * sigma_y)))
        xx0, xx1 = max(0, int(round(cx)) - radius_x), min(w - 1, int(round(cx)) + radius_x)
        yy0, yy1 = max(0, int(round(cy)) - radius_y), min(h - 1, int(round(cy)) + radius_y)
        xs = np.arange(xx0, xx1 + 1, dtype=np.float32)
        ys = np.arange(yy0, yy1 + 1, dtype=np.float32)
        gx = ((xs - cx) ** 2) / (2.0 * sigma_x * sigma_x)
        gy = ((ys - cy) ** 2) / (2.0 * sigma_y * sigma_y)
        patch = np.exp(-(gy[:, None] + gx[None, :])).astype(np.float32) * float(score)
        mask[yy0 : yy1 + 1, xx0 : xx1 + 1] = np.maximum(mask[yy0 : yy1 + 1, xx0 : xx1 + 1], patch)
        return

    raise ValueError(mode)


@torch.no_grad()
def predict_maps(
    model: torch.nn.Module,
    dataset_dir: Path,
    dataset_name: str,
    norm_dataset: str,
    split: str,
    max_samples: int,
    image_size: int,
    conf_thres: float,
    iou_thres: float,
    modes: list[str],
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, list[np.ndarray]], dict[str, float]]:
    _ = norm_dataset
    dataset = SplitSet(dataset_dir, dataset_name, split, max_samples=max_samples)
    samples: list[dict[str, object]] = []
    maps = {mode: [] for mode in modes}
    det_count = 0
    images_with_det = 0
    stride = int(model.stride.max()) if hasattr(model, "stride") else 32

    for idx, name in enumerate(dataset.names):
        mask_np, size = dataset.mask(name)
        height, width = int(size[0]), int(size[1])
        img0 = cv2.imread(str(image_path(dataset_dir, dataset_name, name)), cv2.IMREAD_COLOR)
        if img0 is None:
            raise FileNotFoundError(name)
        img = letterbox(img0, image_size, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        tensor = torch.from_numpy(img).to(device).float() / 255.0
        tensor = tensor.unsqueeze(0)
        pred = model(tensor, augment=False)[0]
        det = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, classes=None, agnostic=True)[0]

        per_mode = {mode: np.zeros((height, width), dtype=np.float32) for mode in modes}
        if det is not None and len(det):
            det[:, :4] = scale_coords(tensor.shape[2:], det[:, :4], img0.shape).round()
            images_with_det += 1
            det_count += int(len(det))
            for row in det.detach().cpu().numpy():
                box = row[:4]
                score = float(row[4])
                for mode in modes:
                    add_box(per_mode[mode], box, score, mode)

        samples.append({"name": name, "size": [height, width], "mask": mask_np, "preds": {}})
        for mode in modes:
            maps[mode].append(per_mode[mode])

        if (idx + 1) % 50 == 0:
            print(f"{dataset_name} {split}: {idx + 1}/{len(dataset)}", flush=True)

    stats = {
        "images": float(len(dataset)),
        "images_with_det": float(images_with_det),
        "detections": float(det_count),
        "detections_per_image": float(det_count) / float(max(len(dataset), 1)),
    }
    return samples, maps, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate EFLNet official detection weights as box-to-mask candidates.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset", default="NUAA-SIRST")
    parser.add_argument("--norm_dataset", default="NUAA-SIRST")
    parser.add_argument("--split", default="test")
    parser.add_argument("--weight", default=str(EFLNET / "runs" / "train" / "NUAA.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--conf_thres", type=float, default=0.01)
    parser.add_argument("--iou_thres", type=float, default=0.50)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--thresholds", default="0.01,0.03,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--modes", default="box,center0.50,center0.33,ellipse0.60,ellipse0.40,dot1,dot2,dot3")
    parser.add_argument("--output", required=True)
    parser.add_argument("--save_maps", default="")
    args = parser.parse_args()

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model = load_eflnet(Path(args.weight), device)
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]

    samples, maps, stats = predict_maps(
        model=model,
        dataset_dir=Path(args.dataset_dir),
        dataset_name=args.dataset,
        norm_dataset=args.norm_dataset,
        split=args.split,
        max_samples=args.max_samples,
        image_size=args.image_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        modes=modes,
        device=device,
    )

    results = []
    for mode in modes:
        rows = [{"threshold": th, **metric_from_maps(samples, maps[mode], th)} for th in thresholds]
        rows.sort(key=lambda row: row["mIoU"], reverse=True)
        best = rows[0]
        print(
            f"{args.dataset} {Path(args.weight).name} {mode}: th={best['threshold']} "
            f"mIoU={best['mIoU'] * 100:.2f} nIoU={best['nIoU'] * 100:.2f} "
            f"Pd={best['Pd'] * 100:.2f} Fa={best['Fa'] * 1e6:.2f}",
            flush=True,
        )
        results.append({"mode": mode, "best": best, "rows": rows})

    Path(args.output).write_text(
        json.dumps(
            {
                "weight": str(Path(args.weight)),
                "dataset": args.dataset,
                "norm_dataset": args.norm_dataset,
                "split": args.split,
                "image_size": args.image_size,
                "conf_thres": args.conf_thres,
                "iou_thres": args.iou_thres,
                "stats": stats,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.save_maps:
        meta = [{"name": sample["name"], "size": sample["size"]} for sample in samples]
        payload = {
            "meta_json": np.array(json.dumps(meta)),
            "names_json": np.array(json.dumps(modes)),
        }
        for idx, sample in enumerate(samples):
            payload[f"mask_{idx}"] = sample["mask"].astype(np.float32)
            for mode in modes:
                payload[f"pred_{idx}_{mode}"] = maps[mode][idx].astype(np.float32)
        np.savez_compressed(args.save_maps, **payload)


if __name__ == "__main__":
    main()
