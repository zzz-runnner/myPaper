#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from skimage import measure
from ultralytics import YOLO
import ultralytics.nn.modules.conv as ultralytics_conv
from ultralytics.utils import nms

ROOT = Path(__file__).resolve().parents[1]


class PConv(nn.Module):
    def __init__(self, c1: int, c2: int, k: int, s: int):
        super().__init__()
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=item) for item in p]
        self.cw = nn.Conv2d(c1, c2 // 4, (1, k), stride=s, padding=0)
        self.ch = nn.Conv2d(c1, c2 // 4, (k, 1), stride=s, padding=0)
        self.cat = nn.Conv2d(c2, c2, 2, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cat(torch.cat([self.cw(self.pad[0](x)), self.cw(self.pad[1](x)), self.ch(self.pad[2](x)), self.ch(self.pad[3](x))], dim=1))


setattr(ultralytics_conv, "PConv", PConv)


class SplitSet:
    def __init__(self, dataset_dir: Path, dataset_name: str, split: str, max_samples: int) -> None:
        self.root = dataset_dir / dataset_name
        index = self.root / "img_idx" / f"{split}_{dataset_name}.txt"
        self.names = [x.strip() for x in index.read_text(encoding="utf-8").splitlines() if x.strip()]
        if max_samples > 0:
            self.names = self.names[:max_samples]

    def mask(self, name: str) -> tuple[np.ndarray, list[int]]:
        for ext in (".png", ".bmp", ".jpg"):
            path = self.root / "masks" / f"{name}{ext}"
            if path.exists():
                arr = np.array(Image.open(path), dtype=np.float32) / 255.0
                if arr.ndim == 3:
                    arr = arr[:, :, 0]
                return arr, [int(arr.shape[0]), int(arr.shape[1])]
        raise FileNotFoundError(name)

    def image_path(self, name: str) -> Path:
        for ext in (".png", ".bmp", ".jpg"):
            path = self.root / "images" / f"{name}{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(name)


def resolve_weight(weight: Path, temp_dir: Path) -> Path:
    if weight.suffix.lower() != ".zip":
        return weight
    with zipfile.ZipFile(weight) as zf:
        candidates = [name for name in zf.namelist() if name.endswith("/weights/best.pt") or name.endswith("weights/best.pt")]
        if not candidates:
            raise FileNotFoundError(f"no weights/best.pt in {weight}")
        member = sorted(candidates, key=len)[0]
        out = temp_dir / (weight.stem + "_best.pt")
        with zf.open(member) as src, out.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return out


def letterbox_gray(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = img.shape
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size), 114, dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, scale, (left, top)


def scale_boxes_xyxy(boxes: np.ndarray, scale: float, pad: tuple[int, int], shape: tuple[int, int]) -> np.ndarray:
    out = boxes.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - pad[0]) / scale
    out[:, [1, 3]] = (out[:, [1, 3]] - pad[1]) / scale
    h, w = shape
    out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, w - 1)
    out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, h - 1)
    return out


def add_box(mask: np.ndarray, box: np.ndarray, score: float, mode: str) -> None:
    h, w = mask.shape
    x0, y0, x1, y1 = box.astype(np.float32)
    bw, bh = max(1.0, x1 - x0 + 1.0), max(1.0, y1 - y0 + 1.0)
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    if mode.startswith("center"):
        frac = float(mode.replace("center", ""))
        hw, hh = max(1.0, bw * frac * 0.5), max(1.0, bh * frac * 0.5)
        xx0, yy0 = int(round(max(0, cx - hw))), int(round(max(0, cy - hh)))
        xx1, yy1 = int(round(min(w - 1, cx + hw))), int(round(min(h - 1, cy + hh)))
        mask[yy0 : yy1 + 1, xx0 : xx1 + 1] = np.maximum(mask[yy0 : yy1 + 1, xx0 : xx1 + 1], score)
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
        rx, ry = max(1, int(round(3 * sigma_x))), max(1, int(round(3 * sigma_y)))
        xx0, xx1 = max(0, int(round(cx)) - rx), min(w - 1, int(round(cx)) + rx)
        yy0, yy1 = max(0, int(round(cy)) - ry), min(h - 1, int(round(cy)) + ry)
        xs = np.arange(xx0, xx1 + 1, dtype=np.float32)
        ys = np.arange(yy0, yy1 + 1, dtype=np.float32)
        patch = np.exp(-(((ys - cy)[:, None] ** 2) / (2 * sigma_y * sigma_y) + ((xs - cx)[None, :] ** 2) / (2 * sigma_x * sigma_x)))
        mask[yy0 : yy1 + 1, xx0 : xx1 + 1] = np.maximum(mask[yy0 : yy1 + 1, xx0 : xx1 + 1], patch.astype(np.float32) * float(score))
        return
    raise ValueError(mode)


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
        pred_regions = list(measure.regionprops(measure.label(pred_bin.astype(np.int64), connectivity=2)))
        gt_regions = list(measure.regionprops(measure.label(mask.astype(np.int64), connectivity=2)))
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
    return {
        "pixAcc": float(total_correct / float(total_label + np.spacing(1))),
        "mIoU": float(total_inter / float(total_union + np.spacing(1))),
        "nIoU": float(np.mean(sample_ious)),
        "Pd": float(pd) / float(max(target, 1)),
        "Fa": float(false_pixels) / float(max(all_pixels, 1)),
    }


@torch.no_grad()
def predict_maps(args: argparse.Namespace, weight: Path) -> tuple[list[dict[str, object]], dict[str, list[np.ndarray]], dict[str, float]]:
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = YOLO(str(weight)).model.to(device).eval()
    dataset = SplitSet(Path(args.dataset_dir), args.dataset, args.split, args.max_samples)
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    samples: list[dict[str, object]] = []
    maps = {mode: [] for mode in modes}
    det_count = 0
    images_with_det = 0
    for idx, name in enumerate(dataset.names):
        mask, size = dataset.mask(name)
        img0 = cv2.imread(str(dataset.image_path(name)), cv2.IMREAD_GRAYSCALE)
        if img0 is None:
            raise FileNotFoundError(name)
        inp, scale, pad = letterbox_gray(img0, args.image_size)
        tensor = torch.from_numpy(inp).to(device).float().view(1, 1, args.image_size, args.image_size) / 255.0
        pred = model(tensor)
        pred_tensor = pred[0] if isinstance(pred, (tuple, list)) else pred
        det = nms.non_max_suppression(pred_tensor, conf_thres=args.conf_thres, iou_thres=args.iou_thres, nc=1, max_det=args.max_det)[0]
        per_mode = {mode: np.zeros((size[0], size[1]), dtype=np.float32) for mode in modes}
        if det is not None and len(det):
            arr = det.detach().cpu().numpy()
            boxes = scale_boxes_xyxy(arr[:, :4], scale, pad, (size[0], size[1]))
            scores = arr[:, 4]
            images_with_det += 1
            det_count += int(len(arr))
            for box, score in zip(boxes, scores):
                for mode in modes:
                    add_box(per_mode[mode], box, float(score), mode)
        samples.append({"name": name, "size": size, "mask": mask})
        for mode in modes:
            maps[mode].append(per_mode[mode])
        if (idx + 1) % 50 == 0:
            print(f"{args.dataset} {args.split}: {idx + 1}/{len(dataset.names)}", flush=True)
    stats = {
        "images": float(len(dataset.names)),
        "images_with_det": float(images_with_det),
        "detections": float(det_count),
        "detections_per_image": float(det_count) / float(max(len(dataset.names), 1)),
    }
    return samples, maps, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PConv-SDloss YOLOv8n-p2 weights as detection heatmap candidates.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset", default="NUAA-SIRST")
    parser.add_argument("--split", default="test")
    parser.add_argument("--weight", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--conf_thres", type=float, default=0.01)
    parser.add_argument("--iou_thres", type=float, default=0.50)
    parser.add_argument("--max_det", type=int, default=300)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--thresholds", default="0.01,0.03,0.05,0.07,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60")
    parser.add_argument("--modes", default="dot1,dot2,dot3,gauss0.20,gauss0.33,gauss0.50,center0.33,center0.50")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    with tempfile.TemporaryDirectory(prefix="pconv_yolo_") as temp:
        weight = resolve_weight(Path(args.weight), Path(temp))
        samples, maps, stats = predict_maps(args, weight)
    results = []
    for mode in modes:
        rows = [{"threshold": th, **metric_from_maps(samples, maps[mode], th)} for th in thresholds]
        rows.sort(key=lambda row: row["mIoU"], reverse=True)
        best = rows[0]
        print(
            f"{Path(args.weight).name} {mode}: th={best['threshold']} mIoU={best['mIoU'] * 100:.2f} "
            f"Pd={best['Pd'] * 100:.2f} Fa={best['Fa'] * 1e6:.2f}",
            flush=True,
        )
        results.append({"mode": mode, "best": best, "rows": rows})
    Path(args.output).write_text(
        json.dumps({"weight": args.weight, "dataset": args.dataset, "split": args.split, "stats": stats, "results": results}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
