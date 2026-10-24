#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage import measure


ROOT = Path(__file__).resolve().parents[1]
SCTRANSNET = ROOT / "external" / "SCTransNet"
sys.path.insert(0, str(SCTRANSNET))

from dataset import TestSetLoader  # noqa: E402
from metrics import PD_FA, mIoU  # noqa: E402
from model.SCTransNet import SCTransNet  # noqa: E402
import model.Config as config  # noqa: E402

from train_our_method import TCDSCTransNetWrapper  # noqa: E402


def normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module.model."):
            normalized[key[len("module.model.") :]] = value
        elif key.startswith("model."):
            normalized[key[len("model.") :]] = value
        elif key.startswith("module."):
            normalized[key[len("module.") :]] = value
        else:
            normalized[key] = value
    return normalized


def parse_thresholds(spec: str) -> list[float]:
    values: list[float] = []
    for part in spec.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" in item:
            bits = [float(x) for x in item.split(":")]
            if len(bits) != 3:
                raise ValueError(f"Bad threshold range: {item}")
            start, stop, step = bits
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
            bits = [int(x) for x in item.split(":")]
            if len(bits) != 3:
                raise ValueError(f"Bad integer range: {item}")
            start, stop, step = bits
            values.extend(range(start, stop + 1, step))
        else:
            values.append(int(item))
    return sorted(set(values))


def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    base_net = SCTransNet(config.get_SCTrans_config(), mode="test", deepsuper=True)
    if ckpt_args.get("input_adapter", False) or ckpt_args.get("output_refiner", False):
        net = TCDSCTransNetWrapper(
            base_net,
            input_adapter=bool(ckpt_args.get("input_adapter", False)),
            adapter_hidden=int(ckpt_args.get("adapter_hidden", 8)),
            adapter_scale=float(ckpt_args.get("adapter_scale", 0.1)),
            output_refiner=bool(ckpt_args.get("output_refiner", False)),
            refiner_hidden=int(ckpt_args.get("refiner_hidden", 8)),
            refiner_scale=float(ckpt_args.get("refiner_scale", 0.1)),
        )
    else:
        net = base_net
    net = net.to(device)
    state_dict = normalize_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    model_keys = net.state_dict().keys()
    if any(key.startswith("model.") for key in model_keys):
        state_dict = {
            (key if key.startswith("adapter.") or key.startswith("model.") or key.startswith("output_refiner.") else f"model.{key}"): value
            for key, value in state_dict.items()
        }
    net.load_state_dict(state_dict, strict=False)
    net.eval()
    return net


def undo_aug(pred: torch.Tensor, aug_name: str) -> torch.Tensor:
    if aug_name == "none":
        return pred
    if aug_name == "hflip":
        return torch.flip(pred, dims=[-1])
    if aug_name == "vflip":
        return torch.flip(pred, dims=[-2])
    if aug_name == "transpose":
        return pred.transpose(-1, -2)
    raise ValueError(aug_name)


def apply_aug(img: torch.Tensor, aug_name: str) -> torch.Tensor:
    if aug_name == "none":
        return img
    if aug_name == "hflip":
        return torch.flip(img, dims=[-1])
    if aug_name == "vflip":
        return torch.flip(img, dims=[-2])
    if aug_name == "transpose":
        return img.transpose(-1, -2)
    raise ValueError(aug_name)


@torch.no_grad()
def collect_predictions(
    net: torch.nn.Module,
    dataset_dir: Path,
    train_dataset_name: str,
    test_dataset_name: str,
    device: torch.device,
    use_tta: bool,
) -> list[tuple[torch.Tensor, torch.Tensor, list[int]]]:
    test_set = TestSetLoader(str(dataset_dir), train_dataset_name, test_dataset_name)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    aug_names = ["none", "hflip", "vflip", "transpose"] if use_tta else ["none"]
    samples: list[tuple[torch.Tensor, torch.Tensor, list[int]]] = []

    for img, gt_mask, size, _name in test_loader:
        img = img.to(device)
        preds = []
        for aug_name in aug_names:
            pred = net(apply_aug(img, aug_name))
            if isinstance(pred, (tuple, list)):
                pred = pred[-1]
            pred = undo_aug(pred, aug_name)
            preds.append(pred)
        pred_avg = torch.stack(preds, dim=0).mean(dim=0)
        h, w = int(size[0]), int(size[1])
        pred_avg = pred_avg[:, :, :h, :w].cpu()
        gt_mask = gt_mask[:, :, :h, :w].cpu()
        samples.append((pred_avg, gt_mask, [h, w]))
    return samples


def samplewise_niou(pred_bin: torch.Tensor, gt: torch.Tensor) -> float:
    pred_np = pred_bin.cpu().numpy().astype(np.uint8)
    gt_np = (gt > 0).cpu().numpy().astype(np.uint8)
    scores = []
    for idx in range(pred_np.shape[0]):
        inter = np.logical_and(pred_np[idx], gt_np[idx]).sum()
        union = np.logical_or(pred_np[idx], gt_np[idx]).sum()
        scores.append(float(inter) / float(union + np.spacing(1)))
    return float(np.mean(scores))


def filter_components(pred_bin: torch.Tensor, min_area: int, max_area: int, max_height: int, max_width: int) -> torch.Tensor:
    if min_area <= 1 and max_area <= 0 and max_height <= 0 and max_width <= 0:
        return pred_bin
    pred_np = pred_bin.cpu().numpy().astype(np.uint8)
    filtered = np.zeros_like(pred_np)
    for b_idx in range(pred_np.shape[0]):
        for c_idx in range(pred_np.shape[1]):
            label_img = measure.label(pred_np[b_idx, c_idx] > 0, connectivity=2)
            for region in measure.regionprops(label_img):
                h = region.bbox[2] - region.bbox[0]
                w = region.bbox[3] - region.bbox[1]
                if region.area < min_area:
                    continue
                if max_area > 0 and region.area > max_area:
                    continue
                if max_height > 0 and h > max_height:
                    continue
                if max_width > 0 and w > max_width:
                    continue
                filtered[b_idx, c_idx][label_img == region.label] = 1
    return torch.from_numpy(filtered.astype(bool))


def evaluate_threshold(
    samples: list[tuple[torch.Tensor, torch.Tensor, list[int]]],
    threshold: float,
    min_area: int,
    max_area: int,
    max_height: int,
    max_width: int,
) -> dict[str, float]:
    iou_metric = mIoU()
    pdfa_metric = PD_FA()
    niou_scores = []
    for pred, gt_mask, size in samples:
        pred_bin = filter_components(pred > threshold, min_area, max_area, max_height, max_width)
        iou_metric.update(pred_bin, gt_mask)
        pdfa_metric.update(pred_bin[0, 0], gt_mask[0, 0], size)
        niou_scores.append(samplewise_niou(pred_bin, gt_mask))
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
        "nIoU": float(np.mean(niou_scores)),
        "Pd": float(pd),
        "Fa": float(fa),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep inference thresholds for a SCTransNet checkpoint. Report separately from default 0.5 metrics."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--train_dataset_name", required=True)
    parser.add_argument("--test_dataset_name", required=True)
    parser.add_argument("--thresholds", default="0.05:0.95:0.05")
    parser.add_argument("--min_areas", default="1")
    parser.add_argument("--max_areas", default="0", help="0 disables upper area filtering.")
    parser.add_argument("--max_heights", default="0", help="0 disables height filtering.")
    parser.add_argument("--max_widths", default="0", help="0 disables width filtering.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tta", action="store_true", help="Average none/hflip/vflip/transpose predictions.")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    thresholds = parse_thresholds(args.thresholds)
    min_areas = parse_ints(args.min_areas)
    max_areas = parse_ints(args.max_areas)
    max_heights = parse_ints(args.max_heights)
    max_widths = parse_ints(args.max_widths)
    net = load_model(Path(args.checkpoint), device)
    samples = collect_predictions(
        net=net,
        dataset_dir=Path(args.dataset_dir),
        train_dataset_name=args.train_dataset_name,
        test_dataset_name=args.test_dataset_name,
        device=device,
        use_tta=args.tta,
    )
    rows = [
        evaluate_threshold(samples, threshold, min_area, max_area, max_height, max_width)
        for threshold in thresholds
        for min_area in min_areas
        for max_area in max_areas
        for max_height in max_heights
        for max_width in max_widths
    ]
    rows.sort(key=lambda row: row["mIoU"], reverse=True)

    best = rows[0]
    print(
        "best "
        f"threshold={best['threshold']:.3f} "
        f"min_area={best['min_area']} "
        f"max_area={best['max_area']} "
        f"max_height={best['max_height']} "
        f"max_width={best['max_width']} "
        f"mIoU={best['mIoU'] * 100:.2f} "
        f"nIoU={best['nIoU'] * 100:.2f} "
        f"Pd={best['Pd'] * 100:.2f} "
        f"Fa={best['Fa'] * 1e6:.2f} "
        f"pixAcc={best['pixAcc'] * 100:.2f} "
        f"tta={args.tta}"
    )
    print("| threshold | min area | max area | max h | max w | mIoU | nIoU | Pd | Fa | pixAcc |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['threshold']:.3f} | "
            f"{row['min_area']} | "
            f"{row['max_area']} | "
            f"{row['max_height']} | "
            f"{row['max_width']} | "
            f"{row['mIoU'] * 100:.2f} | "
            f"{row['nIoU'] * 100:.2f} | "
            f"{row['Pd'] * 100:.2f} | "
            f"{row['Fa'] * 1e6:.2f} | "
            f"{row['pixAcc'] * 100:.2f} |"
        )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "train_dataset_name": args.train_dataset_name,
            "test_dataset_name": args.test_dataset_name,
            "tta": args.tta,
            "best": best,
            "rows": rows,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
