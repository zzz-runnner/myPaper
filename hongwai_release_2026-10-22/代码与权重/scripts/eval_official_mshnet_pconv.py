#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
SCTRANSNET = ROOT / "external" / "SCTransNet"
MSHNET_MODEL = ROOT / "external" / "MSHNet" / "model" / "MSHNet.py"
APCONV_MODEL = ROOT / "external" / "PConv-SDloss-Data" / "model" / "APConv.py"
sys.path.insert(0, str(SCTRANSNET))

from dataset import TestSetLoader  # noqa: E402
from metrics import PD_FA, mIoU  # noqa: E402


def load_symbol(path: Path, module_name: str, symbol: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, symbol)


MSHNetCore = load_symbol(MSHNET_MODEL, "mshnet_eval_file", "MSHNet")
ChannelAttention = load_symbol(MSHNET_MODEL, "mshnet_eval_file_ca", "ChannelAttention")
SpatialAttention = load_symbol(MSHNET_MODEL, "mshnet_eval_file_sa", "SpatialAttention")
OfficialPConv = load_symbol(APCONV_MODEL, "pconv_eval_file", "PConv")


class P43FirstBlockResNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        use_pconv = in_channels == 16 and out_channels == 16
        if use_pconv:
            self.conv1 = OfficialPConv(in_channels, out_channels, k=4, s=stride)
            self.conv2 = OfficialPConv(out_channels, out_channels, k=3, s=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        return self.relu(out)


def normalize_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in state.items():
        if key.startswith("module."):
            out[key[len("module.") :]] = value
        else:
            out[key] = value
    return out


def load_official_weight(model: torch.nn.Module, weight: Path) -> None:
    obj = torch.load(weight, map_location="cpu")
    if isinstance(obj, dict) and "net" in obj:
        state = obj["net"]
    elif isinstance(obj, dict) and "state_dict" in obj:
        state = obj["state_dict"]
    else:
        state = obj
    state = normalize_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"missing_keys={len(missing)}", flush=True)
    if unexpected:
        print(f"unexpected_keys={len(unexpected)}", flush=True)


def samplewise_niou(pred_bin: torch.Tensor, gt: torch.Tensor) -> float:
    pred_np = pred_bin.cpu().numpy().astype(np.uint8)
    gt_np = (gt > 0).cpu().numpy().astype(np.uint8)
    scores = []
    for idx in range(pred_np.shape[0]):
        inter = np.logical_and(pred_np[idx], gt_np[idx]).sum()
        union = np.logical_or(pred_np[idx], gt_np[idx]).sum()
        scores.append(float(inter) / float(union + np.spacing(1)))
    return float(np.mean(scores))


@torch.no_grad()
def collect_probs(
    model: torch.nn.Module,
    dataset_dir: Path,
    dataset_name: str,
    device: torch.device,
    warm_flag: bool,
) -> list[dict[str, object]]:
    dataset = TestSetLoader(str(dataset_dir), dataset_name, dataset_name)
    loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
    model.eval()
    samples: list[dict[str, object]] = []
    for image, mask, size, sample_name in loader:
        image = image.to(device)
        image3 = image.repeat(1, 3, 1, 1)
        masks, logits = model(image3, warm_flag)
        pred = torch.sigmoid(logits)
        h, w = int(size[0]), int(size[1])
        samples.append(
            {
                "name": sample_name[0],
                "pred": pred[:, :, :h, :w].cpu(),
                "mask": mask[:, :, :h, :w].cpu(),
                "size": [h, w],
                "aux_outputs": len(masks),
            }
        )
    return samples


def open_pair(dataset_root: Path, name: str) -> tuple[Image.Image, Image.Image]:
    for ext in (".png", ".bmp", ".jpg"):
        img_path = dataset_root / "images" / f"{name}{ext}"
        mask_path = dataset_root / "masks" / f"{name}{ext}"
        if img_path.exists() and mask_path.exists():
            return Image.open(img_path).convert("RGB"), Image.open(mask_path)
    raise FileNotFoundError(name)


@torch.no_grad()
def collect_probs_official_preprocess(
    model: torch.nn.Module,
    dataset_dir: Path,
    dataset_name: str,
    device: torch.device,
    warm_flag: bool,
    base_size: int,
) -> list[dict[str, object]]:
    dataset_root = dataset_dir / dataset_name
    names = (dataset_root / "img_idx" / f"test_{dataset_name}.txt").read_text().splitlines()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    model.eval()
    samples: list[dict[str, object]] = []
    for name in names:
        image_pil, mask_pil = open_pair(dataset_root, name)
        width, height = image_pil.size
        image = image_pil.resize((base_size, base_size), Image.BILINEAR)
        image_tensor = transform(image).unsqueeze(0).to(device)
        masks, logits = model(image_tensor, warm_flag)
        pred = torch.sigmoid(logits)
        pred = F.interpolate(pred, size=(height, width), mode="bilinear", align_corners=False)
        mask = np.array(mask_pil, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).view(1, 1, height, width).float()
        samples.append(
            {
                "name": name,
                "pred": pred.cpu(),
                "mask": mask_tensor,
                "size": [height, width],
                "aux_outputs": len(masks),
            }
        )
    return samples


def evaluate_samples(samples: list[dict[str, object]], threshold: float) -> dict[str, float]:
    metric_iou = mIoU()
    metric_pdfa = PD_FA()
    niou_scores = []
    losses = []
    for sample in samples:
        pred = sample["pred"]
        mask = sample["mask"]
        size = sample["size"]
        pred_bin = pred > threshold
        metric_iou.update(pred_bin, mask)
        metric_pdfa.update(pred_bin[0, 0], mask[0, 0], size)
        niou_scores.append(samplewise_niou(pred_bin, mask))
        losses.append(float(torch.nn.functional.binary_cross_entropy(pred.clamp(1e-5, 1.0 - 1e-5), mask)))
    pix_acc, miou = metric_iou.get()
    pd = float(metric_pdfa.PD) / float(max(metric_pdfa.target, 1))
    fa = float(metric_pdfa.dismatch_pixel) / float(max(metric_pdfa.all_pixel, 1))
    return {
        "loss": float(np.mean(losses)),
        "pixAcc": float(pix_acc),
        "mIoU": float(miou),
        "nIoU": float(np.mean(niou_scores)),
        "Pd": float(pd),
        "Fa": float(fa),
    }


def parse_thresholds(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate official PConv/MSHNet weights in the local SCTransNet metric path.")
    parser.add_argument("--weight", required=True)
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60")
    parser.add_argument("--warm_flags", default="false,true")
    parser.add_argument("--model_variant", choices=["mshnet", "p43"], default="mshnet")
    parser.add_argument("--preprocess", choices=["sctransnet", "official_mshnet"], default="sctransnet")
    parser.add_argument("--base_size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    if args.model_variant == "p43":
        model = MSHNetCore(3, block=P43FirstBlockResNet).to(device)
    else:
        model = MSHNetCore(3).to(device)
    load_official_weight(model, Path(args.weight))

    thresholds = parse_thresholds(args.thresholds)
    warm_flags = [item.strip().lower() in {"1", "true", "yes", "warm"} for item in args.warm_flags.split(",")]
    report: dict[str, object] = {
        "weight": str(Path(args.weight)),
        "dataset": args.dataset_name,
        "preprocess": args.preprocess,
        "thresholds": thresholds,
        "warm_flags": {},
    }
    for warm_flag in warm_flags:
        if args.preprocess == "official_mshnet":
            samples = collect_probs_official_preprocess(
                model, Path(args.dataset_dir), args.dataset_name, device, warm_flag, args.base_size
            )
        else:
            samples = collect_probs(model, Path(args.dataset_dir), args.dataset_name, device, warm_flag)
        rows = {}
        print(f"dataset={args.dataset_name} warm_flag={warm_flag} samples={len(samples)}", flush=True)
        for threshold in thresholds:
            metrics = evaluate_samples(samples, threshold)
            rows[str(threshold)] = metrics
            print(
                f"  th={threshold:.2f} mIoU={metrics['mIoU'] * 100:.2f} "
                f"nIoU={metrics['nIoU'] * 100:.2f} Pd={metrics['Pd'] * 100:.2f} "
                f"Fa={metrics['Fa'] * 1e6:.2f} pixAcc={metrics['pixAcc'] * 100:.2f}",
                flush=True,
            )
        report["warm_flags"][str(warm_flag)] = rows

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
