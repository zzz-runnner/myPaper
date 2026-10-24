#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SCTRANSNET = ROOT / "external" / "SCTransNet"
sys.path.insert(0, str(SCTRANSNET))

from dataset import TestSetLoader  # noqa: E402
from metrics import PD_FA, mIoU  # noqa: E402
from model.SCTransNet import SCTransNet  # noqa: E402
import model.Config as config  # noqa: E402


def normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module.model."):
            normalized[key[len("module.model."):]] = value
        elif key.startswith("model."):
            normalized[key[len("model."):]] = value
        elif key.startswith("module."):
            normalized[key[len("module."):]] = value
        else:
            normalized[key] = value
    return normalized


def samplewise_niou(pred: torch.Tensor, gt: torch.Tensor) -> float:
    pred_bin = (pred > 0.5).cpu().numpy().astype(np.uint8)
    gt_bin = (gt > 0).cpu().numpy().astype(np.uint8)
    scores = []
    for idx in range(pred_bin.shape[0]):
        inter = np.logical_and(pred_bin[idx], gt_bin[idx]).sum()
        union = np.logical_or(pred_bin[idx], gt_bin[idx]).sum()
        scores.append(float(inter) / float(union + np.spacing(1)))
    return float(np.mean(scores))


def evaluate(
    checkpoint: Path,
    dataset_dir: Path,
    train_dataset_name: str,
    test_dataset_name: str,
    threshold: float,
    device: torch.device,
) -> dict[str, float]:
    test_set = TestSetLoader(str(dataset_dir), train_dataset_name, test_dataset_name)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    net = SCTransNet(config.get_SCTrans_config(), mode="test", deepsuper=True).to(device)
    ckpt = torch.load(checkpoint, map_location="cpu")
    state_dict = normalize_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    net.load_state_dict(state_dict)
    net.eval()

    iou = mIoU()
    pdfa = PD_FA()
    niou_scores = []

    with torch.no_grad():
        for img, gt_mask, size, _ in test_loader:
            img = img.to(device)
            pred = net(img)
            if isinstance(pred, (tuple, list)):
                pred = pred[-1]
            pred = pred[:, :, : size[0], : size[1]]
            gt_mask = gt_mask[:, :, : size[0], : size[1]]

            iou.update((pred > threshold).cpu(), gt_mask.cpu())
            pdfa.update((pred[0, 0] > threshold).cpu(), gt_mask[0, 0], size)
            niou_scores.append(samplewise_niou(pred.cpu(), gt_mask.cpu()))

    pix_acc, miou = iou.get()
    pd, fa = pdfa.get()
    return {
        "pixAcc": float(pix_acc),
        "mIoU": float(miou),
        "nIoU": float(np.mean(niou_scores)),
        "Pd": float(pd),
        "Fa": float(fa),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a SCTransNet checkpoint on one or more datasets.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--train_dataset_name", required=True)
    parser.add_argument("--test_dataset_names", nargs="+", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    for name in args.test_dataset_names:
        metrics = evaluate(
            Path(args.checkpoint),
            Path(args.dataset_dir),
            args.train_dataset_name,
            name,
            args.threshold,
            device,
        )
        print(
            f"{name}: "
            f"mIoU={metrics['mIoU'] * 100:.2f}, "
            f"nIoU={metrics['nIoU'] * 100:.2f}, "
            f"Pd={metrics['Pd'] * 100:.2f}, "
            f"Fa={metrics['Fa'] * 1e6:.2f}, "
            f"pixAcc={metrics['pixAcc'] * 100:.2f}"
        )


if __name__ == "__main__":
    main()
