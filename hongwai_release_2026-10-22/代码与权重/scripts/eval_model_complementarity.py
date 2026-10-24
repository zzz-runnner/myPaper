#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))
sys.path.insert(0, str(ROOT / "scripts"))

from dataset import TestSetLoader  # noqa: E402
from metrics import PD_FA, mIoU  # noqa: E402
from train_hybrid_methods import load_model_checkpoint, make_model  # noqa: E402


def parse_model_spec(spec: str) -> tuple[str, str, str, str]:
    parts = spec.split("|")
    if len(parts) != 4:
        raise ValueError("Model spec must be name|model|base_checkpoint|checkpoint")
    return parts[0], parts[1], parts[2], parts[3]


@torch.no_grad()
def collect_model_preds(
    specs: list[tuple[str, str, str, str]],
    dataset_dir: Path,
    dataset_name: str,
    device: torch.device,
):
    dataset = TestSetLoader(str(dataset_dir), dataset_name, dataset_name)
    loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
    models = []
    for name, model_name, base_checkpoint, checkpoint in specs:
        model = make_model(model_name, warm_epoch=1, base_checkpoint=base_checkpoint).to(device)
        if hasattr(model, "initialize_refiner"):
            model.initialize_refiner()
        load_model_checkpoint(model, Path(checkpoint))
        if hasattr(model, "freeze_inherited_refiner"):
            model.freeze_inherited_refiner()
        model.eval()
        models.append((name, model))

    samples = []
    for image, mask, size, sample_name in loader:
        image = image.to(device)
        h, w = int(size[0]), int(size[1])
        preds = {}
        for name, model in models:
            pred = model(image, epoch=10**9)
            if isinstance(pred, (tuple, list)):
                pred = pred[-1]
            preds[name] = pred[:, :, :h, :w].cpu()
        samples.append({"name": sample_name[0], "mask": mask[:, :, :h, :w].cpu(), "size": [h, w], "preds": preds})
    return samples


def eval_maps(samples, maps: list[torch.Tensor], threshold: float) -> dict[str, float]:
    metric_iou = mIoU()
    metric_pdfa = PD_FA()
    for sample, pred in zip(samples, maps):
        pred_bin = pred > threshold
        mask = sample["mask"]
        metric_iou.update(pred_bin, mask)
        metric_pdfa.update(pred_bin[0, 0], mask[0, 0], sample["size"])
    pix_acc, miou = metric_iou.get()
    pd = float(metric_pdfa.PD) / float(max(metric_pdfa.target, 1))
    fa = float(metric_pdfa.dismatch_pixel) / float(max(metric_pdfa.all_pixel, 1))
    return {"mIoU": float(miou), "pixAcc": float(pix_acc), "Pd": float(pd), "Fa": float(fa)}


def sample_iou(pred_bin: torch.Tensor, mask: torch.Tensor) -> float:
    pred = pred_bin.bool()
    gt = mask > 0
    inter = (pred & gt).sum().item()
    union = (pred | gt).sum().item()
    return float(inter) / float(union + np.spacing(1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate complementarity between hybrid IRSTD checkpoints.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--model_spec", action="append", required=True, help="name|model|base_checkpoint|checkpoint")
    parser.add_argument("--thresholds", default="0.45,0.5")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    specs = [parse_model_spec(spec) for spec in args.model_spec]
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    samples = collect_model_preds(specs, Path(args.dataset_dir), args.dataset_name, device)
    names = [spec[0] for spec in specs]

    report: dict[str, object] = {"dataset": args.dataset_name, "models": names, "thresholds": {}}
    for threshold in thresholds:
        rows: dict[str, dict[str, float]] = {}
        for name in names:
            maps = [sample["preds"][name] for sample in samples]
            rows[name] = eval_maps(samples, maps, threshold)

        avg_maps = [sum(sample["preds"][name] for name in names) / len(names) for sample in samples]
        max_maps = [torch.stack([sample["preds"][name] for name in names], dim=0).max(dim=0).values for sample in samples]
        min_maps = [torch.stack([sample["preds"][name] for name in names], dim=0).min(dim=0).values for sample in samples]
        rows["avg_prob"] = eval_maps(samples, avg_maps, threshold)
        rows["max_prob"] = eval_maps(samples, max_maps, threshold)
        rows["min_prob"] = eval_maps(samples, min_maps, threshold)

        oracle_maps = []
        oracle_choice: dict[str, int] = {name: 0 for name in names}
        for sample in samples:
            best_name = names[0]
            best_iou = -1.0
            for name in names:
                iou = sample_iou(sample["preds"][name] > threshold, sample["mask"])
                if iou > best_iou:
                    best_iou = iou
                    best_name = name
            oracle_choice[best_name] += 1
            oracle_maps.append(sample["preds"][best_name])
        rows["sample_oracle"] = eval_maps(samples, oracle_maps, threshold)
        report["thresholds"][str(threshold)] = {"rows": rows, "oracle_choice": oracle_choice}

        print(f"threshold={threshold:.3f}", flush=True)
        for key, value in sorted(rows.items(), key=lambda item: item[1]["mIoU"], reverse=True):
            print(
                f"  {key}: mIoU={value['mIoU'] * 100:.2f} Pd={value['Pd'] * 100:.2f} "
                f"Fa={value['Fa'] * 1e6:.2f} pixAcc={value['pixAcc'] * 100:.2f}",
                flush=True,
            )
        print(f"  oracle_choice={oracle_choice}", flush=True)

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
