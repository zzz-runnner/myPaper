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
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))
sys.path.insert(0, str(ROOT / "external" / "BGM"))

from eval_component_fusion_learner import FullSplitSet, metric_from_maps  # noqa: E402
from net import Net  # noqa: E402


def normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        out[key] = value
    return out


def checkpoint_specs(root: Path) -> dict[str, Path]:
    base = root / "external" / "BGM" / "checkpoints_download" / "weights"
    return {
        "bgm_irstd": base / "BasicUNet_plus-2024-09-20-19-40-03-IRSTD-1K" / "best_mIoU_on_IRSTD-1K.pth.tar",
        "bgm_nuaa": base / "BasicUNet_plus-2024-09-20-21-17-03-NUAA-SIRST" / "best_mIoU_on_NUAA-SIRST.pth.tar",
        "bgm_nudt": base / "BasicUNet_plus-2024-09-20-21-47-11-NUDT-SIRST" / "best_mIoU_on_NUDT-SIRST.pth.tar",
    }


@torch.no_grad()
def predict_dataset(
    model: torch.nn.Module,
    dataset_dir: Path,
    norm_dataset: str,
    dataset_name: str,
    split: str,
    max_samples: int,
    device: torch.device,
) -> list[dict[str, object]]:
    dataset = FullSplitSet(dataset_dir, norm_dataset, dataset_name, split, max_samples=max_samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    samples: list[dict[str, object]] = []
    model.eval()
    for idx, (image, mask, size, name) in enumerate(loader):
        image = image.to(device)
        pred = model(image)
        height, width = int(size[0]), int(size[1])
        pred = pred[:, :, :height, :width]
        prob_np = pred[0, 0].detach().cpu().numpy().astype(np.float32)
        mask_np = mask[0, 0, :height, :width].numpy().astype(np.float32)
        samples.append({"name": name[0], "size": [height, width], "mask": mask_np, "preds": {"bgm": prob_np}})
        if (idx + 1) % 50 == 0:
            print(f"{norm_dataset}->{dataset_name} {split}: {idx + 1}/{len(dataset)}", flush=True)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate official BGM checkpoints on local prepared datasets.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--datasets", default="IRSTD-1K,NUAA-SIRST")
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoints", default="bgm_irstd,bgm_nuaa,bgm_nudt")
    parser.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    specs = checkpoint_specs(ROOT)
    results = []

    for label in [x.strip() for x in args.checkpoints.split(",") if x.strip()]:
        checkpoint = specs[label]
        ckpt = torch.load(checkpoint, map_location="cpu")
        model = Net("BasicUNet_plus")
        model.load_state_dict(normalize_state_dict(ckpt["state_dict"]), strict=True)
        model.to(device).eval()
        norm_dataset = {
            "bgm_irstd": "IRSTD-1K",
            "bgm_nuaa": "NUAA-SIRST",
            "bgm_nudt": "NUDT-SIRST",
        }[label]
        for dataset_name in [x.strip() for x in args.datasets.split(",") if x.strip()]:
            samples = predict_dataset(
                model,
                Path(args.dataset_dir),
                norm_dataset,
                dataset_name,
                args.split,
                args.max_samples,
                device,
            )
            maps = [sample["preds"]["bgm"] for sample in samples]
            rows = [{"threshold": th, **metric_from_maps(samples, maps, th)} for th in thresholds]
            rows.sort(key=lambda row: row["mIoU"], reverse=True)
            best = rows[0]
            print(
                f"{label} {dataset_name}: th={best['threshold']} "
                f"mIoU={best['mIoU'] * 100:.2f} nIoU={best['nIoU'] * 100:.2f} "
                f"Pd={best['Pd'] * 100:.2f} Fa={best['Fa'] * 1e6:.2f}",
                flush=True,
            )
            results.append(
                {
                    "checkpoint_label": label,
                    "checkpoint": str(checkpoint),
                    "norm_dataset": norm_dataset,
                    "dataset": dataset_name,
                    "split": args.split,
                    "best": best,
                    "rows": rows,
                }
            )

    Path(args.output).write_text(
        json.dumps(
            {
                "thresholds": thresholds,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
