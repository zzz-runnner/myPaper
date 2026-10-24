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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "LRRNet"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))

from models.LRRNet import UNet  # noqa: E402
from eval_component_fusion_learner import FullSplitSet, metric_from_maps  # noqa: E402


@torch.no_grad()
def predict_dataset(
    model: torch.nn.Module,
    dataset_dir: Path,
    dataset_name: str,
    split: str,
    device: torch.device,
    max_samples: int,
    input_size: int,
) -> list[dict[str, object]]:
    dataset = FullSplitSet(dataset_dir, dataset_name, dataset_name, split, max_samples=max_samples)
    loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
    samples: list[dict[str, object]] = []
    for idx, (image, mask, size, name) in enumerate(loader):
        height, width = int(size[0]), int(size[1])
        image_np = image[0, 0, :height, :width].numpy().astype(np.float32)
        inp = cv2.resize(image_np, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
        inp_t = torch.from_numpy(inp[None, None]).float().to(device)
        _d, target_logits = model(inp_t)
        prob = torch.sigmoid(target_logits)
        prob = F.interpolate(prob, size=(height, width), mode="bilinear", align_corners=False)
        prob_np = prob[0, 0].detach().cpu().numpy().astype(np.float32)
        mask_np = mask[0, 0, :height, :width].numpy().astype(np.float32)
        samples.append({"name": name[0], "size": [height, width], "mask": mask_np, "preds": {"lrrnet": prob_np}})
        if (idx + 1) % 50 == 0:
            print(f"{dataset_name} {split}: {idx + 1}/{len(dataset)}", flush=True)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the official LRRNet checkpoint on local prepared datasets.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--datasets", default="IRSTD-1K,NUAA-SIRST")
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint", default=str(ROOT / "external" / "LRRNet" / "checkpoints" / "IRDST_1K_miou_0.7239356450020329.pkl"))
    parser.add_argument("--thresholds", default="0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90")
    parser.add_argument("--input_size", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = UNet(img_channels=1, base_channels=16, channel_mults=(1, 2, 4, 8), num_res_blocks=2, dropout=0.2, norm="gn", num_groups=16)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    results = []
    for dataset_name in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        samples = predict_dataset(
            model,
            Path(args.dataset_dir),
            dataset_name,
            args.split,
            device,
            args.max_samples,
            args.input_size,
        )
        maps = [sample["preds"]["lrrnet"] for sample in samples]
        rows = []
        for threshold in thresholds:
            rows.append({"threshold": threshold, **metric_from_maps(samples, maps, threshold)})
        rows.sort(key=lambda r: r["mIoU"], reverse=True)
        best = rows[0]
        print(
            f"{dataset_name}: best th={best['threshold']} mIoU={best['mIoU']*100:.2f} "
            f"Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.2f}",
            flush=True,
        )
        results.append({"dataset": dataset_name, "split": args.split, "best": best, "rows": rows})

    Path(args.output).write_text(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "input_size": args.input_size,
                "thresholds": thresholds,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
