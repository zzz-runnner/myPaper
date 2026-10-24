#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_component_fusion_learner import metric_from_maps  # noqa: E402
from train_lpsm_anchor_refiner_from_cache import LPSMAnchorResidualUNet  # noqa: E402
from train_teacher_anchor_refiner_from_cache import AnchorTestSet, parse_csv  # noqa: E402


def parse_float_values(value: str) -> list[float]:
    if ":" in value:
        start, stop, step = [float(x) for x in value.split(":")]
        vals = []
        cur = start
        while cur <= stop + 1e-9:
            vals.append(round(cur, 6))
            cur += step
        return vals
    return [float(x) for x in parse_csv(value)]


def parse_int_values(value: str) -> list[int]:
    return [int(x) for x in parse_csv(value)]


def min_area_filter(prob: np.ndarray, threshold: float, min_area: int) -> np.ndarray:
    binary = (prob > threshold).astype(np.uint8)
    if min_area <= 1:
        return binary.astype(np.float32)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros_like(binary)
    for label in range(1, num):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            out[labels == label] = 1
    return out.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LPSM checkpoint with threshold and component min-area sweep.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", default="NUAA-SIRST")
    parser.add_argument("--cache", default=str(ROOT / "reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__test.npz"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher_mode", default="local_rule")
    parser.add_argument("--base_channels", type=int, default=56)
    parser.add_argument("--state_stages", default="e2,mid")
    parser.add_argument("--residual_scale", type=float, default=0.06)
    parser.add_argument("--thresholds", default="0.492:0.508:0.002")
    parser.add_argument("--min_areas", default="1,2,3,4,5,6")
    parser.add_argument("--eval_max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dataset = AnchorTestSet(Path(args.dataset_dir), args.dataset_name, Path(args.cache), args.teacher_mode, args.eval_max_samples)
    feat0, _tlogit0, _mask0, _size0, _name0 = dataset[0]
    model = LPSMAnchorResidualUNet(in_ch=feat0.shape[0], base=args.base_channels, state_stages=args.state_stages).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    model.eval()

    samples = []
    maps = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    with torch.no_grad():
        for feat, tlogit, mask, size, sample_name in loader:
            feat = feat.to(device)
            tlogit = tlogit.to(device)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                residual = args.residual_scale * torch.tanh(model(feat))
                prob = torch.sigmoid(tlogit + residual)[0, 0].detach().float().cpu().numpy()
            h, w = int(size[0]), int(size[1])
            maps.append(prob[:h, :w].astype(np.float32))
            samples.append({"name": str(sample_name[0]), "mask": mask.numpy()[0].astype(np.float32), "size": [h, w]})

    rows = []
    for threshold in parse_float_values(args.thresholds):
        rows.append({"kind": "raw", "threshold": threshold, "min_area": 1, **metric_from_maps(samples, maps, threshold)})
        for min_area in parse_int_values(args.min_areas):
            if min_area <= 1:
                continue
            filtered = [min_area_filter(prob, threshold, min_area) for prob in maps]
            rows.append({"kind": "min_area", "threshold": threshold, "min_area": min_area, **metric_from_maps(samples, filtered, 0.5)})
    rows.sort(key=lambda r: r["mIoU"], reverse=True)
    best = rows[0]
    print(
        f"best kind={best['kind']} th={best['threshold']} min_area={best['min_area']} "
        f"mIoU={best['mIoU']*100:.4f} nIoU={best['nIoU']*100:.4f} "
        f"Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.4f}",
        flush=True,
    )
    Path(args.output).write_text(
        json.dumps({"args": vars(args), "checkpoint_best": ckpt.get("best"), "best": best, "rows": rows}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
