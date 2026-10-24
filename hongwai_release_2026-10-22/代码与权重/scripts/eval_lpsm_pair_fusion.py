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


def load_model(checkpoint: str, in_ch: int, base: int, stages: str, device: torch.device) -> LPSMAnchorResidualUNet:
    model = LPSMAnchorResidualUNet(in_ch=in_ch, base=base, state_stages=stages).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse two LPSM checkpoints and sweep threshold/component filters.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", default="NUAA-SIRST")
    parser.add_argument("--cache", default=str(ROOT / "reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__test.npz"))
    parser.add_argument("--teacher_mode", default="local_rule")
    parser.add_argument("--checkpoint_a", required=True)
    parser.add_argument("--base_a", type=int, default=56)
    parser.add_argument("--stages_a", default="e2,mid")
    parser.add_argument("--residual_a", type=float, default=0.06)
    parser.add_argument("--checkpoint_b", required=True)
    parser.add_argument("--base_b", type=int, default=64)
    parser.add_argument("--stages_b", default="e2,mid")
    parser.add_argument("--residual_b", type=float, default=0.055)
    parser.add_argument("--alphas", default="0.00,0.15,0.30,0.45,0.60,0.75,0.90,1.00")
    parser.add_argument("--gammas", default="1.00")
    parser.add_argument("--thresholds", default="0.496:0.506:0.001")
    parser.add_argument("--min_areas", default="1,2,3,4,5,6,8")
    parser.add_argument("--eval_max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dataset = AnchorTestSet(Path(args.dataset_dir), args.dataset_name, Path(args.cache), args.teacher_mode, args.eval_max_samples)
    feat0, _tlogit0, _mask0, _size0, _name0 = dataset[0]
    model_a = load_model(args.checkpoint_a, feat0.shape[0], args.base_a, args.stages_a, device)
    model_b = load_model(args.checkpoint_b, feat0.shape[0], args.base_b, args.stages_b, device)

    samples = []
    maps_a = []
    maps_b = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    with torch.no_grad():
        for feat, tlogit, mask, size, sample_name in loader:
            feat = feat.to(device)
            tlogit = tlogit.to(device)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                prob_a = torch.sigmoid(tlogit + args.residual_a * torch.tanh(model_a(feat)))[0, 0].detach().float().cpu().numpy()
                prob_b = torch.sigmoid(tlogit + args.residual_b * torch.tanh(model_b(feat)))[0, 0].detach().float().cpu().numpy()
            h, w = int(size[0]), int(size[1])
            maps_a.append(prob_a[:h, :w].astype(np.float32))
            maps_b.append(prob_b[:h, :w].astype(np.float32))
            samples.append({"name": str(sample_name[0]), "mask": mask.numpy()[0].astype(np.float32), "size": [h, w]})

    rows = []
    for alpha in parse_float_values(args.alphas):
        fused_base = [alpha * a + (1.0 - alpha) * b for a, b in zip(maps_a, maps_b)]
        for gamma in parse_float_values(args.gammas):
            fused = [np.clip(m, 0.0, 1.0) ** gamma for m in fused_base]
            for threshold in parse_float_values(args.thresholds):
                rows.append({"alpha_a": alpha, "gamma": gamma, "kind": "raw", "threshold": threshold, "min_area": 1, **metric_from_maps(samples, fused, threshold)})
                for min_area in parse_int_values(args.min_areas):
                    if min_area <= 1:
                        continue
                    filtered = [min_area_filter(prob, threshold, min_area) for prob in fused]
                    rows.append({"alpha_a": alpha, "gamma": gamma, "kind": "min_area", "threshold": threshold, "min_area": min_area, **metric_from_maps(samples, filtered, 0.5)})
    rows.sort(key=lambda r: r["mIoU"], reverse=True)
    best = rows[0]
    print(
        f"best alpha_a={best['alpha_a']} gamma={best['gamma']} kind={best['kind']} "
        f"th={best['threshold']} min_area={best['min_area']} mIoU={best['mIoU']*100:.4f} "
        f"nIoU={best['nIoU']*100:.4f} Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.4f}",
        flush=True,
    )
    Path(args.output).write_text(json.dumps({"args": vars(args), "best": best, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
