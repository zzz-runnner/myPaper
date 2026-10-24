#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_lpsm_anchor_refiner_from_cache import LPSMAnchorResidualUNet  # noqa: E402
from train_teacher_anchor_refiner_from_cache import AnchorTestSet  # noqa: E402


def load_model(checkpoint: str, in_ch: int, base: int, stages: str, device: torch.device) -> LPSMAnchorResidualUNet:
    model = LPSMAnchorResidualUNet(in_ch=in_ch, base=base, state_stages=stages).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fused probability maps from the retained LPSM pair.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", default="NUAA-SIRST")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--teacher_mode", default="local_rule")
    parser.add_argument("--checkpoint_a", required=True)
    parser.add_argument("--base_a", type=int, default=56)
    parser.add_argument("--stages_a", default="e2,mid")
    parser.add_argument("--residual_a", type=float, default=0.06)
    parser.add_argument("--checkpoint_b", required=True)
    parser.add_argument("--base_b", type=int, default=64)
    parser.add_argument("--stages_b", default="e2,mid")
    parser.add_argument("--residual_b", type=float, default=0.055)
    parser.add_argument("--alpha_a", type=float, default=0.45)
    parser.add_argument("--gamma", type=float, default=0.998)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dataset = AnchorTestSet(Path(args.dataset_dir), args.dataset_name, Path(args.cache), args.teacher_mode, 0)
    feat0, _tlogit0, _mask0, _size0, _name0 = dataset[0]
    model_a = load_model(args.checkpoint_a, feat0.shape[0], args.base_a, args.stages_a, device)
    model_b = load_model(args.checkpoint_b, feat0.shape[0], args.base_b, args.stages_b, device)

    names: list[str] = []
    sizes: list[list[int]] = []
    masks: list[np.ndarray] = []
    maps_pair: list[np.ndarray] = []
    maps_a: list[np.ndarray] = []
    maps_b: list[np.ndarray] = []

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    with torch.no_grad():
        for feat, tlogit, mask, size, sample_name in loader:
            feat = feat.to(device)
            tlogit = tlogit.to(device)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                prob_a = torch.sigmoid(tlogit + args.residual_a * torch.tanh(model_a(feat)))[0, 0].detach().float().cpu().numpy()
                prob_b = torch.sigmoid(tlogit + args.residual_b * torch.tanh(model_b(feat)))[0, 0].detach().float().cpu().numpy()
            h, w = int(size[0]), int(size[1])
            prob_a = prob_a[:h, :w].astype(np.float32)
            prob_b = prob_b[:h, :w].astype(np.float32)
            fused = np.clip(args.alpha_a * prob_a + (1.0 - args.alpha_a) * prob_b, 0.0, 1.0) ** args.gamma
            names.append(str(sample_name[0]))
            sizes.append([h, w])
            masks.append(mask.numpy()[0].astype(np.float32))
            maps_a.append(prob_a)
            maps_b.append(prob_b)
            maps_pair.append(fused.astype(np.float32))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        names=np.asarray(names, dtype=object),
        sizes=np.asarray(sizes, dtype=np.int32),
        masks=np.asarray(masks, dtype=object),
        maps_lpsm_pair=np.asarray(maps_pair, dtype=object),
        maps_lpsm_a=np.asarray(maps_a, dtype=object),
        maps_lpsm_b=np.asarray(maps_b, dtype=object),
    )
    print(f"wrote {args.output} samples={len(names)}", flush=True)


if __name__ == "__main__":
    main()
