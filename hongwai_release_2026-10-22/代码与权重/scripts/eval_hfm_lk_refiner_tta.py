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

from eval_anchor_refiner_tta import apply_aug, parse_float_list, undo_aug  # noqa: E402
from eval_component_fusion_learner import metric_from_maps  # noqa: E402
from train_hfm_lk_anchor_refiner_from_cache import HFMLKAnchorResidualUNet  # noqa: E402
from train_teacher_anchor_refiner_from_cache import AnchorTestSet, parse_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate/export HFM-LK anchor-refiner maps with threshold sweep and flip TTA.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher_mode", default="strong_fusion")
    parser.add_argument("--base_channels", type=int, default=48)
    parser.add_argument("--residual_scale", type=float, default=0.32)
    parser.add_argument("--thresholds", default="0.42:0.62:0.005")
    parser.add_argument("--tta", default="none")
    parser.add_argument("--eval_max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--save_maps", default="")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dataset = AnchorTestSet(Path(args.dataset_dir), args.dataset_name, Path(args.cache), args.teacher_mode, args.eval_max_samples)
    feat0, _tlogit0, _mask0, _size0, _name0 = dataset[0]
    model = HFMLKAnchorResidualUNet(in_ch=feat0.shape[0], base=args.base_channels).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=True)
    model.eval()
    thresholds = parse_float_list(args.thresholds)
    aug_names = [x.strip() for x in parse_csv(args.tta) if x.strip()]

    samples = []
    maps_by_mode: dict[str, list[np.ndarray]] = {"no_tta": [], f"tta_{len(aug_names)}": []}
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    with torch.no_grad():
        for feat, tlogit, mask, size, sample_name in loader:
            feat = feat.to(device)
            tlogit = tlogit.to(device)
            probs = []
            for aug in aug_names:
                aug_feat, aug_tlogit = apply_aug(feat, tlogit, aug)
                residual = args.residual_scale * torch.tanh(model(aug_feat))
                prob = torch.sigmoid(aug_tlogit + residual)
                prob = undo_aug(prob, aug)
                probs.append(prob)
                if aug == "none":
                    h0, w0 = int(size[0]), int(size[1])
                    maps_by_mode["no_tta"].append(prob[0, 0, :h0, :w0].detach().cpu().numpy().astype(np.float32))
            avg_prob = torch.stack(probs, dim=0).mean(dim=0)
            h, w = int(size[0]), int(size[1])
            maps_by_mode[f"tta_{len(aug_names)}"].append(avg_prob[0, 0, :h, :w].detach().cpu().numpy().astype(np.float32))
            samples.append({"name": str(sample_name[0]), "mask": mask.numpy()[0].astype(np.float32), "size": [h, w]})

    rows = []
    for mode, maps in maps_by_mode.items():
        for threshold in thresholds:
            rows.append({"mode": mode, "threshold": threshold, **metric_from_maps(samples, maps, threshold)})
    rows.sort(key=lambda r: r["mIoU"], reverse=True)
    best = rows[0]
    print(
        f"best mode={best['mode']} th={best['threshold']} mIoU={best['mIoU']*100:.4f} "
        f"nIoU={best['nIoU']*100:.4f} Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.4f}",
        flush=True,
    )
    Path(args.output).write_text(
        json.dumps({"args": vars(args), "checkpoint_best": ckpt.get("best"), "best": best, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    if args.save_maps:
        payload: dict[str, object] = {
            "names": np.asarray([sample["name"] for sample in samples], dtype=object),
            "sizes": np.asarray([sample["size"] for sample in samples], dtype=np.int32),
            "masks": np.asarray([sample["mask"] for sample in samples], dtype=object),
        }
        for mode, maps in maps_by_mode.items():
            payload[f"maps_{mode}"] = np.asarray(maps, dtype=object)
        np.savez_compressed(args.save_maps, **payload)


if __name__ == "__main__":
    main()
