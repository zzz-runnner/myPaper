#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_cached_prediction_fusion import load_cache  # noqa: E402


def parse_values(text: str, cast=float):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def metrics_from_flat(preds: list[np.ndarray], masks: list[np.ndarray], thresholds: list[float]) -> list[dict[str, float]]:
    rows = []
    all_pixels = float(sum(mask.size for mask in masks))
    for th in thresholds:
        inter_total = 0.0
        union_total = 0.0
        vals = []
        target = 0
        detect = 0
        false_pixels = 0
        for pred, mask in zip(preds, masks):
            pred_bin = pred > th
            mask_bin = mask > 0.5
            inter = float(np.logical_and(pred_bin, mask_bin).sum())
            union = float(np.logical_or(pred_bin, mask_bin).sum())
            inter_total += inter
            union_total += union
            vals.append(inter / (union + np.spacing(1)))
            if mask_bin.any():
                target += 1
                if inter > 0:
                    detect += 1
            false_pixels += float(np.logical_and(pred_bin, ~mask_bin).sum())
        rows.append(
            {
                "threshold": th,
                "mIoU": inter_total / (union_total + np.spacing(1)),
                "nIoU": float(np.mean(vals)),
                "Pd": float(detect) / float(max(target, 1)),
                "Fa": false_pixels / all_pixels,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast threshold-only refusion search for s703 maps and corrected cache sources.")
    parser.add_argument("--s703_maps", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--map_key", default="maps_no_tta")
    parser.add_argument("--source_names", default="weighted_raw,strong_fusion,bgm_nuaa,rpcanet_sirst_s3,bg_anomaly")
    parser.add_argument("--alphas", default="-0.08,-0.04,-0.02,0.01,0.02,0.04,0.08,0.12")
    parser.add_argument("--gammas", default="0.94,0.97,1.0,1.03,1.06")
    parser.add_argument("--thresholds", default="0.440,0.448,0.456,0.462,0.466,0.468,0.470,0.472,0.476,0.480,0.488,0.496,0.504,0.512,0.520")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = np.load(args.s703_maps, allow_pickle=True)
    s703 = [x.astype(np.float32).reshape(-1) for x in list(data[args.map_key])]
    masks = [x.astype(np.float32).reshape(-1) for x in list(data["masks"])]
    names = [str(x) for x in list(data["names"])]
    cache_samples, _ = load_cache(Path(args.cache))
    cache_by_name = {str(sample["name"]): sample for sample in cache_samples}
    source_names = [x.strip() for x in args.source_names.split(",") if x.strip()]
    alphas = parse_values(args.alphas, float)
    gammas = parse_values(args.gammas, float)
    thresholds = parse_values(args.thresholds, float)

    rows = []
    for row in metrics_from_flat(s703, masks, thresholds):
        rows.append({"kind": "s703", "source": "", "alpha": 0.0, "gamma": 1.0, **row})

    for source in source_names:
        src = [cache_by_name[name]["preds"][source].astype(np.float32).reshape(-1) for name in names]
        for alpha in alphas:
            for gamma in gammas:
                maps = []
                for base, smap in zip(s703, src):
                    if alpha < 0:
                        fused = np.clip((1.0 - abs(alpha)) * base + alpha * smap, 0.0, 1.0)
                    else:
                        fused = np.clip((1.0 - alpha) * base + alpha * smap, 0.0, 1.0)
                    maps.append((fused**gamma).astype(np.float32))
                for row in metrics_from_flat(maps, masks, thresholds):
                    rows.append({"kind": "refusion", "source": source, "alpha": alpha, "gamma": gamma, **row})

    rows.sort(key=lambda item: item["mIoU"], reverse=True)
    best = rows[0]
    print(
        f"best kind={best['kind']} source={best['source']} alpha={best['alpha']} gamma={best['gamma']} "
        f"th={best['threshold']} mIoU={best['mIoU']*100:.4f} nIoU={best['nIoU']*100:.4f} "
        f"Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.4f}",
        flush=True,
    )
    Path(args.output).write_text(json.dumps({"args": vars(args), "best": best, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
