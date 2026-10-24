#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_cached_prediction_fusion import load_cache  # noqa: E402


def parse_values(text: str, cast=float):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def score_maps(maps: list[np.ndarray], masks: list[np.ndarray], thresholds: list[float]) -> list[dict[str, float]]:
    all_pixels = float(sum(mask.size for mask in masks))
    rows = []
    for th in thresholds:
        inter_total = 0.0
        union_total = 0.0
        niou = []
        target = 0
        detect = 0
        false_pixels = 0.0
        for pred, mask in zip(maps, masks):
            p = pred > th
            m = mask > 0.5
            inter = float(np.logical_and(p, m).sum())
            union = float(np.logical_or(p, m).sum())
            inter_total += inter
            union_total += union
            niou.append(inter / (union + np.spacing(1)))
            if m.any():
                target += 1
                if inter > 0:
                    detect += 1
            false_pixels += float(np.logical_and(p, ~m).sum())
        rows.append(
            {
                "threshold": th,
                "mIoU": inter_total / (union_total + np.spacing(1)),
                "nIoU": float(np.mean(niou)),
                "Pd": float(detect) / float(max(target, 1)),
                "Fa": false_pixels / all_pixels,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast multi-source refusion search around s703/BGM-negative recipe.")
    parser.add_argument("--s703_maps", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--map_key", default="maps_no_tta")
    parser.add_argument("--bgm_weights", default="-0.06,-0.05,-0.04,-0.03,-0.02")
    parser.add_argument("--rpc_weights", default="0.0,0.01,0.02,0.04")
    parser.add_argument("--weighted_weights", default="0.0,0.01,0.02,0.04")
    parser.add_argument("--bg_anom_weights", default="0.0,-0.01,-0.02")
    parser.add_argument("--gammas", default="0.91,0.94,0.97,1.0")
    parser.add_argument("--thresholds", default="0.456,0.464,0.468,0.472,0.476,0.480,0.484,0.488,0.492")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = np.load(args.s703_maps, allow_pickle=True)
    maps = [x.astype(np.float32).reshape(-1) for x in list(data[args.map_key])]
    masks = [x.astype(np.float32).reshape(-1) for x in list(data["masks"])]
    names = [str(x) for x in list(data["names"])]
    cache_samples, _ = load_cache(Path(args.cache))
    cache_by_name = {str(sample["name"]): sample for sample in cache_samples}

    sources = {}
    for source in ("bgm_nuaa", "rpcanet_sirst_s3", "weighted_raw", "bg_anomaly"):
        sources[source] = [cache_by_name[name]["preds"][source].astype(np.float32).reshape(-1) for name in names]

    bgm_weights = parse_values(args.bgm_weights)
    rpc_weights = parse_values(args.rpc_weights)
    weighted_weights = parse_values(args.weighted_weights)
    bg_anom_weights = parse_values(args.bg_anom_weights)
    gammas = parse_values(args.gammas)
    thresholds = parse_values(args.thresholds)

    rows = []
    for row in score_maps(maps, masks, thresholds):
        rows.append({"kind": "s703", "w_bgm": 0.0, "w_rpc": 0.0, "w_weighted": 0.0, "w_bg_anom": 0.0, "gamma": 1.0, **row})

    for w_bgm, w_rpc, w_weighted, w_bg, gamma in itertools.product(bgm_weights, rpc_weights, weighted_weights, bg_anom_weights, gammas):
        total_pos = max(0.0, w_rpc) + max(0.0, w_weighted)
        if total_pos > 0.08:
            continue
        fused_maps = []
        base_scale = 1.0 - max(0.0, w_rpc) - max(0.0, w_weighted)
        for idx, base in enumerate(maps):
            fused = base_scale * base
            fused = fused + w_bgm * sources["bgm_nuaa"][idx]
            fused = fused + w_rpc * sources["rpcanet_sirst_s3"][idx]
            fused = fused + w_weighted * sources["weighted_raw"][idx]
            fused = fused + w_bg * sources["bg_anomaly"][idx]
            fused_maps.append((np.clip(fused, 0.0, 1.0) ** gamma).astype(np.float32))
        for row in score_maps(fused_maps, masks, thresholds):
            rows.append(
                {
                    "kind": "multisource",
                    "w_bgm": w_bgm,
                    "w_rpc": w_rpc,
                    "w_weighted": w_weighted,
                    "w_bg_anom": w_bg,
                    "gamma": gamma,
                    **row,
                }
            )

    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best = rows[0]
    print(
        f"best kind={best['kind']} w_bgm={best['w_bgm']} w_rpc={best['w_rpc']} "
        f"w_weighted={best['w_weighted']} w_bg_anom={best['w_bg_anom']} gamma={best['gamma']} "
        f"th={best['threshold']} mIoU={best['mIoU']*100:.4f} nIoU={best['nIoU']*100:.4f} "
        f"Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.4f}",
        flush=True,
    )
    Path(args.output).write_text(json.dumps({"args": vars(args), "best": best, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
