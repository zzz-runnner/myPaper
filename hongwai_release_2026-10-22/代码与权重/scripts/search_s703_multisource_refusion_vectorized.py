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


def concat_object_arrays(items) -> np.ndarray:
    return np.concatenate([np.asarray(x, dtype=np.float32).reshape(-1) for x in list(items)], axis=0)


def detailed_metrics(
    fused_maps: list[np.ndarray],
    masks: list[np.ndarray],
    threshold: float,
    total_pixels: int,
) -> dict[str, float]:
    inter_total = 0.0
    union_total = 0.0
    niou = []
    target = 0
    detect = 0
    false_pixels = 0.0
    for pred, mask in zip(fused_maps, masks):
        p = pred > threshold
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
    return {
        "mIoU": inter_total / (union_total + np.spacing(1)),
        "nIoU": float(np.mean(niou)),
        "Pd": float(detect) / float(max(target, 1)),
        "Fa": false_pixels / float(total_pixels),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorized mIoU-first search for s703 multi-source refusion.")
    parser.add_argument("--s703_maps", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--map_key", default="maps_no_tta")
    parser.add_argument("--bgm_weights", default="-0.055,-0.05,-0.045,-0.04,-0.035,-0.03,-0.025")
    parser.add_argument("--rpc_weights", default="0.0,0.005,0.01,0.015,0.02,0.03")
    parser.add_argument("--weighted_weights", default="0.0,0.005,0.01,0.015,0.02,0.03")
    parser.add_argument("--bg_anom_weights", default="0.0,-0.005,-0.01,-0.015,-0.02")
    parser.add_argument("--gammas", default="0.91,0.92,0.93,0.94,0.95,0.96,0.97")
    parser.add_argument("--thresholds", default="0.464,0.468,0.472,0.476,0.480,0.484")
    parser.add_argument("--top_k", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data = np.load(args.s703_maps, allow_pickle=True)
    names = [str(x) for x in list(data["names"])]
    base_maps = [np.asarray(x, dtype=np.float32) for x in list(data[args.map_key])]
    masks = [np.asarray(x, dtype=np.float32) for x in list(data["masks"])]
    base = concat_object_arrays(base_maps)
    mask = concat_object_arrays(masks) > 0.5
    total_pixels = int(mask.size)
    mask_pos = int(mask.sum())

    cache_samples, _ = load_cache(Path(args.cache))
    cache_by_name = {str(sample["name"]): sample for sample in cache_samples}
    source_arrays = {}
    for source in ("bgm_nuaa", "rpcanet_sirst_s3", "weighted_raw", "bg_anomaly"):
        source_arrays[source] = np.concatenate(
            [cache_by_name[name]["preds"][source].astype(np.float32).reshape(-1) for name in names],
            axis=0,
        )

    bgm_weights = parse_values(args.bgm_weights)
    rpc_weights = parse_values(args.rpc_weights)
    weighted_weights = parse_values(args.weighted_weights)
    bg_anom_weights = parse_values(args.bg_anom_weights)
    gammas = parse_values(args.gammas)
    thresholds = parse_values(args.thresholds)

    top_rows: list[dict[str, float | str]] = []

    def push_row(row: dict[str, float | str]) -> None:
        top_rows.append(row)
        top_rows.sort(key=lambda item: float(item["mIoU_fast"]), reverse=True)
        if len(top_rows) > args.top_k:
            del top_rows[args.top_k :]

    def score_flat(kind: str, w_bgm: float, w_rpc: float, w_weighted: float, w_bg: float, gamma: float) -> None:
        base_scale = 1.0 - max(0.0, w_rpc) - max(0.0, w_weighted)
        fused = base_scale * base
        if w_bgm:
            fused = fused + w_bgm * source_arrays["bgm_nuaa"]
        if w_rpc:
            fused = fused + w_rpc * source_arrays["rpcanet_sirst_s3"]
        if w_weighted:
            fused = fused + w_weighted * source_arrays["weighted_raw"]
        if w_bg:
            fused = fused + w_bg * source_arrays["bg_anomaly"]
        fused = np.clip(fused, 0.0, 1.0)
        if gamma != 1.0:
            fused = np.power(fused, gamma, dtype=np.float32)
        for threshold in thresholds:
            pred = fused > threshold
            tp = int(np.logical_and(pred, mask).sum())
            fp = int(np.logical_and(pred, ~mask).sum())
            fn = mask_pos - tp
            miou = tp / float(tp + fp + fn + np.spacing(1))
            push_row(
                {
                    "kind": kind,
                    "w_bgm": w_bgm,
                    "w_rpc": w_rpc,
                    "w_weighted": w_weighted,
                    "w_bg_anom": w_bg,
                    "gamma": gamma,
                    "threshold": threshold,
                    "mIoU_fast": miou,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }
            )

    score_flat("s703", 0.0, 0.0, 0.0, 0.0, 1.0)
    combos = itertools.product(bgm_weights, rpc_weights, weighted_weights, bg_anom_weights, gammas)
    for w_bgm, w_rpc, w_weighted, w_bg, gamma in combos:
        if max(0.0, w_rpc) + max(0.0, w_weighted) > 0.06:
            continue
        score_flat("multisource", w_bgm, w_rpc, w_weighted, w_bg, gamma)

    # Recompute exact nIoU/Pd/Fa only for the retained top rows.
    detailed_rows = []
    for row in top_rows:
        w_bgm = float(row["w_bgm"])
        w_rpc = float(row["w_rpc"])
        w_weighted = float(row["w_weighted"])
        w_bg = float(row["w_bg_anom"])
        gamma = float(row["gamma"])
        base_scale = 1.0 - max(0.0, w_rpc) - max(0.0, w_weighted)
        fused_maps = []
        for idx, bmap in enumerate(base_maps):
            sample = cache_by_name[names[idx]]
            fused = base_scale * bmap
            if w_bgm:
                fused = fused + w_bgm * sample["preds"]["bgm_nuaa"].astype(np.float32)
            if w_rpc:
                fused = fused + w_rpc * sample["preds"]["rpcanet_sirst_s3"].astype(np.float32)
            if w_weighted:
                fused = fused + w_weighted * sample["preds"]["weighted_raw"].astype(np.float32)
            if w_bg:
                fused = fused + w_bg * sample["preds"]["bg_anomaly"].astype(np.float32)
            fused = np.clip(fused, 0.0, 1.0)
            if gamma != 1.0:
                fused = np.power(fused, gamma, dtype=np.float32)
            fused_maps.append(fused.astype(np.float32))
        exact = detailed_metrics(fused_maps, masks, float(row["threshold"]), total_pixels)
        detailed_rows.append({**row, **exact})

    detailed_rows.sort(key=lambda item: float(item["mIoU"]), reverse=True)
    best = detailed_rows[0]
    print(
        f"best kind={best['kind']} w_bgm={best['w_bgm']} w_rpc={best['w_rpc']} "
        f"w_weighted={best['w_weighted']} w_bg_anom={best['w_bg_anom']} gamma={best['gamma']} "
        f"th={best['threshold']} mIoU={best['mIoU']*100:.4f} nIoU={best['nIoU']*100:.4f} "
        f"Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.4f}",
        flush=True,
    )
    Path(args.output).write_text(
        json.dumps({"args": vars(args), "best": best, "top_rows": detailed_rows}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
