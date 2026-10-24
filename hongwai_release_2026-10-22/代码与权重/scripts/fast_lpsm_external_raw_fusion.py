#!/usr/bin/env python
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_obj_npz(path: Path, key: str):
    data = np.load(path, allow_pickle=True)
    names = [str(x) for x in list(data["names"])]
    maps = [np.asarray(x, dtype=np.float32).reshape(-1) for x in list(data[key])]
    masks = [np.asarray(x, dtype=bool).reshape(-1) for x in list(data["masks"])]
    return names, maps, masks


def metric(maps, masks, threshold: float):
    inter = 0.0
    union = 0.0
    niou = []
    target = 0
    detect = 0
    false_pixels = 0.0
    all_pixels = float(sum(m.size for m in masks))
    for prob, mask in zip(maps, masks):
        pred = prob > threshold
        i = float(np.logical_and(pred, mask).sum())
        u = float(np.logical_or(pred, mask).sum())
        inter += i
        union += u
        niou.append(i / (u + np.spacing(1)))
        if mask.any():
            target += 1
            if i > 0:
                detect += 1
        false_pixels += float(np.logical_and(pred, ~mask).sum())
    return {
        "mIoU": inter / (union + np.spacing(1)),
        "nIoU": float(np.mean(niou)),
        "Pd": detect / float(max(target, 1)),
        "Fa": false_pixels / all_pixels,
    }


def main() -> None:
    specs = [
        ("lpsm", ROOT / "reports/tmp_lpsm_pair_test_maps.npz", "maps_lpsm_pair"),
        ("dfaware", ROOT / "reports/tmp_dfaware_test_maps.npz", "maps_candidate"),
        ("hfmnet", ROOT / "reports/tmp_hfmnet_test_maps.npz", "maps_candidate"),
        ("s703", ROOT / "reports/tmp_s703_test_maps.npz", "maps_no_tta"),
    ]
    labels = [x[0] for x in specs]
    ref_names = None
    masks = None
    sources = []
    for label, path, key in specs:
        names, maps, cur_masks = load_obj_npz(path, key)
        if ref_names is None:
            ref_names = names
            masks = cur_masks
        elif names != ref_names:
            raise ValueError(f"name mismatch for {label}")
        sources.append(maps)
    assert masks is not None

    candidate_extras = {(0.0, 0.0, 0.0)}
    for idx in range(3):
        for value in [-0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.06]:
            row = [0.0, 0.0, 0.0]
            row[idx] = value
            candidate_extras.add(tuple(row))
    for i, j in itertools.combinations(range(3), 2):
        for value_i in [-0.02, 0.02, 0.04]:
            for value_j in [-0.02, 0.02, 0.04]:
                row = [0.0, 0.0, 0.0]
                row[i] = value_i
                row[j] = value_j
                candidate_extras.add(tuple(row))
    gammas = [0.996, 0.998, 1.0, 1.002]
    thresholds = [0.499, 0.500, 0.5005, 0.501, 0.5015, 0.502, 0.503]

    rows = []
    for extras in sorted(candidate_extras):
        if sum(abs(x) for x in extras) > 0.10:
            continue
        w0 = 1.0 - sum(extras)
        if w0 <= 0.0 or w0 > 1.15:
            continue
        weights = (w0, *extras)
        base_maps = []
        for per_image in zip(*sources):
            fused = np.zeros_like(per_image[0], dtype=np.float32)
            for weight, prob in zip(weights, per_image):
                fused += float(weight) * prob
            base_maps.append(np.clip(fused, 0.0, 1.0))
        for gamma in gammas:
            maps = [x**gamma for x in base_maps]
            for threshold in thresholds:
                rows.append({"weights": weights, "gamma": gamma, "threshold": threshold, **metric(maps, masks, threshold)})
    rows.sort(key=lambda x: x["mIoU"], reverse=True)
    out = {"labels": labels, "best": rows[0], "rows": rows[:80]}
    path = ROOT / "reports/nuaa_lpsm_external_sources_raw_fast.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    best = rows[0]
    print(
        f"best weights={best['weights']} gamma={best['gamma']} th={best['threshold']} "
        f"mIoU={best['mIoU']*100:.4f} nIoU={best['nIoU']*100:.4f} "
        f"Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
