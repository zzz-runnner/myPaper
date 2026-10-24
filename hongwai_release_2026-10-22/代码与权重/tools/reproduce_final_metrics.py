#!/usr/bin/env python3
"""Evaluate only the frozen historical recipes; no threshold or weight search."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_component_fusion_learner import metric_from_maps
RECIPES = {
    "IRSTD-1K": ("tmp_anchor_irstd_no_tta_maps.npz", "maps_no_tta", "tmp_dfaware_official_irstd_maps.npz", "tmp_hfmnet_official_irstd_maps.npz", 0.400, 201),
    "NUAA-SIRST": ("tmp_lpsm_pair_test_maps.npz", "maps_lpsm_pair", "tmp_dfaware_test_maps.npz", "tmp_hfmnet_test_maps.npz", 0.4995, 214),
}
def load(path, key):
    # Historical NPZ files contain trusted object arrays; do not use untrusted files.
    with np.load(path, allow_pickle=True) as data:
        names = list(map(str, data["names"]))
        sizes = np.asarray(data["sizes"], dtype=np.int32)
        masks = [np.asarray(m, dtype=np.float32).squeeze() for m in data["masks"]]
        maps = [np.asarray(m, dtype=np.float32) for m in data[key]]
    if len(names) != len(set(names)) or not (len(names) == len(sizes) == len(masks) == len(maps)):
        raise ValueError(f"Invalid sample identities/counts: {path}")
    for name, size, mask, prob in zip(names, sizes, masks, maps):
        if mask.shape != tuple(size) or prob.shape != tuple(size):
            raise ValueError(f"Invalid shape for {path}:{name}")
        if not np.isfinite(prob).all() or prob.min() < 0 or prob.max() > 1:
            raise ValueError(f"Invalid probabilities for {path}:{name}")
    return names, sizes, masks, maps

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=[*RECIPES, "all"], default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "results/final_metrics.recomputed.json")
    args = parser.parse_args()
    reference = json.loads((ROOT / "results/final_metrics.reference.json").read_text())
    results = {}
    for ds in RECIPES if args.dataset == "all" else [args.dataset]:
        primary, key, df, hfm, threshold, expected_n = RECIPES[ds]
        print(f"Loading {ds} retained maps...", flush=True)
        names, sizes, masks, base = load(ROOT / "reports" / primary, key)
        if len(names) != expected_n:
            raise ValueError(f"{ds}: expected {expected_n} images, got {len(names)}")
        sources = [base]
        for filename in (df, hfm):
            other_names, other_sizes, other_masks, maps = load(ROOT / "reports" / filename, "maps_candidate")
            if names != other_names or not np.array_equal(sizes, other_sizes):
                raise ValueError(f"Sample order/size mismatch: {filename}")
            if any(not np.array_equal(a, b) for a, b in zip(masks, other_masks)):
                raise ValueError(f"Mask mismatch: {filename}")
            sources.append(maps)
        binaries = []
        for a, b, c in zip(*sources):
            prob = np.clip(0.876*a + 0.062*b + 0.062*c, 0.0, 1.0) ** 0.994
            binary = (prob > threshold).astype(np.uint8)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            keep = np.zeros(n, dtype=np.uint8)
            keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 3
            binaries.append(keep[labels].astype(np.float32))
        samples = [{"name": name, "size": size.tolist(), "mask": mask} for name, size, mask in zip(names, sizes, masks)]
        metric = metric_from_maps(samples, binaries, 0.5)
        tp = sum(int(((p > 0) & (m > 0)).sum()) for p, m in zip(binaries, masks))
        fp = sum(int(((p > 0) & (m <= 0)).sum()) for p, m in zip(binaries, masks))
        fn = sum(int(((p <= 0) & (m > 0)).sum()) for p, m in zip(binaries, masks))
        precision = tp / (tp + fp + 0.001)
        recall = tp / (tp + fn + 0.001)
        metric.update(precision=precision, recall=recall, F_measure=2*precision*recall/(precision+recall+0.00001))
        for field in ("mIoU", "nIoU", "F_measure", "Pd", "Fa"):
            if not np.isclose(metric[field], reference[ds][field], rtol=0, atol=1e-10):
                raise AssertionError(f"{ds} {field}: {metric[field]} != {reference[ds][field]}")
        results[ds] = {**metric, "images": len(names), "TP": tp, "FP": fp, "FN": fn, "reference_matches": True}
        print(f"{ds}: mIoU={100*metric['mIoU']:.4f}, F1={100*metric['F_measure']:.4f}; reference matches", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
if __name__ == "__main__":
    main()
