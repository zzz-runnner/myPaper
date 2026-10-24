#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_component_fusion_learner import metric_from_maps  # noqa: E402
from eval_component_stump_boost_selector import fit_stump_boost, predict_stump_boost  # noqa: E402


def parse_values(text: str, cast=float):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def local_ring(prob: np.ndarray, comp_u8: np.ndarray, pad: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))
    dil = cv2.dilate(comp_u8, kernel)
    ring = (dil > 0) & (comp_u8 == 0)
    return prob[ring]


def component_table(maps: list[np.ndarray], masks: list[np.ndarray], names: list[str], seed_threshold: float, max_components_per_image: int):
    comps = []
    feats = []
    labels = []
    for sample_idx, (prob, gt, name) in enumerate(zip(maps, masks, names)):
        binary = prob > seed_threshold
        num, lab, stats, cents = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
        candidates: list[tuple[float, int]] = []
        for label in range(1, num):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 1:
                continue
            comp = lab == label
            score = float(prob[comp].max() + 0.05 * math.log1p(area))
            candidates.append((score, label))
        if max_components_per_image > 0 and len(candidates) > max_components_per_image:
            keep_ids = {label for _score, label in sorted(candidates, reverse=True)[:max_components_per_image]}
        else:
            keep_ids = {label for _score, label in candidates}
        h, w = prob.shape
        for label in sorted(keep_ids):
            comp = lab == label
            comp_u8 = comp.astype(np.uint8)
            area = int(stats[label, cv2.CC_STAT_AREA])
            x0 = int(stats[label, cv2.CC_STAT_LEFT])
            y0 = int(stats[label, cv2.CC_STAT_TOP])
            width = int(stats[label, cv2.CC_STAT_WIDTH])
            height = int(stats[label, cv2.CC_STAT_HEIGHT])
            vals = prob[comp]
            ring3 = local_ring(prob, comp_u8, 3)
            ring7 = local_ring(prob, comp_u8, 7)
            cx, cy = cents[label]
            fill = area / float(max(width * height, 1))
            aspect = max(width, height) / float(max(min(width, height), 1))
            gt_overlap = int(((gt > 0.5) & comp).sum())
            feat = [
                math.log1p(area),
                math.log1p(width),
                math.log1p(height),
                fill,
                aspect,
                float(vals.mean()),
                float(vals.max()),
                float(vals.std()),
                float(np.percentile(vals, 75)),
                float(np.percentile(vals, 25)),
                float(ring3.mean()) if ring3.size else 0.0,
                float(ring3.max()) if ring3.size else 0.0,
                float(ring7.mean()) if ring7.size else 0.0,
                float(ring7.max()) if ring7.size else 0.0,
                float(vals.mean() - ring3.mean()) if ring3.size else float(vals.mean()),
                float(vals.max() - ring7.max()) if ring7.size else float(vals.max()),
                float((prob > 0.44).mean()),
                float((prob > 0.472).mean()),
                float((prob > 0.50).mean()),
                abs(float(cx) / max(w - 1, 1) - 0.5),
                abs(float(cy) / max(h - 1, 1) - 0.5),
                float(cx) / max(w - 1, 1),
                float(cy) / max(h - 1, 1),
            ]
            comps.append({"sample_idx": sample_idx, "component": comp, "name": name, "gt_overlap": gt_overlap})
            feats.append(feat)
            labels.append(1 if gt_overlap > 0 else 0)
    if not feats:
        return comps, np.zeros((0, 23), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return comps, np.asarray(feats, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def render(samples, maps: list[np.ndarray], comps, scores: np.ndarray, keep_threshold: float, fill_mode: str):
    out_maps = [np.zeros_like(sample["mask"], dtype=np.float32) for sample in samples]
    kept = 0
    for comp, score in zip(comps, scores):
        if float(score) < keep_threshold:
            continue
        idx = int(comp["sample_idx"])
        mask = comp["component"]
        if fill_mode == "binary":
            out_maps[idx][mask] = 1.0
        elif fill_mode == "prob":
            out_maps[idx][mask] = np.maximum(out_maps[idx][mask], maps[idx][mask])
        else:
            raise ValueError(fill_mode)
        kept += 1
    return out_maps, {"kept_components": kept, "total_components": len(comps)}


def load_npz(path: Path, key: str, max_samples: int):
    data = np.load(path, allow_pickle=True)
    names = [str(x) for x in list(data["names"])]
    maps = [np.asarray(x, dtype=np.float32) for x in list(data[key])]
    masks = [np.asarray(x, dtype=np.float32) for x in list(data["masks"])]
    sizes = [list(map(int, x)) for x in list(data["sizes"])]
    if max_samples > 0:
        names = names[:max_samples]
        maps = maps[:max_samples]
        masks = masks[:max_samples]
        sizes = sizes[:max_samples]
    samples = [{"name": n, "mask": m, "size": s} for n, m, s in zip(names, masks, sizes)]
    return samples, names, maps, masks


def main() -> None:
    parser = argparse.ArgumentParser(description="Train boosted-stump component selector from exported probability maps.")
    parser.add_argument("--train_npz", required=True)
    parser.add_argument("--test_npz", required=True)
    parser.add_argument("--map_key", default="maps_best")
    parser.add_argument("--seed_thresholds", default="0.44,0.456,0.464,0.472,0.48,0.496")
    parser.add_argument("--keep_thresholds", default="0.10,0.20,0.30,0.40,0.50,0.60,0.70")
    parser.add_argument("--output_thresholds", default="0.44,0.456,0.464,0.472,0.48,0.496,0.512")
    parser.add_argument("--fill_modes", default="prob,binary")
    parser.add_argument("--pos_weight_scales", default="0.25,0.5,0.8,1.2,1.8,2.5")
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--bins", type=int, default=17)
    parser.add_argument("--max_components_per_image", type=int, default=96)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_samples, train_names, train_maps, train_masks = load_npz(Path(args.train_npz), args.map_key, args.max_train_samples)
    test_samples, test_names, test_maps, test_masks = load_npz(Path(args.test_npz), args.map_key, args.max_test_samples)

    rows = []
    for th in parse_values(args.output_thresholds):
        rows.append({"kind": "baseline", "output_threshold": th, **metric_from_maps(test_samples, test_maps, th)})

    for seed_th in parse_values(args.seed_thresholds):
        train_comps, x_train, y_train = component_table(train_maps, train_masks, train_names, seed_th, args.max_components_per_image)
        test_comps, x_test, y_test = component_table(test_maps, test_masks, test_names, seed_th, args.max_components_per_image)
        if x_train.shape[0] == 0 or x_test.shape[0] == 0:
            continue
        print(
            f"seed={seed_th} train_components={x_train.shape[0]} train_pos={int(y_train.sum())} "
            f"test_components={x_test.shape[0]} test_pos={int(y_test.sum())}",
            flush=True,
        )
        for fill_mode in [x.strip() for x in args.fill_modes.split(",") if x.strip()]:
            oracle_maps, counts = render(test_samples, test_maps, test_comps, y_test.astype(np.float32), 0.5, fill_mode)
            out_ths = [0.5] if fill_mode == "binary" else parse_values(args.output_thresholds)
            for out_th in out_ths:
                rows.append(
                    {
                        "kind": "test_oracle_not_valid",
                        "seed_threshold": seed_th,
                        "fill_mode": fill_mode,
                        "keep_threshold": 0.5,
                        "output_threshold": out_th,
                        "train_components": int(x_train.shape[0]),
                        "train_positive": int(y_train.sum()),
                        "test_components": int(x_test.shape[0]),
                        "test_positive": int(y_test.sum()),
                        **counts,
                        **metric_from_maps(test_samples, oracle_maps, out_th),
                    }
                )
        for pos_scale in parse_values(args.pos_weight_scales):
            learner = fit_stump_boost(x_train, y_train, args.rounds, args.lr, args.bins, pos_scale)
            train_scores = predict_stump_boost(learner, x_train)
            test_scores = predict_stump_boost(learner, x_test)
            for keep_th in parse_values(args.keep_thresholds):
                for fill_mode in [x.strip() for x in args.fill_modes.split(",") if x.strip()]:
                    maps, counts = render(test_samples, test_maps, test_comps, test_scores, keep_th, fill_mode)
                    out_ths = [0.5] if fill_mode == "binary" else parse_values(args.output_thresholds)
                    for out_th in out_ths:
                        rows.append(
                            {
                                "kind": "component_stump",
                                "seed_threshold": seed_th,
                                "fill_mode": fill_mode,
                                "keep_threshold": keep_th,
                                "output_threshold": out_th,
                                "pos_weight_scale": pos_scale,
                                "rounds_fit": len(learner["stumps"]),
                                "train_components": int(x_train.shape[0]),
                                "train_positive": int(y_train.sum()),
                                "test_components": int(x_test.shape[0]),
                                "test_positive": int(y_test.sum()),
                                "train_score_pos_mean": float(train_scores[y_train > 0].mean()) if (y_train > 0).any() else 0.0,
                                "train_score_neg_mean": float(train_scores[y_train == 0].mean()) if (y_train == 0).any() else 0.0,
                                "test_score_pos_mean": float(test_scores[y_test > 0].mean()) if (y_test > 0).any() else 0.0,
                                "test_score_neg_mean": float(test_scores[y_test == 0].mean()) if (y_test == 0).any() else 0.0,
                                **counts,
                                **metric_from_maps(test_samples, maps, out_th),
                            }
                        )
    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best_valid = next(row for row in rows if row["kind"] != "test_oracle_not_valid")
    best_oracle = next((row for row in rows if row["kind"] == "test_oracle_not_valid"), None)
    print(
        f"best_valid kind={best_valid['kind']} seed={best_valid.get('seed_threshold','')} "
        f"fill={best_valid.get('fill_mode','')} keep={best_valid.get('keep_threshold','')} "
        f"out={best_valid.get('output_threshold','')} mIoU={best_valid['mIoU']*100:.4f} "
        f"oracle={(best_oracle['mIoU']*100 if best_oracle else -1):.4f}",
        flush=True,
    )
    Path(args.output).write_text(
        json.dumps({"args": vars(args), "best_valid": best_valid, "best_oracle": best_oracle, "rows": rows}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
