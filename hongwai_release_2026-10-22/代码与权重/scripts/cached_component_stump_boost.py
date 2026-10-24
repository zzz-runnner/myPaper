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
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))

from eval_component_fusion_learner import metric_from_maps  # noqa: E402
from eval_component_stump_boost_selector import fit_stump_boost, predict_stump_boost  # noqa: E402
from inmemory_cached_weight_search import load_cache_dense  # noqa: E402


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def load_many_cache(paths: list[Path]):
    all_samples = []
    all_preds = []
    names_ref: list[str] | None = None
    for path in paths:
        samples, names, preds, _masks = load_cache_dense(path)
        if names_ref is None:
            names_ref = names
        elif names != names_ref:
            raise ValueError(f"name mismatch in {path}: {names} != {names_ref}")
        all_samples.extend(samples)
        all_preds.extend(preds)
    if names_ref is None:
        raise RuntimeError("no caches")
    return all_samples, names_ref, all_preds


def map_stack(stack: np.ndarray, mode: str, names: list[str]) -> np.ndarray:
    if mode == "avg":
        return stack.mean(axis=0).astype(np.float32)
    if mode == "max":
        return stack.max(axis=0).astype(np.float32)
    if mode in names:
        return stack[names.index(mode)].astype(np.float32)
    raise ValueError(f"bad map mode: {mode}")


def component_table(samples, preds, names: list[str], seed_threshold: float, seed_mode: str, max_components_per_image: int):
    rows: list[dict[str, object]] = []
    labels: list[int] = []
    features: list[list[float]] = []
    for sample_idx, (sample, stack) in enumerate(zip(samples, preds)):
        mask_gt = sample["mask"].astype(bool)
        seed_map = map_stack(stack, seed_mode, names)
        seed = seed_map > seed_threshold
        num_labels, label_img, stats, centroids = cv2.connectedComponentsWithStats(seed.astype(np.uint8), 8)
        candidates: list[tuple[float, int]] = []
        for label_id in range(1, num_labels):
            comp = label_img == label_id
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < 1:
                continue
            score = float(seed_map[comp].max() + 0.05 * math.log1p(area))
            candidates.append((score, label_id))
        if max_components_per_image > 0 and len(candidates) > max_components_per_image:
            keep_ids = {label_id for _score, label_id in sorted(candidates, reverse=True)[:max_components_per_image]}
        else:
            keep_ids = {label_id for _score, label_id in candidates}
        for label_id in sorted(keep_ids):
            comp = label_img == label_id
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            x0 = int(stats[label_id, cv2.CC_STAT_LEFT])
            y0 = int(stats[label_id, cv2.CC_STAT_TOP])
            width = int(stats[label_id, cv2.CC_STAT_WIDTH])
            height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            x1, y1 = x0 + width, y0 + height
            probs = stack[:, comp]
            model_means = probs.mean(axis=1)
            model_max = probs.max(axis=1)
            model_std = probs.std(axis=1)
            avg = stack.mean(axis=0)
            maxp = stack.max(axis=0)
            minp = stack.min(axis=0)
            cx, cy = centroids[label_id]
            fill = area / float(max(width * height, 1))
            gt_overlap = int((mask_gt & comp).sum())
            feature = [
                math.log1p(area),
                math.log1p(width),
                math.log1p(height),
                fill,
                float(avg[comp].mean()),
                float(avg[comp].max()),
                float(maxp[comp].mean()),
                float(maxp[comp].max()),
                float(minp[comp].mean()),
                float(seed_map[comp].mean()),
                float(seed_map[comp].max()),
                float(((stack[:, comp] > seed_threshold).any(axis=1)).sum()) / float(max(len(names), 1)),
                abs(float(cx) / max(mask_gt.shape[1] - 1, 1) - 0.5),
                abs(float(cy) / max(mask_gt.shape[0] - 1, 1) - 0.5),
            ]
            feature.extend(model_means.astype(float).tolist())
            feature.extend(model_max.astype(float).tolist())
            feature.extend(model_std.astype(float).tolist())
            rows.append({"sample_idx": sample_idx, "component": comp, "bbox": [y0, x0, y1, x1], "gt_overlap": gt_overlap})
            labels.append(1 if gt_overlap > 0 else 0)
            features.append(feature)
    if not features:
        return rows, np.zeros((0, 14 + 3 * len(names)), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return rows, np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def render(samples, preds, names: list[str], comps, scores: np.ndarray, keep_threshold: float, fill_mode: str):
    maps = [np.zeros_like(sample["mask"], dtype=np.float32) for sample in samples]
    total_kept = 0
    for comp, score in zip(comps, scores):
        if float(score) < keep_threshold:
            continue
        sample_idx = int(comp["sample_idx"])
        mask = comp["component"]
        stack = preds[sample_idx]
        if fill_mode == "avg":
            maps[sample_idx][mask] = np.maximum(maps[sample_idx][mask], stack.mean(axis=0)[mask])
        elif fill_mode == "max":
            maps[sample_idx][mask] = np.maximum(maps[sample_idx][mask], stack.max(axis=0)[mask])
        elif fill_mode in names:
            maps[sample_idx][mask] = np.maximum(maps[sample_idx][mask], stack[names.index(fill_mode)][mask])
        elif fill_mode == "binary":
            maps[sample_idx][mask] = 1.0
        else:
            raise ValueError(fill_mode)
        total_kept += 1
    return maps, {"kept_components": total_kept, "total_components": len(comps)}


def baseline_maps(samples, preds, names: list[str], mode: str):
    return [map_stack(stack, mode, names) for stack in preds]


def main() -> None:
    parser = argparse.ArgumentParser(description="Cached full-data boosted-stump component selector.")
    parser.add_argument("--train_cache", action="append", required=True)
    parser.add_argument("--test_cache", required=True)
    parser.add_argument("--seed_thresholds", default="0.35,0.40,0.42,0.45")
    parser.add_argument("--seed_modes", default="avg,max")
    parser.add_argument("--fill_modes", default="avg,max,binary")
    parser.add_argument("--keep_thresholds", default="0.10,0.20,0.30,0.40,0.50,0.60")
    parser.add_argument("--output_thresholds", default="0.35,0.40,0.42,0.45,0.50")
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.30)
    parser.add_argument("--bins", type=int, default=17)
    parser.add_argument("--pos_weight_scales", default="0.25,0.50,0.80,1.20")
    parser.add_argument("--max_components_per_image", type=int, default=96)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_samples, names, train_preds = load_many_cache([Path(x) for x in args.train_cache])
    test_samples, test_names, test_preds, _test_masks = load_cache_dense(Path(args.test_cache))
    if test_names != names:
        raise ValueError(f"test names mismatch: {test_names} != {names}")
    if args.max_train_samples > 0:
        train_samples = train_samples[: args.max_train_samples]
        train_preds = train_preds[: args.max_train_samples]
    if args.max_test_samples > 0:
        test_samples = test_samples[: args.max_test_samples]
        test_preds = test_preds[: args.max_test_samples]

    rows: list[dict[str, object]] = []
    output_thresholds = parse_float_list(args.output_thresholds)
    for mode in [*names, "avg", "max"]:
        maps = baseline_maps(test_samples, test_preds, names, mode)
        for out_th in output_thresholds:
            rows.append({"kind": "baseline", "mode": mode, "output_threshold": out_th, **metric_from_maps(test_samples, maps, out_th)})

    for seed_mode in [x.strip() for x in args.seed_modes.split(",") if x.strip()]:
        for seed_th in parse_float_list(args.seed_thresholds):
            print(f"building tables seed_mode={seed_mode} seed={seed_th}", flush=True)
            train_comps, x_train, y_train = component_table(
                train_samples, train_preds, names, seed_th, seed_mode, args.max_components_per_image
            )
            test_comps, x_test, y_test = component_table(
                test_samples, test_preds, names, seed_th, seed_mode, args.max_components_per_image
            )
            if x_train.shape[0] == 0:
                continue
            print(
                f"components train={x_train.shape[0]} pos={int(y_train.sum())} "
                f"test={x_test.shape[0]} pos={int(y_test.sum())}",
                flush=True,
            )
            for fill_mode in [x.strip() for x in args.fill_modes.split(",") if x.strip()]:
                oracle_scores = y_test.astype(np.float32)
                oracle_maps, oracle_count = render(test_samples, test_preds, names, test_comps, oracle_scores, 0.5, fill_mode)
                out_th = 0.5 if fill_mode == "binary" else seed_th
                rows.append(
                    {
                        "kind": "test_oracle_not_valid",
                        "seed_mode": seed_mode,
                        "seed_threshold": seed_th,
                        "fill_mode": fill_mode,
                        "keep_threshold": 0.5,
                        "output_threshold": out_th,
                        "train_components": int(x_train.shape[0]),
                        "train_positive": int(y_train.sum()),
                        "test_components": int(x_test.shape[0]),
                        "test_positive": int(y_test.sum()),
                        **oracle_count,
                        **metric_from_maps(test_samples, oracle_maps, out_th),
                    }
                )
            for pos_scale in parse_float_list(args.pos_weight_scales):
                learner = fit_stump_boost(x_train, y_train, args.rounds, args.lr, args.bins, pos_scale)
                train_scores = predict_stump_boost(learner, x_train)
                test_scores = predict_stump_boost(learner, x_test)
                for fill_mode in [x.strip() for x in args.fill_modes.split(",") if x.strip()]:
                    out_ths = [0.5] if fill_mode == "binary" else output_thresholds
                    for keep_th in parse_float_list(args.keep_thresholds):
                        maps, counts = render(test_samples, test_preds, names, test_comps, test_scores, keep_th, fill_mode)
                        for out_th in out_ths:
                            rows.append(
                                {
                                    "kind": "cached_stump_boost",
                                    "seed_mode": seed_mode,
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
        f"best_valid kind={best_valid['kind']} mode={best_valid.get('mode', '')} "
        f"seed_mode={best_valid.get('seed_mode', '')} seed={best_valid.get('seed_threshold', '')} "
        f"fill={best_valid.get('fill_mode', '')} keep={best_valid.get('keep_threshold', '')} "
        f"out={best_valid.get('output_threshold', '')} mIoU={best_valid['mIoU']*100:.2f} "
        f"Pd={best_valid['Pd']*100:.2f} Fa={best_valid['Fa']*1e6:.2f}",
        flush=True,
    )
    if best_oracle is not None:
        print(
            f"best_oracle_not_valid seed_mode={best_oracle.get('seed_mode', '')} "
            f"seed={best_oracle.get('seed_threshold', '')} fill={best_oracle.get('fill_mode', '')} "
            f"mIoU={best_oracle['mIoU']*100:.2f} Pd={best_oracle['Pd']*100:.2f} Fa={best_oracle['Fa']*1e6:.2f}",
            flush=True,
        )
    Path(args.output).write_text(
        json.dumps(
            {
                "train_caches": args.train_cache,
                "test_cache": args.test_cache,
                "models": names,
                "selector": "cached_boosted_decision_stumps",
                "best_valid": best_valid,
                "best_oracle_not_valid": best_oracle,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
