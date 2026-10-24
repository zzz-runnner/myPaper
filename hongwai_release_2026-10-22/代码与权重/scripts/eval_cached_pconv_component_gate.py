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

from eval_component_fusion_learner import fit_logistic, metric_from_maps, predict_logistic  # noqa: E402


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_cache(path: Path) -> tuple[list[dict[str, object]], list[str]]:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta_json"]))
    names = json.loads(str(data["names_json"]))
    samples: list[dict[str, object]] = []
    for idx, item in enumerate(meta):
        height, width = [int(x) for x in item["size"]]
        mask = data[f"mask_{idx}"].astype(np.float32)[:height, :width]
        preds = {
            name: data[f"pred_{idx}_{name}"].astype(np.float32)[:height, :width]
            for name in names
        }
        samples.append({"name": item["name"], "size": [height, width], "mask": mask, "preds": preds})
    return samples, names


def load_many(paths: list[Path]) -> tuple[list[dict[str, object]], list[str]]:
    all_samples: list[dict[str, object]] = []
    ref_names: list[str] | None = None
    for path in paths:
        samples, names = load_cache(path)
        if ref_names is None:
            ref_names = names
        elif names != ref_names:
            raise ValueError(f"Cache model names differ in {path}: {names} != {ref_names}")
        all_samples.extend(samples)
    return all_samples, ref_names or []


def component_rows(
    sample: dict[str, object],
    names: list[str],
    seed_threshold: float,
    min_area: int,
    max_area: int,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    preds = sample["preds"]
    mask = sample["mask"] > 0
    stack = np.stack([preds[name] for name in names], axis=0)
    avg = stack.mean(axis=0)
    maxp = stack.max(axis=0)
    minp = stack.min(axis=0)
    seed = avg > seed_threshold
    num_labels, label_img, stats, centroids = cv2.connectedComponentsWithStats(seed.astype(np.uint8), 8)
    rows: list[dict[str, object]] = []
    feats: list[list[float]] = []
    labels: list[int] = []
    height_all, width_all = mask.shape
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_area or (max_area > 0 and area > max_area):
            continue
        x0 = int(stats[label_id, cv2.CC_STAT_LEFT])
        y0 = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        x1 = x0 + width
        y1 = y0 + height
        comp = label_img == label_id
        comp_stack = stack[:, comp]
        comp_avg = avg[comp]
        comp_maxp = maxp[comp]
        comp_minp = minp[comp]
        model_means = comp_stack.mean(axis=1)
        model_maxes = comp_stack.max(axis=1)
        model_votes = (comp_stack > seed_threshold).mean(axis=1)
        vote_models = float(((comp_stack > seed_threshold).any(axis=1)).sum())
        cx, cy = centroids[label_id]
        fill = float(area) / max(float(width * height), 1.0)
        aspect = float(max(width, height)) / max(float(min(width, height)), 1.0)
        border = min(x0, y0, width_all - x1, height_all - y1) / max(float(max(height_all, width_all)), 1.0)
        gt_overlap = int((mask & comp).sum())
        p43_idx = names.index("pconv_p43") if "pconv_p43" in names else -1
        d05_idx = names.index("pconv_d05") if "pconv_d05" in names else -1
        base_idx = names.index("baseq") if "baseq" in names else -1
        ref_idx = names.index("refiner") if "refiner" in names else -1
        p43_minus_base = float(model_means[p43_idx] - model_means[base_idx]) if p43_idx >= 0 and base_idx >= 0 else 0.0
        p43_minus_ref = float(model_means[p43_idx] - model_means[ref_idx]) if p43_idx >= 0 and ref_idx >= 0 else 0.0
        p43_minus_d05 = float(model_means[p43_idx] - model_means[d05_idx]) if p43_idx >= 0 and d05_idx >= 0 else 0.0
        feat = [
            math.log1p(float(area)),
            math.log1p(float(height)),
            math.log1p(float(width)),
            fill,
            aspect,
            border,
            abs(cy / max(height_all - 1, 1) - 0.5),
            abs(cx / max(width_all - 1, 1) - 0.5),
            float(comp_avg.mean()),
            float(comp_avg.max()),
            float(comp_avg.std()),
            float(comp_maxp.mean()),
            float(comp_maxp.max()),
            float(comp_minp.mean()),
            vote_models / max(float(len(names)), 1.0),
            p43_minus_base,
            p43_minus_ref,
            p43_minus_d05,
        ]
        feat.extend(model_means.astype(float).tolist())
        feat.extend(model_maxes.astype(float).tolist())
        feat.extend(model_votes.astype(float).tolist())
        rows.append({"component": comp, "gt_overlap": gt_overlap})
        feats.append(feat)
        labels.append(1 if gt_overlap > 0 else 0)
    if not feats:
        return rows, np.zeros((0, 18 + 3 * len(names)), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return rows, np.asarray(feats, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def build_cache(samples: list[dict[str, object]], names: list[str], seed_threshold: float, min_area: int, max_area: int):
    cache = []
    xs = []
    ys = []
    for sample in samples:
        rows, x, y = component_rows(sample, names, seed_threshold, min_area, max_area)
        cache.append((sample, rows, x, y))
        if x.shape[0]:
            xs.append(x)
            ys.append(y)
    if not xs:
        return cache, np.zeros((0, 18 + 3 * len(names)), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return cache, np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def avg_maps(samples: list[dict[str, object]], names: list[str]) -> list[np.ndarray]:
    return [np.stack([sample["preds"][name] for name in names], axis=0).mean(axis=0) for sample in samples]


def render(cache, names: list[str], learner: dict[str, object] | None, keep_threshold: float) -> tuple[list[np.ndarray], dict[str, int]]:
    maps: list[np.ndarray] = []
    kept = 0
    total = 0
    for sample, rows, x, y in cache:
        out = np.zeros_like(sample["mask"], dtype=np.float32)
        avg = np.stack([sample["preds"][name] for name in names], axis=0).mean(axis=0)
        if learner is None:
            scores = y.astype(np.float32)
        else:
            scores = predict_logistic(learner, x)
        for row, score in zip(rows, scores):
            if float(score) >= keep_threshold:
                comp = row["component"]
                out[comp] = np.maximum(out[comp], avg[comp])
                kept += 1
            total += 1
        maps.append(out)
    return maps, {"kept_components": int(kept), "total_components": int(total)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cached component gate for IRSTD dual-PConv predictions.")
    parser.add_argument("--train_cache", action="append", required=True)
    parser.add_argument("--test_cache", required=True)
    parser.add_argument("--models", default="baseq,refiner,pconv_d05,pconv_p43")
    parser.add_argument("--seed_thresholds", default="0.30,0.35,0.40,0.45,0.50")
    parser.add_argument("--keep_thresholds", default="0.20,0.30,0.40,0.50,0.60,0.70,0.80")
    parser.add_argument("--output_thresholds", default="0.30,0.35,0.38,0.40,0.42,0.45,0.50")
    parser.add_argument("--min_area", type=int, default=1)
    parser.add_argument("--max_area", type=int, default=600)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    train_samples, cache_names = load_many([Path(p) for p in args.train_cache])
    test_samples, test_names = load_cache(Path(args.test_cache))
    if cache_names != test_names:
        raise ValueError("Train/test cache model names differ")
    names = parse_csv(args.models)
    missing = [name for name in names if name not in cache_names]
    if missing:
        raise ValueError(f"Missing models in cache: {missing}")
    seed_thresholds = [float(x) for x in parse_csv(args.seed_thresholds)]
    keep_thresholds = [float(x) for x in parse_csv(args.keep_thresholds)]
    output_thresholds = [float(x) for x in parse_csv(args.output_thresholds)]
    rows: list[dict[str, object]] = []

    base_maps = avg_maps(test_samples, names)
    for out_th in output_thresholds:
        rows.append({"kind": "baseline_avg", "output_threshold": out_th, **metric_from_maps(test_samples, base_maps, out_th)})

    for seed_th in seed_thresholds:
        train_cache, x_train, y_train = build_cache(train_samples, names, seed_th, args.min_area, args.max_area)
        test_cache, x_test, y_test = build_cache(test_samples, names, seed_th, args.min_area, args.max_area)
        if x_train.shape[0] == 0 or x_test.shape[0] == 0:
            continue
        learner = fit_logistic(x_train, y_train, args.l2, args.steps, args.lr)
        train_scores = predict_logistic(learner, x_train)
        test_scores = predict_logistic(learner, x_test)
        oracle_maps, oracle_counts = render(test_cache, names, None, 0.5)
        for out_th in output_thresholds:
            rows.append(
                {
                    "kind": "test_oracle_not_valid",
                    "seed_threshold": seed_th,
                    "output_threshold": out_th,
                    "train_components": int(x_train.shape[0]),
                    "train_positive": int(y_train.sum()),
                    "test_components": int(x_test.shape[0]),
                    "test_positive": int(y_test.sum()),
                    **oracle_counts,
                    **metric_from_maps(test_samples, oracle_maps, out_th),
                }
            )
        for keep_th in keep_thresholds:
            maps, counts = render(test_cache, names, learner, keep_th)
            for out_th in output_thresholds:
                rows.append(
                    {
                        "kind": "train_fit_eval",
                        "seed_threshold": seed_th,
                        "keep_threshold": keep_th,
                        "output_threshold": out_th,
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
        f"best_valid kind={best_valid['kind']} seed={best_valid.get('seed_threshold', '')} "
        f"keep={best_valid.get('keep_threshold', '')} out={best_valid.get('output_threshold', '')} "
        f"mIoU={best_valid['mIoU']*100:.2f} Pd={best_valid['Pd']*100:.2f} Fa={best_valid['Fa']*1e6:.2f}",
        flush=True,
    )
    if best_oracle is not None:
        print(
            f"best_oracle_not_valid seed={best_oracle.get('seed_threshold', '')} "
            f"out={best_oracle.get('output_threshold', '')} mIoU={best_oracle['mIoU']*100:.2f}",
            flush=True,
        )
    Path(args.output).write_text(
        json.dumps(
            {
                "train_cache": args.train_cache,
                "test_cache": args.test_cache,
                "models": names,
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
