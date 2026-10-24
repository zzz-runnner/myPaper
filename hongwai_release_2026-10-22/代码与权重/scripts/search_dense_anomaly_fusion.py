#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_cached_prediction_fusion import load_cache, render  # noqa: E402
from eval_component_fusion_learner import metric_from_maps  # noqa: E402
from train_teacher_anchor_refiner_from_cache import robust01  # noqa: E402


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def open_image(dataset_dir: Path, dataset_name: str, name: str) -> np.ndarray:
    root = dataset_dir / dataset_name
    for ext in (".png", ".bmp", ".jpg"):
        path = root / "images" / f"{name}{ext}"
        if path.exists():
            return np.asarray(Image.open(path).convert("I"), dtype=np.float32)
    raise FileNotFoundError(f"{dataset_name}/{name}")


def dense_anomaly_score(image: np.ndarray) -> np.ndarray:
    raw = image.astype(np.float32)
    orig_h, orig_w = raw.shape
    max_side = max(orig_h, orig_w)
    scale = 1.0
    if max_side > 320:
        scale = 320.0 / float(max_side)
        raw = cv2.resize(raw, (max(8, int(orig_w * scale)), max(8, int(orig_h * scale))), interpolation=cv2.INTER_AREA)
    src = robust01(raw, 0.5, 99.5)
    cues: list[np.ndarray] = []
    for k in (5, 11, 19):
        mean = cv2.blur(src, (k, k))
        sq = cv2.blur(src * src, (k, k))
        std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
        cues.append(robust01((src - mean) / (std + 1e-3), 55.0, 99.85))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        cues.append(robust01(cv2.morphologyEx(src, cv2.MORPH_TOPHAT, kernel), 55.0, 99.85))
    for sigma in (0.9, 1.6, 2.6):
        small = cv2.GaussianBlur(src, (0, 0), sigmaX=sigma, sigmaY=sigma)
        large = cv2.GaussianBlur(src, (0, 0), sigmaX=sigma * 2.8, sigmaY=sigma * 2.8)
        cues.append(robust01(small - large, 55.0, 99.85))
        cues.append(robust01(np.abs(cv2.Laplacian(small, cv2.CV_32F, ksize=3)), 60.0, 99.9))
    h, w = src.shape
    bg_small = cv2.resize(src, (max(8, w // 24), max(8, h // 24)), interpolation=cv2.INTER_AREA)
    bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_CUBIC)
    cues.append(robust01(src - bg, 55.0, 99.85))
    score = np.stack(cues, axis=0).max(axis=0)
    score = cv2.GaussianBlur(score.astype(np.float32), (0, 0), sigmaX=0.45, sigmaY=0.45)
    score = robust01(score, 40.0, 99.8).astype(np.float32)
    if scale != 1.0:
        score = cv2.resize(score, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return score


def sparse_topk(score: np.ndarray, threshold: float, topk: int, max_area: int, max_side: int, dilate: int) -> np.ndarray:
    seed = score > threshold
    num_labels, label_img, stats, _ = cv2.connectedComponentsWithStats(seed.astype(np.uint8), 8)
    candidates: list[tuple[float, int]] = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        if not (1 <= area <= max_area and width <= max_side and height <= max_side):
            continue
        comp = label_img == label_id
        fill = area / float(max(width * height, 1))
        shape_penalty = abs(np.log(max(width / float(max(height, 1)), 1e-3)))
        comp_score = float(score[comp].mean() + 0.8 * score[comp].max() + 0.08 * fill - 0.025 * shape_penalty)
        candidates.append((comp_score, label_id))
    out = np.zeros_like(score, dtype=np.uint8)
    for _score, label_id in sorted(candidates, reverse=True)[:topk]:
        out[label_img == label_id] = 1
    if dilate > 0:
        k = dilate * 2 + 1
        out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1)
    return out.astype(np.float32)


def build_score_maps(samples: list[dict[str, object]], dataset_dir: Path, dataset_name: str) -> list[np.ndarray]:
    scores = []
    for sample in samples:
        h, w = [int(x) for x in sample["size"]]
        image = open_image(dataset_dir, dataset_name, str(sample["name"]))[:h, :w]
        scores.append(dense_anomaly_score(image)[:h, :w])
    return scores


def build_anomaly_maps_from_scores(scores: list[np.ndarray], mode: str, threshold: float, topk: int, max_area: int, max_side: int, dilate: int) -> list[np.ndarray]:
    maps = []
    for score in scores:
        if mode == "dense":
            maps.append(score)
        elif mode == "sparse":
            maps.append(sparse_topk(score, threshold, topk, max_area, max_side, dilate))
        else:
            raise ValueError(mode)
    return maps


def fuse_maps(teacher: list[np.ndarray], anomaly: list[np.ndarray], alpha: float, mode: str) -> list[np.ndarray]:
    out = []
    for t, a in zip(teacher, anomaly):
        if mode == "avg":
            fused = (1.0 - alpha) * t + alpha * a
        elif mode == "rescue":
            fused = np.maximum(t, alpha * a)
        elif mode == "suppress":
            fused = t * (1.0 - alpha * (a < 0.10).astype(np.float32))
        else:
            raise ValueError(mode)
        out.append(np.clip(fused, 0.0, 1.0).astype(np.float32))
    return out


def score_for_selection(metrics: dict[str, float], kind: str) -> float:
    if kind == "mIoU":
        return float(metrics["mIoU"])
    if kind == "miou_fa":
        return float(metrics["mIoU"]) - 0.000002 * float(metrics["Fa"] * 1e6)
    raise ValueError(kind)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search dense handcrafted anomaly candidates fused with cached teachers.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--test_cache", required=True)
    parser.add_argument("--teacher_mode", default="avg")
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--anomaly_modes", default="dense,sparse")
    parser.add_argument("--fusion_modes", default="avg,rescue")
    parser.add_argument("--anomaly_thresholds", default="0.72,0.78,0.84,0.90")
    parser.add_argument("--topks", default="4,8,16,32")
    parser.add_argument("--max_areas", default="80,160,320")
    parser.add_argument("--max_sides", default="20,32,48")
    parser.add_argument("--dilates", default="0,1")
    parser.add_argument("--alphas", default="0.05,0.10,0.15,0.20,0.30,0.40")
    parser.add_argument("--output_thresholds", default="0.30,0.35,0.38,0.40,0.42,0.45,0.50,0.55,0.60")
    parser.add_argument("--selection", default="miou_fa", choices=["mIoU", "miou_fa"])
    parser.add_argument("--test_topk", type=int, default=24)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    train_samples, train_names = load_cache(Path(args.train_cache))
    test_samples, test_names = load_cache(Path(args.test_cache))
    if args.max_train_samples > 0:
        train_samples = train_samples[: args.max_train_samples]
    if args.max_test_samples > 0:
        test_samples = test_samples[: args.max_test_samples]
    if train_names != test_names:
        raise ValueError(f"name mismatch: {train_names} != {test_names}")
    train_teacher = render(train_samples, train_names, args.teacher_mode)
    test_teacher = render(test_samples, test_names, args.teacher_mode)
    train_scores = build_score_maps(train_samples, dataset_dir, args.dataset_name)
    test_scores = build_score_maps(test_samples, dataset_dir, args.dataset_name)
    anomaly_modes = [x for x in args.anomaly_modes.split(",") if x]
    fusion_modes = [x for x in args.fusion_modes.split(",") if x]
    anomaly_thresholds = parse_float_list(args.anomaly_thresholds)
    topks = [int(x) for x in args.topks.split(",") if x.strip()]
    max_areas = [int(x) for x in args.max_areas.split(",") if x.strip()]
    max_sides = [int(x) for x in args.max_sides.split(",") if x.strip()]
    dilates = [int(x) for x in args.dilates.split(",") if x.strip()]
    alphas = parse_float_list(args.alphas)
    out_thresholds = parse_float_list(args.output_thresholds)

    train_map_cache: dict[tuple[object, ...], list[np.ndarray]] = {}
    test_map_cache: dict[tuple[object, ...], list[np.ndarray]] = {}
    rows = []
    for anomaly_mode in anomaly_modes:
        if anomaly_mode == "dense":
            param_rows = [(0.0, 0, 0, 0, 0)]
        else:
            param_rows = [(th, topk, area, side, dil) for th in anomaly_thresholds for topk in topks for area in max_areas for side in max_sides for dil in dilates]
        for anomaly_threshold, topk, max_area, max_side, dilate in param_rows:
            key = (anomaly_mode, anomaly_threshold, topk, max_area, max_side, dilate)
            train_map_cache[key] = build_anomaly_maps_from_scores(train_scores, anomaly_mode, anomaly_threshold, topk, max_area, max_side, dilate)
            for fusion_mode in fusion_modes:
                for alpha in alphas:
                    maps = fuse_maps(train_teacher, train_map_cache[key], alpha, fusion_mode)
                    for output_threshold in out_thresholds:
                        metrics = metric_from_maps(train_samples, maps, output_threshold)
                        rows.append(
                            {
                                "anomaly_mode": anomaly_mode,
                                "anomaly_threshold": anomaly_threshold,
                                "topk": topk,
                                "max_area": max_area,
                                "max_side": max_side,
                                "dilate": dilate,
                                "fusion_mode": fusion_mode,
                                "alpha": alpha,
                                "output_threshold": output_threshold,
                                "train_score": score_for_selection(metrics, args.selection),
                                **{f"train_{k}": v for k, v in metrics.items()},
                            }
                        )
    rows.sort(key=lambda row: row["train_score"], reverse=True)

    test_rows = []
    for row in rows[: max(args.test_topk, 1)]:
        key = (row["anomaly_mode"], row["anomaly_threshold"], row["topk"], row["max_area"], row["max_side"], row["dilate"])
        if key not in test_map_cache:
            test_map_cache[key] = build_anomaly_maps_from_scores(test_scores, str(row["anomaly_mode"]), float(row["anomaly_threshold"]), int(row["topk"]), int(row["max_area"]), int(row["max_side"]), int(row["dilate"]))
        maps = fuse_maps(test_teacher, test_map_cache[key], float(row["alpha"]), str(row["fusion_mode"]))
        metrics = metric_from_maps(test_samples, maps, float(row["output_threshold"]))
        test_rows.append({**row, **{f"test_{k}": v for k, v in metrics.items()}})
    selected = test_rows[0]
    best_test = sorted(test_rows, key=lambda row: row["test_mIoU"], reverse=True)[0]
    baseline_rows = []
    for th in out_thresholds:
        baseline_rows.append({"mode": args.teacher_mode, "threshold": th, **metric_from_maps(test_samples, test_teacher, th)})
    baseline_rows.sort(key=lambda row: row["mIoU"], reverse=True)
    print(
        f"selected {selected['anomaly_mode']} {selected['fusion_mode']} alpha={selected['alpha']} "
        f"train_mIoU={selected['train_mIoU']*100:.2f} test_mIoU={selected['test_mIoU']*100:.2f} "
        f"Pd={selected['test_Pd']*100:.2f} Fa={selected['test_Fa']*1e6:.2f}",
        flush=True,
    )
    print(
        f"best_test_in_topk {best_test['anomaly_mode']} {best_test['fusion_mode']} alpha={best_test['alpha']} "
        f"test_mIoU={best_test['test_mIoU']*100:.2f} Pd={best_test['test_Pd']*100:.2f} Fa={best_test['test_Fa']*1e6:.2f}",
        flush=True,
    )
    print(
        f"baseline_test mode={baseline_rows[0]['mode']} th={baseline_rows[0]['threshold']} "
        f"mIoU={baseline_rows[0]['mIoU']*100:.2f} Pd={baseline_rows[0]['Pd']*100:.2f} Fa={baseline_rows[0]['Fa']*1e6:.2f}",
        flush=True,
    )
    Path(args.output).write_text(
        json.dumps(
            {
                "args": vars(args),
                "models": train_names,
                "selected_by_train": selected,
                "best_test_in_train_topk_reference": best_test,
                "baseline_test_best": baseline_rows[0],
                "train_rows": rows,
                "test_rows": test_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
