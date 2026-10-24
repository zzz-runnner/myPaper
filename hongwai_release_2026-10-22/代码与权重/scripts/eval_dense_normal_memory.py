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
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))

from eval_component_fusion_learner import metric_from_maps  # noqa: E402


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def robust01(x: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(x, [low, high])
    if hi <= lo + 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def open_pair(root: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    for ext in (".png", ".bmp", ".jpg"):
        image_path = root / "images" / f"{name}{ext}"
        mask_path = root / "masks" / f"{name}{ext}"
        if image_path.exists() and mask_path.exists():
            image = np.asarray(Image.open(image_path).convert("I"), dtype=np.float32)
            mask = np.asarray(Image.open(mask_path), dtype=np.float32)
            if mask.ndim > 2:
                mask = mask[:, :, 0]
            return image, (mask > 127).astype(np.uint8)
    raise FileNotFoundError(name)


def dense_features(image: np.ndarray, patch: int) -> np.ndarray:
    src = robust01(image)
    mean = cv2.blur(src, (patch, patch))
    sq = cv2.blur(src * src, (patch, patch))
    std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    z = (src - mean) / (std + 1e-3)
    dog = cv2.GaussianBlur(src, (0, 0), 1.0) - cv2.GaussianBlur(src, (0, 0), 2.5)
    lap = cv2.Laplacian(cv2.GaussianBlur(src, (0, 0), 1.0), cv2.CV_32F, ksize=3)
    grad_x = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (patch, patch))
    tophat = cv2.morphologyEx(src, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(src, cv2.MORPH_BLACKHAT, kernel)
    feats = np.stack(
        [
            src,
            mean,
            std,
            z,
            dog,
            lap,
            grad,
            robust01(tophat, 50.0, 99.8),
            robust01(blackhat, 50.0, 99.8),
        ],
        axis=-1,
    ).astype(np.float32)
    return feats


def sample_background_features(
    dataset_dir: Path,
    datasets: list[str],
    split: str,
    max_images_per_dataset: int,
    samples_per_image: int,
    patch: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunks = []
    for dataset in datasets:
        root = dataset_dir / dataset
        names = (root / "img_idx" / f"{split}_{dataset}.txt").read_text(encoding="utf-8").splitlines()
        if max_images_per_dataset > 0:
            names = names[:max_images_per_dataset]
        for name in names:
            image, mask = open_pair(root, name)
            feats = dense_features(image, patch).reshape(-1, 9)
            bg = (mask.reshape(-1) == 0)
            idx = np.flatnonzero(bg)
            if idx.size == 0:
                continue
            take = min(samples_per_image, idx.size)
            chosen = rng.choice(idx, size=take, replace=False)
            chunks.append(feats[chosen])
    if not chunks:
        raise RuntimeError("No background features sampled")
    return np.concatenate(chunks, axis=0).astype(np.float32)


def fit_memory(feats: np.ndarray, memory_size: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    std[std < 1e-6] = 1.0
    z = ((feats - mean) / std).astype(np.float32)
    if z.shape[0] > memory_size:
        idx = rng.choice(z.shape[0], size=memory_size, replace=False)
        z = z[idx]
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "memory": z.astype(np.float32)}


def anomaly_map(image: np.ndarray, model: dict[str, np.ndarray], patch: int, chunk_pixels: int) -> np.ndarray:
    feats = dense_features(image, patch).reshape(-1, 9)
    z = ((feats - model["mean"]) / model["std"]).astype(np.float32)
    memory = model["memory"]
    scores = np.empty((z.shape[0],), dtype=np.float32)
    mem_norm = np.sum(memory * memory, axis=1)[None, :]
    for start in range(0, z.shape[0], chunk_pixels):
        block = z[start : start + chunk_pixels]
        dist = np.sum(block * block, axis=1, keepdims=True) + mem_norm - 2.0 * (block @ memory.T)
        scores[start : start + chunk_pixels] = np.sqrt(np.maximum(dist.min(axis=1), 0.0))
    amap = scores.reshape(image.shape).astype(np.float32)
    return robust01(amap, 50.0, 99.8)


def eval_dataset(
    dataset_dir: Path,
    dataset: str,
    model: dict[str, np.ndarray],
    patch: int,
    chunk_pixels: int,
    max_samples: int,
    thresholds: list[float],
) -> dict[str, object]:
    root = dataset_dir / dataset
    names = (root / "img_idx" / f"test_{dataset}.txt").read_text(encoding="utf-8").splitlines()
    if max_samples > 0:
        names = names[:max_samples]
    samples = []
    maps = []
    for idx, name in enumerate(names):
        image, mask = open_pair(root, name)
        maps.append(anomaly_map(image, model, patch, chunk_pixels))
        samples.append({"name": name, "mask": mask.astype(np.float32), "size": [int(mask.shape[0]), int(mask.shape[1])]})
        if (idx + 1) % 50 == 0:
            print(f"{dataset} evaluated {idx + 1}/{len(names)}", flush=True)
    rows = [{"threshold": th, **metric_from_maps(samples, maps, th)} for th in thresholds]
    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    return {"dataset": dataset, "best": rows[0], "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Dense normal-memory anomaly map for IRSTD.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--fit_datasets", default="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3")
    parser.add_argument("--eval_datasets", default="IRSTD-1K,NUAA-SIRST")
    parser.add_argument("--max_fit_images_per_dataset", type=int, default=200)
    parser.add_argument("--samples_per_image", type=int, default=256)
    parser.add_argument("--memory_size", type=int, default=4096)
    parser.add_argument("--patch", type=int, default=9)
    parser.add_argument("--chunk_pixels", type=int, default=65536)
    parser.add_argument("--eval_max_samples", type=int, default=0)
    parser.add_argument("--thresholds", default="0.50,0.60,0.70,0.80,0.85,0.90,0.95")
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    fit_datasets = parse_csv(args.fit_datasets)
    eval_datasets = parse_csv(args.eval_datasets)
    thresholds = parse_float_list(args.thresholds)
    print("sampling background memory", flush=True)
    feats = sample_background_features(
        dataset_dir,
        fit_datasets,
        "train",
        args.max_fit_images_per_dataset,
        args.samples_per_image,
        args.patch,
        args.seed,
    )
    print(f"sampled {feats.shape[0]} background vectors", flush=True)
    model = fit_memory(feats, args.memory_size, args.seed)
    results = [
        eval_dataset(dataset_dir, dataset, model, args.patch, args.chunk_pixels, args.eval_max_samples, thresholds)
        for dataset in eval_datasets
    ]
    for item in results:
        best = item["best"]
        print(
            f"{item['dataset']} best th={best['threshold']} mIoU={best['mIoU']*100:.2f} "
            f"Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.2f}",
            flush=True,
        )
    Path(args.output).write_text(
        json.dumps(
            {
                "args": vars(args),
                "fit_vectors": int(feats.shape[0]),
                "memory_vectors": int(model["memory"].shape[0]),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
