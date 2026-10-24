#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))

from eval_component_fusion_learner import (  # noqa: E402
    FullSplitSet,
    build_models,
    parse_model_spec,
    robust01,
    sigmoid,
    standardize_apply,
    standardize_fit,
)
from metrics import PD_FA, mIoU  # noqa: E402


@torch.no_grad()
def collect_prediction_samples(
    models: list[tuple[str, torch.nn.Module]],
    dataset_dir: Path,
    dataset_name: str,
    split: str,
    max_samples: int,
    device: torch.device,
) -> list[dict[str, object]]:
    dataset = FullSplitSet(dataset_dir, dataset_name, dataset_name, split, max_samples=max_samples)
    loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
    samples: list[dict[str, object]] = []
    for idx, (image, mask, size, name) in enumerate(loader):
        image = image.to(device)
        height = int(size[0])
        width = int(size[1])
        preds: dict[str, np.ndarray] = {}
        for model_name, model in models:
            pred = model(image, epoch=10**9)
            if isinstance(pred, (tuple, list)):
                pred = pred[-1]
            preds[model_name] = pred[0, 0, :height, :width].detach().cpu().numpy().astype(np.float32)
        samples.append(
            {
                "dataset": dataset_name,
                "name": name[0],
                "image": image[0, 0, :height, :width].detach().cpu().numpy().astype(np.float32),
                "mask": mask[0, 0, :height, :width].numpy().astype(np.float32),
                "size": [height, width],
                "preds": preds,
            }
        )
        if (idx + 1) % 100 == 0:
            print(f"collected {dataset_name} {split} {idx + 1}/{len(dataset)}", flush=True)
    return samples


def collect_fit_samples_multi(
    models: list[tuple[str, torch.nn.Module]],
    dataset_dir: Path,
    datasets: list[str],
    split: str,
    max_samples_per_dataset: int,
    device: torch.device,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for dataset_name in datasets:
        out.extend(collect_prediction_samples(models, dataset_dir, dataset_name, split, max_samples_per_dataset, device))
    return out


def local_feature_stack(image: np.ndarray, pred_stack: np.ndarray) -> np.ndarray:
    src01 = robust01(image.astype(np.float32), 0.5, 99.5)
    pred_mean = pred_stack.mean(axis=0)
    pred_max = pred_stack.max(axis=0)
    pred_min = pred_stack.min(axis=0)
    pred_std = pred_stack.std(axis=0)
    feats = [src01, pred_mean, pred_max, pred_min, pred_std]
    for k in (3, 7, 15):
        mean = cv2.blur(src01, (k, k))
        sq = cv2.blur(src01 * src01, (k, k))
        std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
        feats.append(robust01(src01 - mean, 50.0, 99.7))
        feats.append(robust01((src01 - mean) / (std + 1e-3), 50.0, 99.7))
        feats.append(cv2.blur(pred_max, (k, k)))
    for sigma in (1.0, 2.0):
        small = cv2.GaussianBlur(src01, (0, 0), sigmaX=sigma, sigmaY=sigma)
        large = cv2.GaussianBlur(src01, (0, 0), sigmaX=sigma * 2.5, sigmaY=sigma * 2.5)
        feats.append(robust01(small - large, 50.0, 99.7))
        feats.append(robust01(np.abs(cv2.Laplacian(small, cv2.CV_32F, ksize=3)), 50.0, 99.7))
    return np.stack(feats, axis=-1).astype(np.float32)


def sample_pixels(
    samples: list[dict[str, object]],
    names: list[str],
    pos_per_image: int,
    neg_per_image: int,
    hard_neg_quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260613)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for sample in samples:
        pred_stack = np.stack([sample["preds"][name] for name in names], axis=0)
        feature_map = local_feature_stack(sample["image"], pred_stack)
        feature_map = np.concatenate([pred_stack.transpose(1, 2, 0), feature_map], axis=-1)
        flat = feature_map.reshape(-1, feature_map.shape[-1])
        mask = sample["mask"].reshape(-1) > 0
        pos_idx = np.flatnonzero(mask)
        if pos_idx.size:
            size = min(pos_per_image, pos_idx.size)
            xs.append(flat[rng.choice(pos_idx, size=size, replace=pos_idx.size < size)])
            ys.append(np.ones((size,), dtype=np.float32))
        neg_idx = np.flatnonzero(~mask)
        if neg_idx.size:
            hardness = pred_stack.max(axis=0).reshape(-1)[neg_idx]
            cutoff = np.percentile(hardness, hard_neg_quantile)
            hard_idx = neg_idx[hardness >= cutoff]
            if hard_idx.size < max(8, neg_per_image // 4):
                hard_idx = neg_idx
            size = min(neg_per_image, hard_idx.size)
            xs.append(flat[rng.choice(hard_idx, size=size, replace=hard_idx.size < size)])
            ys.append(np.zeros((size,), dtype=np.float32))
    if not xs:
        raise RuntimeError("No pixel samples collected.")
    return np.concatenate(xs, axis=0).astype(np.float32), np.concatenate(ys, axis=0).astype(np.float32)


def fit_logistic(
    x: np.ndarray,
    y: np.ndarray,
    steps: int,
    lr: float,
    l2: float,
    batch_pixels: int,
) -> dict[str, np.ndarray]:
    mean, std = standardize_fit(x)
    xs = standardize_apply(x, mean, std)
    xb = np.concatenate([xs, np.ones((xs.shape[0], 1), dtype=np.float32)], axis=1)
    pos = max(float(y.sum()), 1.0)
    neg = max(float((1.0 - y).sum()), 1.0)
    weights = np.where(y > 0, min(neg / pos, 80.0), 1.0).astype(np.float32)
    coef = np.zeros((xb.shape[1],), dtype=np.float32)
    rng = np.random.default_rng(20260613)
    n = xb.shape[0]
    batch = n if batch_pixels <= 0 else min(batch_pixels, n)
    for step in range(steps):
        if batch < n:
            idx = rng.choice(n, size=batch, replace=False)
            xb_step = xb[idx]
            y_step = y[idx]
            weights_step = weights[idx]
        else:
            xb_step = xb
            y_step = y
            weights_step = weights
        pred = sigmoid(xb_step @ coef)
        grad = (xb_step.T @ ((pred - y_step) * weights_step)) / max(float(weights_step.sum()), 1.0)
        grad[:-1] += l2 * coef[:-1]
        coef -= lr * grad.astype(np.float32)
        if (step + 1) % 200 == 0 or step + 1 == steps:
            print(f"fit logistic step {step + 1}/{steps}", flush=True)
    return {"coef": coef.astype(np.float32), "mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def predict_map(sample: dict[str, object], names: list[str], model: dict[str, np.ndarray], chunk: int = 262144) -> np.ndarray:
    pred_stack = np.stack([sample["preds"][name] for name in names], axis=0)
    feature_map = local_feature_stack(sample["image"], pred_stack)
    feature_map = np.concatenate([pred_stack.transpose(1, 2, 0), feature_map], axis=-1)
    flat = feature_map.reshape(-1, feature_map.shape[-1])
    out = np.zeros((flat.shape[0],), dtype=np.float32)
    for start in range(0, flat.shape[0], chunk):
        part = flat[start : start + chunk]
        xs = standardize_apply(part, model["mean"], model["std"])
        xb = np.concatenate([xs, np.ones((xs.shape[0], 1), dtype=np.float32)], axis=1)
        out[start : start + chunk] = sigmoid(xb @ model["coef"]).astype(np.float32)
    return out.reshape(sample["mask"].shape)


def metric_from_maps(samples: list[dict[str, object]], maps: list[np.ndarray], threshold: float) -> dict[str, float]:
    iou_metric = mIoU()
    pdfa_metric = PD_FA()
    sample_ious: list[float] = []
    for sample, pred in zip(samples, maps):
        mask = sample["mask"] > 0
        pred_bin = pred > threshold
        inter = int((pred_bin & mask).sum())
        union = int((pred_bin | mask).sum())
        sample_ious.append(inter / float(union + np.spacing(1)))
        pred_tensor = torch.from_numpy(pred_bin.astype(np.float32)).view(1, 1, *pred_bin.shape)
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).view(1, 1, *mask.shape)
        iou_metric.update(pred_tensor, mask_tensor)
        pdfa_metric.update(pred_tensor[0, 0], mask_tensor[0, 0], sample["size"])
    pix_acc, miou = iou_metric.get()
    pd = float(pdfa_metric.PD) / float(max(pdfa_metric.target, 1))
    fa = float(pdfa_metric.dismatch_pixel) / float(max(pdfa_metric.all_pixel, 1))
    return {"pixAcc": float(pix_acc), "mIoU": float(miou), "nIoU": float(np.mean(sample_ious)), "Pd": pd, "Fa": fa}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a pixel-level multi-teacher stacker on full IRSTD datasets.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--model_spec", action="append", required=True)
    parser.add_argument("--fit_datasets", default="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3")
    parser.add_argument("--fit_split", default="train")
    parser.add_argument("--eval_split", default="test")
    parser.add_argument("--max_fit_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--pos_per_image", type=int, default=192)
    parser.add_argument("--neg_per_image", type=int, default=768)
    parser.add_argument("--hard_neg_quantile", type=float, default=94.0)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--lr", type=float, default=0.06)
    parser.add_argument("--l2", type=float, default=0.002)
    parser.add_argument("--batch_pixels", type=int, default=65536)
    parser.add_argument("--thresholds", default="0.35,0.40,0.45,0.50,0.55,0.60")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    specs = [parse_model_spec(spec) for spec in args.model_spec]
    names = [spec[0] for spec in specs]
    models = build_models(specs, device)
    fit_datasets = [item.strip() for item in args.fit_datasets.split(",") if item.strip()]
    fit_samples = collect_fit_samples_multi(
        models,
        Path(args.dataset_dir),
        fit_datasets,
        args.fit_split,
        args.max_fit_samples,
        device,
    )
    eval_samples = collect_prediction_samples(
        models,
        Path(args.dataset_dir),
        args.dataset_name,
        args.eval_split,
        args.max_eval_samples,
        device,
    )
    x, y = sample_pixels(
        fit_samples,
        names,
        pos_per_image=args.pos_per_image,
        neg_per_image=args.neg_per_image,
        hard_neg_quantile=args.hard_neg_quantile,
    )
    print(f"sampled pixels total={x.shape[0]} positive={int(y.sum())}", flush=True)
    learner = fit_logistic(x, y, steps=args.steps, lr=args.lr, l2=args.l2, batch_pixels=args.batch_pixels)
    stacker_maps = []
    for idx, sample in enumerate(eval_samples):
        stacker_maps.append(predict_map(sample, names, learner))
        if (idx + 1) % 50 == 0 or idx + 1 == len(eval_samples):
            print(f"predicted stacker map {idx + 1}/{len(eval_samples)}", flush=True)
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    rows: list[dict[str, object]] = []
    for th in thresholds:
        rows.append({"kind": "pixel_stacker", "threshold": th, **metric_from_maps(eval_samples, stacker_maps, th)})
    for name in names:
        maps = [sample["preds"][name] for sample in eval_samples]
        for th in thresholds:
            rows.append({"kind": "baseline", "mode": name, "threshold": th, **metric_from_maps(eval_samples, maps, th)})
    avg_maps = [np.stack([sample["preds"][name] for name in names], axis=0).mean(axis=0) for sample in eval_samples]
    max_maps = [np.stack([sample["preds"][name] for name in names], axis=0).max(axis=0) for sample in eval_samples]
    for mode, maps in [("avg", avg_maps), ("max", max_maps)]:
        for th in thresholds:
            rows.append({"kind": "baseline", "mode": mode, "threshold": th, **metric_from_maps(eval_samples, maps, th)})
    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best_valid = rows[0]
    print(
        f"best_valid kind={best_valid['kind']} mode={best_valid.get('mode', '')} "
        f"th={best_valid['threshold']} mIoU={best_valid['mIoU'] * 100:.2f} "
        f"Pd={best_valid['Pd'] * 100:.2f} Fa={best_valid['Fa'] * 1e6:.2f}",
        flush=True,
    )
    for row in rows[:20]:
        print(
            f"{row['kind']} mode={row.get('mode', '')} th={row['threshold']} "
            f"mIoU={row['mIoU'] * 100:.2f} Pd={row['Pd'] * 100:.2f} Fa={row['Fa'] * 1e6:.2f}",
            flush=True,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "dataset": args.dataset_name,
                "fit_datasets": fit_datasets,
                "models": names,
                "pos_per_image": int(args.pos_per_image),
                "neg_per_image": int(args.neg_per_image),
                "hard_neg_quantile": float(args.hard_neg_quantile),
                "steps": int(args.steps),
                "lr": float(args.lr),
                "l2": float(args.l2),
                "batch_pixels": int(args.batch_pixels),
                "fit_pixels": int(x.shape[0]),
                "fit_positive": int(y.sum()),
                "best_valid": best_valid,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
