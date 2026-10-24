#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_nuaa_external_mamba_residual_pixel as extpix  # noqa: E402
import train_nuaa_official_mamba_residual_pixel as basepix  # noqa: E402
from eval_cached_prediction_fusion import load_cache  # noqa: E402
from eval_component_fusion_learner import metric_from_maps  # noqa: E402


def binarize(prob: np.ndarray, threshold: float, min_area: int) -> np.ndarray:
    b = (prob >= threshold).astype(np.uint8)
    if min_area <= 1:
        return b.astype(np.float32)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    out = np.zeros_like(b)
    for label in range(1, num):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            out[labels == label] = 1
    return out.astype(np.float32)


def metric_binary(samples: list[dict], maps: list[np.ndarray]) -> dict:
    return metric_from_maps(samples, maps, 0.5)


def image_iou(mask: np.ndarray, pred: np.ndarray) -> float:
    gt = mask > 0.5
    pr = pred > 0.5
    inter = float(np.logical_and(gt, pr).sum())
    union = float(np.logical_or(gt, pr).sum())
    if union <= 0:
        return 1.0
    return inter / union


def attach_external(samples: list[dict], split: str) -> None:
    if split == "train":
        paths = {
            "dfaware_ext": (ROOT / "reports/tmp_dfaware_train_maps.npz", "maps_candidate"),
            "hfm_ext": (ROOT / "reports/tmp_hfmnet_train_maps.npz", "maps_candidate"),
            "s703_ext": (ROOT / "reports/tmp_s703_train_maps.npz", "maps_no_tta"),
        }
    else:
        paths = {
            "dfaware_ext": (ROOT / "reports/tmp_dfaware_test_maps.npz", "maps_candidate"),
            "hfm_ext": (ROOT / "reports/tmp_hfmnet_test_maps.npz", "maps_candidate"),
            "s703_ext": (ROOT / "reports/tmp_s703_test_maps.npz", "maps_no_tta"),
        }
    extpix.attach_external(samples, paths)


def train_base(train_samples: list[dict], args: argparse.Namespace, device: torch.device):
    x, y = basepix.sample_pixels(train_samples, args.pos_per_image, args.neg_per_image, args.seed)
    return basepix.train_model(x, y, args, device)


def train_external(train_samples: list[dict], args: argparse.Namespace, device: torch.device):
    x, y = extpix.sample_pixels(train_samples, args.pos_per_image, args.neg_per_image, args.seed)
    return extpix.train_model(x, y, args, device)


def per_image_features(sample: dict, mask_a: np.ndarray, mask_b: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray) -> dict:
    union = np.logical_or(mask_a > 0.5, mask_b > 0.5)
    inter = np.logical_and(mask_a > 0.5, mask_b > 0.5)
    area_a = float((mask_a > 0.5).sum())
    area_b = float((mask_b > 0.5).sum())
    return {
        "name": sample["name"],
        "area_a": area_a,
        "area_b": area_b,
        "area_delta": area_b - area_a,
        "union_extra": float(union.sum() - (mask_a > 0.5).sum()),
        "inter_drop": float((mask_a > 0.5).sum() - inter.sum()),
        "mean_a": float(prob_a[union].mean()) if union.any() else 0.0,
        "mean_b": float(prob_b[union].mean()) if union.any() else 0.0,
        "max_a": float(prob_a.max()),
        "max_b": float(prob_b.max()),
        "mean_delta": float((prob_b[union].mean() - prob_a[union].mean()) if union.any() else 0.0),
        "max_delta": float(prob_b.max() - prob_a.max()),
        "component_delta": float(cv2.connectedComponents((mask_b > 0.5).astype(np.uint8), 8)[0] - cv2.connectedComponents((mask_a > 0.5).astype(np.uint8), 8)[0]),
        "choose_b": 0,
    }


FEATURE_KEYS = [
    "area_a",
    "area_b",
    "area_delta",
    "union_extra",
    "inter_drop",
    "mean_a",
    "mean_b",
    "max_a",
    "max_b",
    "mean_delta",
    "max_delta",
    "component_delta",
]


def train_logistic_selector(features: list[dict], labels: list[int], seed: int, reg: float) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    x = np.asarray([[float(f[k]) for k in FEATURE_KEYS] for f in features], dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)[:, None]
    mu = x.mean(axis=0, keepdims=True)
    sig = x.std(axis=0, keepdims=True) + 1e-5
    xn = (x - mu) / sig
    torch.manual_seed(seed)
    w = torch.zeros((x.shape[1], 1), dtype=torch.float32, requires_grad=True)
    b = torch.zeros((1,), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.AdamW([w, b], lr=0.05, weight_decay=reg)
    xt = torch.from_numpy(xn)
    yt = torch.from_numpy(y)
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    pos_weight = torch.tensor([min(neg / pos, 10.0)], dtype=torch.float32)
    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(xt @ w + b, yt, pos_weight=pos_weight)
        loss.backward()
        opt.step()
    return w.detach().numpy().reshape(-1), float(b.detach().numpy()[0]), mu.reshape(-1), sig.reshape(-1)


def selector_scores(features: list[dict], params: tuple[np.ndarray, float, np.ndarray, np.ndarray]) -> np.ndarray:
    w, b, mu, sig = params
    x = np.asarray([[float(f[k]) for k in FEATURE_KEYS] for f in features], dtype=np.float32)
    logits = ((x - mu) / sig) @ w + b
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_cache", default=str(ROOT / "reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__train.npz"))
    ap.add_argument("--test_cache", default=str(ROOT / "reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__test.npz"))
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--pos_per_image", type=int, default=512)
    ap.add_argument("--neg_per_image", type=int, default=2048)
    ap.add_argument("--batch_pixels", type=int, default=65536)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--residual_scale", type=float, default=0.04)
    ap.add_argument("--max_pos_weight", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=20260615)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default=str(ROOT / "reports/nuaa_residual_candidate_complementarity.json"))
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    train_samples, _ = load_cache(Path(args.train_cache))
    test_samples, _ = load_cache(Path(args.test_cache))
    attach_external(train_samples, "train")
    attach_external(test_samples, "test")

    print("training base residual", flush=True)
    base_model = train_base(train_samples, args, device)
    base_train_probs = basepix.predict_maps(base_model, train_samples, device)
    base_test_probs = basepix.predict_maps(base_model, test_samples, device)

    print("training external residual", flush=True)
    ext_args = argparse.Namespace(**vars(args))
    ext_args.residual_scale = 0.035
    ext_args.pos_per_image = 768
    ext_args.neg_per_image = 3072
    ext_args.max_pos_weight = 24.0
    ext_model = train_external(train_samples, ext_args, device)
    ext_train_probs = extpix.predict_maps(ext_model, train_samples, device)
    ext_test_probs = extpix.predict_maps(ext_model, test_samples, device)

    # Use the known best operating points for the two near-tied maps.
    base_train_masks = [binarize(p, 0.498, 3) for p in base_train_probs]
    base_test_masks = [binarize(p, 0.498, 3) for p in base_test_probs]
    ext_train_masks = [binarize(p, 0.494, 3) for p in ext_train_probs]
    ext_test_masks = [binarize(p, 0.494, 3) for p in ext_test_probs]

    rows = {
        "base_train": metric_binary(train_samples, base_train_masks),
        "base_test": metric_binary(test_samples, base_test_masks),
        "external_train": metric_binary(train_samples, ext_train_masks),
        "external_test": metric_binary(test_samples, ext_test_masks),
        "union_test": metric_binary(test_samples, [np.logical_or(a > 0.5, b > 0.5).astype(np.float32) for a, b in zip(base_test_masks, ext_test_masks)]),
        "intersection_test": metric_binary(test_samples, [np.logical_and(a > 0.5, b > 0.5).astype(np.float32) for a, b in zip(base_test_masks, ext_test_masks)]),
    }

    oracle_test = []
    for sample, a, b in zip(test_samples, base_test_masks, ext_test_masks):
        ia = image_iou(sample["mask"], a)
        ib = image_iou(sample["mask"], b)
        oracle_test.append(b if ib > ia else a)
    rows["per_image_oracle_test"] = metric_binary(test_samples, oracle_test)

    # Train a tiny transparent selector from train per-image oracle labels.
    train_feat = []
    y = []
    for sample, ma, mb, pa, pb in zip(train_samples, base_train_masks, ext_train_masks, base_train_probs, ext_train_probs):
        feat = per_image_features(sample, ma, mb, pa, pb)
        ia = image_iou(sample["mask"], ma)
        ib = image_iou(sample["mask"], mb)
        feat["choose_b"] = int(ib > ia)
        train_feat.append(feat)
        y.append(feat["choose_b"])
    y_arr = np.asarray(y, dtype=np.float32)
    selector = "base"
    rule = None
    best_train = rows["base_train"]["mIoU"]
    # Simple one-dimensional rules only; avoids overfitting a 213-image split too aggressively.
    for key in ["area_delta", "union_extra", "inter_drop", "mean_b", "max_b"]:
        vals = np.asarray([f[key] for f in train_feat], dtype=np.float32)
        for q in np.linspace(0.1, 0.9, 17):
            th = float(np.quantile(vals, q))
            for direction in [">", "<="]:
                pred_train = []
                for sample, ma, mb, feat in zip(train_samples, base_train_masks, ext_train_masks, train_feat):
                    choose = feat[key] > th if direction == ">" else feat[key] <= th
                    pred_train.append(mb if choose else ma)
                row = metric_binary(train_samples, pred_train)
                if row["mIoU"] > best_train:
                    best_train = row["mIoU"]
                    selector = "rule"
                    rule = {"key": key, "threshold": th, "direction": direction, "train": row}
    if rule is not None:
        pred_test = []
        test_feat = []
        for sample, ma, mb, pa, pb in zip(test_samples, base_test_masks, ext_test_masks, base_test_probs, ext_test_probs):
            feat = per_image_features(sample, ma, mb, pa, pb)
            test_feat.append(feat)
            choose = feat[rule["key"]] > rule["threshold"] if rule["direction"] == ">" else feat[rule["key"]] <= rule["threshold"]
            pred_test.append(mb if choose else ma)
        rows["train_selected_rule_test"] = metric_binary(test_samples, pred_test)
    else:
        rows["train_selected_rule_test"] = rows["base_test"]

    # A tiny image-level logistic selector, selected by train-fold validation.
    test_feat = [
        per_image_features(sample, ma, mb, pa, pb)
        for sample, ma, mb, pa, pb in zip(test_samples, base_test_masks, ext_test_masks, base_test_probs, ext_test_probs)
    ]
    best_selector = None
    best_selector_score = rows["base_train"]["mIoU"]
    regs = [0.0, 1e-4, 1e-3, 1e-2]
    prob_thresholds = [0.2, 0.35, 0.5, 0.65, 0.8]
    for reg in regs:
        for pth in prob_thresholds:
            fold_maps: list[np.ndarray | None] = [None] * len(train_samples)
            for fold in range(3):
                tr = [i for i in range(len(train_samples)) if i % 3 != fold]
                va = [i for i in range(len(train_samples)) if i % 3 == fold]
                params = train_logistic_selector([train_feat[i] for i in tr], [y[i] for i in tr], args.seed + fold, reg)
                scores = selector_scores([train_feat[i] for i in va], params)
                for local_idx, score in zip(va, scores):
                    fold_maps[local_idx] = ext_train_masks[local_idx] if float(score) >= pth else base_train_masks[local_idx]
            pred_train = [m for m in fold_maps if m is not None]
            row = metric_binary(train_samples, pred_train)
            if row["mIoU"] > best_selector_score:
                best_selector_score = row["mIoU"]
                best_selector = {"reg": reg, "prob_threshold": pth, "cv_train": row}
    if best_selector is not None:
        params = train_logistic_selector(train_feat, y, args.seed + 99, best_selector["reg"])
        scores = selector_scores(test_feat, params)
        pred_test = [mb if float(score) >= best_selector["prob_threshold"] else ma for score, ma, mb in zip(scores, base_test_masks, ext_test_masks)]
        rows["logistic_selector_test"] = metric_binary(test_samples, pred_test)
    else:
        rows["logistic_selector_test"] = rows["base_test"]

    out = {
        "args": vars(args),
        "rows": rows,
        "selector": selector,
        "rule": rule,
        "logistic_selector": best_selector,
        "test_oracle_gain_over_base": rows["per_image_oracle_test"]["mIoU"] - rows["base_test"]["mIoU"],
    }
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"rows": rows, "selector": selector, "rule": rule}, indent=2), flush=True)


if __name__ == "__main__":
    main()
