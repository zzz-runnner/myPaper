#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_component_fusion_learner import (  # noqa: E402
    build_models,
    collect_fit_samples_multi,
    collect_samples,
    fit_background_normality_model,
    load_official_bgm_models,
    load_rpcanet_models,
    metric_from_maps,
    parse_model_spec,
    parse_official_bgm_spec,
    parse_rpcanet_spec,
)


NAMES = ["tpc_a", "tpc_b", "lca", "bgm_nuaa", "rpcanet_sirst_s3", "bg_anomaly"]


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def local_cues(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    img01 = (img - mn) / max(mx - mn, 1e-6)
    blur5 = cv2.GaussianBlur(img01, (5, 5), 0)
    blur11 = cv2.GaussianBlur(img01, (11, 11), 0)
    dog = np.maximum(blur5 - blur11, 0.0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    top = cv2.morphologyEx(img01, cv2.MORPH_TOPHAT, k)
    mean = cv2.blur(img01, (9, 9))
    sq = cv2.blur(img01 * img01, (9, 9))
    std = np.sqrt(np.maximum(sq - mean * mean, 1e-6))
    z = np.maximum((img01 - mean) / (std + 1e-6), 0.0)
    z = np.clip(z / 6.0, 0.0, 1.0)
    return np.stack([img01, dog, top, z], axis=0).astype(np.float32)


def sample_pixels(
    samples: list[dict[str, object]],
    max_pos_per_image: int,
    max_neg_per_image: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for sample in samples:
        pred_stack = np.stack([sample["preds"][name].astype(np.float32) for name in NAMES], axis=0)
        cues = local_cues(sample["image"].astype(np.float32))
        feat = np.concatenate([pred_stack, cues], axis=0).reshape(10, -1).T
        mask = sample["mask"].reshape(-1) > 0
        pos = np.flatnonzero(mask)
        neg = np.flatnonzero(~mask)
        if pos.size:
            if pos.size > max_pos_per_image:
                pos = rng.choice(pos, size=max_pos_per_image, replace=False)
            xs.append(feat[pos])
            ys.append(np.ones((pos.size,), dtype=np.float32))
        hard_score = pred_stack.max(axis=0).reshape(-1) + 0.5 * cues[3].reshape(-1)
        if neg.size:
            if neg.size > max_neg_per_image:
                hard_pool = neg[np.argsort(hard_score[neg])[-min(max_neg_per_image * 4, neg.size) :]]
                neg = rng.choice(hard_pool, size=min(max_neg_per_image, hard_pool.size), replace=False)
            xs.append(feat[neg])
            ys.append(np.zeros((neg.size,), dtype=np.float32))
    return np.concatenate(xs, axis=0).astype(np.float32), np.concatenate(ys, axis=0).astype(np.float32)


class PixelFusionMLP(nn.Module):
    def __init__(self, in_dim: int = 10, hidden: int = 48) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ResidualCalibrator(nn.Module):
    def __init__(self, in_dim: int = 10, residual_scale: float = 0.35) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.residual_scale = float(residual_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = (
            0.14 * x[:, 0]
            + 0.14 * x[:, 1]
            + 0.14 * x[:, 2]
            + 0.22 * x[:, 3]
            + 0.28 * x[:, 4]
            + 0.08 * x[:, 5]
        )
        base = base.clamp(1e-4, 1.0 - 1e-4)
        base_logit = torch.logit(base)
        residual = self.residual_scale * torch.tanh(self.linear(x).squeeze(-1))
        return base_logit + residual


def train_model(
    x: np.ndarray,
    y: np.ndarray,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
    model_kind: str,
    residual_scale: float,
) -> nn.Module:
    torch.manual_seed(seed)
    if model_kind == "mlp":
        model = PixelFusionMLP(x.shape[1]).to(device)
    elif model_kind == "residual_linear":
        model = ResidualCalibrator(x.shape[1], residual_scale=residual_scale).to(device)
    else:
        raise ValueError(f"Unsupported model kind: {model_kind}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x_t = torch.from_numpy(x).to(device)
    y_t = torch.from_numpy(y).to(device)
    pos = float(y.mean())
    pos_weight = torch.tensor([(1.0 - pos) / max(pos, 1e-6)], device=device).clamp(1.0, 80.0)
    n = x.shape[0]
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    for step in range(steps):
        idx = torch.randint(0, n, (batch_size,), generator=gen, device=device)
        logits = model(x_t[idx])
        loss = F.binary_cross_entropy_with_logits(logits, y_t[idx], pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"step {step+1}/{steps} loss={float(loss.detach().cpu()):.5f}", flush=True)
    return model


@torch.no_grad()
def predict_maps(model: PixelFusionMLP, samples: list[dict[str, object]], device: torch.device) -> list[np.ndarray]:
    model.eval()
    maps = []
    for idx, sample in enumerate(samples):
        pred_stack = np.stack([sample["preds"][name].astype(np.float32) for name in NAMES], axis=0)
        cues = local_cues(sample["image"].astype(np.float32))
        feat = np.concatenate([pred_stack, cues], axis=0).reshape(10, -1).T.astype(np.float32)
        out_chunks = []
        for start in range(0, feat.shape[0], 262144):
            xt = torch.from_numpy(feat[start : start + 262144]).to(device)
            out_chunks.append(torch.sigmoid(model(xt)).detach().cpu().numpy())
        prob = np.concatenate(out_chunks, axis=0).reshape(sample["mask"].shape).astype(np.float32)
        maps.append(prob)
        if (idx + 1) % 50 == 0:
            print(f"predicted {idx+1}/{len(samples)}", flush=True)
    return maps


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a pixel MLP fusion over TPC/LCA/BGM/RPCANet/bg candidates.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", default="NUAA-SIRST")
    parser.add_argument("--fit_datasets", default="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3")
    parser.add_argument("--model_spec", action="append", required=True)
    parser.add_argument("--official_bgm_spec", action="append", required=True)
    parser.add_argument("--rpcanet_spec", action="append", required=True)
    parser.add_argument("--background_sample_pixels", type=int, default=2048)
    parser.add_argument("--background_topk", type=int, default=16)
    parser.add_argument("--background_threshold", type=float, default=0.88)
    parser.add_argument("--max_fit_samples", type=int, default=220)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--max_pos_per_image", type=int, default=256)
    parser.add_argument("--max_neg_per_image", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--model_kind", default="residual_linear", choices=["residual_linear", "mlp"])
    parser.add_argument("--residual_scale", type=float, default=0.20)
    parser.add_argument("--thresholds", default="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    specs = [parse_model_spec(spec) for spec in args.model_spec]
    models = build_models(specs, device)
    bgm_specs = [parse_official_bgm_spec(spec) for spec in args.official_bgm_spec]
    bgms = load_official_bgm_models(bgm_specs, device)
    rpc_specs = [parse_rpcanet_spec(spec) for spec in args.rpcanet_spec]
    rpcanets = load_rpcanet_models(rpc_specs, device)
    fit_datasets = [x.strip() for x in args.fit_datasets.split(",") if x.strip()]
    background_model = fit_background_normality_model(
        Path(args.dataset_dir),
        fit_datasets,
        "train",
        args.max_fit_samples,
        args.background_sample_pixels,
    )
    fit_samples = collect_fit_samples_multi(
        models,
        [],
        bgms,
        rpcanets,
        Path(args.dataset_dir),
        fit_datasets,
        "train",
        args.max_fit_samples,
        device,
        background_model=background_model,
        background_as_candidate=True,
        background_topk=args.background_topk,
        background_threshold=args.background_threshold,
    )
    eval_samples = collect_samples(
        models,
        [],
        bgms,
        rpcanets,
        Path(args.dataset_dir),
        args.dataset_name,
        "test",
        args.max_eval_samples,
        device,
        background_model=background_model,
        background_as_candidate=True,
        background_topk=args.background_topk,
        background_threshold=args.background_threshold,
    )
    rng = np.random.default_rng(args.seed)
    x, y = sample_pixels(fit_samples, args.max_pos_per_image, args.max_neg_per_image, rng)
    print(f"training pixels={x.shape[0]} pos={int(y.sum())} neg={int((1-y).sum())}", flush=True)
    model = train_model(x, y, args.steps, args.batch_size, args.lr, args.seed, device, args.model_kind, args.residual_scale)
    maps = predict_maps(model, eval_samples, device)
    rows = []
    for threshold in parse_float_list(args.thresholds):
        rows.append({"threshold": threshold, **metric_from_maps(eval_samples, maps, threshold)})
    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best = rows[0]
    print(
        f"best th={best['threshold']} mIoU={best['mIoU']*100:.2f} nIoU={best['nIoU']*100:.2f} "
        f"Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.2f}",
        flush=True,
    )
    output = Path(args.output)
    output.write_text(
        json.dumps(
            {
                "dataset": args.dataset_name,
                "fit_datasets": fit_datasets,
                "best": best,
                "rows": rows,
                "train_pixels": int(x.shape[0]),
                "train_positive": int(y.sum()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.checkpoint:
        torch.save({"model": model.state_dict(), "best": best}, args.checkpoint)


if __name__ == "__main__":
    main()
