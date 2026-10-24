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

from eval_cached_prediction_fusion import load_cache  # noqa: E402
from eval_component_fusion_learner import metric_from_maps  # noqa: E402
from search_nuaa_official_mamba_local_rules import build_maps  # noqa: E402


LOCAL_CFG = {
    "bg_th": 0.08,
    "support_th": 0.22,
    "rank_th": 0.985,
    "suppress_scale": 0.85,
    "agree_th": 0.1,
    "mx_th": 0.5,
    "restore_cap": 0.58,
    "range_th": 0.05,
    "restore_mix": 0.0,
    "gamma": 1.0,
}


def rank01(x: np.ndarray) -> np.ndarray:
    flat = x.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(flat), dtype=np.float32)
    return ranks.reshape(x.shape)


def load_external(path: Path, key: str) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    names = [str(x) for x in data["names"]]
    maps = [np.asarray(x, dtype=np.float32) for x in data[key]]
    return dict(zip(names, maps))


def attach_external(samples: list[dict], paths: dict[str, tuple[Path, str]]) -> None:
    loaded = {name: load_external(path, key) for name, (path, key) in paths.items()}
    for sample in samples:
        sample_name = str(sample["name"])
        preds = sample["preds"]
        for name, maps in loaded.items():
            if sample_name not in maps:
                raise KeyError(f"{sample_name} missing in {name}")
            value = maps[sample_name]
            if value.shape != sample["mask"].shape:
                h, w = sample["mask"].shape
                value = value[:h, :w]
            preds[name] = value.astype(np.float32)


def logit_np(x: np.ndarray) -> np.ndarray:
    x = np.clip(x.astype(np.float32), 1e-4, 1.0 - 1e-4)
    return np.log(x / (1.0 - x)).astype(np.float32)


def feature_stack(sample: dict) -> tuple[np.ndarray, np.ndarray]:
    preds = sample["preds"]
    local = build_maps(sample, LOCAL_CFG).astype(np.float32)
    base = preds["official_mamba_fused"].astype(np.float32)
    strong = preds["strong_fusion"].astype(np.float32)
    weighted = preds["weighted_raw"].astype(np.float32)
    bg = preds["bgm_nuaa"].astype(np.float32)
    rpc = preds["rpcanet_sirst_s3"].astype(np.float32)
    dfaware = preds["dfaware_ext"].astype(np.float32)
    hfm = preds["hfm_ext"].astype(np.float32)
    s703 = preds["s703_ext"].astype(np.float32)
    s703_bgm = np.clip(0.96 * s703 - 0.04 * bg, 0.0, 1.0).astype(np.float32) ** 0.94
    ext_mean = ((dfaware + hfm + s703_bgm) / 3.0).astype(np.float32)
    ext_max = np.maximum.reduce([dfaware, hfm, s703_bgm]).astype(np.float32)
    core_max = np.maximum.reduce([local, base, strong, weighted, s703_bgm, dfaware, hfm]).astype(np.float32)
    core_min = np.minimum.reduce([local, base, strong, weighted, s703_bgm, dfaware, hfm]).astype(np.float32)
    mean7 = cv2.blur(local, (7, 7)).astype(np.float32)
    mean11 = cv2.blur(local, (11, 11)).astype(np.float32)
    ext_support = cv2.blur(ext_max, (7, 7)).astype(np.float32)
    feats = np.stack(
        [
            local,
            base,
            strong,
            weighted,
            bg,
            rpc,
            dfaware,
            hfm,
            s703_bgm,
            ext_mean,
            ext_max,
            core_max,
            core_max - core_min,
            np.maximum(ext_max - local, 0.0),
            np.maximum(local - ext_max, 0.0),
            mean7,
            mean11,
            ext_support,
            rank01(local),
            rank01(ext_max),
            logit_np(local),
            logit_np(ext_max),
        ],
        axis=-1,
    )
    return feats.astype(np.float32), local


def sample_pixels(samples: list[dict], pos_per_image: int, neg_per_image: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs = []
    ys = []
    for sample in samples:
        feats, local = feature_stack(sample)
        flat = feats.reshape(-1, feats.shape[-1])
        mask = sample["mask"].reshape(-1) > 0
        pos = np.flatnonzero(mask)
        if pos.size:
            k = min(pos_per_image, pos.size)
            xs.append(flat[rng.choice(pos, size=k, replace=pos.size < k)])
            ys.append(np.ones((k,), dtype=np.float32))
        neg = np.flatnonzero(~mask)
        if neg.size:
            hard_score = np.maximum(local.reshape(-1)[neg], flat[neg, 10])
            pool_n = min(max(neg_per_image * 10, neg_per_image), neg.size)
            pool = neg[np.argsort(hard_score)[-pool_n:]]
            k = min(neg_per_image, pool.size)
            xs.append(flat[rng.choice(pool, size=k, replace=pool.size < k)])
            ys.append(np.zeros((k,), dtype=np.float32))
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.float32)


class ResidualPixel(nn.Module):
    def __init__(self, in_dim: int, residual_scale: float) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.residual_scale = float(residual_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x[:, 0].clamp(1e-4, 1.0 - 1e-4)
        return torch.logit(base) + self.residual_scale * torch.tanh(self.linear(x).squeeze(-1))


def train_model(x: np.ndarray, y: np.ndarray, args: argparse.Namespace, device: torch.device) -> ResidualPixel:
    torch.manual_seed(args.seed)
    model = ResidualPixel(x.shape[1], args.residual_scale).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    x_t = torch.from_numpy(x).to(device)
    y_t = torch.from_numpy(y).to(device)
    pos = max(float(y.mean()), 1e-6)
    pos_weight = torch.tensor([(1.0 - pos) / pos], device=device).clamp(1.0, args.max_pos_weight)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    n = x.shape[0]
    batch = min(args.batch_pixels, n)
    for step in range(args.steps):
        idx = torch.randint(0, n, (batch,), generator=gen, device=device)
        logits = model(x_t[idx])
        loss = F.binary_cross_entropy_with_logits(logits, y_t[idx], pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 300 == 0 or step + 1 == args.steps:
            print(f"step {step+1}/{args.steps} loss={float(loss.detach().cpu()):.5f}", flush=True)
    return model


@torch.no_grad()
def predict_maps(model: ResidualPixel, samples: list[dict], device: torch.device) -> list[np.ndarray]:
    model.eval()
    maps = []
    for sample in samples:
        feats, _local = feature_stack(sample)
        flat = feats.reshape(-1, feats.shape[-1]).astype(np.float32)
        out = []
        for start in range(0, flat.shape[0], 262144):
            x = torch.from_numpy(flat[start : start + 262144]).to(device)
            out.append(torch.sigmoid(model(x)).detach().cpu().numpy())
        maps.append(np.concatenate(out).reshape(sample["mask"].shape).astype(np.float32))
    return maps


def min_area_map(prob: np.ndarray, threshold: float, min_area: int) -> np.ndarray:
    binary = (prob >= threshold).astype(np.uint8)
    if min_area <= 1:
        return binary.astype(np.float32)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros_like(binary)
    for label in range(1, num):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            out[labels == label] = 1
    return out.astype(np.float32)


def parse_values(text: str, cast=float):
    return [cast(x) for x in text.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_cache", default=str(ROOT / "reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__train.npz"))
    ap.add_argument("--test_cache", default=str(ROOT / "reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__test.npz"))
    ap.add_argument("--dfaware_train", default=str(ROOT / "reports/tmp_dfaware_train_maps.npz"))
    ap.add_argument("--dfaware_test", default=str(ROOT / "reports/tmp_dfaware_test_maps.npz"))
    ap.add_argument("--hfm_train", default=str(ROOT / "reports/tmp_hfmnet_train_maps.npz"))
    ap.add_argument("--hfm_test", default=str(ROOT / "reports/tmp_hfmnet_test_maps.npz"))
    ap.add_argument("--s703_train", default=str(ROOT / "reports/tmp_s703_train_maps.npz"))
    ap.add_argument("--s703_test", default=str(ROOT / "reports/tmp_s703_test_maps.npz"))
    ap.add_argument("--pos_per_image", type=int, default=768)
    ap.add_argument("--neg_per_image", type=int, default=3072)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--batch_pixels", type=int, default=65536)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--residual_scale", type=float, default=0.035)
    ap.add_argument("--max_pos_weight", type=float, default=24.0)
    ap.add_argument("--thresholds", default="0.490,0.492,0.494,0.496,0.498,0.500,0.502,0.504")
    ap.add_argument("--min_areas", default="1,2,3,4")
    ap.add_argument("--seed", type=int, default=20260616)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default=str(ROOT / "reports/nuaa_external_mamba_residual_pixel.json"))
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    train_samples, _ = load_cache(Path(args.train_cache))
    test_samples, _ = load_cache(Path(args.test_cache))
    attach_external(
        train_samples,
        {
            "dfaware_ext": (Path(args.dfaware_train), "maps_candidate"),
            "hfm_ext": (Path(args.hfm_train), "maps_candidate"),
            "s703_ext": (Path(args.s703_train), "maps_no_tta"),
        },
    )
    attach_external(
        test_samples,
        {
            "dfaware_ext": (Path(args.dfaware_test), "maps_candidate"),
            "hfm_ext": (Path(args.hfm_test), "maps_candidate"),
            "s703_ext": (Path(args.s703_test), "maps_no_tta"),
        },
    )
    x, y = sample_pixels(train_samples, args.pos_per_image, args.neg_per_image, args.seed)
    print(f"train pixels={x.shape[0]} pos={int(y.sum())} neg={int((1-y).sum())} dim={x.shape[1]}", flush=True)
    model = train_model(x, y, args, device)
    maps = predict_maps(model, test_samples, device)
    thresholds = parse_values(args.thresholds, float)
    min_areas = parse_values(args.min_areas, int)
    rows = []
    for th in thresholds:
        rows.append({"kind": "threshold", "threshold": th, "min_area": 1, **metric_from_maps(test_samples, maps, th)})
    for th in thresholds:
        for area in min_areas:
            if area <= 1:
                continue
            binary_maps = [min_area_map(m, th, area) for m in maps]
            rows.append({"kind": "component", "threshold": th, "min_area": area, **metric_from_maps(test_samples, binary_maps, 0.5)})
    rows.sort(key=lambda r: r["mIoU"], reverse=True)
    out = {"args": vars(args), "train_pixels": int(x.shape[0]), "train_positive": int(y.sum()), "best": rows[0], "rows": rows}
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"best": rows[0], "train_pixels": int(x.shape[0]), "train_positive": int(y.sum())}, indent=2), flush=True)


if __name__ == "__main__":
    main()
