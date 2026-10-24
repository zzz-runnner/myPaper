#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_component_fusion_learner import metric_from_maps  # noqa: E402
from train_teacher_anchor_refiner_from_cache import (  # noqa: E402
    AnchorTestSet,
    AnchorTrainSet,
    ConvBlock,
    dice_loss,
    parse_csv,
    weighted_dice_loss,
)

import numpy as np  # noqa: E402
from torch.utils.data import ConcatDataset, DataLoader  # noqa: E402


class HaarFrequencyGate(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        lo = torch.tensor([0.70710678, 0.70710678], dtype=torch.float32)
        hi = torch.tensor([-0.70710678, 0.70710678], dtype=torch.float32)
        kernels = torch.stack(
            [
                torch.outer(lo, hi),
                torch.outer(hi, lo),
                torch.outer(hi, hi),
                torch.outer(lo, lo),
            ],
            dim=0,
        ).unsqueeze(1)
        self.register_buffer("kernels", kernels)
        self.reduce = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(max(1, min(16, channels // 4)), channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate") if pad_h or pad_w else x
        filters = self.kernels.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
        bands = F.conv2d(x_pad, filters, stride=2, groups=c)
        bands = bands.view(b, c, 4, bands.shape[-2], bands.shape[-1]).abs().mean(dim=1)
        gate = self.reduce(F.interpolate(bands, size=x_pad.shape[-2:], mode="bilinear", align_corners=False))
        gate = gate[:, :, :h, :w]
        return x + gate * self.proj(x)


class WaveletAnchorResidualUNet(nn.Module):
    def __init__(self, in_ch: int, base: int = 48) -> None:
        super().__init__()
        self.e1 = ConvBlock(in_ch, base)
        self.f1 = HaarFrequencyGate(base)
        self.e2 = ConvBlock(base, base * 2)
        self.f2 = HaarFrequencyGate(base * 2)
        self.e3 = ConvBlock(base * 2, base * 4)
        self.e4 = ConvBlock(base * 4, base * 8)
        self.mid = ConvBlock(base * 8, base * 8)
        self.fm = HaarFrequencyGate(base * 8)
        self.d3 = ConvBlock(base * 12, base * 4)
        self.d2 = ConvBlock(base * 6, base * 2)
        self.d1 = ConvBlock(base * 3, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.f1(self.e1(x))
        e2 = self.f2(self.e2(F.max_pool2d(e1, 2)))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        mid = self.fm(self.mid(e4))
        d3 = F.interpolate(mid, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PAL-inspired wavelet-frequency teacher-anchor refiner.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--train_cache_spec", action="append", required=True, help="DATASET|CACHE")
    parser.add_argument("--test_cache_spec", action="append", required=True, help="DATASET|CACHE")
    parser.add_argument("--teacher_mode", default="avg")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--base_channels", type=int, default=56)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--residual_scale", type=float, default=0.4)
    parser.add_argument("--anchor_weight", type=float, default=0.02)
    parser.add_argument("--pos_weight", type=float, default=30.0)
    parser.add_argument("--hard_clutter_weight", type=float, default=0.0)
    parser.add_argument("--hard_clutter_threshold", type=float, default=0.35)
    parser.add_argument("--miss_weight", type=float, default=0.0)
    parser.add_argument("--miss_threshold", type=float, default=0.20)
    parser.add_argument("--uncertainty_weight", type=float, default=0.0)
    parser.add_argument("--weighted_dice", action="store_true")
    parser.add_argument("--thresholds", default="0.30,0.34,0.38,0.42,0.46,0.50,0.54,0.58,0.62,0.66")
    parser.add_argument("--eval_interval", type=int, default=2)
    parser.add_argument("--max_train_samples_per_cache", type=int, default=0)
    parser.add_argument("--eval_max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dataset_dir = Path(args.dataset_dir)
    train_sets = []
    for raw in args.train_cache_spec:
        dataset_name, cache = raw.split("|", 1)
        train_sets.append(
            AnchorTrainSet(dataset_dir, dataset_name, Path(cache), args.teacher_mode, args.patch_size, args.max_train_samples_per_cache, args.seed + len(train_sets))
        )
    test_sets = []
    for raw in args.test_cache_spec:
        dataset_name, cache = raw.split("|", 1)
        test_sets.append((dataset_name, AnchorTestSet(dataset_dir, dataset_name, Path(cache), args.teacher_mode, args.eval_max_samples)))

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    loader = DataLoader(ConcatDataset(train_sets), batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=device.type == "cuda")
    in_ch = train_sets[0][0][0].shape[0]
    model = WaveletAnchorResidualUNet(in_ch=in_ch, base=args.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    thresholds = [float(x) for x in parse_csv(args.thresholds)]
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    best = None
    history = []
    pos_weight = torch.tensor([args.pos_weight], dtype=torch.float32, device=device).view(1, 1, 1, 1)

    with (run_dir / "train.log").open("w", encoding="utf-8") as log:
        log.write(json.dumps(vars(args), indent=2) + "\n")
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            model.train()
            losses = []
            for feat, gt, tlogit in loader:
                feat = feat.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)
                tlogit = tlogit.to(device, non_blocking=True)
                with torch.no_grad():
                    teacher_prob = torch.sigmoid(tlogit)
                    hard_clutter = (teacher_prob > args.hard_clutter_threshold).float() * (1.0 - gt)
                    missed_target = (teacher_prob < args.miss_threshold).float() * gt
                    uncertainty = 4.0 * teacher_prob * (1.0 - teacher_prob)
                    pixel_weight = 1.0 + args.hard_clutter_weight * hard_clutter + args.miss_weight * missed_target + args.uncertainty_weight * uncertainty
                with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                    residual = args.residual_scale * torch.tanh(model(feat))
                    logits = tlogit + residual
                    bce = F.binary_cross_entropy_with_logits(logits, gt, weight=pixel_weight, pos_weight=pos_weight)
                    dloss = weighted_dice_loss(logits, gt, pixel_weight) if args.weighted_dice else dice_loss(logits, gt)
                    loss = bce + dloss + args.anchor_weight * residual.square().mean()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
            sched.step()
            row = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else 0.0, "lr": float(sched.get_last_lr()[0]), "seconds": time.time() - start}
            if epoch % args.eval_interval == 0 or epoch == args.epochs:
                evals = {}
                model.eval()
                for name, ds in test_sets:
                    samples = []
                    maps = []
                    for feat, tlogit, mask, size, sample_name in DataLoader(ds, batch_size=1, shuffle=False, num_workers=1):
                        feat = feat.to(device)
                        tlogit = tlogit.to(device)
                        with torch.no_grad(), torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                            residual = args.residual_scale * torch.tanh(model(feat))
                            prob = torch.sigmoid(tlogit + residual)[0, 0].detach().float().cpu().numpy()
                        h, w = int(size[0]), int(size[1])
                        maps.append(prob[:h, :w].astype(np.float32))
                        samples.append({"name": str(sample_name[0]), "mask": mask.numpy()[0].astype(np.float32), "size": [h, w]})
                    rows = [{"threshold": th, **metric_from_maps(samples, maps, th)} for th in thresholds]
                    rows.sort(key=lambda r: r["mIoU"], reverse=True)
                    evals[name] = rows[0]
                row["eval"] = evals
                mean_miou = float(np.mean([item["mIoU"] for item in evals.values()]))
                if best is None or mean_miou > best["mean_mIoU"]:
                    best = {"epoch": epoch, "mean_mIoU": mean_miou, "eval": evals}
                    torch.save({"epoch": epoch, "state_dict": model.state_dict(), "best": best}, run_dir / "best.pth.tar")
            history.append(row)
            line = json.dumps(row, sort_keys=True)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
    (run_dir / "summary.json").write_text(json.dumps({"args": vars(args), "best": best, "history": history}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
