#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

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


def gn(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(max(1, min(16, channels // 4)), channels)


class LargeKernelCue(nn.Module):
    """HFM/DFAware-inspired large-kernel context block without Mamba/selective-scan."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.dw13 = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1), groups=channels)
        self.dw31 = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0), groups=channels)
        self.dw15 = nn.Conv2d(channels, channels, (1, 5), padding=(0, 2), groups=channels)
        self.dw51 = nn.Conv2d(channels, channels, (5, 1), padding=(2, 0), groups=channels)
        self.dw17 = nn.Conv2d(channels, channels, (1, 7), padding=(0, 3), groups=channels)
        self.dw71 = nn.Conv2d(channels, channels, (7, 1), padding=(3, 0), groups=channels)
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1),
            gn(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.dw3(x)
        k3 = self.dw31(self.dw13(base))
        k5 = self.dw51(self.dw15(base))
        k7 = self.dw71(self.dw17(base))
        return x + self.mix(torch.cat([base, k3, k5, k7], dim=1))


class DynamicFrequencyGate(nn.Module):
    """Cheap dynamic high-frequency cue using DoG/Laplacian-like depthwise filters."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        lap = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
        blur = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=torch.float32) / 16.0
        self.register_buffer("lap_kernel", lap.view(1, 1, 3, 3))
        self.register_buffer("blur_kernel", blur.view(1, 1, 3, 3))
        self.gate = nn.Sequential(
            nn.Conv2d(3, 12, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(12, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, 1),
            gn(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        lap_kernel = self.lap_kernel.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
        blur_kernel = self.blur_kernel.to(dtype=x.dtype, device=x.device).repeat(c, 1, 1, 1)
        lap = F.conv2d(x, lap_kernel, padding=1, groups=c).abs().mean(dim=1, keepdim=True)
        blur = F.conv2d(x, blur_kernel, padding=1, groups=c)
        dog = (x - blur).abs().mean(dim=1, keepdim=True)
        local = (x - F.avg_pool2d(x, 7, stride=1, padding=3)).abs().mean(dim=1, keepdim=True)
        gate = self.gate(torch.cat([lap, dog, local], dim=1))
        return x + gate * self.proj(x)


class HFMLKBlock(nn.Module):
    def __init__(self, cin: int, cout: int, use_context: bool = True) -> None:
        super().__init__()
        self.stem = ConvBlock(cin, cout)
        self.context = LargeKernelCue(cout) if use_context else nn.Identity()
        self.freq = DynamicFrequencyGate(cout)
        self.out = nn.Sequential(nn.Conv2d(cout, cout, 3, padding=1), gn(cout), nn.SiLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.stem(x)
        y = self.context(y)
        y = self.freq(y)
        return self.out(y)


class HFMLKAnchorResidualUNet(nn.Module):
    def __init__(self, in_ch: int, base: int = 48) -> None:
        super().__init__()
        self.e1 = HFMLKBlock(in_ch, base)
        self.e2 = HFMLKBlock(base, base * 2)
        self.e3 = HFMLKBlock(base * 2, base * 4)
        self.e4 = ConvBlock(base * 4, base * 8)
        self.mid = HFMLKBlock(base * 8, base * 8)
        self.d3 = HFMLKBlock(base * 12, base * 4, use_context=False)
        self.d2 = HFMLKBlock(base * 6, base * 2, use_context=False)
        self.d1 = HFMLKBlock(base * 3, base, use_context=False)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        mid = self.mid(e4)
        d3 = F.interpolate(mid, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.d3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HFM/DFAware-inspired large-kernel frequency teacher-anchor refiner.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--train_cache_spec", action="append", required=True, help="DATASET|CACHE")
    parser.add_argument("--test_cache_spec", action="append", required=True, help="DATASET|CACHE")
    parser.add_argument("--teacher_mode", default="avg")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--base_channels", type=int, default=48)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--residual_scale", type=float, default=0.4)
    parser.add_argument("--anchor_weight", type=float, default=0.02)
    parser.add_argument("--pos_weight", type=float, default=30.0)
    parser.add_argument("--hard_clutter_weight", type=float, default=0.0)
    parser.add_argument("--hard_clutter_threshold", type=float, default=0.35)
    parser.add_argument("--miss_weight", type=float, default=0.0)
    parser.add_argument("--miss_threshold", type=float, default=0.20)
    parser.add_argument("--uncertainty_weight", type=float, default=0.0)
    parser.add_argument("--rank_weight", type=float, default=0.0)
    parser.add_argument("--rank_margin", type=float, default=0.35)
    parser.add_argument("--rank_topk", type=int, default=512)
    parser.add_argument("--fp_topk_weight", type=float, default=0.0)
    parser.add_argument("--fp_topk_ratio", type=float, default=0.002)
    parser.add_argument("--fp_topk_min", type=int, default=128)
    parser.add_argument("--pos_peak_weight", type=float, default=0.0)
    parser.add_argument("--pos_peak_margin", type=float, default=4.0)
    parser.add_argument("--weighted_dice", action="store_true")
    parser.add_argument("--thresholds", default="0.30,0.34,0.38,0.42,0.46,0.50,0.54,0.58,0.62,0.66")
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--max_train_samples_per_cache", type=int, default=0)
    parser.add_argument("--eval_max_samples", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--focus_dataset", default="")
    parser.add_argument("--focus_patch_prob", type=float, default=0.0)
    parser.add_argument("--focus_hard_threshold", type=float, default=0.45)
    parser.add_argument("--focus_miss_threshold", type=float, default=0.18)
    parser.add_argument("--copy_paste_prob", type=float, default=0.0)
    parser.add_argument("--copy_paste_max_area", type=int, default=96)
    parser.add_argument("--gamma_prob", type=float, default=0.0)
    parser.add_argument("--noise_std", type=float, default=0.0)
    parser.add_argument("--image_aug_mode", choices=["image", "feature", "none"], default="image")
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume_checkpoint", default="")
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
            AnchorTrainSet(
                dataset_dir,
                dataset_name,
                Path(cache),
                args.teacher_mode,
                args.patch_size,
                args.max_train_samples_per_cache,
                args.seed + len(train_sets),
                args.focus_dataset,
                args.focus_patch_prob,
                args.focus_hard_threshold,
                args.focus_miss_threshold,
                args.copy_paste_prob,
                args.copy_paste_max_area,
                args.gamma_prob,
                args.noise_std,
                args.image_aug_mode,
            )
        )
    test_sets = []
    for raw in args.test_cache_spec:
        dataset_name, cache = raw.split("|", 1)
        test_sets.append((dataset_name, AnchorTestSet(dataset_dir, dataset_name, Path(cache), args.teacher_mode, args.eval_max_samples)))

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    loader = DataLoader(
        ConcatDataset(train_sets),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    in_ch = train_sets[0][0][0].shape[0]
    model = HFMLKAnchorResidualUNet(in_ch=in_ch, base=args.base_channels).to(device)
    if args.resume_checkpoint:
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=True)
        print(f"loaded resume checkpoint {args.resume_checkpoint}", flush=True)
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
                    rank_loss = logits.new_tensor(0.0)
                    if args.rank_weight > 0.0:
                        flat_logits = logits.flatten(1)
                        flat_gt = gt.flatten(1)
                        flat_teacher = teacher_prob.flatten(1)
                        rank_terms = []
                        for bi in range(flat_logits.shape[0]):
                            pos_mask = flat_gt[bi] > 0.5
                            neg_mask = flat_gt[bi] <= 0.5
                            if not bool(pos_mask.any()) or not bool(neg_mask.any()):
                                continue
                            pos_logits = flat_logits[bi][pos_mask]
                            neg_logits = flat_logits[bi][neg_mask]
                            neg_teacher = flat_teacher[bi][neg_mask]
                            k_pos = min(args.rank_topk, int(pos_logits.numel()))
                            k_neg = min(args.rank_topk, int(neg_logits.numel()))
                            pos_pick = torch.topk(pos_logits, k_pos, largest=False).values
                            neg_idx = torch.topk(neg_teacher, k_neg, largest=True).indices
                            neg_pick = neg_logits[neg_idx]
                            rank_terms.append(F.softplus(args.rank_margin - pos_pick.mean() + neg_pick.mean()))
                        if rank_terms:
                            rank_loss = torch.stack(rank_terms).mean()
                    fp_topk_loss = logits.new_tensor(0.0)
                    if args.fp_topk_weight > 0.0:
                        flat_logits = logits.flatten(1)
                        flat_gt = gt.flatten(1)
                        fp_terms = []
                        for bi in range(flat_logits.shape[0]):
                            bg_logits = flat_logits[bi][flat_gt[bi] <= 0.5]
                            if bg_logits.numel() == 0:
                                continue
                            k = max(args.fp_topk_min, int(bg_logits.numel() * args.fp_topk_ratio))
                            k = min(k, int(bg_logits.numel()))
                            hard_bg = torch.topk(bg_logits, k, largest=True).values
                            fp_terms.append(F.softplus(hard_bg).mean())
                        if fp_terms:
                            fp_topk_loss = torch.stack(fp_terms).mean()
                    pos_peak_loss = logits.new_tensor(0.0)
                    if args.pos_peak_weight > 0.0:
                        flat_logits = logits.flatten(1)
                        flat_gt = gt.flatten(1)
                        peak_terms = []
                        for bi in range(flat_logits.shape[0]):
                            pos_logits = flat_logits[bi][flat_gt[bi] > 0.5]
                            if pos_logits.numel() == 0:
                                continue
                            peak_terms.append(F.softplus(args.pos_peak_margin - pos_logits.max()))
                        if peak_terms:
                            pos_peak_loss = torch.stack(peak_terms).mean()
                    loss = (
                        bce
                        + dloss
                        + args.anchor_weight * residual.square().mean()
                        + args.rank_weight * rank_loss
                        + args.fp_topk_weight * fp_topk_loss
                        + args.pos_peak_weight * pos_peak_loss
                    )
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
