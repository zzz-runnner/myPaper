#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_component_fusion_learner import metric_from_maps  # noqa: E402
from train_teacher_anchor_refiner_from_cache import (  # noqa: E402
    AnchorResidualUNet,
    AnchorTestSet,
    AnchorTrainSet,
    dice_loss,
    parse_csv,
)


def soft_sdm_loss(logits: torch.Tensor, target: torch.Tensor, delta: float = 0.5, with_distance: bool = True) -> torch.Tensor:
    pred = torch.sigmoid(logits).clamp(1e-5, 1.0 - 1e-5)
    target = target.float()
    smooth = 1e-8
    batch, _, height, width = pred.shape
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    pred_sum = pred_flat.sum(dim=1)
    target_sum = target_flat.sum(dim=1)
    union = pred_sum + target_sum - intersection
    iou = (intersection + smooth) / (union + smooth)
    scale_delta = ((pred_sum - target_sum) * 0.5).pow(2)
    alpha = (torch.minimum(pred_sum, target_sum) + scale_delta + smooth) / (
        torch.maximum(pred_sum, target_sum) + scale_delta + smooth
    )
    scaled_iou = alpha * iou
    if not with_distance:
        return 1.0 - scaled_iou.mean()
    positive = target_sum > 0
    if not torch.any(positive):
        return 1.0 - scaled_iou.mean()
    x_index = torch.arange(0, width, device=pred.device, dtype=pred.dtype).view(1, 1, 1, width) / float(width)
    y_index = torch.arange(0, height, device=pred.device, dtype=pred.dtype).view(1, 1, height, 1) / float(height)
    pred_mass = pred.sum(dim=(1, 2, 3)).clamp_min(smooth)
    target_mass = target.sum(dim=(1, 2, 3)).clamp_min(smooth)
    pred_cx = (x_index * pred).sum(dim=(1, 2, 3)) / pred_mass
    pred_cy = (y_index * pred).sum(dim=(1, 2, 3)) / pred_mass
    target_cx = (x_index * target).sum(dim=(1, 2, 3)) / target_mass
    target_cy = (y_index * target).sum(dim=(1, 2, 3)) / target_mass
    center_dist = ((pred_cx - target_cx).pow(2) + (pred_cy - target_cy).pow(2))[positive].mean()
    resize_factor = 512.0 * 512.0 / float(max(1, width * height))
    beta = target_sum * delta * resize_factor / 81.0
    beta = torch.where(beta > delta, torch.full_like(beta, delta), beta).mean()
    return (1.0 + beta) * (1.0 - scaled_iou.mean()) + (1.0 - beta) * center_dist


def thermal_spot_loss(logits: torch.Tensor, target: torch.Tensor, sigma_scale: float, sigma_min: float, sigma_max: float, guard: int, contrast_weight: float, scale_weight: float) -> torch.Tensor:
    pred = torch.sigmoid(logits).clamp(1e-5, 1.0 - 1e-5)
    target = target.float()
    batch, _, height, width = pred.shape
    smooth = 1e-6
    target_sum = target.flatten(1).sum(dim=1)
    positive = (target_sum > 0).float().view(batch, 1, 1, 1)
    if float(positive.sum().detach().cpu()) <= 0:
        return pred.new_tensor(0.0)
    x_index = torch.linspace(0.0, float(width - 1), width, device=pred.device, dtype=pred.dtype).view(1, 1, 1, width)
    y_index = torch.linspace(0.0, float(height - 1), height, device=pred.device, dtype=pred.dtype).view(1, 1, height, 1)
    mass = target_sum.clamp_min(smooth).view(batch, 1, 1, 1)
    center_x = (target * x_index).sum(dim=(1, 2, 3)).view(batch, 1, 1, 1) / mass
    center_y = (target * y_index).sum(dim=(1, 2, 3)).view(batch, 1, 1, 1) / mass
    radius = torch.sqrt(target_sum.clamp_min(1.0) / math.pi).view(batch, 1, 1, 1)
    sigma = (radius * sigma_scale).clamp(min=sigma_min, max=sigma_max)
    dist2 = (x_index - center_x).pow(2) + (y_index - center_y).pow(2)
    gaussian = torch.exp(-dist2 / (2.0 * sigma.pow(2) + smooth)) * positive
    if guard % 2 == 0:
        guard += 1
    guard_mask = F.max_pool2d(target, max(1, guard), stride=1, padding=max(1, guard) // 2).detach()
    soft_target = (gaussian * guard_mask).detach()
    weight = (guard_mask * (1.0 + soft_target)).detach()
    shape = (F.binary_cross_entropy(pred, soft_target, reduction="none") * weight).sum() / weight.sum().clamp_min(1.0)
    core_mass = soft_target.flatten(1).sum(dim=1).clamp_min(smooth).view(batch, 1, 1, 1)
    center_score = (pred * soft_target).sum(dim=(1, 2, 3)).view(batch, 1, 1, 1) / core_mass
    ring = (guard_mask * (1.0 - soft_target)).detach()
    ring_mass = ring.flatten(1).sum(dim=1).clamp_min(smooth).view(batch, 1, 1, 1)
    ring_score = (pred * ring).sum(dim=(1, 2, 3)).view(batch, 1, 1, 1) / ring_mass
    contrast = F.relu(0.05 + ring_score - center_score)
    contrast = (contrast * positive).sum() / positive.sum().clamp_min(1.0)
    pred_guard_mass = (pred * guard_mask).flatten(1).sum(dim=1)
    soft_mass = soft_target.flatten(1).sum(dim=1).clamp_min(1.0)
    scale = ((pred_guard_mass - soft_mass).pow(2) / soft_mass.pow(2)).view(batch, 1, 1, 1)
    scale = (scale * positive).sum() / positive.sum().clamp_min(1.0)
    return shape + contrast_weight * contrast + scale_weight * scale


def background_hf_loss(logits: torch.Tensor, target: torch.Tensor, feat: torch.Tensor, weight: float, mass_weight: float, dilate: int) -> torch.Tensor:
    if weight <= 0 and mass_weight <= 0:
        return logits.new_tensor(0.0)
    pred = torch.sigmoid(logits)
    image = feat[:, 0:1]
    if dilate % 2 == 0:
        dilate += 1
    guard = F.max_pool2d(target, max(1, dilate), stride=1, padding=max(1, dilate) // 2).detach()
    bg = (1.0 - guard).detach()
    with torch.no_grad():
        kernel = logits.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
        hf = F.conv2d(image, kernel, padding=1).abs()
        hf = hf / (hf.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        hf = hf.clamp(max=6.0)
    hf_term = (pred.pow(2) * bg * hf).sum() / (bg * hf).sum().clamp_min(1.0)
    mass_term = (pred * bg).sum() / bg.sum().clamp_min(1.0)
    return weight * hf_term + mass_weight * mass_term


def main() -> None:
    parser = argparse.ArgumentParser(description="Train teacher-anchored refiner with SLS/SDM and thermal-spot high-frequency clutter losses.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--train_cache_spec", action="append", required=True, help="DATASET|CACHE")
    parser.add_argument("--test_cache_spec", action="append", required=True, help="DATASET|CACHE")
    parser.add_argument("--teacher_mode", default="avg")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--residual_scale", type=float, default=0.5)
    parser.add_argument("--anchor_weight", type=float, default=0.02)
    parser.add_argument("--pos_weight", type=float, default=25.0)
    parser.add_argument("--sdm_weight", type=float, default=0.25)
    parser.add_argument("--spot_weight", type=float, default=0.08)
    parser.add_argument("--bg_hf_weight", type=float, default=0.02)
    parser.add_argument("--bg_mass_weight", type=float, default=0.01)
    parser.add_argument("--thresholds", default="0.30,0.35,0.368,0.38,0.40,0.42,0.45,0.50,0.55,0.60")
    parser.add_argument("--eval_interval", type=int, default=2)
    parser.add_argument("--max_train_samples_per_cache", type=int, default=0)
    parser.add_argument("--eval_max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--device", default="cuda:0")
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
        train_sets.append(AnchorTrainSet(dataset_dir, dataset_name, Path(cache), args.teacher_mode, args.patch_size, args.max_train_samples_per_cache, args.seed + len(train_sets)))
    test_sets = []
    for raw in args.test_cache_spec:
        dataset_name, cache = raw.split("|", 1)
        test_sets.append((dataset_name, AnchorTestSet(dataset_dir, dataset_name, Path(cache), args.teacher_mode, args.eval_max_samples)))
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    loader = DataLoader(ConcatDataset(train_sets), batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=device.type == "cuda")
    in_ch = train_sets[0][0][0].shape[0]
    model = AnchorResidualUNet(in_ch=in_ch, base=args.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)
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
                residual = args.residual_scale * torch.tanh(model(feat))
                logits = tlogit + residual
                bce = F.binary_cross_entropy_with_logits(logits, gt, pos_weight=pos_weight)
                loss = bce + dice_loss(logits, gt) + args.anchor_weight * residual.square().mean()
                loss = loss + args.sdm_weight * soft_sdm_loss(logits, gt)
                loss = loss + args.spot_weight * thermal_spot_loss(logits, gt, 1.2, 1.0, 6.0, 9, 0.5, 0.1)
                loss = loss + background_hf_loss(logits, gt, feat, args.bg_hf_weight, args.bg_mass_weight, 9)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
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
                        with torch.no_grad():
                            residual = args.residual_scale * torch.tanh(model(feat))
                            prob = torch.sigmoid(tlogit + residual)[0, 0].detach().cpu().numpy()
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
