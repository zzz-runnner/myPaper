#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn import init
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


ROOT = Path(__file__).resolve().parents[1]
SCTRANSNET = ROOT / "external" / "SCTransNet"
sys.path.insert(0, str(SCTRANSNET))

from dataset import TestSetLoader, augumentation  # noqa: E402
from metrics import PD_FA, mIoU  # noqa: E402
from model.SCTransNet import SCTransNet  # noqa: E402
from utils import Normalized, get_img_norm_cfg, random_crop, seed_pytorch  # noqa: E402
import model.Config as config  # noqa: E402


def weights_init_kaiming(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and hasattr(m, "weight"):
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("Linear") != -1 and hasattr(m, "weight"):
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("BatchNorm") != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module.model.model."):
            normalized["model." + key[len("module.model.model.") :]] = value
        elif key.startswith("model.model."):
            normalized["model." + key[len("model.model.") :]] = value
        elif key.startswith("module.model."):
            normalized[key[len("module.model.") :]] = value
        elif key.startswith("model."):
            normalized[key[len("model.") :]] = value
        elif key.startswith("module."):
            normalized[key[len("module.") :]] = value
        else:
            normalized[key] = value
    return normalized


def load_checkpoint_into_model(model: nn.Module, checkpoint: Path) -> tuple[list[str], list[str]]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    state_dict = normalize_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    model_keys = model.state_dict().keys()
    if any(key.startswith("model.") for key in model_keys):
        state_dict = {
            (key if key.startswith("adapter.") or key.startswith("model.") else f"model.{key}"): value
            for key, value in state_dict.items()
        }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return list(missing), list(unexpected)


class IRSTDOurTrainSet(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        dataset_name: str,
        patch_size: int,
        pos_prob: float,
        noise_std: float,
        gamma_prob: float,
        intensity_jitter: float,
        bgm_prob: float,
        bgm_region_size: int,
        bgm_mask_ratio: float,
        paired_bgm: bool,
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir / dataset_name
        self.patch_size = patch_size
        self.pos_prob = pos_prob
        self.noise_std = noise_std
        self.gamma_prob = gamma_prob
        self.intensity_jitter = intensity_jitter
        self.bgm_prob = bgm_prob
        self.bgm_region_size = bgm_region_size
        self.bgm_mask_ratio = bgm_mask_ratio
        self.paired_bgm = paired_bgm
        self.img_norm_cfg = get_img_norm_cfg(dataset_name, str(dataset_dir))
        self.transform = augumentation()
        with (self.dataset_dir / "img_idx" / f"train_{dataset_name}.txt").open("r") as f:
            self.train_list = f.read().splitlines()

    def __len__(self) -> int:
        return len(self.train_list)

    def _open_pair(self, name: str) -> tuple[Image.Image, Image.Image]:
        for ext in (".png", ".bmp", ".jpg"):
            img_path = self.dataset_dir / "images" / f"{name}{ext}"
            mask_path = self.dataset_dir / "masks" / f"{name}{ext}"
            if img_path.exists() and mask_path.exists():
                return Image.open(img_path).convert("I"), Image.open(mask_path)
        raise FileNotFoundError(name)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_pil, mask_pil = self._open_pair(self.train_list[idx])
        img = Normalized(np.array(img_pil, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask_pil, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]

        if self.intensity_jitter > 0 and random.random() < 0.8:
            scale = random.uniform(1.0 - self.intensity_jitter, 1.0 + self.intensity_jitter)
            shift = random.uniform(-self.intensity_jitter, self.intensity_jitter)
            img = img * scale + shift

        if self.gamma_prob > 0 and random.random() < self.gamma_prob:
            min_v = float(img.min())
            rng = float(img.max() - min_v)
            if rng > 1e-6:
                gamma = random.uniform(0.65, 1.55)
                x = np.clip((img - min_v) / rng, 0.0, 1.0)
                img = np.power(x, gamma) * rng + min_v

        if self.noise_std > 0 and random.random() < 0.7:
            img = img + np.random.normal(0.0, self.noise_std, size=img.shape).astype(np.float32)

        img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=self.pos_prob)
        img_patch, mask_patch = self.transform(img_patch, mask_patch)
        if self.paired_bgm:
            bgm_patch = self._background_masked_view(img_patch, mask_patch)
            img_patch = np.stack([img_patch, bgm_patch], axis=0)
            mask_patch = np.stack([mask_patch, mask_patch], axis=0)
        elif self.bgm_prob > 0 and random.random() < self.bgm_prob:
            img_patch = self._background_masked_view(img_patch, mask_patch)
            img_patch = img_patch[np.newaxis, :]
            mask_patch = mask_patch[np.newaxis, :]
        else:
            img_patch = img_patch[np.newaxis, :]
            mask_patch = mask_patch[np.newaxis, :]
        img_patch = torch.from_numpy(np.ascontiguousarray(img_patch)).float()
        mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch)).float()
        return img_patch, mask_patch

    def _background_masked_view(self, img_patch: np.ndarray, mask_patch: np.ndarray) -> np.ndarray:
        region_size = max(1, int(self.bgm_region_size))
        h, w = img_patch.shape
        grid_h = int(math.ceil(h / region_size))
        grid_w = int(math.ceil(w / region_size))
        keep = np.zeros(grid_h * grid_w, dtype=np.float32)
        keep[: int(round(keep.size * self.bgm_mask_ratio))] = 1.0
        np.random.shuffle(keep)
        keep = keep.reshape(grid_h, grid_w)
        keep = keep.repeat(region_size, axis=0)[:h, :].repeat(region_size, axis=1)[:, :w]

        guard = torch.from_numpy(mask_patch.astype(np.float32)).view(1, 1, h, w)
        guard = F.max_pool2d(guard, kernel_size=7, stride=1, padding=3).view(h, w).numpy()
        keep[guard > 0] = 0.0
        out = img_patch.copy()
        out[keep > 0] = float(img_patch.mean())
        return out


class ModelEma:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for key, ema_value in self.module.state_dict().items():
            model_value = msd[key].detach()
            if torch.is_floating_point(ema_value):
                ema_value.copy_(ema_value * self.decay + model_value * (1.0 - self.decay))
            else:
                ema_value.copy_(model_value)


class TCDInputAdapter(nn.Module):
    def __init__(self, hidden_channels: int = 8, residual_scale: float = 0.10) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.proj = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        laplace = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
        self.register_buffer("laplace_kernel", laplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        high = F.conv2d(x, self.laplace_kernel.to(dtype=x.dtype), padding=1)
        local_mean = F.avg_pool2d(x, kernel_size=9, stride=1, padding=4)
        contrast = x - local_mean
        features = torch.cat([x, high, contrast], dim=1)
        return x + self.residual_scale * self.proj(features)


class SCTransNetWithInputAdapter(nn.Module):
    def __init__(self, base_model: nn.Module, hidden_channels: int, residual_scale: float) -> None:
        super().__init__()
        self.adapter = TCDInputAdapter(hidden_channels=hidden_channels, residual_scale=residual_scale)
        self.model = base_model

    def forward(self, x: torch.Tensor):
        return self.model(self.adapter(x))


class TCDOutputRefiner(nn.Module):
    def __init__(self, hidden_channels: int = 8, residual_scale: float = 0.10) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.proj = nn.Sequential(
            nn.Conv2d(4, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        laplace = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
        self.register_buffer("laplace_kernel", laplace)
        self.reset_output()

    def reset_output(self) -> None:
        init.constant_(self.proj[-1].weight, 0.0)
        init.constant_(self.proj[-1].bias, 0.0)

    def _refine_one(self, image: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        if image.shape[-2:] != pred.shape[-2:]:
            image = F.interpolate(image, size=pred.shape[-2:], mode="bilinear", align_corners=False)
        high = F.conv2d(image, self.laplace_kernel.to(dtype=image.dtype), padding=1).abs()
        high = high / (high.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        contrast = image - F.avg_pool2d(image, kernel_size=9, stride=1, padding=4)
        features = torch.cat([pred.detach(), image, high.clamp(max=6.0), contrast], dim=1)
        pred = pred.clamp(1e-5, 1.0 - 1e-5)
        logits = torch.logit(pred)
        return torch.sigmoid(logits + self.residual_scale * self.proj(features))

    def forward(self, image: torch.Tensor, preds):
        if isinstance(preds, (tuple, list)):
            refined = list(preds)
            refined[-1] = self._refine_one(image, refined[-1])
            return tuple(refined)
        return self._refine_one(image, preds)


class TCDSCTransNetWrapper(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        input_adapter: bool,
        adapter_hidden: int,
        adapter_scale: float,
        output_refiner: bool,
        refiner_hidden: int,
        refiner_scale: float,
    ) -> None:
        super().__init__()
        self.adapter = TCDInputAdapter(hidden_channels=adapter_hidden, residual_scale=adapter_scale) if input_adapter else None
        self.model = base_model
        self.output_refiner = TCDOutputRefiner(hidden_channels=refiner_hidden, residual_scale=refiner_scale) if output_refiner else None

    def forward(self, x: torch.Tensor):
        model_input = self.adapter(x) if self.adapter is not None else x
        preds = self.model(model_input)
        if self.output_refiner is not None:
            preds = self.output_refiner(x, preds)
        return preds


def set_base_model_trainable(model: nn.Module, trainable: bool) -> None:
    for name, param in model.named_parameters():
        if name.startswith("model."):
            param.requires_grad_(trainable)


def split_weight_decay_params(model: nn.Module) -> list[dict[str, object]]:
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower() or "bn" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]


def make_scheduler(optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int, min_lr_ratio: float):
    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    inter = (pred_flat * target_flat).sum(dim=1)
    denom = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    return (1.0 - (2.0 * inter + smooth) / (denom + smooth)).mean()


def focal_bce_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
    pred = pred.clamp(1e-5, 1.0 - 1e-5)
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    pt = torch.where(target > 0.5, pred, 1.0 - pred)
    alpha_t = torch.where(target > 0.5, torch.full_like(target, alpha), torch.full_like(target, 1.0 - alpha))
    return (alpha_t * (1.0 - pt).pow(gamma) * bce).mean()


def sls_iou_loss(pred: torch.Tensor, target: torch.Tensor, location_weight: float) -> torch.Tensor:
    pred = pred.clamp(1e-5, 1.0 - 1e-5)
    target = target.float()
    smooth = 1e-6

    intersection = (pred * target).flatten(1).sum(dim=1)
    pred_sum = pred.flatten(1).sum(dim=1)
    target_sum = target.flatten(1).sum(dim=1)
    union = pred_sum + target_sum - intersection
    iou = (intersection + smooth) / (union + smooth)

    scale_delta = ((pred_sum - target_sum) * 0.5).pow(2)
    scale_alpha = (torch.minimum(pred_sum, target_sum) + scale_delta + smooth) / (
        torch.maximum(pred_sum, target_sum) + scale_delta + smooth
    )
    loss = 1.0 - (scale_alpha * iou).mean()

    if location_weight <= 0:
        return loss

    batch, _, height, width = pred.shape
    x_index = torch.linspace(0.0, 1.0, width, device=pred.device, dtype=pred.dtype).view(1, 1, 1, width)
    y_index = torch.linspace(0.0, 1.0, height, device=pred.device, dtype=pred.dtype).view(1, 1, height, 1)
    pred_mass = pred_sum.clamp_min(smooth).view(batch, 1, 1, 1)
    target_mass = target_sum.clamp_min(smooth).view(batch, 1, 1, 1)
    pred_cx = (pred * x_index).sum(dim=(1, 2, 3)) / pred_mass.flatten()
    pred_cy = (pred * y_index).sum(dim=(1, 2, 3)) / pred_mass.flatten()
    target_cx = (target * x_index).sum(dim=(1, 2, 3)) / target_mass.flatten()
    target_cy = (target * y_index).sum(dim=(1, 2, 3)) / target_mass.flatten()
    positive = (target_sum > 0).float()
    loc = ((pred_cx - target_cx).pow(2) + (pred_cy - target_cy).pow(2)) * positive
    loc = loc.sum() / positive.sum().clamp_min(1.0)
    return loss + location_weight * loc


def background_hf_suppression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    image: torch.Tensor,
    bg_hf_weight: float,
    bg_pred_weight: float,
    target_dilate: int,
    hf_clip: float,
) -> torch.Tensor:
    if bg_hf_weight <= 0 and bg_pred_weight <= 0:
        return pred.new_tensor(0.0)

    pred = pred.clamp(1e-5, 1.0 - 1e-5)
    if target_dilate > 1:
        if target_dilate % 2 == 0:
            target_dilate += 1
        target_bg_guard = F.max_pool2d(target, kernel_size=target_dilate, stride=1, padding=target_dilate // 2)
    else:
        target_bg_guard = target
    bg = (1.0 - target_bg_guard).clamp(0.0, 1.0).detach()

    with torch.no_grad():
        laplace_kernel = pred.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
        hf = F.conv2d(image, laplace_kernel, padding=1).abs()
        hf = hf / (hf.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        hf = hf.clamp(max=hf_clip)

    denom_hf = (bg * hf).sum().clamp_min(1.0)
    denom_bg = bg.sum().clamp_min(1.0)
    hf_penalty = ((pred.pow(2) * bg * hf).sum() / denom_hf) if bg_hf_weight > 0 else pred.new_tensor(0.0)
    bg_mass_penalty = ((pred * bg).sum() / denom_bg) if bg_pred_weight > 0 else pred.new_tensor(0.0)
    return bg_hf_weight * hf_penalty + bg_pred_weight * bg_mass_penalty


def tcd_loss_single(
    pred: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float,
    dice_weight: float,
    focal_weight: float,
    sls_weight: float,
    pos_weight: float,
    hard_neg_weight: float,
    focal_alpha: float,
    focal_gamma: float,
) -> torch.Tensor:
    pred = pred.clamp(1e-5, 1.0 - 1e-5)
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    with torch.no_grad():
        pos = (target > 0.5).float()
        hard_neg = ((pred > 0.2) & (target <= 0.5)).float()
        area = pos.flatten(1).sum(dim=1).view(-1, 1, 1, 1)
        small_target_boost = (1.0 + sls_weight / torch.sqrt(area + 1.0)).clamp(max=1.0 + sls_weight)
        weights = 1.0 + pos_weight * pos * small_target_boost + hard_neg_weight * hard_neg
    weighted_bce = (bce * weights).mean()
    return (
        bce_weight * weighted_bce
        + dice_weight * dice_loss(pred, target)
        + focal_weight * focal_bce_loss(pred, target, focal_alpha, focal_gamma)
    )


def tcd_loss(preds, target: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if isinstance(preds, (tuple, list)):
        weights = [0.25, 0.30, 0.40, 0.55, 0.75, 1.0]
        weights = weights[-len(preds):]
        total = 0.0
        norm = 0.0
        for weight, pred in zip(weights, preds):
            total = total + weight * tcd_loss_single(
                pred,
                target,
                args.bce_weight,
                args.dice_weight,
                args.focal_weight,
                args.sls_weight,
                args.pos_weight,
                args.hard_neg_weight,
                args.focal_alpha,
                args.focal_gamma,
            )
            norm += weight
        return total / norm
    return tcd_loss_single(
        preds,
        target,
        args.bce_weight,
        args.dice_weight,
        args.focal_weight,
        args.sls_weight,
        args.pos_weight,
        args.hard_neg_weight,
        args.focal_alpha,
        args.focal_gamma,
    )


def our_loss(preds, target: torch.Tensor, image: torch.Tensor, args: argparse.Namespace, epoch: int = 0) -> torch.Tensor:
    base = tcd_loss(preds, target, args)
    final_pred = preds[-1] if isinstance(preds, (tuple, list)) else preds
    bg_loss = background_hf_suppression_loss(
        final_pred,
        target,
        image,
        args.bg_hf_weight,
        args.bg_pred_weight,
        args.target_dilate,
        args.hf_clip,
    )
    sls_iou = final_pred.new_tensor(0.0)
    if args.sls_iou_weight > 0 and epoch > args.sls_iou_warm_epochs:
        sls_iou = args.sls_iou_weight * sls_iou_loss(final_pred, target, args.sls_location_weight)
    return base + bg_loss + sls_iou


@torch.no_grad()
def evaluate(model: nn.Module, dataset_dir: Path, dataset_name: str, threshold: float, device: torch.device) -> dict[str, float]:
    test_set = TestSetLoader(str(dataset_dir), dataset_name, dataset_name)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    model.eval()
    eval_miou = mIoU()
    eval_pdfa = PD_FA()
    losses = []
    for img, gt_mask, size, _ in test_loader:
        img = img.to(device)
        gt_mask = gt_mask.to(device)
        pred = model(img)
        if isinstance(pred, (tuple, list)):
            pred = pred[-1]
        pred = pred[:, :, : size[0], : size[1]]
        gt_mask = gt_mask[:, :, : size[0], : size[1]]
        losses.append(float(F.binary_cross_entropy(pred.clamp(1e-5, 1.0 - 1e-5), gt_mask).detach().cpu()))
        eval_miou.update((pred > threshold).cpu(), gt_mask.cpu())
        eval_pdfa.update((pred[0, 0] > threshold).cpu(), gt_mask[0, 0].cpu(), size)
    pix_acc, miou = eval_miou.get()
    pd, fa = eval_pdfa.get()
    return {"loss": float(np.mean(losses)), "pixAcc": float(pix_acc), "mIoU": float(miou), "Pd": float(pd), "Fa": float(fa)}


def save_checkpoint(path: Path, model: nn.Module, epoch: int, total_loss: list[float], args: argparse.Namespace, ema: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "total_loss": total_loss,
            "method": args.method_name,
            "ema": ema,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TCD-SCTransNet our-method variants on full IRSTD datasets.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--method_name", default="TCD-SCTransNet")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--min_lr_ratio", type=float, default=0.01)
    parser.add_argument("--pos_prob", type=float, default=0.75)
    parser.add_argument("--noise_std", type=float, default=0.015)
    parser.add_argument("--gamma_prob", type=float, default=0.35)
    parser.add_argument("--intensity_jitter", type=float, default=0.08)
    parser.add_argument("--bgm_prob", type=float, default=0.0)
    parser.add_argument("--bgm_region_size", type=int, default=16)
    parser.add_argument("--bgm_mask_ratio", type=float, default=0.2)
    parser.add_argument("--paired_bgm", action="store_true")
    parser.add_argument("--bce_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=0.7)
    parser.add_argument("--focal_weight", type=float, default=0.35)
    parser.add_argument("--sls_weight", type=float, default=2.0)
    parser.add_argument("--pos_weight", type=float, default=4.0)
    parser.add_argument("--hard_neg_weight", type=float, default=1.2)
    parser.add_argument("--focal_alpha", type=float, default=0.75)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--sls_iou_weight", type=float, default=0.0)
    parser.add_argument("--sls_iou_warm_epochs", type=int, default=0)
    parser.add_argument("--sls_location_weight", type=float, default=0.20)
    parser.add_argument("--bg_hf_weight", type=float, default=0.0)
    parser.add_argument("--bg_pred_weight", type=float, default=0.0)
    parser.add_argument("--target_dilate", type=int, default=7)
    parser.add_argument("--hf_clip", type=float, default=6.0)
    parser.add_argument("--input_adapter", action="store_true")
    parser.add_argument("--adapter_hidden", type=int, default=8)
    parser.add_argument("--adapter_scale", type=float, default=0.10)
    parser.add_argument("--output_refiner", action="store_true")
    parser.add_argument("--refiner_hidden", type=int, default=8)
    parser.add_argument("--refiner_scale", type=float, default=0.10)
    parser.add_argument("--freeze_base_epochs", type=int, default=0)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--begin_test", type=int, default=350)
    parser.add_argument("--every_test", type=int, default=1)
    parser.add_argument("--every_print", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--no_ema", action="store_true")
    args = parser.parse_args()

    seed_pytorch(args.seed)
    dataset_dir = Path(args.dataset_dir)
    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints" / args.dataset_name
    log_dir = run_dir / "tensorboard" / args.dataset_name
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    text_log_path = ckpt_dir / f"{args.dataset_name}_{args.method_name}_{time.ctime().replace(' ', '_').replace(':', '_')}.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = IRSTDOurTrainSet(
        dataset_dir=dataset_dir,
        dataset_name=args.dataset_name,
        patch_size=args.patch_size,
        pos_prob=args.pos_prob,
        noise_std=args.noise_std,
        gamma_prob=args.gamma_prob,
        intensity_jitter=args.intensity_jitter,
        bgm_prob=args.bgm_prob,
        bgm_region_size=args.bgm_region_size,
        bgm_mask_ratio=args.bgm_mask_ratio,
        paired_bgm=args.paired_bgm,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.threads,
        pin_memory=True,
        drop_last=True,
    )

    base_net = SCTransNet(config.get_SCTrans_config(), mode="train", deepsuper=True)
    net = (
        TCDSCTransNetWrapper(
            base_net,
            input_adapter=args.input_adapter,
            adapter_hidden=args.adapter_hidden,
            adapter_scale=args.adapter_scale,
            output_refiner=args.output_refiner,
            refiner_hidden=args.refiner_hidden,
            refiner_scale=args.refiner_scale,
        )
        if args.input_adapter or args.output_refiner
        else base_net
    ).to(device)
    net.apply(weights_init_kaiming)
    if args.resume_checkpoint:
        missing, unexpected = load_checkpoint_into_model(net, Path(args.resume_checkpoint))
        print(f"loaded checkpoint={args.resume_checkpoint} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if args.output_refiner and hasattr(net, "output_refiner") and net.output_refiner is not None:
        net.output_refiner.reset_output()
    net.train()
    ema = None if args.no_ema else ModelEma(net, args.ema_decay)

    optimizer = torch.optim.AdamW(split_weight_decay_params(net), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999), eps=1e-8)
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs, args.min_lr_ratio)
    writer = SummaryWriter(str(log_dir))

    total_loss_list: list[float] = []
    loss_window: list[float] = []
    best_miou = -1.0
    best_epoch = -1
    best_metrics: dict[str, float] | None = None

    with text_log_path.open("w") as f:
        f.write(f"method={args.method_name} dataset={args.dataset_name} train_size={len(train_set)} args={vars(args)}\n")
        print(f"{args.dataset_name}\t{args.method_name}")
        print(f"train_size={len(train_set)} batch_size={args.batch_size} run_dir={run_dir}")
        for idx_epoch in range(args.epochs):
            freeze_base = args.freeze_base_epochs > 0 and idx_epoch < args.freeze_base_epochs
            set_base_model_trainable(net, not freeze_base)
            net.train()
            for img, gt_mask in train_loader:
                if img.ndim == 5:
                    bsz, views, channels, height, width = img.shape
                    img = img.view(bsz * views, channels, height, width)
                    gt_mask = gt_mask.view(bsz * views, channels, height, width)
                elif img.ndim == 4 and img.shape[1] != 1:
                    bsz, views, height, width = img.shape
                    img = img.view(bsz * views, 1, height, width)
                    gt_mask = gt_mask.view(bsz * views, 1, height, width)
                img = img.to(device, non_blocking=True)
                gt_mask = gt_mask.to(device, non_blocking=True)
                preds = net(img)
                loss = our_loss(preds, gt_mask, img, args, idx_epoch + 1)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.clip_grad > 0:
                    nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad)
                optimizer.step()
                if ema is not None:
                    ema.update(net)
                loss_window.append(float(loss.detach().cpu()))
            scheduler.step()

            epoch_num = idx_epoch + 1
            if epoch_num % args.every_print == 0:
                mean_loss = float(np.mean(loss_window)) if loss_window else 0.0
                total_loss_list.append(mean_loss)
                lr_now = optimizer.param_groups[0]["lr"]
                line = time.ctime()[4:-5] + f" Epoch---{epoch_num}, total_loss---{mean_loss:.6f}, lr---{lr_now:.8f},"
                print(line, flush=True)
                f.write(line + "\n")
                f.flush()
                writer.add_scalar("loss", mean_loss, epoch_num)
                writer.add_scalar("lr", lr_now, epoch_num)
                loss_window = []

            if epoch_num >= args.begin_test and epoch_num % args.every_test == 0:
                eval_model = ema.module if ema is not None else net
                metrics = evaluate(eval_model, dataset_dir, args.dataset_name, args.threshold, device)
                writer.add_scalar("mIOU", metrics["mIoU"], epoch_num)
                writer.add_scalar("testloss", metrics["loss"], epoch_num)
                if metrics["mIoU"] > best_miou:
                    best_miou = metrics["mIoU"]
                    best_epoch = epoch_num
                    best_metrics = metrics
                    print(f"------save the best model epoch {args.method_name} _{epoch_num} ------", flush=True)
                    print(f"pixAcc, mIoU:\t({metrics['pixAcc']}, {metrics['mIoU']})", flush=True)
                    print(f"testloss:\t{metrics['loss']}", flush=True)
                    print(f"PD, FA:\t({metrics['Pd']}, {metrics['Fa']})", flush=True)
                    f.write(f"the best model epoch \t_{epoch_num}\n")
                    f.write(f"pixAcc, mIoU:\t({metrics['pixAcc']}, {metrics['mIoU']})\n")
                    f.write(f"PD, FA:\t({metrics['Pd']}, {metrics['Fa']})\n")
                    f.flush()
                    save_checkpoint(ckpt_dir / f"{args.method_name}_{epoch_num}_best.pth.tar", eval_model, epoch_num, total_loss_list, args, ema is not None)

            if epoch_num % args.save_every == 0 or epoch_num == args.epochs:
                save_checkpoint(ckpt_dir / f"{args.method_name}_{epoch_num}.pth.tar", net, epoch_num, total_loss_list, args, False)

        if best_metrics is not None:
            f.write(
                "final_best "
                f"epoch={best_epoch} "
                f"mIoU={best_metrics['mIoU'] * 100:.2f} "
                f"Pd={best_metrics['Pd'] * 100:.2f} "
                f"Fa={best_metrics['Fa'] * 1e6:.2f}\n"
            )
        writer.close()


if __name__ == "__main__":
    main()
