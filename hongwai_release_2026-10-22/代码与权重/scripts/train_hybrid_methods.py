#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch.nn import init
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


ROOT = Path(__file__).resolve().parents[1]
SCTRANSNET = ROOT / "external" / "SCTransNet"
BGM_MODEL = ROOT / "external" / "BGM" / "model" / "BasicUNet" / "model_BasicUNet.py"
MSHNET_MODEL = ROOT / "external" / "MSHNet" / "model" / "MSHNet.py"
APCONV_MODEL = ROOT / "external" / "PConv-SDloss-Data" / "model" / "APConv.py"

sys.path.insert(0, str(SCTRANSNET))
from dataset import TestSetLoader, augumentation  # noqa: E402
from metrics import PD_FA, mIoU  # noqa: E402
from utils import Normalized, get_img_norm_cfg, random_crop, seed_pytorch  # noqa: E402
from model.SCTransNet import SCTransNet  # noqa: E402
import model.Config as config  # noqa: E402


def load_symbol(path: Path, module_name: str, symbol: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, symbol)


BasicUNetPlus = load_symbol(BGM_MODEL, "bgm_basicunet_file", "BasicUNet_plus")
MSHNetCore = load_symbol(MSHNET_MODEL, "mshnet_file", "MSHNet")
MSHNetChannelAttention = load_symbol(MSHNET_MODEL, "mshnet_file_ca", "ChannelAttention")
MSHNetSpatialAttention = load_symbol(MSHNET_MODEL, "mshnet_file_sa", "SpatialAttention")
OfficialPConv = load_symbol(APCONV_MODEL, "apconv_file", "PConv")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def parse_dataset_list(value: str, fallback: str) -> list[str]:
    if not value.strip():
        return [fallback]
    datasets = [item.strip() for item in value.split(",") if item.strip()]
    return datasets or [fallback]


def parse_dataset_float_overrides(value: str) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        key, raw = item.split("=", 1)
        overrides[key.strip()] = float(raw.strip())
    return overrides


def parse_dataset_int_overrides(value: str) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        key, raw = item.split("=", 1)
        overrides[key.strip()] = int(raw.strip())
    return overrides


def open_dataset_pair(dataset_root: Path, name: str, rgb: bool = False) -> tuple[Image.Image, Image.Image]:
    for ext in (".png", ".bmp", ".jpg"):
        img_path = dataset_root / "images" / f"{name}{ext}"
        mask_path = dataset_root / "masks" / f"{name}{ext}"
        if img_path.exists() and mask_path.exists():
            return Image.open(img_path).convert("RGB" if rgb else "I"), Image.open(mask_path)
    raise FileNotFoundError(name)


def normalize_imagenet_rgb(image: np.ndarray) -> np.ndarray:
    image01 = image.astype(np.float32) / 255.0
    return ((image01 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


def random_crop_rgb(image: np.ndarray, mask: np.ndarray, patch_size: int, pos_prob: float | None) -> tuple[np.ndarray, np.ndarray]:
    height, width = mask.shape
    if min(height, width) < patch_size:
        pad_h = max(height, patch_size) - height
        pad_w = max(width, patch_size) - width
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
        height, width = mask.shape

    while True:
        h_start = random.randint(0, height - patch_size)
        w_start = random.randint(0, width - patch_size)
        img_patch = image[h_start : h_start + patch_size, w_start : w_start + patch_size, :]
        mask_patch = mask[h_start : h_start + patch_size, w_start : w_start + patch_size]
        if pos_prob is None or random.random() > pos_prob or mask_patch.sum() > 0:
            return img_patch, mask_patch


def random_crop_triplet(
    image: np.ndarray,
    mask: np.ndarray,
    pseudo: np.ndarray,
    patch_size: int,
    pos_prob: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = mask.shape
    if min(height, width) < patch_size:
        pad_h = max(height, patch_size) - height
        pad_w = max(width, patch_size) - width
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="constant")
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
        pseudo = np.pad(pseudo, ((0, pad_h), (0, pad_w)), mode="constant")
        height, width = mask.shape

    while True:
        h_start = random.randint(0, height - patch_size)
        w_start = random.randint(0, width - patch_size)
        h_end = h_start + patch_size
        w_end = w_start + patch_size
        img_patch = image[h_start:h_end, w_start:w_end]
        mask_patch = mask[h_start:h_end, w_start:w_end]
        pseudo_patch = pseudo[h_start:h_end, w_start:w_end]
        if pos_prob is None or random.random() > pos_prob or mask_patch.sum() > 0:
            return img_patch, mask_patch, pseudo_patch


def augment_triplet(
    image: np.ndarray,
    mask: np.ndarray,
    pseudo: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        image = image[::-1, :]
        mask = mask[::-1, :]
        pseudo = pseudo[::-1, :]
    if random.random() < 0.5:
        image = image[:, ::-1]
        mask = mask[:, ::-1]
        pseudo = pseudo[:, ::-1]
    if random.random() < 0.5:
        image = image.transpose(1, 0)
        mask = mask.transpose(1, 0)
        pseudo = pseudo.transpose(1, 0)
    return image, mask, pseudo


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


class FullIRSTDTrainSet(Dataset):
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
        pseudo_dir: Path | None = None,
        copy_paste_prob: float = 0.0,
        copy_paste_max_targets: int = 1,
        copy_paste_min_area: int = 1,
        copy_paste_max_area: int = 900,
        copy_paste_scale_min: float = 0.75,
        copy_paste_scale_max: float = 1.35,
        copy_paste_alpha_min: float = 0.75,
        copy_paste_alpha_max: float = 1.0,
        copy_paste_contrast_jitter: float = 0.20,
        copy_paste_guard: int = 9,
        copy_paste_pseudo_value: float = 1.0,
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
        self.pseudo_dir = pseudo_dir
        self.img_norm_cfg = get_img_norm_cfg(dataset_name, str(dataset_dir))
        self.transform = augumentation()
        with (self.dataset_dir / "img_idx" / f"train_{dataset_name}.txt").open("r") as f:
            self.names = f.read().splitlines()
        self.copy_paste_prob = copy_paste_prob
        self.copy_paste_max_targets = max(1, int(copy_paste_max_targets))
        self.copy_paste_min_area = max(1, int(copy_paste_min_area))
        self.copy_paste_max_area = max(self.copy_paste_min_area, int(copy_paste_max_area))
        self.copy_paste_scale_min = max(0.1, float(copy_paste_scale_min))
        self.copy_paste_scale_max = max(self.copy_paste_scale_min, float(copy_paste_scale_max))
        self.copy_paste_alpha_min = float(copy_paste_alpha_min)
        self.copy_paste_alpha_max = max(self.copy_paste_alpha_min, float(copy_paste_alpha_max))
        self.copy_paste_contrast_jitter = max(0.0, float(copy_paste_contrast_jitter))
        self.copy_paste_guard = max(1, int(copy_paste_guard))
        self.copy_paste_pseudo_value = float(copy_paste_pseudo_value)
        self.target_bank = self._build_target_bank() if self.copy_paste_prob > 0 else []

    def __len__(self) -> int:
        return len(self.names)

    def _open_pair(self, name: str) -> tuple[Image.Image, Image.Image]:
        for ext in (".png", ".bmp", ".jpg"):
            img_path = self.dataset_dir / "images" / f"{name}{ext}"
            mask_path = self.dataset_dir / "masks" / f"{name}{ext}"
            if img_path.exists() and mask_path.exists():
                return Image.open(img_path).convert("I"), Image.open(mask_path)
        raise FileNotFoundError(name)

    def _open_pseudo(self, name: str, shape: tuple[int, int]) -> np.ndarray:
        if self.pseudo_dir is None:
            raise RuntimeError("Pseudo-label directory is not configured.")
        pseudo_root = self.pseudo_dir / self.dataset_name / "train"
        npz_path = pseudo_root / f"{name}.npz"
        png_path = pseudo_root / f"{name}.png"
        if npz_path.exists():
            with np.load(npz_path) as data:
                pseudo = np.array(data["pseudo"], dtype=np.float32)
        elif png_path.exists():
            pseudo = np.array(Image.open(png_path).convert("L"), dtype=np.float32) / 255.0
        else:
            raise FileNotFoundError(f"Missing pseudo label for {self.dataset_name}/{name} under {pseudo_root}")
        if pseudo.shape != shape:
            pseudo_img = Image.fromarray(np.clip(pseudo * 255.0, 0, 255).astype(np.uint8))
            pseudo = np.array(pseudo_img.resize((shape[1], shape[0]), Image.BILINEAR), dtype=np.float32) / 255.0
        return np.clip(pseudo, 0.0, 1.0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        name = self.names[idx]
        img_pil, mask_pil = self._open_pair(name)
        image = Normalized(np.array(img_pil, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask_pil, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        pseudo = self._open_pseudo(name, mask.shape) if self.pseudo_dir is not None else None

        if self.intensity_jitter > 0 and random.random() < 0.8:
            scale = random.uniform(1.0 - self.intensity_jitter, 1.0 + self.intensity_jitter)
            shift = random.uniform(-self.intensity_jitter, self.intensity_jitter)
            image = image * scale + shift

        if self.gamma_prob > 0 and random.random() < self.gamma_prob:
            min_v = float(image.min())
            span = float(image.max() - min_v)
            if span > 1e-6:
                gamma = random.uniform(0.70, 1.45)
                image01 = np.clip((image - min_v) / span, 0.0, 1.0)
                image = np.power(image01, gamma) * span + min_v

        if self.noise_std > 0 and random.random() < 0.6:
            image = image + np.random.normal(0.0, self.noise_std, image.shape).astype(np.float32)

        if pseudo is not None:
            image, mask, pseudo = random_crop_triplet(image, mask, pseudo, self.patch_size, pos_prob=self.pos_prob)
            image, mask, pseudo = augment_triplet(image, mask, pseudo)
        else:
            image, mask = random_crop(image, mask, self.patch_size, pos_prob=self.pos_prob)
            image, mask = self.transform(image, mask)
        if self.copy_paste_prob > 0 and self.target_bank and random.random() < self.copy_paste_prob:
            image, mask, pseudo = self._copy_paste_targets(image, mask, pseudo)

        if self.paired_bgm:
            bgm_image = self._background_masked_view(image, mask)
            image = np.stack([image, bgm_image], axis=0)
            mask = np.stack([mask, mask], axis=0)
            if pseudo is not None:
                pseudo = np.stack([pseudo, pseudo], axis=0)
        elif self.bgm_prob > 0 and random.random() < self.bgm_prob:
            image = self._background_masked_view(image, mask)[np.newaxis, :]
            mask = mask[np.newaxis, :]
            if pseudo is not None:
                pseudo = pseudo[np.newaxis, :]
        else:
            image = image[np.newaxis, :]
            mask = mask[np.newaxis, :]
            if pseudo is not None:
                pseudo = pseudo[np.newaxis, :]

        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).float()
        if pseudo is None:
            return image_tensor, mask_tensor
        pseudo_tensor = torch.from_numpy(np.ascontiguousarray(pseudo)).float()
        return image_tensor, mask_tensor, pseudo_tensor

    def _build_target_bank(self) -> list[tuple[np.ndarray, np.ndarray]]:
        bank: list[tuple[np.ndarray, np.ndarray]] = []
        for name in self.names:
            try:
                img_pil, mask_pil = self._open_pair(name)
            except FileNotFoundError:
                continue
            image = Normalized(np.array(img_pil, dtype=np.float32), self.img_norm_cfg)
            mask = np.array(mask_pil, dtype=np.float32)
            if mask.ndim > 2:
                mask = mask[:, :, 0]
            binary = (mask > 127).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
            for label_idx in range(1, num_labels):
                x, y, w, h, area = stats[label_idx]
                area = int(area)
                if area < self.copy_paste_min_area or area > self.copy_paste_max_area:
                    continue
                pad = 3
                x1 = max(0, int(x) - pad)
                y1 = max(0, int(y) - pad)
                x2 = min(image.shape[1], int(x + w) + pad)
                y2 = min(image.shape[0], int(y + h) + pad)
                patch_mask = (labels[y1:y2, x1:x2] == label_idx).astype(np.float32)
                if int(patch_mask.sum()) <= 0:
                    continue
                patch_img = image[y1:y2, x1:x2].astype(np.float32)
                bank.append((patch_img, patch_mask))
        return bank

    def _copy_paste_targets(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        pseudo: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        height, width = image.shape
        target_count = random.randint(1, self.copy_paste_max_targets)
        guard_kernel = self.copy_paste_guard + (1 - self.copy_paste_guard % 2)
        guard = cv2.dilate((mask > 0.5).astype(np.uint8), np.ones((guard_kernel, guard_kernel), dtype=np.uint8), 1)
        out_image = image.copy()
        out_mask = mask.copy()
        out_pseudo = None if pseudo is None else pseudo.copy()
        for _ in range(target_count):
            patch_img, patch_mask = random.choice(self.target_bank)
            scale = random.uniform(self.copy_paste_scale_min, self.copy_paste_scale_max)
            new_w = max(1, int(round(patch_img.shape[1] * scale)))
            new_h = max(1, int(round(patch_img.shape[0] * scale)))
            if new_w >= width or new_h >= height:
                continue
            resized_img = cv2.resize(patch_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
            resized_mask = cv2.resize(patch_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
            resized_mask = (resized_mask > 0.5).astype(np.float32)
            if int(resized_mask.sum()) < 1:
                continue
            paste_x = paste_y = None
            for _trial in range(24):
                x = random.randint(0, width - new_w)
                y = random.randint(0, height - new_h)
                if np.any(guard[y : y + new_h, x : x + new_w] * resized_mask):
                    continue
                paste_x, paste_y = x, y
                break
            if paste_x is None or paste_y is None:
                continue
            dst = out_image[paste_y : paste_y + new_h, paste_x : paste_x + new_w]
            src_bg = float(resized_img[resized_mask < 0.5].mean()) if np.any(resized_mask < 0.5) else float(resized_img.mean())
            dst_bg = float(dst[resized_mask < 0.5].mean()) if np.any(resized_mask < 0.5) else float(dst.mean())
            contrast_scale = random.uniform(
                1.0 - self.copy_paste_contrast_jitter,
                1.0 + self.copy_paste_contrast_jitter,
            )
            adjusted = dst_bg + (resized_img - src_bg) * contrast_scale
            alpha = random.uniform(self.copy_paste_alpha_min, self.copy_paste_alpha_max)
            soft_alpha = cv2.GaussianBlur(resized_mask, (3, 3), 0).astype(np.float32)
            soft_alpha = np.clip(soft_alpha * alpha, 0.0, 1.0)
            pasted = dst * (1.0 - soft_alpha) + adjusted * soft_alpha
            out_image[paste_y : paste_y + new_h, paste_x : paste_x + new_w] = pasted
            out_mask[paste_y : paste_y + new_h, paste_x : paste_x + new_w] = np.maximum(
                out_mask[paste_y : paste_y + new_h, paste_x : paste_x + new_w],
                resized_mask,
            )
            if out_pseudo is not None:
                out_pseudo[paste_y : paste_y + new_h, paste_x : paste_x + new_w] = np.maximum(
                    out_pseudo[paste_y : paste_y + new_h, paste_x : paste_x + new_w],
                    resized_mask * self.copy_paste_pseudo_value,
                )
            guard[paste_y : paste_y + new_h, paste_x : paste_x + new_w] = np.maximum(
                guard[paste_y : paste_y + new_h, paste_x : paste_x + new_w],
                resized_mask.astype(np.uint8),
            )
        return out_image, out_mask, out_pseudo

    def _background_masked_view(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        region = max(1, int(self.bgm_region_size))
        height, width = image.shape
        grid_h = int(math.ceil(height / region))
        grid_w = int(math.ceil(width / region))
        keep = np.zeros(grid_h * grid_w, dtype=np.float32)
        keep[: int(round(keep.size * self.bgm_mask_ratio))] = 1.0
        np.random.shuffle(keep)
        keep = keep.reshape(grid_h, grid_w).repeat(region, 0)[:height, :].repeat(region, 1)[:, :width]

        guard = torch.from_numpy(mask.astype(np.float32)).view(1, 1, height, width)
        guard = F.max_pool2d(guard, kernel_size=7, stride=1, padding=3).view(height, width).numpy()
        keep[guard > 0] = 0.0

        out = image.copy()
        out[keep > 0] = float(image.mean())
        return out


class OfficialMSHNetTrainSet(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        dataset_name: str,
        patch_size: int,
        pos_prob: float,
        base_size: int,
        scale_aug: bool,
        blur_prob: float,
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir / dataset_name
        self.patch_size = patch_size
        self.pos_prob = pos_prob
        self.base_size = base_size
        self.scale_aug = scale_aug
        self.blur_prob = blur_prob
        with (self.dataset_dir / "img_idx" / f"train_{dataset_name}.txt").open("r") as f:
            self.names = f.read().splitlines()

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_pil, mask_pil = open_dataset_pair(self.dataset_dir, self.names[idx], rgb=True)
        if random.random() < 0.5:
            image_pil = image_pil.transpose(Image.FLIP_LEFT_RIGHT)
            mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)
        if self.scale_aug:
            long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
            width, height = image_pil.size
            if height > width:
                new_h = long_size
                new_w = int(width * long_size / height + 0.5)
                short_size = new_w
            else:
                new_w = long_size
                new_h = int(height * long_size / width + 0.5)
                short_size = new_h
            image_pil = image_pil.resize((new_w, new_h), Image.BILINEAR)
            mask_pil = mask_pil.resize((new_w, new_h), Image.NEAREST)
            if short_size < self.patch_size:
                pad_w = self.patch_size - new_w if new_w < self.patch_size else 0
                pad_h = self.patch_size - new_h if new_h < self.patch_size else 0
                padded_image = Image.new("RGB", (new_w + pad_w, new_h + pad_h))
                padded_mask = Image.new("L", (new_w + pad_w, new_h + pad_h))
                padded_image.paste(image_pil, (0, 0))
                padded_mask.paste(mask_pil, (0, 0))
                image_pil = padded_image
                mask_pil = padded_mask
        if self.blur_prob > 0 and random.random() < self.blur_prob:
            image_pil = image_pil.filter(ImageFilter.GaussianBlur(radius=random.random()))

        image = np.array(image_pil, dtype=np.float32)
        mask = np.array(mask_pil, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        image, mask = random_crop_rgb(image, mask, self.patch_size, pos_prob=self.pos_prob)
        image = normalize_imagenet_rgb(image)
        mask = mask[np.newaxis, :]
        return (
            torch.from_numpy(np.ascontiguousarray(image)).float(),
            torch.from_numpy(np.ascontiguousarray(mask)).float(),
        )


class ChannelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(1, channels // 16)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.fc2(F.relu(self.fc1(F.adaptive_avg_pool2d(x, 1)), inplace=True))
        maxv = self.fc2(F.relu(self.fc1(F.adaptive_max_pool2d(x, 1)), inplace=True))
        return torch.sigmoid(avg + maxv)


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv(torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)))


class ResCBAMBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.short = None
        if in_channels != out_channels:
            self.short = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.short is None else self.short(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out * self.ca(out)
        out = out * self.sa(out)
        return F.relu(out + residual, inplace=True)


class PinwheelConv(nn.Module):
    """Asymmetric multi-branch convolution for tiny thermal-spot evidence."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        branch_channels = max(4, out_channels // 4)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, (1, 3), padding=(0, 1), bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(branch_channels, branch_channels, (3, 1), padding=(1, 0), bias=False),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, (3, 1), padding=(1, 0), bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(branch_channels, branch_channels, (1, 3), padding=(0, 1), bias=False),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, (1, 5), padding=(0, 2), bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(branch_channels, branch_channels, (5, 1), padding=(2, 0), bias=False),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels, branch_channels, (5, 1), padding=(2, 0), bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(branch_channels, branch_channels, (1, 5), padding=(0, 2), bias=False),
                ),
            ]
        )
        self.merge = nn.Sequential(
            nn.BatchNorm2d(branch_channels * 4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(branch_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.merge(torch.cat([branch(x) for branch in self.branches], dim=1))


class PinwheelCBAMBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pconv = PinwheelConv(in_channels, out_channels)
        self.conv = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.short = None
        if in_channels != out_channels:
            self.short = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.short is None else self.short(x)
        out = self.conv(self.pconv(x))
        out = out * self.ca(out)
        out = out * self.sa(out)
        return F.relu(out + residual, inplace=True)


class AsymPadPConvCBAMBlock(nn.Module):
    """Official PConv-style asymmetric-padding block for IRST Gaussian spot evidence."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 4) -> None:
        super().__init__()
        self.pconv = OfficialPConv(in_channels, out_channels, kernel_size, 1)
        self.conv = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.short = None
        if in_channels != out_channels:
            self.short = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.short is None else self.short(x)
        out = self.conv(self.pconv(x))
        out = out * self.ca(out)
        out = out * self.sa(out)
        return F.relu(out + residual, inplace=True)


class TinyFeatureUpProjector(nn.Module):
    """FeatUp-inspired high-resolution projection guided by IR local contrast."""

    def __init__(self, in_channels: int, out_channels: int, guide_channels: int = 2) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels + guide_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.gate = nn.Sequential(nn.Conv2d(out_channels + guide_channels, out_channels, 1), nn.Sigmoid())

    def forward(self, feature: torch.Tensor, size: tuple[int, int], guide: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(feature, size=size, mode="bilinear", align_corners=False)
        up = self.reduce(up)
        guide = F.interpolate(guide, size=size, mode="bilinear", align_corners=False)
        fused = torch.cat([up, guide], dim=1)
        return F.relu(up + self.gate(fused) * self.refine(fused), inplace=True)


class BackgroundNormalityBank(nn.Module):
    """Multi-prototype background memory adapted from anomaly detection."""

    def __init__(self, channels: int, momentum: float = 0.98, scale: float = 0.30, prototypes: int = 16) -> None:
        super().__init__()
        self.momentum = momentum
        self.scale = scale
        self.prototypes = prototypes
        self.register_buffer("prototype", torch.zeros(1, channels, 1, 1))
        self.register_buffer("memory", torch.zeros(prototypes, channels))
        self.register_buffer("ready", torch.zeros(1))
        self.project = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(8, 1, 1),
        )

    def forward(self, feature: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            current = feature.detach().mean(dim=(0, 2, 3), keepdim=True)
            if self.training:
                if float(self.ready.item()) < 0.5:
                    self.prototype.copy_(current)
                    flat = feature.detach().permute(0, 2, 3, 1).reshape(-1, feature.shape[1])
                    if flat.shape[0] >= self.prototypes:
                        indices = torch.linspace(0, flat.shape[0] - 1, self.prototypes, device=flat.device).long()
                        self.memory.copy_(flat[indices])
                    else:
                        self.memory.copy_(current.view(1, -1).repeat(self.prototypes, 1))
                    self.ready.fill_(1.0)
                else:
                    self.prototype.mul_(self.momentum).add_(current * (1.0 - self.momentum))
                    score = guide.detach().mean(1, keepdim=True)
                    score = F.interpolate(score, size=feature.shape[-2:], mode="bilinear", align_corners=False)
                    flat = feature.detach().permute(0, 2, 3, 1).reshape(-1, feature.shape[1])
                    bg_score = score.flatten()
                    count = min(self.prototypes, flat.shape[0])
                    if count > 0:
                        chosen = torch.topk(bg_score, k=count, largest=False).indices
                        updates = flat[chosen]
                        self.memory[:count].mul_(self.momentum).add_(updates * (1.0 - self.momentum))
        distance = (feature - self.prototype.to(dtype=feature.dtype)).abs().mean(1, keepdim=True)
        distance = distance / (distance.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        flat_feature = feature.permute(0, 2, 3, 1).reshape(-1, feature.shape[1])
        memory = self.memory.to(device=feature.device, dtype=feature.dtype)
        proto_distance = torch.cdist(flat_feature, memory).min(dim=1).values.view(feature.shape[0], 1, *feature.shape[-2:])
        proto_distance = proto_distance / (proto_distance.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        guide = F.interpolate(guide, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        anomaly = torch.sigmoid(
            self.project(torch.cat([distance.clamp(max=6.0), proto_distance.clamp(max=6.0), guide], dim=1))
        )
        return feature * (1.0 + self.scale * anomaly)


class TCFBasicUNet(nn.Module):
    """BGM-style UNet plus FeatUp projection and background-normality guidance."""

    def __init__(self, filters: tuple[int, int, int, int, int] = (16, 32, 64, 128, 256)) -> None:
        super().__init__()
        c0, c1, c2, c3, c4 = filters
        self.pool = nn.MaxPool2d(2, 2)
        self.conv0 = self._make_layer(1, c0)
        self.conv1 = self._make_layer(c0, c1, 2)
        self.conv2 = self._make_layer(c1, c2, 2)
        self.conv3 = self._make_layer(c2, c3, 2)
        self.conv4 = self._make_layer(c3, c4, 2)
        self.dec3 = self._make_layer(c3 + c4, c3, 2)
        self.dec2 = self._make_layer(c2 + c3, c2, 2)
        self.dec1 = self._make_layer(c1 + c2, c1, 2)
        self.dec0 = self._make_layer(c0 + c1, c0)

        self.up4 = TinyFeatureUpProjector(c4, c0)
        self.up3 = TinyFeatureUpProjector(c3, c0)
        self.up2 = TinyFeatureUpProjector(c2, c0)
        self.up1 = TinyFeatureUpProjector(c1, c0)
        self.fuse = self._make_layer(c0, c0)
        self.normality = BackgroundNormalityBank(c0)
        self.final = nn.Conv2d(c0, 1, 1)
        laplace = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
        self.register_buffer("laplace_kernel", laplace)

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int = 1) -> nn.Sequential:
        layers: list[nn.Module] = [ResCBAMBlock(in_channels, out_channels)]
        for _ in range(blocks - 1):
            layers.append(ResCBAMBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _guide(self, image: torch.Tensor) -> torch.Tensor:
        lap = F.conv2d(image, self.laplace_kernel.to(dtype=image.dtype), padding=1).abs()
        lap = lap / (lap.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        contrast = image - F.avg_pool2d(image, 9, stride=1, padding=4)
        contrast = contrast.abs()
        contrast = contrast / (contrast.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        return torch.cat([lap.clamp(max=6.0) / 6.0, contrast.clamp(max=6.0) / 6.0], dim=1)

    def forward(self, image: torch.Tensor, epoch: int | None = None) -> torch.Tensor:
        guide = self._guide(image)
        size = image.shape[-2:]
        x0 = self.conv0(image)
        x1 = self.conv1(self.pool(x0))
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))

        d3 = self.dec3(torch.cat([x3, F.interpolate(x4, size=x3.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        d2 = self.dec2(torch.cat([x2, F.interpolate(d3, size=x2.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        d1 = self.dec1(torch.cat([x1, F.interpolate(d2, size=x1.shape[-2:], mode="bilinear", align_corners=False)], dim=1))
        d0 = self.dec0(torch.cat([x0, F.interpolate(d1, size=x0.shape[-2:], mode="bilinear", align_corners=False)], dim=1))

        fused = self.fuse(self.up4(x4, size, guide) + self.up3(d3, size, guide) + self.up2(d2, size, guide) + self.up1(d1, size, guide) + d0)
        fused = self.normality(fused, guide)
        return torch.sigmoid(self.final(fused))


class PConvTCFBasicUNet(TCFBasicUNet):
    """TCF-BGM with pinwheel-shaped local evidence blocks."""

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int = 1) -> nn.Sequential:
        layers: list[nn.Module] = [PinwheelCBAMBlock(in_channels, out_channels)]
        for _ in range(blocks - 1):
            layers.append(PinwheelCBAMBlock(out_channels, out_channels))
        return nn.Sequential(*layers)


class APConvTCFBasicUNet(TCFBasicUNet):
    """TCF-BGM using the official asymmetric-padding PConv operator."""

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int = 1) -> nn.Sequential:
        layers: list[nn.Module] = [AsymPadPConvCBAMBlock(in_channels, out_channels)]
        for _ in range(blocks - 1):
            layers.append(AsymPadPConvCBAMBlock(out_channels, out_channels))
        return nn.Sequential(*layers)


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

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        high = F.conv2d(image, self.laplace_kernel.to(dtype=image.dtype), padding=1)
        local_mean = F.avg_pool2d(image, kernel_size=9, stride=1, padding=4)
        contrast = image - local_mean
        features = torch.cat([image, high, contrast], dim=1)
        return image + self.residual_scale * self.proj(features)


class SCTRefiner(nn.Module):
    """Frozen SCTransNet coarse predictor plus learnable target-clutter residual refiner."""

    def __init__(self, checkpoint: Path, freeze_base: bool = True, residual_scale: float = 0.35) -> None:
        super().__init__()
        self.base = SCTransNet(config.get_SCTrans_config(), mode="train", deepsuper=True)
        ckpt = torch.load(checkpoint, map_location="cpu")
        state = normalize_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
        args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
        adapter_state = {key[len("adapter.") :]: value for key, value in state.items() if key.startswith("adapter.")}
        self.adapter = None
        if adapter_state:
            hidden = int(args.get("adapter_hidden", adapter_state["proj.0.weight"].shape[0]))
            scale = float(args.get("adapter_scale", 0.10))
            self.adapter = TCDInputAdapter(hidden_channels=hidden, residual_scale=scale)
            adapter_missing, adapter_unexpected = self.adapter.load_state_dict(adapter_state, strict=False)
            if adapter_unexpected:
                print(f"SCTRefiner unexpected adapter keys: {len(adapter_unexpected)}", flush=True)
            if adapter_missing:
                print(f"SCTRefiner missing adapter keys: {len(adapter_missing)}", flush=True)

        base_state = {key: value for key, value in state.items() if not key.startswith(("adapter.", "output_refiner."))}
        missing, unexpected = self.base.load_state_dict(base_state, strict=False)
        if unexpected:
            print(f"SCTRefiner unexpected checkpoint keys: {len(unexpected)}", flush=True)
        if missing:
            print(f"SCTRefiner missing checkpoint keys: {len(missing)}", flush=True)
        self.freeze_base = freeze_base
        if freeze_base:
            self.base.eval()
            for param in self.base.parameters():
                param.requires_grad_(False)
            if self.adapter is not None:
                self.adapter.eval()
                for param in self.adapter.parameters():
                    param.requires_grad_(False)

        self.residual_scale = residual_scale
        self.pconv = PinwheelConv(5, 16)
        self.featup = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.normality = BackgroundNormalityBank(16, momentum=0.98, scale=0.25)
        self.out = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(8, 1, 1),
        )
        laplace = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
        self.register_buffer("laplace_kernel", laplace)
        init.constant_(self.out[-1].weight, 0.0)
        init.constant_(self.out[-1].bias, 0.0)

    def initialize_refiner(self) -> None:
        """Initialize only the new residual branch, never the loaded SCTransNet base."""
        self.pconv.apply(weights_init_kaiming)
        self.featup.apply(weights_init_kaiming)
        self.normality.apply(weights_init_kaiming)
        self.out.apply(weights_init_kaiming)
        init.constant_(self.out[-1].weight, 0.0)
        init.constant_(self.out[-1].bias, 0.0)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_base:
            self.base.eval()
            if self.adapter is not None:
                self.adapter.eval()
        return self

    def _base_pred(self, image: torch.Tensor):
        with torch.no_grad() if self.freeze_base else torch.enable_grad():
            base_input = self.adapter(image) if self.adapter is not None else image
            pred = self.base(base_input)
            if isinstance(pred, (tuple, list)):
                pred = pred[-1]
        pred = pred.clamp(1e-5, 1.0 - 1e-5)
        self._last_base_pred = pred.detach()
        return pred

    def _guides(self, image: torch.Tensor, base_pred: torch.Tensor) -> torch.Tensor:
        lap = F.conv2d(image, self.laplace_kernel.to(dtype=image.dtype), padding=1).abs()
        lap = lap / (lap.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        contrast = (image - F.avg_pool2d(image, 9, stride=1, padding=4)).abs()
        contrast = contrast / (contrast.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        local_prob = F.avg_pool2d(base_pred, 7, stride=1, padding=3)
        sharp_prob = (base_pred - local_prob).abs()
        return torch.cat(
            [
                image,
                base_pred,
                lap.clamp(max=6.0) / 6.0,
                contrast.clamp(max=6.0) / 6.0,
                sharp_prob,
            ],
            dim=1,
        )

    def forward(self, image: torch.Tensor, epoch: int | None = None) -> torch.Tensor:
        base_pred = self._base_pred(image)
        features = self.pconv(self._guides(image, base_pred))
        guide = features[:, :2].abs()
        features = self.featup(features)
        features = self.normality(features, guide)
        residual = self.out(features)
        logits = torch.logit(base_pred)
        return torch.sigmoid(logits + self.residual_scale * residual)


class DynamicHighFreqGate(nn.Module):
    """DHiF/SET-inspired dynamic gate for separating target high-frequency from clutter."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.residual_gain = nn.Parameter(torch.zeros(1))
        self.depthwise = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.context = nn.Sequential(
            nn.Conv2d(channels + 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, feature: torch.Tensor, image: torch.Tensor, base_pred: torch.Tensor, laplace: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(image, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        base_pred = F.interpolate(base_pred, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        laplace = F.interpolate(laplace, size=feature.shape[-2:], mode="bilinear", align_corners=False)
        high = self.depthwise(feature)
        gate = self.context(torch.cat([feature, image, base_pred, laplace], dim=1))
        return feature + self.residual_gain * gate * high


class SCTDHiFRefiner(SCTRefiner):
    """SCTransNet refiner with dynamic high-frequency target-clutter filtering."""

    def __init__(self, checkpoint: Path, freeze_base: bool = True, residual_scale: float = 0.35) -> None:
        super().__init__(checkpoint=checkpoint, freeze_base=freeze_base, residual_scale=residual_scale)
        self.dhif_gate = DynamicHighFreqGate(16)

    def initialize_refiner(self) -> None:
        super().initialize_refiner()
        self.dhif_gate.apply(weights_init_kaiming)

    def forward(self, image: torch.Tensor, epoch: int | None = None) -> torch.Tensor:
        base_pred = self._base_pred(image)
        guides = self._guides(image, base_pred)
        features = self.pconv(guides)
        guide = features[:, :2].abs()
        features = self.featup(features)
        features = self.dhif_gate(features, image, base_pred, guides[:, 2:3])
        features = self.normality(features, guide)
        residual = self.out(features)
        logits = torch.logit(base_pred)
        return torch.sigmoid(logits + self.residual_scale * residual)


class HardClutterCorrectionGate(nn.Module):
    """Pixel-wise residual router for target-like clutter and missed tiny spots."""

    def __init__(self, in_channels: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(8, 1, 1),
        )
        init.constant_(self.net[-1].weight, 0.0)
        init.constant_(self.net[-1].bias, 0.0)

    def forward(self, guides: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(guides))


class SCTTPCDHiFRefiner(SCTDHiFRefiner):
    """Teacher-preserved DHiF refiner with hard-clutter residual routing."""

    def __init__(self, checkpoint: Path, freeze_base: bool = True, residual_scale: float = 0.35) -> None:
        super().__init__(checkpoint=checkpoint, freeze_base=freeze_base, residual_scale=residual_scale)
        self.correction_gate = HardClutterCorrectionGate(in_channels=5)

    def initialize_refiner(self) -> None:
        super().initialize_refiner()
        self.correction_gate.apply(weights_init_kaiming)
        init.constant_(self.correction_gate.net[-1].weight, 0.0)
        init.constant_(self.correction_gate.net[-1].bias, 0.0)

    def forward(self, image: torch.Tensor, epoch: int | None = None) -> torch.Tensor:
        base_pred = self._base_pred(image)
        guides = self._guides(image, base_pred)
        features = self.pconv(guides)
        guide = features[:, :2].abs()
        features = self.featup(features)
        features = self.dhif_gate(features, image, base_pred, guides[:, 2:3])
        features = self.normality(features, guide)
        residual = self.out(features)
        correction_gate = self.correction_gate(guides)
        logits = torch.logit(base_pred)
        return torch.sigmoid(logits + self.residual_scale * (0.50 + correction_gate) * residual)


class SCTAPConvTPCRefiner(SCTTPCDHiFRefiner):
    """Strong-baseline refiner using official PConv plus TPC/DHiF background control."""

    def __init__(self, checkpoint: Path, freeze_base: bool = True, residual_scale: float = 0.35) -> None:
        super().__init__(checkpoint=checkpoint, freeze_base=freeze_base, residual_scale=residual_scale)
        self.pconv = AsymPadPConvCBAMBlock(5, 16, kernel_size=4)


class MultiScaleTinyFeatUpBranch(nn.Module):
    """FeatUp/UniVAD-inspired branch for tiny evidence recovery and background normality."""

    def __init__(self, in_channels: int = 5, channels: int = 16) -> None:
        super().__init__()
        self.stem = PinwheelConv(in_channels, channels)
        self.down1 = nn.Sequential(nn.MaxPool2d(2, 2), ResCBAMBlock(channels, channels))
        self.down2 = nn.Sequential(nn.MaxPool2d(2, 2), ResCBAMBlock(channels, channels))
        self.down3 = nn.Sequential(nn.MaxPool2d(2, 2), ResCBAMBlock(channels, channels))
        self.up1 = TinyFeatureUpProjector(channels, channels)
        self.up2 = TinyFeatureUpProjector(channels, channels)
        self.up3 = TinyFeatureUpProjector(channels, channels)
        self.scale_weights = nn.Parameter(torch.zeros(4))
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.normality = BackgroundNormalityBank(channels, momentum=0.99, scale=0.35)
        self.out = nn.Sequential(
            nn.Conv2d(channels, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(8, 1, 1),
        )
        init.constant_(self.out[-1].weight, 0.0)
        init.constant_(self.out[-1].bias, 0.0)

    def forward(self, guides: torch.Tensor) -> torch.Tensor:
        guide = guides[:, 2:4]
        size = guides.shape[-2:]
        x0 = self.stem(guides)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        p1 = self.up1(x1, size, guide)
        p2 = self.up2(x2, size, guide)
        p3 = self.up3(x3, size, guide)
        weights = torch.softmax(self.scale_weights, dim=0)
        fused = self.fuse(weights[0] * x0 + weights[1] * p1 + weights[2] * p2 + weights[3] * p3)
        fused = self.normality(fused, guide)
        return self.out(fused)


class SCTBNGFeatUpRefiner(SCTTPCDHiFRefiner):
    """Background-normality guided FeatUp branch on top of the teacher-preserved refiner."""

    def __init__(self, checkpoint: Path, freeze_base: bool = True, residual_scale: float = 0.35) -> None:
        super().__init__(checkpoint=checkpoint, freeze_base=freeze_base, residual_scale=residual_scale)
        self.bng_featup = MultiScaleTinyFeatUpBranch(in_channels=5, channels=16)
        self.bng_scale = nn.Parameter(torch.tensor(0.10))

    def initialize_refiner(self) -> None:
        super().initialize_refiner()
        self.bng_featup.apply(weights_init_kaiming)
        init.constant_(self.bng_featup.out[-1].weight, 0.0)
        init.constant_(self.bng_featup.out[-1].bias, 0.0)

    def forward(self, image: torch.Tensor, epoch: int | None = None) -> torch.Tensor:
        base_pred = self._base_pred(image)
        guides = self._guides(image, base_pred)
        features = self.pconv(guides)
        guide = features[:, :2].abs()
        features = self.featup(features)
        features = self.dhif_gate(features, image, base_pred, guides[:, 2:3])
        features = self.normality(features, guide)
        residual = self.out(features)
        correction_gate = self.correction_gate(guides)
        bng_residual = self.bng_featup(guides)
        bng_scale = self.bng_scale.clamp(0.0, 0.50)
        logits = torch.logit(base_pred)
        residual = (0.50 + correction_gate) * residual + bng_scale * bng_residual
        return torch.sigmoid(logits + self.residual_scale * residual)


class LocalContrastAnomalyBranch(nn.Module):
    """Bounded tiny residual from local contrast, high-frequency, and normality cues."""

    def __init__(self, channels: int = 12) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(4, channels)
        self.stem = nn.Sequential(
            nn.Conv2d(10, channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=False),
            nn.GroupNorm(4, channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.GroupNorm(4, channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.router = nn.Sequential(
            nn.Conv2d(10, 8, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid(),
        )
        self.out = nn.Conv2d(channels, 1, 1)
        init.constant_(self.out.weight, 0.0)
        init.constant_(self.out.bias, 0.0)

    @staticmethod
    def _positive_norm(x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        x = x / (x.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        return x.clamp(max=6.0) / 6.0

    def forward(self, image: torch.Tensor, base_pred: torch.Tensor, guides: torch.Tensor) -> torch.Tensor:
        local5 = F.avg_pool2d(image, 5, stride=1, padding=2)
        local15 = F.avg_pool2d(image, 15, stride=1, padding=7)
        hot5 = self._positive_norm(image - local5)
        hot15 = self._positive_norm(image - local15)
        cold15 = self._positive_norm(local15 - image)

        base_smooth = F.avg_pool2d(base_pred, 9, stride=1, padding=4)
        base_peak = self._positive_norm(base_pred - base_smooth)
        base_halo = self._positive_norm(base_smooth - base_pred)

        feats = torch.cat(
            [
                image,
                base_pred,
                guides[:, 2:3],
                guides[:, 3:4],
                guides[:, 4:5],
                hot5,
                hot15,
                cold15,
                base_peak,
                base_halo,
            ],
            dim=1,
        )
        gate = self.router(feats)
        return torch.tanh(self.out(self.stem(feats))) * (0.25 + gate)


class TargetClutterDecoupleBranch(nn.Module):
    """Explicitly separates target-like high frequency from hard background clutter."""

    def __init__(self, channels: int = 16, max_residual: float = 0.55) -> None:
        super().__init__()
        self.max_residual = max_residual
        self.encoder = nn.Sequential(
            PinwheelConv(12, channels),
            nn.LeakyReLU(0.1, inplace=True),
            DynamicHighFreqGate(channels),
            ResCBAMBlock(channels, channels),
        )
        self.target_head = nn.Sequential(
            nn.Conv2d(channels + 4, channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 1, 1),
        )
        self.clutter_head = nn.Sequential(
            nn.Conv2d(channels + 4, channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 1, 1),
        )
        self.router = nn.Sequential(
            nn.Conv2d(12, channels, 1, bias=False),
            nn.GroupNorm(4, channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 2, 1),
        )
        init.constant_(self.target_head[-1].weight, 0.0)
        init.constant_(self.target_head[-1].bias, 0.0)
        init.constant_(self.clutter_head[-1].weight, 0.0)
        init.constant_(self.clutter_head[-1].bias, 0.0)
        init.constant_(self.router[-1].weight, 0.0)
        init.constant_(self.router[-1].bias, 0.0)

    @staticmethod
    def _positive_norm(x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        x = x / (x.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        return x.clamp(max=6.0) / 6.0

    def forward(
        self,
        image: torch.Tensor,
        primary: torch.Tensor,
        guides: torch.Tensor,
        laplace: torch.Tensor,
        contrast: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local7 = F.avg_pool2d(image, 7, stride=1, padding=3)
        local17 = F.avg_pool2d(image, 17, stride=1, padding=8)
        hot7 = self._positive_norm(image - local7)
        hot17 = self._positive_norm(image - local17)
        cold17 = self._positive_norm(local17 - image)

        base_smooth = F.avg_pool2d(primary, 9, stride=1, padding=4)
        base_peak = self._positive_norm(primary - base_smooth)
        base_halo = self._positive_norm(base_smooth - primary)
        clutter_edge = self._positive_norm(laplace * (1.0 - primary))
        weak_target = self._positive_norm(hot7 * (0.25 + primary + base_peak))

        feats = torch.cat(
            [
                image,
                primary,
                guides[:, 2:3],
                guides[:, 3:4],
                hot7,
                hot17,
                cold17,
                base_peak,
                base_halo,
                clutter_edge,
                weak_target,
                contrast,
            ],
            dim=1,
        )
        encoded = self.encoder[0](feats)
        encoded = self.encoder[1](encoded)
        encoded = self.encoder[2](encoded, image, primary, laplace)
        encoded = self.encoder[3](encoded)
        aux = torch.cat([primary, laplace, clutter_edge, weak_target], dim=1)
        target_delta = F.softplus(self.target_head(torch.cat([encoded, aux], dim=1))) - math.log(2.0)
        clutter_delta = F.softplus(self.clutter_head(torch.cat([encoded, aux], dim=1))) - math.log(2.0)
        route = torch.softmax(self.router(feats), dim=1)
        target_gate = route[:, 0:1] * (0.35 + primary + weak_target).clamp(max=1.5)
        clutter_gate = route[:, 1:2] * (0.35 + clutter_edge + base_halo).clamp(max=1.5)
        residual = self.max_residual * torch.tanh(target_gate * target_delta - clutter_gate * clutter_delta)
        return residual, target_gate, clutter_gate


class SCTTargetClutterDecoupleRefiner(SCTRefiner):
    """SCTransNet refiner with target-vs-clutter high-frequency decoupling."""

    def __init__(self, checkpoint: Path, freeze_base: bool = True, residual_scale: float = 0.35) -> None:
        super().__init__(checkpoint=checkpoint, freeze_base=freeze_base, residual_scale=residual_scale)
        self.tcd_branch = TargetClutterDecoupleBranch(channels=16)
        self.freeze_parent_refiner = True

    def initialize_refiner(self) -> None:
        super().initialize_refiner()
        self.tcd_branch.apply(weights_init_kaiming)
        init.constant_(self.tcd_branch.target_head[-1].weight, 0.0)
        init.constant_(self.tcd_branch.target_head[-1].bias, 0.0)
        init.constant_(self.tcd_branch.clutter_head[-1].weight, 0.0)
        init.constant_(self.tcd_branch.clutter_head[-1].bias, 0.0)
        init.constant_(self.tcd_branch.router[-1].weight, 0.0)
        init.constant_(self.tcd_branch.router[-1].bias, 0.0)

    def freeze_inherited_refiner(self) -> None:
        for module in (self.pconv, self.featup, self.normality, self.out):
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_parent_refiner:
            self.freeze_inherited_refiner()
        return self

    def forward(self, image: torch.Tensor, epoch: int | None = None) -> torch.Tensor:
        base_pred = self._base_pred(image)
        guides = self._guides(image, base_pred)
        with torch.no_grad() if self.freeze_parent_refiner else torch.enable_grad():
            features = self.pconv(guides)
            guide = features[:, :2].abs()
            features = self.featup(features)
            features = self.normality(features, guide)
            inherited_residual = self.out(features)
        residual, target_gate, clutter_gate = self.tcd_branch(
            image,
            base_pred,
            guides,
            guides[:, 2:3],
            guides[:, 3:4],
        )
        self._last_tcd_target_gate = target_gate
        self._last_tcd_clutter_gate = clutter_gate
        self._last_base_pred = base_pred.detach()
        logits = torch.logit(base_pred)
        return torch.sigmoid(logits + self.residual_scale * (inherited_residual + residual))


class SCTLocalContrastAnomalyRefiner(SCTRefiner):
    """SCTransNet with a small anomaly-style local contrast logit calibrator."""

    def __init__(self, checkpoint: Path, freeze_base: bool = True, residual_scale: float = 0.35) -> None:
        super().__init__(checkpoint=checkpoint, freeze_base=freeze_base, residual_scale=residual_scale)
        self.lca_branch = LocalContrastAnomalyBranch(channels=12)
        self.lca_gain = nn.Parameter(torch.tensor(0.60))
        self.freeze_parent_refiner = True

    def freeze_inherited_refiner(self) -> None:
        for module in (self.pconv, self.featup, self.normality, self.out):
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

    def initialize_refiner(self) -> None:
        super().initialize_refiner()
        self.lca_branch.apply(weights_init_kaiming)
        init.constant_(self.lca_branch.out.weight, 0.0)
        init.constant_(self.lca_branch.out.bias, 0.0)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_parent_refiner:
            self.freeze_inherited_refiner()
        return self

    def forward(self, image: torch.Tensor, epoch: int | None = None) -> torch.Tensor:
        base_pred = self._base_pred(image)
        guides = self._guides(image, base_pred)
        with torch.no_grad() if self.freeze_parent_refiner else torch.enable_grad():
            features = self.pconv(guides)
            guide = features[:, :2].abs()
            features = self.featup(features)
            features = self.normality(features, guide)
            residual = self.out(features)
        delta = self.lca_branch(image, base_pred, guides)
        gain = self.lca_gain.clamp(0.0, 1.20)
        logits = torch.logit(base_pred)
        return torch.sigmoid(logits + self.residual_scale * (residual + gain * delta))


class SCTTCDLCABNGRefiner(SCTTargetClutterDecoupleRefiner):
    """Hybrid TCD/LCA/FeatUp refiner for bounded innovation on a strong SCTransNet base."""

    def __init__(self, checkpoint: Path, freeze_base: bool = True, residual_scale: float = 0.35) -> None:
        super().__init__(checkpoint=checkpoint, freeze_base=freeze_base, residual_scale=residual_scale)
        self.lca_branch = LocalContrastAnomalyBranch(channels=12)
        self.bng_featup = MultiScaleTinyFeatUpBranch(in_channels=5, channels=16)
        self.tcd_gain = nn.Parameter(torch.tensor(1.00))
        self.lca_gain = nn.Parameter(torch.tensor(0.45))
        self.bng_gain = nn.Parameter(torch.tensor(0.08))

    def initialize_refiner(self) -> None:
        super().initialize_refiner()
        self.lca_branch.apply(weights_init_kaiming)
        self.bng_featup.apply(weights_init_kaiming)
        init.constant_(self.lca_branch.out.weight, 0.0)
        init.constant_(self.lca_branch.out.bias, 0.0)
        init.constant_(self.bng_featup.out[-1].weight, 0.0)
        init.constant_(self.bng_featup.out[-1].bias, 0.0)

    def forward(self, image: torch.Tensor, epoch: int | None = None) -> torch.Tensor:
        base_pred = self._base_pred(image)
        guides = self._guides(image, base_pred)
        with torch.no_grad() if self.freeze_parent_refiner else torch.enable_grad():
            features = self.pconv(guides)
            guide = features[:, :2].abs()
            features = self.featup(features)
            features = self.normality(features, guide)
            inherited_residual = self.out(features)
        tcd_residual, target_gate, clutter_gate = self.tcd_branch(
            image,
            base_pred,
            guides,
            guides[:, 2:3],
            guides[:, 3:4],
        )
        lca_residual = self.lca_branch(image, base_pred, guides)
        bng_residual = self.bng_featup(guides)
        self._last_tcd_target_gate = target_gate
        self._last_tcd_clutter_gate = clutter_gate
        self._last_base_pred = base_pred.detach()
        residual = (
            inherited_residual
            + self.tcd_gain.clamp(0.0, 1.25) * tcd_residual
            + self.lca_gain.clamp(0.0, 1.00) * lca_residual
            + self.bng_gain.clamp(0.0, 0.35) * bng_residual
        )
        logits = torch.logit(base_pred)
        return torch.sigmoid(logits + self.residual_scale * residual)


class SpectralNormalityBranch(nn.Module):
    """SET/UniVAD-inspired spectral anomaly branch for tiny thermal evidence."""

    def __init__(self, channels: int = 16, max_residual: float = 0.45) -> None:
        super().__init__()
        self.max_residual = max_residual
        self.stem = nn.Sequential(
            PinwheelConv(13, channels),
            nn.LeakyReLU(0.1, inplace=True),
            DynamicHighFreqGate(channels),
            ResCBAMBlock(channels, channels),
        )
        self.normality = BackgroundNormalityBank(channels, momentum=0.99, scale=0.28, prototypes=24)
        self.spectral_router = nn.Sequential(
            nn.Conv2d(13, channels, 1, bias=False),
            nn.GroupNorm(4, channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(channels, 8, 3, padding=1, bias=False),
            nn.GroupNorm(4, 8),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(8, 1, 1),
        )
        init.constant_(self.spectral_router[-2].weight, 0.0)
        init.constant_(self.spectral_router[-2].bias, -2.0)
        init.constant_(self.out[-1].weight, 0.0)
        init.constant_(self.out[-1].bias, 0.0)

    @staticmethod
    def _norm_pos(x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        x = x / (x.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        return x.clamp(max=6.0) / 6.0

    @staticmethod
    def _norm_abs(x: torch.Tensor) -> torch.Tensor:
        x = x.abs()
        x = x / (x.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        return x.clamp(max=6.0) / 6.0

    @staticmethod
    def _spectral_highpass(image: torch.Tensor) -> torch.Tensor:
        dtype = image.dtype
        image_f = image.float()
        _, _, height, width = image_f.shape
        fy = torch.fft.fftfreq(height, d=1.0, device=image.device).view(1, 1, height, 1)
        fx = torch.fft.fftfreq(width, d=1.0, device=image.device).view(1, 1, 1, width)
        radius = torch.sqrt(fx.pow(2) + fy.pow(2))
        radius = radius / radius.max().clamp_min(1e-6)
        fft = torch.fft.fft2(image_f, norm="ortho")
        high = torch.fft.ifft2(fft * radius, norm="ortho").abs().to(dtype=dtype)
        high = high / (high.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        return high.clamp(max=6.0) / 6.0

    @staticmethod
    def _spectral_residual(image: torch.Tensor) -> torch.Tensor:
        dtype = image.dtype
        image_f = image.float()
        fft = torch.fft.fft2(image_f, norm="ortho")
        amp = torch.abs(fft).clamp_min(1e-6)
        phase = torch.angle(fft)
        log_amp = torch.log(amp)
        smooth = F.avg_pool2d(log_amp, 9, stride=1, padding=4)
        residual = log_amp - smooth
        spectral = torch.polar(torch.exp(residual), phase)
        saliency = torch.fft.ifft2(spectral, norm="ortho").abs().pow(2).to(dtype=dtype)
        saliency = F.avg_pool2d(saliency, 3, stride=1, padding=1)
        saliency = saliency / (saliency.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        return saliency.clamp(max=6.0) / 6.0

    def forward(self, image: torch.Tensor, base_pred: torch.Tensor, guides: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        local3 = F.avg_pool2d(image, 3, stride=1, padding=1)
        local7 = F.avg_pool2d(image, 7, stride=1, padding=3)
        local15 = F.avg_pool2d(image, 15, stride=1, padding=7)
        local31 = F.avg_pool2d(image, 31, stride=1, padding=15)
        hot3 = self._norm_pos(image - local3)
        hot7 = self._norm_pos(image - local7)
        hot15 = self._norm_pos(image - local15)
        dog = self._norm_abs(local3 - local31)
        cold15 = self._norm_pos(local15 - image)
        spectral_high = self._spectral_highpass(image)
        spectral_residual = self._spectral_residual(image)
        base_smooth = F.avg_pool2d(base_pred, 9, stride=1, padding=4)
        base_peak = self._norm_pos(base_pred - base_smooth)
        base_halo = self._norm_pos(base_smooth - base_pred)
        feats = torch.cat(
            [
                image,
                base_pred,
                guides[:, 2:3],
                guides[:, 3:4],
                guides[:, 4:5],
                hot3,
                hot7,
                hot15,
                dog,
                cold15,
                spectral_high,
                spectral_residual,
                base_peak + base_halo,
            ],
            dim=1,
        )
        encoded = self.stem[0](feats)
        encoded = self.stem[1](encoded)
        encoded = self.stem[2](encoded, image, base_pred, spectral_high)
        encoded = self.stem[3](encoded)
        encoded = self.normality(encoded, torch.cat([spectral_high, spectral_residual], dim=1))
        gate = self.spectral_router(feats)
        residual = self.max_residual * torch.tanh(self.out(encoded)) * (0.30 + gate)
        return residual, gate


class SCTSpectrumAnomalyRefiner(SCTTPCDHiFRefiner):
    """Bounded spectral-normality residual on top of a strong TPC/DHiF checkpoint."""

    def __init__(self, checkpoint: Path, freeze_base: bool = True, residual_scale: float = 0.35) -> None:
        super().__init__(checkpoint=checkpoint, freeze_base=freeze_base, residual_scale=residual_scale)
        self.spectral_branch = SpectralNormalityBranch(channels=16, max_residual=0.45)
        self.inherited_gain = nn.Parameter(torch.tensor(1.0))
        self.spectral_gain = nn.Parameter(torch.tensor(0.45))
        self.freeze_parent_refiner = True

    def freeze_inherited_refiner(self) -> None:
        for module in (self.pconv, self.featup, self.normality, self.out, self.dhif_gate, self.correction_gate):
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

    def initialize_refiner(self) -> None:
        super().initialize_refiner()
        self.spectral_branch.apply(weights_init_kaiming)
        init.constant_(self.spectral_branch.spectral_router[-2].weight, 0.0)
        init.constant_(self.spectral_branch.spectral_router[-2].bias, -2.0)
        init.constant_(self.spectral_branch.out[-1].weight, 0.0)
        init.constant_(self.spectral_branch.out[-1].bias, 0.0)
        self.freeze_inherited_refiner()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_parent_refiner:
            self.freeze_inherited_refiner()
        return self

    def forward(self, image: torch.Tensor, epoch: int | None = None) -> torch.Tensor:
        base_pred = self._base_pred(image)
        guides = self._guides(image, base_pred)
        with torch.no_grad() if self.freeze_parent_refiner else torch.enable_grad():
            features = self.pconv(guides)
            guide = features[:, :2].abs()
            features = self.featup(features)
            features = self.dhif_gate(features, image, base_pred, guides[:, 2:3])
            features = self.normality(features, guide)
            inherited_residual = (0.50 + self.correction_gate(guides)) * self.out(features)
        spectral_residual, spectral_gate = self.spectral_branch(image, base_pred, guides)
        self._last_spectral_gate = spectral_gate
        self._last_base_pred = base_pred.detach()
        residual = (
            self.inherited_gain.clamp(0.0, 1.15) * inherited_residual
            + self.spectral_gain.clamp(0.0, 0.90) * spectral_residual
        )
        logits = torch.logit(base_pred)
        return torch.sigmoid(logits + self.residual_scale * residual)


class PromptMorphRefinementBranch(nn.Module):
    """SAMRefiner-inspired coarse-mask prompt excavation with thermal spot shape cues."""

    def __init__(self, channels: int = 20, max_residual: float = 0.70) -> None:
        super().__init__()
        self.max_residual = max_residual
        in_channels = 18
        self.stem = nn.Sequential(
            PinwheelConv(in_channels, channels),
            nn.LeakyReLU(0.1, inplace=True),
            ResCBAMBlock(channels, channels),
        )
        self.split_context = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                    nn.GroupNorm(4, channels),
                    nn.LeakyReLU(0.1, inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=False),
                    nn.GroupNorm(4, channels),
                    nn.LeakyReLU(0.1, inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=4, dilation=4, groups=channels, bias=False),
                    nn.GroupNorm(4, channels),
                    nn.LeakyReLU(0.1, inplace=True),
                ),
            ]
        )
        self.merge = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.GroupNorm(4, channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, bias=False),
            nn.GroupNorm(4, channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 3, 1),
        )
        self.add_head = nn.Sequential(
            nn.Conv2d(channels + 5, channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 1, 1),
        )
        self.suppress_head = nn.Sequential(
            nn.Conv2d(channels + 5, channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 1, 1),
        )
        self.boundary_head = nn.Sequential(
            nn.Conv2d(channels + 5, channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 1, 1),
        )
        for head in (self.add_head, self.suppress_head, self.boundary_head):
            init.constant_(head[-1].weight, 0.0)
            init.constant_(head[-1].bias, 0.0)
        init.constant_(self.router[-1].weight, 0.0)
        init.constant_(self.router[-1].bias, 0.0)

    @staticmethod
    def _soft_dilate(x: torch.Tensor, kernel: int) -> torch.Tensor:
        if kernel % 2 == 0:
            kernel += 1
        return F.max_pool2d(x, kernel, stride=1, padding=kernel // 2)

    @staticmethod
    def _soft_erode(x: torch.Tensor, kernel: int) -> torch.Tensor:
        if kernel % 2 == 0:
            kernel += 1
        return -F.max_pool2d(-x, kernel, stride=1, padding=kernel // 2)

    @staticmethod
    def _norm_pos(x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        x = x / (x.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        return x.clamp(max=6.0) / 6.0

    @staticmethod
    def _norm_abs(x: torch.Tensor) -> torch.Tensor:
        x = x.abs()
        x = x / (x.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        return x.clamp(max=6.0) / 6.0

    def forward(self, image: torch.Tensor, primary: torch.Tensor, guides: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dil3 = self._soft_dilate(primary, 3)
        dil7 = self._soft_dilate(primary, 7)
        dil15 = self._soft_dilate(primary, 15)
        ero3 = self._soft_erode(primary, 3)
        ero7 = self._soft_erode(primary, 7)
        core = torch.sigmoid((primary - 0.55) * 14.0)
        loose = torch.sigmoid((primary - 0.22) * 12.0)
        boundary = (dil7 - ero7).clamp(0.0, 1.0)
        ring = (dil15 - dil3).clamp(0.0, 1.0)
        uncertain = (1.0 - (primary - 0.5).abs() * 2.0).clamp(0.0, 1.0)
        hole = (dil7 - primary).clamp(0.0, 1.0) * loose

        local3 = F.avg_pool2d(image, 3, stride=1, padding=1)
        local9 = F.avg_pool2d(image, 9, stride=1, padding=4)
        local21 = F.avg_pool2d(image, 21, stride=1, padding=10)
        hot3 = self._norm_pos(image - local3)
        hot9 = self._norm_pos(image - local9)
        cold21 = self._norm_pos(local21 - image)
        dog = self._norm_abs(local3 - local21)
        target_like = (hot3 + hot9 + guides[:, 2:3] + dog) * (0.20 + dil7 + core)
        target_like = target_like / (target_like.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        target_like = target_like.clamp(max=6.0) / 6.0
        clutter_like = (guides[:, 2:3] + dog + cold21) * (1.0 - dil15)
        clutter_like = clutter_like / (clutter_like.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        clutter_like = clutter_like.clamp(max=6.0) / 6.0

        feats = torch.cat(
            [
                image,
                primary,
                guides[:, 2:3],
                guides[:, 3:4],
                guides[:, 4:5],
                core,
                loose,
                ero3,
                ero7,
                dil3,
                dil7,
                boundary,
                ring,
                uncertain,
                hole,
                target_like,
                clutter_like,
                cold21,
            ],
            dim=1,
        )
        encoded = self.stem(feats)
        split = [branch(encoded) for branch in self.split_context]
        encoded = self.merge(torch.cat(split, dim=1))
        aux = torch.cat([primary, boundary, ring, target_like, clutter_like], dim=1)
        route = torch.softmax(self.router(feats), dim=1)
        add_delta = F.softplus(self.add_head(torch.cat([encoded, aux], dim=1))) - math.log(2.0)
        suppress_delta = F.softplus(self.suppress_head(torch.cat([encoded, aux], dim=1))) - math.log(2.0)
        boundary_delta = self.boundary_head(torch.cat([encoded, aux], dim=1))
        add_gate = route[:, 0:1] * (0.35 + target_like + hole + uncertain * dil7).clamp(max=2.0)
        suppress_gate = route[:, 1:2] * (0.35 + clutter_like + ring * (1.0 - core)).clamp(max=2.0)
        boundary_gate = route[:, 2:3] * (0.35 + boundary + uncertain).clamp(max=2.0)
        residual = self.max_residual * torch.tanh(
            add_gate * add_delta - suppress_gate * suppress_delta + 0.35 * boundary_gate * boundary_delta
        )
        prompt_strength = torch.cat([add_gate, suppress_gate, boundary_gate], dim=1).amax(dim=1, keepdim=True)
        return residual, prompt_strength


class SCTPromptMorphRefiner(SCTTPCDHiFRefiner):
    """Coarse-mask multi-prompt refinement on a strong TPC/DHiF teacher state."""

    def __init__(self, checkpoint: Path, freeze_base: bool = True, residual_scale: float = 0.35) -> None:
        super().__init__(checkpoint=checkpoint, freeze_base=freeze_base, residual_scale=residual_scale)
        self.prompt_branch = PromptMorphRefinementBranch(channels=20, max_residual=0.70)
        self.prompt_gain = nn.Parameter(torch.tensor(0.55))
        self.freeze_parent_refiner = True

    def freeze_inherited_refiner(self) -> None:
        for module in (self.pconv, self.featup, self.normality, self.out, self.dhif_gate, self.correction_gate):
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

    def initialize_refiner(self) -> None:
        super().initialize_refiner()
        self.prompt_branch.apply(weights_init_kaiming)
        for head in (self.prompt_branch.add_head, self.prompt_branch.suppress_head, self.prompt_branch.boundary_head):
            init.constant_(head[-1].weight, 0.0)
            init.constant_(head[-1].bias, 0.0)
        init.constant_(self.prompt_branch.router[-1].weight, 0.0)
        init.constant_(self.prompt_branch.router[-1].bias, 0.0)
        self.freeze_inherited_refiner()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_parent_refiner:
            self.freeze_inherited_refiner()
        return self

    def forward(self, image: torch.Tensor, epoch: int | None = None) -> torch.Tensor:
        base_pred = self._base_pred(image)
        guides = self._guides(image, base_pred)
        with torch.no_grad() if self.freeze_parent_refiner else torch.enable_grad():
            features = self.pconv(guides)
            guide = features[:, :2].abs()
            features = self.featup(features)
            features = self.dhif_gate(features, image, base_pred, guides[:, 2:3])
            features = self.normality(features, guide)
            inherited_residual = (0.50 + self.correction_gate(guides)) * self.out(features)
        primary_logits = torch.logit(base_pred) + self.residual_scale * inherited_residual
        primary = torch.sigmoid(primary_logits)
        prompt_residual, prompt_strength = self.prompt_branch(image, primary, self._guides(image, primary))
        self._last_prompt_strength = prompt_strength
        self._last_base_pred = primary.detach()
        return torch.sigmoid(primary_logits + self.residual_scale * self.prompt_gain.clamp(0.0, 1.20) * prompt_residual)


class DiffusionPriorBranch(nn.Module):
    """Diffuse-to-Detect-style heat diffusion prior for tiny target mask correction."""

    def __init__(self, channels: int = 20, max_residual: float = 0.75, steps: int = 5) -> None:
        super().__init__()
        self.max_residual = max_residual
        self.steps = steps
        in_channels = 16
        self.conductance = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid(),
        )
        self.stem = nn.Sequential(
            PinwheelConv(in_channels + 4, channels),
            nn.LeakyReLU(0.1, inplace=True),
            DynamicHighFreqGate(channels),
            ResCBAMBlock(channels, channels),
        )
        self.add_head = nn.Sequential(
            nn.Conv2d(channels + 5, channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 1, 1),
        )
        self.suppress_head = nn.Sequential(
            nn.Conv2d(channels + 5, channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 1, 1),
        )
        self.router = nn.Sequential(
            nn.Conv2d(in_channels + 4, channels, 1, bias=False),
            nn.GroupNorm(4, channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 2, 1),
        )
        for head in (self.add_head, self.suppress_head):
            init.constant_(head[-1].weight, 0.0)
            init.constant_(head[-1].bias, 0.0)
        init.constant_(self.router[-1].weight, 0.0)
        init.constant_(self.router[-1].bias, 0.0)
        init.constant_(self.conductance[-2].weight, 0.0)
        init.constant_(self.conductance[-2].bias, 1.1)

    @staticmethod
    def _norm_pos(x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        x = x / (x.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        return x.clamp(max=6.0) / 6.0

    @staticmethod
    def _norm_abs(x: torch.Tensor) -> torch.Tensor:
        x = x.abs()
        x = x / (x.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        return x.clamp(max=6.0) / 6.0

    @staticmethod
    def _diffuse(seed: torch.Tensor, conductance: torch.Tensor, steps: int) -> torch.Tensor:
        heat = seed
        for _ in range(max(1, steps)):
            mean = F.avg_pool2d(heat, 3, stride=1, padding=1)
            heat = (heat + 0.42 * conductance * (mean - heat)).clamp(0.0, 1.0)
        return heat

    def forward(self, image: torch.Tensor, primary: torch.Tensor, guides: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local3 = F.avg_pool2d(image, 3, stride=1, padding=1)
        local7 = F.avg_pool2d(image, 7, stride=1, padding=3)
        local15 = F.avg_pool2d(image, 15, stride=1, padding=7)
        local31 = F.avg_pool2d(image, 31, stride=1, padding=15)
        hot3 = self._norm_pos(image - local3)
        hot7 = self._norm_pos(image - local7)
        hot15 = self._norm_pos(image - local15)
        cold15 = self._norm_pos(local15 - image)
        dog = self._norm_abs(local3 - local31)
        edge = guides[:, 2:3]
        contrast = guides[:, 3:4]
        primary_peak = self._norm_pos(primary - F.avg_pool2d(primary, 9, stride=1, padding=4))
        primary_halo = self._norm_pos(F.avg_pool2d(primary, 11, stride=1, padding=5) - primary)
        core = torch.sigmoid((primary - 0.52) * 14.0)
        loose = torch.sigmoid((primary - 0.18) * 11.0)
        target_likelihood = (hot3 + hot7 + hot15 + dog + edge + contrast) / 6.0
        clutter_likelihood = (edge + dog + cold15 + primary_halo) * (1.0 - loose)
        clutter_likelihood = clutter_likelihood / (clutter_likelihood.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        clutter_likelihood = clutter_likelihood.clamp(max=6.0) / 6.0
        candidate_seed = torch.sigmoid((target_likelihood - 0.22) * 8.0) * (0.25 + loose + primary_peak).clamp(max=1.5)
        seed = torch.maximum(core, candidate_seed.clamp(0.0, 1.0))
        barrier = (1.0 - clutter_likelihood).clamp(0.0, 1.0)

        cond_feats = torch.cat(
            [
                image,
                primary,
                edge,
                contrast,
                guides[:, 4:5],
                hot3,
                hot7,
                hot15,
                dog,
                cold15,
                primary_peak,
                primary_halo,
                core,
                loose,
                target_likelihood,
                clutter_likelihood,
            ],
            dim=1,
        )
        conductance = self.conductance(cond_feats) * barrier
        heat = self._diffuse(seed, conductance, self.steps)
        heat_gap = heat - primary
        heat_boundary = self._norm_abs(F.avg_pool2d(heat, 7, stride=1, padding=3) - heat)
        feats = torch.cat([cond_feats, heat, heat_gap, heat_boundary, conductance], dim=1)
        encoded = self.stem[0](feats)
        encoded = self.stem[1](encoded)
        encoded = self.stem[2](encoded, image, primary, edge)
        encoded = self.stem[3](encoded)
        aux = torch.cat([primary, heat, heat_gap, target_likelihood, clutter_likelihood], dim=1)
        route = torch.softmax(self.router(feats), dim=1)
        add_delta = F.softplus(self.add_head(torch.cat([encoded, aux], dim=1))) - math.log(2.0)
        suppress_delta = F.softplus(self.suppress_head(torch.cat([encoded, aux], dim=1))) - math.log(2.0)
        add_gate = route[:, 0:1] * (0.25 + F.relu(heat_gap) + target_likelihood + core).clamp(max=2.2)
        suppress_gate = route[:, 1:2] * (0.25 + F.relu(-heat_gap) + clutter_likelihood + primary_halo).clamp(max=2.2)
        residual = self.max_residual * torch.tanh(add_gate * add_delta - suppress_gate * suppress_delta)
        return residual, heat, conductance


class SCTDiffusionPriorRefiner(SCTTPCDHiFRefiner):
    """Strong-checkpoint refiner with learned heat-diffusion pseudo-mask priors."""

    def __init__(self, checkpoint: Path, freeze_base: bool = True, residual_scale: float = 0.35) -> None:
        super().__init__(checkpoint=checkpoint, freeze_base=freeze_base, residual_scale=residual_scale)
        self.diffusion_branch = DiffusionPriorBranch(channels=20, max_residual=0.75, steps=5)
        self.diffusion_gain = nn.Parameter(torch.tensor(0.58))
        self.freeze_parent_refiner = True

    def freeze_inherited_refiner(self) -> None:
        for module in (self.pconv, self.featup, self.normality, self.out, self.dhif_gate, self.correction_gate):
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

    def initialize_refiner(self) -> None:
        super().initialize_refiner()
        self.diffusion_branch.apply(weights_init_kaiming)
        for head in (self.diffusion_branch.add_head, self.diffusion_branch.suppress_head):
            init.constant_(head[-1].weight, 0.0)
            init.constant_(head[-1].bias, 0.0)
        init.constant_(self.diffusion_branch.router[-1].weight, 0.0)
        init.constant_(self.diffusion_branch.router[-1].bias, 0.0)
        init.constant_(self.diffusion_branch.conductance[-2].weight, 0.0)
        init.constant_(self.diffusion_branch.conductance[-2].bias, 1.1)
        self.freeze_inherited_refiner()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_parent_refiner:
            self.freeze_inherited_refiner()
        return self

    def forward(self, image: torch.Tensor, epoch: int | None = None) -> torch.Tensor:
        base_pred = self._base_pred(image)
        guides = self._guides(image, base_pred)
        with torch.no_grad() if self.freeze_parent_refiner else torch.enable_grad():
            features = self.pconv(guides)
            guide = features[:, :2].abs()
            features = self.featup(features)
            features = self.dhif_gate(features, image, base_pred, guides[:, 2:3])
            features = self.normality(features, guide)
            inherited_residual = (0.50 + self.correction_gate(guides)) * self.out(features)
        primary_logits = torch.logit(base_pred) + self.residual_scale * inherited_residual
        primary = torch.sigmoid(primary_logits)
        diffusion_residual, heat, conductance = self.diffusion_branch(image, primary, self._guides(image, primary))
        self._last_diffusion_prior = heat
        self._last_diffusion_conductance = conductance
        self._last_base_pred = primary.detach()
        return torch.sigmoid(
            primary_logits + self.residual_scale * self.diffusion_gain.clamp(0.0, 1.25) * diffusion_residual
        )


class DirectionalIntegralStateBlock(nn.Module):
    """MambaIR-inspired long-range state mixing with an explicit local-preserving path."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 5, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _prefix_mean(x: torch.Tensor, dim: int) -> torch.Tensor:
        size = x.shape[dim]
        denom = torch.arange(1, size + 1, device=x.device, dtype=x.dtype)
        view = [1] * x.ndim
        view[dim] = size
        return torch.cumsum(x, dim=dim) / denom.view(*view)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local(x)
        left = self._prefix_mean(x, dim=-1)
        right = torch.flip(self._prefix_mean(torch.flip(x, dims=[-1]), dim=-1), dims=[-1])
        top = self._prefix_mean(x, dim=-2)
        bottom = torch.flip(self._prefix_mean(torch.flip(x, dims=[-2]), dim=-2), dims=[-2])
        state = self.mix(torch.cat([local, left, right, top, bottom], dim=1))
        gate = self.gate(torch.cat([x, local], dim=1))
        return x + self.alpha.clamp(0.0, 1.0) * gate * state


class LocalPreservingSSMBranch(nn.Module):
    """Restoration-style SSM branch for global background modeling without erasing tiny evidence."""

    def __init__(self, channels: int = 20, max_residual: float = 0.70) -> None:
        super().__init__()
        self.max_residual = max_residual
        in_channels = 15
        self.stem = nn.Sequential(
            PinwheelConv(in_channels, channels),
            nn.LeakyReLU(0.1, inplace=True),
            DynamicHighFreqGate(channels),
            ResCBAMBlock(channels, channels),
        )
        self.state1 = DirectionalIntegralStateBlock(channels)
        self.state2 = DirectionalIntegralStateBlock(channels)
        self.local_preserve = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GroupNorm(4, channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, bias=False),
            nn.GroupNorm(4, channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 2, 1),
        )
        self.out = nn.Sequential(
            nn.Conv2d(channels * 2 + 4, channels, 3, padding=1, bias=False),
            nn.GroupNorm(4, channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, 1, 1),
        )
        init.constant_(self.router[-1].weight, 0.0)
        init.constant_(self.router[-1].bias, 0.0)
        init.constant_(self.out[-1].weight, 0.0)
        init.constant_(self.out[-1].bias, 0.0)

    @staticmethod
    def _norm_pos(x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        x = x / (x.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        return x.clamp(max=6.0) / 6.0

    @staticmethod
    def _norm_abs(x: torch.Tensor) -> torch.Tensor:
        x = x.abs()
        x = x / (x.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        return x.clamp(max=6.0) / 6.0

    def forward(self, image: torch.Tensor, primary: torch.Tensor, guides: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        local3 = F.avg_pool2d(image, 3, stride=1, padding=1)
        local9 = F.avg_pool2d(image, 9, stride=1, padding=4)
        local21 = F.avg_pool2d(image, 21, stride=1, padding=10)
        hot3 = self._norm_pos(image - local3)
        hot9 = self._norm_pos(image - local9)
        cold21 = self._norm_pos(local21 - image)
        dog = self._norm_abs(local3 - local21)
        edge = guides[:, 2:3]
        contrast = guides[:, 3:4]
        primary_peak = self._norm_pos(primary - F.avg_pool2d(primary, 7, stride=1, padding=3))
        primary_context = F.avg_pool2d(primary, 17, stride=1, padding=8)
        row_state = torch.mean(primary, dim=-1, keepdim=True).expand_as(primary)
        col_state = torch.mean(primary, dim=-2, keepdim=True).expand_as(primary)
        bg_state = F.avg_pool2d((1.0 - primary).clamp(0.0, 1.0), 31, stride=1, padding=15)
        feats = torch.cat(
            [
                image,
                primary,
                edge,
                contrast,
                guides[:, 4:5],
                hot3,
                hot9,
                dog,
                cold21,
                primary_peak,
                primary_context,
                row_state,
                col_state,
                bg_state,
                F.relu(primary_context - primary),
            ],
            dim=1,
        )
        encoded = self.stem[0](feats)
        encoded = self.stem[1](encoded)
        encoded = self.stem[2](encoded, image, primary, edge)
        encoded = self.stem[3](encoded)
        local = self.local_preserve(feats)
        state = self.state2(self.state1(encoded))
        route = torch.softmax(self.router(feats), dim=1)
        aux = torch.cat([primary, primary_peak, hot3, F.relu(primary_context - primary)], dim=1)
        raw_delta = F.softplus(self.out(torch.cat([state, local, aux], dim=1))) - math.log(2.0)
        target_gate = route[:, 0:1] * (0.30 + primary_peak + hot3 + hot9).clamp(max=2.0)
        clutter_gate = route[:, 1:2] * (0.30 + dog + edge + F.relu(primary_context - primary)).clamp(max=2.0)
        residual = self.max_residual * torch.tanh((target_gate - 0.65 * clutter_gate) * raw_delta)
        state_strength = (target_gate + clutter_gate).clamp(0.0, 2.0)
        return residual, state_strength


class SCTLPSMRefiner(SCTTPCDHiFRefiner):
    """MambaIR/MSVMamba-inspired local-preserving state-space residual refiner."""

    def __init__(self, checkpoint: Path, freeze_base: bool = True, residual_scale: float = 0.35) -> None:
        super().__init__(checkpoint=checkpoint, freeze_base=freeze_base, residual_scale=residual_scale)
        self.lpsm_branch = LocalPreservingSSMBranch(channels=20, max_residual=0.70)
        self.lpsm_gain = nn.Parameter(torch.tensor(0.55))
        self.freeze_parent_refiner = True

    def freeze_inherited_refiner(self) -> None:
        for module in (self.pconv, self.featup, self.normality, self.out, self.dhif_gate, self.correction_gate):
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

    def initialize_refiner(self) -> None:
        super().initialize_refiner()
        self.lpsm_branch.apply(weights_init_kaiming)
        init.constant_(self.lpsm_branch.router[-1].weight, 0.0)
        init.constant_(self.lpsm_branch.router[-1].bias, 0.0)
        init.constant_(self.lpsm_branch.out[-1].weight, 0.0)
        init.constant_(self.lpsm_branch.out[-1].bias, 0.0)
        init.constant_(self.lpsm_branch.state1.alpha, 0.0)
        init.constant_(self.lpsm_branch.state2.alpha, 0.0)
        self.freeze_inherited_refiner()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_parent_refiner:
            self.freeze_inherited_refiner()
        return self

    def forward(self, image: torch.Tensor, epoch: int | None = None) -> torch.Tensor:
        base_pred = self._base_pred(image)
        guides = self._guides(image, base_pred)
        with torch.no_grad() if self.freeze_parent_refiner else torch.enable_grad():
            features = self.pconv(guides)
            guide = features[:, :2].abs()
            features = self.featup(features)
            features = self.dhif_gate(features, image, base_pred, guides[:, 2:3])
            features = self.normality(features, guide)
            inherited_residual = (0.50 + self.correction_gate(guides)) * self.out(features)
        primary_logits = torch.logit(base_pred) + self.residual_scale * inherited_residual
        primary = torch.sigmoid(primary_logits)
        lpsm_residual, state_strength = self.lpsm_branch(image, primary, self._guides(image, primary))
        self._last_lpsm_state_strength = state_strength
        self._last_base_pred = primary.detach()
        return torch.sigmoid(primary_logits + self.residual_scale * self.lpsm_gain.clamp(0.0, 1.25) * lpsm_residual)


class ExpertDeltaGate(nn.Module):
    """Small gate that borrows only bounded expert logit differences."""

    def __init__(self, in_channels: int = 8, hidden: int = 12, max_gate: float = 0.45) -> None:
        super().__init__()
        self.max_gate = max_gate
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(4, hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2, groups=hidden, bias=False),
            nn.GroupNorm(4, hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )
        init.constant_(self.net[-1].weight, 0.0)
        init.constant_(self.net[-1].bias, -8.0)

    @staticmethod
    def _norm_pos(x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        x = x / (x.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        return x.clamp(max=6.0) / 6.0

    def forward(self, image: torch.Tensor, primary: torch.Tensor, expert: torch.Tensor, laplace: torch.Tensor) -> torch.Tensor:
        contrast = self._norm_pos(image - F.avg_pool2d(image, 9, stride=1, padding=4))
        expert_gain = self._norm_pos(expert - primary)
        expert_drop = self._norm_pos(primary - expert)
        disagreement = (expert - primary).abs().clamp(max=1.0)
        primary_peak = self._norm_pos(primary - F.avg_pool2d(primary, 7, stride=1, padding=3))
        features = torch.cat(
            [
                image,
                primary,
                expert,
                disagreement,
                expert_gain,
                expert_drop,
                laplace,
                primary_peak,
            ],
            dim=1,
        )
        return self.max_gate * torch.sigmoid(self.net(features))


class SCTExpertGateRefiner(nn.Module):
    """Frozen primary/expert predictors with a trainable target-clutter gate."""

    def __init__(self, checkpoint: Path, freeze_base: bool = True, residual_scale: float = 1.0) -> None:
        super().__init__()
        self.primary = SCTRefiner(checkpoint=checkpoint, freeze_base=freeze_base)
        self.expert: nn.Module | None = None
        self.gate = ExpertDeltaGate()
        self.residual_scale = residual_scale
        laplace = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
        self.register_buffer("laplace_kernel", laplace)

    def initialize_refiner(self) -> None:
        self.primary.initialize_refiner()
        self.primary.eval()
        for param in self.primary.parameters():
            param.requires_grad_(False)
        self.gate.apply(weights_init_kaiming)
        init.constant_(self.gate.net[-1].weight, 0.0)
        init.constant_(self.gate.net[-1].bias, -8.0)

    def load_primary_checkpoint(self, checkpoint: Path) -> None:
        load_model_checkpoint(self.primary, checkpoint)
        self.primary.eval()
        for param in self.primary.parameters():
            param.requires_grad_(False)

    def load_expert_checkpoint(self, model_name: str, base_checkpoint: str, checkpoint: str) -> None:
        if not checkpoint:
            raise ValueError("--expert_checkpoint is required for sct_expert_gate_refiner")
        device = self.laplace_kernel.device
        self.expert = make_model(model_name, warm_epoch=1, base_checkpoint=base_checkpoint).to(device)
        if isinstance(self.expert, SCTRefiner):
            self.expert.initialize_refiner()
        else:
            self.expert.apply(weights_init_kaiming)
        load_model_checkpoint(self.expert, Path(checkpoint))
        self.expert.eval()
        for param in self.expert.parameters():
            param.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.primary.eval()
        if self.expert is not None:
            self.expert.eval()
        return self

    def _laplace(self, image: torch.Tensor) -> torch.Tensor:
        lap = F.conv2d(image, self.laplace_kernel.to(dtype=image.dtype), padding=1).abs()
        lap = lap / (lap.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        return lap.clamp(max=6.0) / 6.0

    def forward(self, image: torch.Tensor, epoch: int | None = None) -> torch.Tensor:
        if self.expert is None:
            raise RuntimeError("Expert checkpoint has not been loaded.")
        with torch.no_grad():
            primary = self.primary(image, epoch=epoch).clamp(1e-5, 1.0 - 1e-5)
            expert = self.expert(image, epoch=epoch)
            if isinstance(expert, (tuple, list)):
                expert = expert[-1]
            expert = expert.clamp(1e-5, 1.0 - 1e-5)
        self._last_base_pred = primary.detach()
        delta = (torch.logit(expert) - torch.logit(primary)).clamp(-3.0, 3.0)
        gate = self.gate(image, primary, expert, self._laplace(image))
        self._last_expert_pred = expert.detach()
        self._last_expert_gate = gate
        self._last_expert_gate_prob = (gate / self.gate.max_gate).clamp(1e-5, 1.0 - 1e-5)
        return torch.sigmoid(torch.logit(primary) + self.residual_scale * gate * delta)


class BGMWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = BasicUNetPlus()

    def forward(self, image: torch.Tensor, epoch: int | None = None):
        return self.net(image)


class MSHNetWrapper(nn.Module):
    def __init__(self, warm_epoch: int) -> None:
        super().__init__()
        self.net = MSHNetCore(1)
        self.warm_epoch = warm_epoch

    def forward(self, image: torch.Tensor, epoch: int | None = None):
        warm = epoch is not None and epoch > self.warm_epoch
        masks, logits = self.net(image, warm)
        probs = [torch.sigmoid(mask) for mask in masks]
        probs.append(torch.sigmoid(logits))
        return probs


class P43FirstBlockResNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        use_pconv = in_channels == 16 and out_channels == 16
        if use_pconv:
            self.conv1 = OfficialPConv(in_channels, out_channels, k=4, s=stride)
            self.conv2 = OfficialPConv(out_channels, out_channels, k=3, s=1)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = None
        self.ca = MSHNetChannelAttention(out_channels)
        self.sa = MSHNetSpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.shortcut is None else self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        return self.relu(out)


class OfficialMSHNetWrapper(nn.Module):
    def __init__(self, warm_epoch: int, p43: bool = False) -> None:
        super().__init__()
        self.net = MSHNetCore(3, block=P43FirstBlockResNet) if p43 else MSHNetCore(3)
        self.warm_epoch = warm_epoch

    def forward(self, image: torch.Tensor, epoch: int | None = None):
        warm = epoch is not None and epoch > self.warm_epoch
        masks, logits = self.net(image, warm)
        probs = [torch.sigmoid(mask) for mask in masks]
        probs.append(torch.sigmoid(logits))
        return probs


def weights_init_kaiming(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if classname.find("Conv") != -1 and hasattr(module, "weight"):
        init.kaiming_normal_(module.weight.data, a=0, mode="fan_in")
        if getattr(module, "bias", None) is not None:
            init.constant_(module.bias.data, 0.0)
    elif classname.find("BatchNorm") != -1:
        init.normal_(module.weight.data, 1.0, 0.02)
        init.constant_(module.bias.data, 0.0)


def make_model(name: str, warm_epoch: int, base_checkpoint: str = "") -> nn.Module:
    if name == "bgm_basic":
        return BGMWrapper()
    if name == "mshnet_sls":
        return MSHNetWrapper(warm_epoch=warm_epoch)
    if name == "mshnet3_sls":
        return OfficialMSHNetWrapper(warm_epoch=warm_epoch, p43=False)
    if name == "mshnet_p43_sls":
        return OfficialMSHNetWrapper(warm_epoch=warm_epoch, p43=True)
    if name == "tcf_bgm":
        return TCFBasicUNet()
    if name == "pconv_tcf":
        return PConvTCFBasicUNet()
    if name == "apconv_tcf":
        return APConvTCFBasicUNet()
    if name == "sct_refiner":
        if not base_checkpoint:
            raise ValueError("--base_checkpoint is required for sct_refiner")
        return SCTRefiner(Path(base_checkpoint), freeze_base=True)
    if name == "sct_dhif_refiner":
        if not base_checkpoint:
            raise ValueError("--base_checkpoint is required for sct_dhif_refiner")
        return SCTDHiFRefiner(Path(base_checkpoint), freeze_base=True)
    if name == "sct_tpc_dhif_refiner":
        if not base_checkpoint:
            raise ValueError("--base_checkpoint is required for sct_tpc_dhif_refiner")
        return SCTTPCDHiFRefiner(Path(base_checkpoint), freeze_base=True)
    if name == "sct_apconv_tpc_refiner":
        if not base_checkpoint:
            raise ValueError("--base_checkpoint is required for sct_apconv_tpc_refiner")
        return SCTAPConvTPCRefiner(Path(base_checkpoint), freeze_base=True)
    if name == "sct_apconv_tpc_unfrozen_refiner":
        if not base_checkpoint:
            raise ValueError("--base_checkpoint is required for sct_apconv_tpc_unfrozen_refiner")
        return SCTAPConvTPCRefiner(Path(base_checkpoint), freeze_base=False)
    if name == "sct_bng_featup_refiner":
        if not base_checkpoint:
            raise ValueError("--base_checkpoint is required for sct_bng_featup_refiner")
        return SCTBNGFeatUpRefiner(Path(base_checkpoint), freeze_base=True)
    if name == "sct_lca_refiner":
        if not base_checkpoint:
            raise ValueError("--base_checkpoint is required for sct_lca_refiner")
        return SCTLocalContrastAnomalyRefiner(Path(base_checkpoint), freeze_base=True)
    if name == "sct_tcd_decouple_refiner":
        if not base_checkpoint:
            raise ValueError("--base_checkpoint is required for sct_tcd_decouple_refiner")
        return SCTTargetClutterDecoupleRefiner(Path(base_checkpoint), freeze_base=True)
    if name == "sct_tcd_lca_bng_refiner":
        if not base_checkpoint:
            raise ValueError("--base_checkpoint is required for sct_tcd_lca_bng_refiner")
        return SCTTCDLCABNGRefiner(Path(base_checkpoint), freeze_base=True)
    if name == "sct_spectral_anomaly_refiner":
        if not base_checkpoint:
            raise ValueError("--base_checkpoint is required for sct_spectral_anomaly_refiner")
        return SCTSpectrumAnomalyRefiner(Path(base_checkpoint), freeze_base=True)
    if name == "sct_prompt_morph_refiner":
        if not base_checkpoint:
            raise ValueError("--base_checkpoint is required for sct_prompt_morph_refiner")
        return SCTPromptMorphRefiner(Path(base_checkpoint), freeze_base=True)
    if name == "sct_diffusion_prior_refiner":
        if not base_checkpoint:
            raise ValueError("--base_checkpoint is required for sct_diffusion_prior_refiner")
        return SCTDiffusionPriorRefiner(Path(base_checkpoint), freeze_base=True)
    if name == "sct_lpsm_refiner":
        if not base_checkpoint:
            raise ValueError("--base_checkpoint is required for sct_lpsm_refiner")
        return SCTLPSMRefiner(Path(base_checkpoint), freeze_base=True)
    if name == "sct_expert_gate_refiner":
        if not base_checkpoint:
            raise ValueError("--base_checkpoint is required for sct_expert_gate_refiner")
        return SCTExpertGateRefiner(Path(base_checkpoint), freeze_base=True)
    raise ValueError(f"Unknown model: {name}")


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    inter = (pred_flat * target_flat).sum(dim=1)
    denom = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    return (1.0 - (2.0 * inter + smooth) / (denom + smooth)).mean()


def soft_iou_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    inter = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - inter
    return (1.0 - (inter + smooth) / (union + smooth)).mean()


def focal_bce_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
    pred = pred.clamp(1e-5, 1.0 - 1e-5)
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    pt = torch.where(target > 0.5, pred, 1.0 - pred)
    alpha_t = torch.where(target > 0.5, torch.full_like(target, alpha), torch.full_like(target, 1.0 - alpha))
    return (alpha_t * (1.0 - pt).pow(gamma) * bce).mean()


def sls_iou_loss(pred: torch.Tensor, target: torch.Tensor, location_weight: float) -> torch.Tensor:
    pred = pred.clamp(1e-5, 1.0 - 1e-5)
    smooth = 1e-6
    intersection = (pred * target).flatten(1).sum(dim=1)
    pred_sum = pred.flatten(1).sum(dim=1)
    target_sum = target.flatten(1).sum(dim=1)
    iou = (intersection + smooth) / (pred_sum + target_sum - intersection + smooth)
    scale_delta = ((pred_sum - target_sum) * 0.5).pow(2)
    alpha = (torch.minimum(pred_sum, target_sum) + scale_delta + smooth) / (
        torch.maximum(pred_sum, target_sum) + scale_delta + smooth
    )
    loss = 1.0 - (alpha * iou).mean()

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
    return loss + location_weight * (loc.sum() / positive.sum().clamp_min(1.0))


def sdm_iou_loss(pred: torch.Tensor, target: torch.Tensor, epoch: int, args: argparse.Namespace) -> torch.Tensor:
    """Mask-format SDM loss adapted from official PConv+SD loss."""

    pred = pred.clamp(1e-5, 1.0 - 1e-5)
    smooth = 1e-8
    batch, _, height, width = pred.shape
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    pred_sum = pred_flat.sum(dim=1)
    target_sum = target_flat.sum(dim=1)
    union = pred_sum + target_sum - intersection
    iou = (intersection + smooth) / (union + smooth)
    if epoch <= args.sdm_warm_epochs:
        return 1.0 - iou.mean()

    scale_delta = ((pred_sum - target_sum) * 0.5).pow(2)
    alpha = (torch.minimum(pred_sum, target_sum) + scale_delta + smooth) / (
        torch.maximum(pred_sum, target_sum) + scale_delta + smooth
    )
    scaled_iou = alpha * iou
    if not args.sdm_with_distance:
        return 1.0 - scaled_iou.mean()

    x_index = torch.arange(0, width, device=pred.device, dtype=pred.dtype).view(1, 1, 1, width) / float(width)
    y_index = torch.arange(0, height, device=pred.device, dtype=pred.dtype).view(1, 1, height, 1) / float(height)
    pred_cx = (x_index * pred).mean(dim=(1, 2, 3))
    pred_cy = (y_index * pred).mean(dim=(1, 2, 3))
    target_cx = (x_index * target).mean(dim=(1, 2, 3))
    target_cy = (y_index * target).mean(dim=(1, 2, 3))
    positive = target_sum > 0
    if not torch.any(positive):
        return 1.0 - scaled_iou.mean()
    angle = (4.0 / (math.pi**2)) * (
        torch.atan(pred_cy / (pred_cx + smooth)) - torch.atan(target_cy / (target_cx + smooth))
    ).pow(2)
    pred_length = torch.sqrt(pred_cx.pow(2) + pred_cy.pow(2) + smooth)
    target_length = torch.sqrt(target_cx.pow(2) + target_cy.pow(2) + smooth)
    length = torch.minimum(pred_length, target_length) / (torch.maximum(pred_length, target_length) + smooth)
    location = (1.0 - length + angle)[positive].mean()

    if args.sdm_dynamic:
        resize_factor = 512.0 * 512.0 / float(max(1, width * height))
        beta = target_sum * args.sdm_delta * resize_factor / 81.0
        beta = torch.where(beta > args.sdm_delta, torch.full_like(beta, args.sdm_delta), beta).mean()
        return (1.0 + beta) * (1.0 - scaled_iou.mean()) + (1.0 - beta) * location
    return 1.0 - scaled_iou.mean() + location


def image_to_luma(image: torch.Tensor) -> torch.Tensor:
    if image.shape[1] == 1:
        return image
    weights = image.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    if image.shape[1] == 3:
        return (image * weights).sum(dim=1, keepdim=True)
    return image.mean(dim=1, keepdim=True)


def background_hf_loss(pred: torch.Tensor, target: torch.Tensor, image: torch.Tensor, weight: float, mass_weight: float, dilate: int) -> torch.Tensor:
    if weight <= 0 and mass_weight <= 0:
        return pred.new_tensor(0.0)
    if target.shape[-2:] != pred.shape[-2:]:
        target = F.interpolate(target, size=pred.shape[-2:], mode="nearest")
    if image.shape[-2:] != pred.shape[-2:]:
        image = F.interpolate(image, size=pred.shape[-2:], mode="bilinear", align_corners=False)
    image = image_to_luma(image)
    if dilate > 1:
        if dilate % 2 == 0:
            dilate += 1
        guard = F.max_pool2d(target, dilate, stride=1, padding=dilate // 2)
    else:
        guard = target
    bg = (1.0 - guard).detach()
    with torch.no_grad():
        kernel = pred.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
        hf = F.conv2d(image, kernel, padding=1).abs()
        hf = hf / (hf.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        hf = hf.clamp(max=6.0)
    hf_term = (pred.pow(2) * bg * hf).sum() / (bg * hf).sum().clamp_min(1.0)
    mass_term = (pred * bg).sum() / bg.sum().clamp_min(1.0)
    return weight * hf_term + mass_weight * mass_term


def teacher_preservation_loss(
    pred: torch.Tensor,
    teacher_pred: torch.Tensor | None,
    target: torch.Tensor,
    image: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if teacher_pred is None or args.teacher_weight <= 0:
        return pred.new_tensor(0.0)
    if target.shape[-2:] != pred.shape[-2:]:
        target = F.interpolate(target, size=pred.shape[-2:], mode="nearest")
    if teacher_pred.shape[-2:] != pred.shape[-2:]:
        teacher_pred = F.interpolate(teacher_pred, size=pred.shape[-2:], mode="bilinear", align_corners=False)
    if image.shape[-2:] != pred.shape[-2:]:
        image = F.interpolate(image, size=pred.shape[-2:], mode="bilinear", align_corners=False)
    image = image_to_luma(image)

    pred = pred.clamp(1e-5, 1.0 - 1e-5)
    teacher_pred = teacher_pred.detach().clamp(1e-5, 1.0 - 1e-5)
    dilate = max(1, int(args.teacher_target_dilate))
    if dilate % 2 == 0:
        dilate += 1
    guard = F.max_pool2d(target, dilate, stride=1, padding=dilate // 2)
    bg = (1.0 - guard).detach()

    with torch.no_grad():
        kernel = pred.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
        hf = F.conv2d(image, kernel, padding=1).abs()
        hf = hf / (hf.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        hf = 1.0 + args.teacher_hf_boost * (hf.clamp(max=6.0) / 6.0)

    teacher_bg = (teacher_pred < args.teacher_bg_threshold).float()
    bg_weight = bg * teacher_bg * hf
    bg_rise = F.relu(pred - teacher_pred - args.teacher_margin)
    bg_loss = (bg_rise.pow(2) * bg_weight).sum() / bg_weight.sum().clamp_min(1.0)

    teacher_pos = target * (teacher_pred > args.teacher_pos_threshold).float()
    pos_drop = F.relu(teacher_pred - pred - args.teacher_margin)
    pos_loss = (pos_drop.pow(2) * teacher_pos).sum() / teacher_pos.sum().clamp_min(1.0)

    missed = (target * (1.0 - teacher_pred)).detach()
    miss_bce = F.binary_cross_entropy(pred, target, reduction="none")
    miss_loss = (miss_bce * missed).sum() / missed.sum().clamp_min(1.0)
    logit_loss = F.mse_loss(torch.logit(pred), torch.logit(teacher_pred))

    return args.teacher_weight * (
        args.teacher_bg_weight * bg_loss
        + args.teacher_pos_weight * pos_loss
        + args.teacher_miss_weight * miss_loss
        + args.teacher_logit_weight * logit_loss
    )


def pseudo_label_loss(
    pred: torch.Tensor,
    pseudo: torch.Tensor | None,
    target: torch.Tensor,
    image: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if pseudo is None or args.pseudo_weight <= 0:
        return pred.new_tensor(0.0)
    if pseudo.shape[-2:] != pred.shape[-2:]:
        pseudo = F.interpolate(pseudo, size=pred.shape[-2:], mode="bilinear", align_corners=False)
    if target.shape[-2:] != pred.shape[-2:]:
        target = F.interpolate(target, size=pred.shape[-2:], mode="nearest")
    if image.shape[-2:] != pred.shape[-2:]:
        image = F.interpolate(image, size=pred.shape[-2:], mode="bilinear", align_corners=False)
    image = image_to_luma(image)

    pred = pred.clamp(1e-5, 1.0 - 1e-5)
    pseudo = pseudo.detach().clamp(1e-5, 1.0 - 1e-5)
    dilate = max(1, int(args.pseudo_target_dilate))
    if dilate % 2 == 0:
        dilate += 1
    target_guard = F.max_pool2d(target, dilate, stride=1, padding=dilate // 2)
    pseudo_pos = (pseudo > args.pseudo_pos_threshold).float()
    pseudo_bg = (pseudo < args.pseudo_bg_threshold).float()
    pos_weight = torch.maximum(target_guard, pseudo_pos)
    bg_weight = (1.0 - target_guard) * pseudo_bg
    soft_weight = 1.0 + args.pseudo_pos_weight * pos_weight + args.pseudo_bg_weight * bg_weight

    if args.pseudo_hf_boost > 0:
        with torch.no_grad():
            kernel = pred.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
            hf = F.conv2d(image, kernel, padding=1).abs()
            hf = hf / (hf.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
            soft_weight = soft_weight * (1.0 + args.pseudo_hf_boost * (hf.clamp(max=6.0) / 6.0))

    soft_bce = F.binary_cross_entropy(pred, pseudo, reduction="none")
    soft_term = (soft_bce * soft_weight).sum() / soft_weight.sum().clamp_min(1.0)
    logit_term = F.mse_loss(torch.logit(pred), torch.logit(pseudo))
    iou_term = soft_iou_loss(pred, pseudo)
    rim_term = pred.new_tensor(0.0)
    if args.pseudo_rim_weight > 0:
        rim = ((pseudo >= args.pseudo_rim_low) & (pseudo <= args.pseudo_rim_high)).float()
        rim = rim * target_guard.detach()
        rim_target = pseudo.clamp(max=args.pseudo_rim_target)
        rim_term = (F.relu(rim_target - pred).pow(2) * rim).sum() / rim.sum().clamp_min(1.0)
    return args.pseudo_weight * (
        args.pseudo_bce_weight * soft_term
        + args.pseudo_logit_weight * logit_term
        + args.pseudo_iou_weight * iou_term
        + args.pseudo_rim_weight * rim_term
    )


def base_anchor_loss(
    pred: torch.Tensor,
    base_pred: torch.Tensor | None,
    target: torch.Tensor,
    image: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if base_pred is None or args.base_anchor_weight <= 0:
        return pred.new_tensor(0.0)
    if target.shape[-2:] != pred.shape[-2:]:
        target = F.interpolate(target, size=pred.shape[-2:], mode="nearest")
    if base_pred.shape[-2:] != pred.shape[-2:]:
        base_pred = F.interpolate(base_pred, size=pred.shape[-2:], mode="bilinear", align_corners=False)
    if image.shape[-2:] != pred.shape[-2:]:
        image = F.interpolate(image, size=pred.shape[-2:], mode="bilinear", align_corners=False)
    image = image_to_luma(image)

    pred = pred.clamp(1e-5, 1.0 - 1e-5)
    base_pred = base_pred.detach().clamp(1e-5, 1.0 - 1e-5)
    dilate = max(1, int(args.base_anchor_target_dilate))
    if dilate % 2 == 0:
        dilate += 1
    guard = F.max_pool2d(target, dilate, stride=1, padding=dilate // 2)
    bg = (1.0 - guard).detach()

    with torch.no_grad():
        kernel = pred.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
        hf = F.conv2d(image, kernel, padding=1).abs()
        hf = hf / (hf.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        hf_weight = 1.0 + args.base_anchor_hf_boost * (hf.clamp(max=6.0) / 6.0)

    bg_rise = F.relu(pred - base_pred - args.base_anchor_margin)
    bg_weight = bg * (base_pred < args.base_anchor_bg_threshold).float() * hf_weight
    bg_loss = (bg_rise.pow(2) * bg_weight).sum() / bg_weight.sum().clamp_min(1.0)

    target_drop = F.relu(base_pred - pred - args.base_anchor_margin)
    target_weight = target * (base_pred > args.base_anchor_pos_threshold).float()
    pos_loss = (target_drop.pow(2) * target_weight).sum() / target_weight.sum().clamp_min(1.0)

    logit_loss = F.mse_loss(torch.logit(pred), torch.logit(base_pred))
    return args.base_anchor_weight * (
        args.base_anchor_bg_weight * bg_loss
        + args.base_anchor_pos_weight * pos_loss
        + args.base_anchor_logit_weight * logit_loss
    )


def expert_gate_oracle_loss(
    primary_pred: torch.Tensor | None,
    expert_pred: torch.Tensor | None,
    gate_prob: torch.Tensor | None,
    target: torch.Tensor,
    image: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if (
        primary_pred is None
        or expert_pred is None
        or gate_prob is None
        or (args.expert_gate_oracle_weight <= 0 and args.expert_gate_sparsity_weight <= 0)
    ):
        fallback = target if isinstance(target, torch.Tensor) else image
        return fallback.new_tensor(0.0)
    if target.shape[-2:] != gate_prob.shape[-2:]:
        target = F.interpolate(target, size=gate_prob.shape[-2:], mode="nearest")
    if primary_pred.shape[-2:] != gate_prob.shape[-2:]:
        primary_pred = F.interpolate(primary_pred, size=gate_prob.shape[-2:], mode="bilinear", align_corners=False)
    if expert_pred.shape[-2:] != gate_prob.shape[-2:]:
        expert_pred = F.interpolate(expert_pred, size=gate_prob.shape[-2:], mode="bilinear", align_corners=False)
    if image.shape[-2:] != gate_prob.shape[-2:]:
        image = F.interpolate(image, size=gate_prob.shape[-2:], mode="bilinear", align_corners=False)
    image = image_to_luma(image)

    primary_pred = primary_pred.detach().clamp(1e-5, 1.0 - 1e-5)
    expert_pred = expert_pred.detach().clamp(1e-5, 1.0 - 1e-5)
    gate_prob = gate_prob.clamp(1e-5, 1.0 - 1e-5)

    dilate = max(1, int(args.expert_gate_target_dilate))
    if dilate % 2 == 0:
        dilate += 1
    with torch.no_grad():
        target_guard = F.max_pool2d(target, dilate, stride=1, padding=dilate // 2)
        primary_err = F.binary_cross_entropy(primary_pred, target, reduction="none")
        expert_err = F.binary_cross_entropy(expert_pred, target, reduction="none")
        err_delta = primary_err - expert_err
        expert_better = (err_delta > args.expert_gate_oracle_margin).float()
        primary_better = (err_delta < -args.expert_gate_oracle_margin).float()
        valid = expert_better + primary_better

        disagreement = (expert_pred - primary_pred).abs()
        if args.expert_gate_min_disagree > 0:
            valid = valid * (disagreement > args.expert_gate_min_disagree).float()

        kernel = gate_prob.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
        hf = F.conv2d(image, kernel, padding=1).abs()
        hf = hf / (hf.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        hf = hf.clamp(max=6.0) / 6.0
        disagree_norm = disagreement / (disagreement.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        disagree_norm = disagree_norm.clamp(max=6.0) / 6.0
        oracle_target = expert_better.detach()
        oracle_weight = valid * (
            1.0
            + args.expert_gate_pos_boost * target_guard
            + args.expert_gate_disagree_boost * disagree_norm
            + args.expert_gate_hf_boost * hf
        )

    loss = gate_prob.new_tensor(0.0)
    if args.expert_gate_oracle_weight > 0:
        oracle_bce = F.binary_cross_entropy(gate_prob, oracle_target, reduction="none")
        loss = loss + args.expert_gate_oracle_weight * (
            (oracle_bce * oracle_weight).sum() / oracle_weight.sum().clamp_min(1.0)
        )
    if args.expert_gate_sparsity_weight > 0:
        bg = (1.0 - target_guard).detach()
        loss = loss + args.expert_gate_sparsity_weight * ((gate_prob * bg).sum() / bg.sum().clamp_min(1.0))
    return loss


def tcd_gate_loss(
    target_gate: torch.Tensor | None,
    clutter_gate: torch.Tensor | None,
    target: torch.Tensor,
    image: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if target_gate is None or clutter_gate is None or args.tcd_gate_weight <= 0:
        return target.new_tensor(0.0)
    if target.shape[-2:] != target_gate.shape[-2:]:
        target = F.interpolate(target, size=target_gate.shape[-2:], mode="nearest")
    if image.shape[-2:] != target_gate.shape[-2:]:
        image = F.interpolate(image, size=target_gate.shape[-2:], mode="bilinear", align_corners=False)
    image = image_to_luma(image)

    target_gate = target_gate.clamp(1e-5, 1.0 - 1e-5)
    clutter_gate = clutter_gate.clamp(1e-5, 1.0 - 1e-5)
    dilate = max(1, int(args.tcd_target_dilate))
    if dilate % 2 == 0:
        dilate += 1
    target_guard = F.max_pool2d(target, dilate, stride=1, padding=dilate // 2).detach()

    with torch.no_grad():
        kernel = image.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
        hf = F.conv2d(image, kernel, padding=1).abs()
        hf = hf / (hf.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        hf = hf.clamp(max=6.0) / 6.0
        bg_hf = (1.0 - target_guard) * hf
        target_weight = target_guard * (1.0 + args.tcd_target_hf_boost * hf)
        clutter_weight = bg_hf

    target_loss = ((1.0 - target_gate).pow(2) * target_weight).sum() / target_weight.sum().clamp_min(1.0)
    clutter_loss = ((1.0 - clutter_gate).pow(2) * clutter_weight).sum() / clutter_weight.sum().clamp_min(1.0)
    anti_target = (clutter_gate.pow(2) * target_guard).sum() / target_guard.sum().clamp_min(1.0)
    return args.tcd_gate_weight * (
        args.tcd_target_gate_weight * target_loss
        + args.tcd_clutter_gate_weight * clutter_loss
        + args.tcd_anti_target_weight * anti_target
    )


def thermal_spot_shape_loss(pred: torch.Tensor, target: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    if args.thermal_spot_weight <= 0:
        return pred.new_tensor(0.0)
    if target.shape[-2:] != pred.shape[-2:]:
        target = F.interpolate(target, size=pred.shape[-2:], mode="nearest")
    pred = pred.clamp(1e-5, 1.0 - 1e-5)
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
    sigma = (radius * args.thermal_spot_sigma_scale).clamp(
        min=args.thermal_spot_sigma_min,
        max=args.thermal_spot_sigma_max,
    )
    dist2 = (x_index - center_x).pow(2) + (y_index - center_y).pow(2)
    gaussian = torch.exp(-dist2 / (2.0 * sigma.pow(2) + smooth)) * positive

    guard_kernel = max(1, int(args.thermal_spot_guard))
    if guard_kernel % 2 == 0:
        guard_kernel += 1
    guard = F.max_pool2d(target, guard_kernel, stride=1, padding=guard_kernel // 2).detach()
    soft_target = (gaussian * guard).detach()
    shape_weight = (guard * (1.0 + args.thermal_spot_core_boost * soft_target)).detach()
    shape_loss = (F.binary_cross_entropy(pred, soft_target, reduction="none") * shape_weight).sum()
    shape_loss = shape_loss / shape_weight.sum().clamp_min(1.0)

    core_mass = soft_target.flatten(1).sum(dim=1).clamp_min(smooth).view(batch, 1, 1, 1)
    center_score = (pred * soft_target).sum(dim=(1, 2, 3)).view(batch, 1, 1, 1) / core_mass
    ring = (guard * (1.0 - soft_target)).detach()
    ring_mass = ring.flatten(1).sum(dim=1).clamp_min(smooth).view(batch, 1, 1, 1)
    ring_score = (pred * ring).sum(dim=(1, 2, 3)).view(batch, 1, 1, 1) / ring_mass
    contrast = F.relu(args.thermal_spot_margin + ring_score - center_score)
    contrast = (contrast * positive).sum() / positive.sum().clamp_min(1.0)

    pred_guard_mass = (pred * guard).flatten(1).sum(dim=1)
    soft_mass = soft_target.flatten(1).sum(dim=1).clamp_min(1.0)
    scale = ((pred_guard_mass - soft_mass).pow(2) / soft_mass.pow(2)).view(batch, 1, 1, 1)
    scale = (scale * positive).sum() / positive.sum().clamp_min(1.0)

    return args.thermal_spot_weight * (
        shape_loss
        + args.thermal_spot_contrast_weight * contrast
        + args.thermal_spot_scale_weight * scale
    )


def diffusion_prior_loss(
    pred: torch.Tensor,
    diffusion_prior: torch.Tensor | None,
    conductance: torch.Tensor | None,
    target: torch.Tensor,
    image: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    if diffusion_prior is None or args.diffusion_prior_weight <= 0:
        return pred.new_tensor(0.0)
    if target.shape[-2:] != diffusion_prior.shape[-2:]:
        target = F.interpolate(target, size=diffusion_prior.shape[-2:], mode="nearest")
    if pred.shape[-2:] != diffusion_prior.shape[-2:]:
        pred = F.interpolate(pred, size=diffusion_prior.shape[-2:], mode="bilinear", align_corners=False)
    if image.shape[-2:] != diffusion_prior.shape[-2:]:
        image = F.interpolate(image, size=diffusion_prior.shape[-2:], mode="bilinear", align_corners=False)
    image = image_to_luma(image)
    prior = diffusion_prior.clamp(1e-5, 1.0 - 1e-5)
    pred = pred.clamp(1e-5, 1.0 - 1e-5)

    dilate = max(1, int(args.diffusion_target_dilate))
    if dilate % 2 == 0:
        dilate += 1
    target_guard = F.max_pool2d(target, dilate, stride=1, padding=dilate // 2).detach()
    with torch.no_grad():
        local3 = F.avg_pool2d(image, 3, stride=1, padding=1)
        local15 = F.avg_pool2d(image, 15, stride=1, padding=7)
        hot = F.relu(image - local3)
        hot = hot / (hot.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        hot = hot.clamp(max=6.0) / 6.0
        dog = (local3 - local15).abs()
        dog = dog / (dog.flatten(1).mean(dim=1).view(-1, 1, 1, 1) + 1e-6)
        dog = dog.clamp(max=6.0) / 6.0
        bg_hf = (1.0 - target_guard) * (hot + dog).clamp(max=1.5)

    pos_weight = target_guard * (1.0 + args.diffusion_pos_boost * hot)
    bg_weight = bg_hf * (1.0 + args.diffusion_bg_boost * bg_hf)
    target_term = (F.binary_cross_entropy(prior, target_guard, reduction="none") * (pos_weight + 0.15 * target)).sum()
    target_term = target_term / (pos_weight + 0.15 * target).sum().clamp_min(1.0)
    bg_term = (prior.pow(2) * bg_weight).sum() / bg_weight.sum().clamp_min(1.0)
    align_term = F.mse_loss(torch.logit(pred), torch.logit(prior.detach()))
    loss = target_term + args.diffusion_bg_weight * bg_term + args.diffusion_align_weight * align_term
    if conductance is not None and args.diffusion_conductance_weight > 0:
        if conductance.shape[-2:] != target_guard.shape[-2:]:
            conductance = F.interpolate(conductance, size=target_guard.shape[-2:], mode="bilinear", align_corners=False)
        barrier_target = (target_guard + hot * target_guard).clamp(0.0, 1.0).detach()
        barrier_bg = bg_hf.detach()
        cond_pos = ((1.0 - conductance).pow(2) * barrier_target).sum() / barrier_target.sum().clamp_min(1.0)
        cond_bg = (conductance.pow(2) * barrier_bg).sum() / barrier_bg.sum().clamp_min(1.0)
        loss = loss + args.diffusion_conductance_weight * (cond_pos + args.diffusion_bg_weight * cond_bg)
    return args.diffusion_prior_weight * loss


def single_loss(pred: torch.Tensor, target: torch.Tensor, args: argparse.Namespace, epoch: int) -> torch.Tensor:
    if target.shape[-2:] != pred.shape[-2:]:
        target = F.interpolate(target, size=pred.shape[-2:], mode="nearest")
    pred = pred.clamp(1e-5, 1.0 - 1e-5)
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    with torch.no_grad():
        pos = (target > 0.5).float()
        hard_neg = ((pred > args.hard_neg_threshold) & (target <= 0.5)).float()
        area = pos.flatten(1).sum(dim=1).view(-1, 1, 1, 1)
        small_boost = (1.0 + args.small_weight / torch.sqrt(area + 1.0)).clamp(max=1.0 + args.small_weight)
        weights = 1.0 + args.pos_weight * pos * small_boost + args.hard_neg_weight * hard_neg
    loss = args.bce_weight * (bce * weights).mean()
    loss = loss + args.dice_weight * dice_loss(pred, target)
    loss = loss + args.soft_iou_weight * soft_iou_loss(pred, target)
    loss = loss + args.focal_weight * focal_bce_loss(pred, target, args.focal_alpha, args.focal_gamma)
    if args.sls_iou_weight > 0 and epoch > args.sls_iou_warm_epochs:
        loss = loss + args.sls_iou_weight * sls_iou_loss(pred, target, args.sls_location_weight)
    if args.sdm_iou_weight > 0:
        loss = loss + args.sdm_iou_weight * sdm_iou_loss(pred, target, epoch, args)
    return loss


def total_loss(
    preds,
    target: torch.Tensor,
    image: torch.Tensor,
    args: argparse.Namespace,
    epoch: int,
    teacher_pred: torch.Tensor | None = None,
    pseudo: torch.Tensor | None = None,
    base_pred: torch.Tensor | None = None,
    expert_pred: torch.Tensor | None = None,
    expert_gate_prob: torch.Tensor | None = None,
    tcd_target_gate: torch.Tensor | None = None,
    tcd_clutter_gate: torch.Tensor | None = None,
    diffusion_prior: torch.Tensor | None = None,
    diffusion_conductance: torch.Tensor | None = None,
) -> torch.Tensor:
    pred_list = preds if isinstance(preds, (tuple, list)) else [preds]
    weights = [0.25, 0.35, 0.50, 0.70, 1.0]
    weights = weights[-len(pred_list):]
    base = sum(w * single_loss(pred, target, args, epoch) for w, pred in zip(weights, pred_list)) / sum(weights)
    final_pred = pred_list[-1]
    return (
        base
        + background_hf_loss(final_pred, target, image, args.bg_hf_weight, args.bg_pred_weight, args.target_dilate)
        + teacher_preservation_loss(final_pred, teacher_pred, target, image, args)
        + pseudo_label_loss(final_pred, pseudo, target, image, args)
        + base_anchor_loss(final_pred, base_pred, target, image, args)
        + expert_gate_oracle_loss(base_pred, expert_pred, expert_gate_prob, target, image, args)
        + tcd_gate_loss(tcd_target_gate, tcd_clutter_gate, target, image, args)
        + thermal_spot_shape_loss(final_pred, target, args)
        + diffusion_prior_loss(final_pred, diffusion_prior, diffusion_conductance, target, image, args)
    )


@torch.no_grad()
def evaluate(model: nn.Module, dataset_dir: Path, dataset_name: str, threshold: float, device: torch.device) -> dict[str, float]:
    dataset = TestSetLoader(str(dataset_dir), dataset_name, dataset_name)
    loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
    model.eval()
    metric_iou = mIoU()
    metric_pdfa = PD_FA()
    losses = []
    for image, mask, size, _ in loader:
        image = image.to(device)
        mask = mask.to(device)
        pred = model(image, epoch=10**9)
        if isinstance(pred, (tuple, list)):
            pred = pred[-1]
        pred = pred[:, :, : size[0], : size[1]]
        mask = mask[:, :, : size[0], : size[1]]
        losses.append(float(F.binary_cross_entropy(pred.clamp(1e-5, 1.0 - 1e-5), mask).cpu()))
        metric_iou.update((pred > threshold).cpu(), mask.cpu())
        metric_pdfa.update((pred[0, 0] > threshold).cpu(), mask[0, 0].cpu(), size)
    pix_acc, miou = metric_iou.get()
    pd, fa = metric_pdfa.get()
    return {"loss": float(np.mean(losses)), "pixAcc": float(pix_acc), "mIoU": float(miou), "Pd": float(pd), "Fa": float(fa)}


@torch.no_grad()
def evaluate_official_mshnet(
    model: nn.Module,
    dataset_dir: Path,
    dataset_name: str,
    threshold: float,
    device: torch.device,
    base_size: int,
) -> dict[str, float]:
    dataset_root = dataset_dir / dataset_name
    names = (dataset_root / "img_idx" / f"test_{dataset_name}.txt").read_text().splitlines()
    model.eval()
    metric_iou = mIoU()
    metric_pdfa = PD_FA()
    losses = []
    for name in names:
        image_pil, mask_pil = open_dataset_pair(dataset_root, name, rgb=True)
        width, height = image_pil.size
        image = image_pil.resize((base_size, base_size), Image.BILINEAR)
        image_tensor = normalize_imagenet_rgb(np.array(image, dtype=np.float32))
        image_tensor = torch.from_numpy(np.ascontiguousarray(image_tensor)).float().unsqueeze(0).to(device)
        mask = np.array(mask_pil, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).float().view(1, 1, height, width).to(device)
        pred = model(image_tensor, epoch=10**9)
        if isinstance(pred, (tuple, list)):
            pred = pred[-1]
        pred = F.interpolate(pred, size=(height, width), mode="bilinear", align_corners=False)
        losses.append(float(F.binary_cross_entropy(pred.clamp(1e-5, 1.0 - 1e-5), mask_tensor).cpu()))
        metric_iou.update((pred > threshold).cpu(), mask_tensor.cpu())
        metric_pdfa.update((pred[0, 0] > threshold).cpu(), mask_tensor[0, 0].cpu(), [height, width])
    pix_acc, miou = metric_iou.get()
    pd = float(metric_pdfa.PD) / float(max(metric_pdfa.target, 1))
    fa = float(metric_pdfa.dismatch_pixel) / float(max(metric_pdfa.all_pixel, 1))
    return {"loss": float(np.mean(losses)), "pixAcc": float(pix_acc), "mIoU": float(miou), "Pd": float(pd), "Fa": float(fa)}


def make_scheduler(optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int, min_lr_ratio: float):
    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint(path: Path, model: nn.Module, epoch: int, metrics: dict[str, float], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "state_dict": model.state_dict(), "metrics": metrics, "args": vars(args)}, path)


def load_model_checkpoint(model: nn.Module, checkpoint: Path) -> None:
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if isinstance(state, dict) and "net" in state:
        state = state["net"]
    state = normalize_state_dict(state)
    current = model.state_dict()
    if not any(key in current for key in state):
        prefixed = {f"net.{key}": value for key, value in state.items()}
        if any(key in current for key in prefixed):
            state = prefixed
    compatible = {}
    skipped = []
    for key, value in state.items():
        if key in current and current[key].shape != value.shape:
            skipped.append(key)
            continue
        compatible[key] = value
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if skipped:
        print(f"Resume skipped shape-mismatched keys: {len(skipped)}", flush=True)
    if missing:
        print(f"Resume missing keys: {len(missing)}", flush=True)
    if unexpected:
        print(f"Resume unexpected keys: {len(unexpected)}", flush=True)


def make_teacher_model(args: argparse.Namespace, device: torch.device) -> nn.Module | None:
    if not args.teacher_checkpoint:
        return None
    teacher_base_checkpoint = args.teacher_base_checkpoint or args.base_checkpoint
    teacher = make_model(args.teacher_model, args.sls_iou_warm_epochs, teacher_base_checkpoint).to(device)
    if isinstance(teacher, SCTRefiner):
        teacher.initialize_refiner()
    else:
        teacher.apply(weights_init_kaiming)
    load_model_checkpoint(teacher, Path(args.teacher_checkpoint))
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def run_evaluate(model: nn.Module, dataset_dir: Path, args: argparse.Namespace, device: torch.device) -> dict[str, float]:
    if args.image_preprocess == "official_mshnet":
        return evaluate_official_mshnet(
            model,
            dataset_dir,
            args.dataset_name,
            args.threshold,
            device,
            args.official_base_size,
        )
    return evaluate(model, dataset_dir, args.dataset_name, args.threshold, device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train full-data IRSTD hybrid methods beyond SCTransNet parameter tuning.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--train_datasets", default="")
    parser.add_argument("--target_repeat", type=int, default=1)
    parser.add_argument("--pseudo_dir", default="")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--method_name", required=True)
    parser.add_argument(
        "--model",
        choices=[
            "bgm_basic",
            "mshnet_sls",
            "mshnet3_sls",
            "mshnet_p43_sls",
            "tcf_bgm",
            "pconv_tcf",
            "apconv_tcf",
            "sct_refiner",
            "sct_dhif_refiner",
            "sct_tpc_dhif_refiner",
            "sct_apconv_tpc_refiner",
            "sct_apconv_tpc_unfrozen_refiner",
            "sct_bng_featup_refiner",
            "sct_lca_refiner",
            "sct_tcd_decouple_refiner",
            "sct_tcd_lca_bng_refiner",
            "sct_spectral_anomaly_refiner",
            "sct_prompt_morph_refiner",
            "sct_diffusion_prior_refiner",
            "sct_lpsm_refiner",
            "sct_expert_gate_refiner",
        ],
        required=True,
    )
    parser.add_argument("--base_checkpoint", default="")
    parser.add_argument("--resume_model_checkpoint", default="")
    parser.add_argument("--expert_checkpoint", default="")
    parser.add_argument("--expert_base_checkpoint", default="")
    parser.add_argument(
        "--expert_model",
        choices=[
            "sct_refiner",
            "sct_dhif_refiner",
            "sct_tpc_dhif_refiner",
            "sct_apconv_tpc_refiner",
            "sct_apconv_tpc_unfrozen_refiner",
            "sct_bng_featup_refiner",
            "sct_lca_refiner",
            "sct_tcd_decouple_refiner",
            "sct_tcd_lca_bng_refiner",
            "sct_spectral_anomaly_refiner",
            "sct_prompt_morph_refiner",
            "sct_diffusion_prior_refiner",
            "sct_lpsm_refiner",
        ],
        default="sct_tpc_dhif_refiner",
    )
    parser.add_argument("--teacher_checkpoint", default="")
    parser.add_argument(
        "--teacher_model",
        choices=[
            "sct_refiner",
            "sct_dhif_refiner",
            "sct_tpc_dhif_refiner",
            "sct_apconv_tpc_refiner",
            "sct_apconv_tpc_unfrozen_refiner",
            "sct_bng_featup_refiner",
            "sct_lca_refiner",
            "sct_tcd_decouple_refiner",
            "sct_tcd_lca_bng_refiner",
            "sct_spectral_anomaly_refiner",
            "sct_prompt_morph_refiner",
            "sct_diffusion_prior_refiner",
            "sct_lpsm_refiner",
        ],
        default="sct_refiner",
    )
    parser.add_argument("--teacher_base_checkpoint", default="")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--image_preprocess", choices=["sctransnet", "official_mshnet"], default="sctransnet")
    parser.add_argument("--official_base_size", type=int, default=256)
    parser.add_argument("--official_scale_aug", action="store_true")
    parser.add_argument("--official_blur_prob", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--base_lr_scale", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=["adamw", "adagrad"], default="adamw")
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--min_lr_ratio", type=float, default=0.03)
    parser.add_argument("--pos_prob", type=float, default=0.65)
    parser.add_argument("--noise_std", type=float, default=0.006)
    parser.add_argument("--gamma_prob", type=float, default=0.12)
    parser.add_argument("--intensity_jitter", type=float, default=0.025)
    parser.add_argument("--bgm_prob", type=float, default=0.25)
    parser.add_argument("--bgm_region_size", type=int, default=16)
    parser.add_argument("--bgm_mask_ratio", type=float, default=0.12)
    parser.add_argument("--paired_bgm", action="store_true")
    parser.add_argument("--copy_paste_prob", type=float, default=0.0)
    parser.add_argument("--copy_paste_prob_by_dataset", default="")
    parser.add_argument("--copy_paste_max_targets", type=int, default=1)
    parser.add_argument("--copy_paste_max_targets_by_dataset", default="")
    parser.add_argument("--copy_paste_min_area", type=int, default=1)
    parser.add_argument("--copy_paste_max_area", type=int, default=900)
    parser.add_argument("--copy_paste_scale_min", type=float, default=0.75)
    parser.add_argument("--copy_paste_scale_max", type=float, default=1.35)
    parser.add_argument("--copy_paste_alpha_min", type=float, default=0.75)
    parser.add_argument("--copy_paste_alpha_max", type=float, default=1.0)
    parser.add_argument("--copy_paste_contrast_jitter", type=float, default=0.20)
    parser.add_argument("--copy_paste_guard", type=int, default=9)
    parser.add_argument("--copy_paste_pseudo_value", type=float, default=1.0)
    parser.add_argument("--bce_weight", type=float, default=1.0)
    parser.add_argument("--dice_weight", type=float, default=0.35)
    parser.add_argument("--soft_iou_weight", type=float, default=0.0)
    parser.add_argument("--focal_weight", type=float, default=0.04)
    parser.add_argument("--small_weight", type=float, default=0.55)
    parser.add_argument("--pos_weight", type=float, default=0.75)
    parser.add_argument("--hard_neg_weight", type=float, default=2.6)
    parser.add_argument("--hard_neg_threshold", type=float, default=0.20)
    parser.add_argument("--focal_alpha", type=float, default=0.58)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--sls_iou_weight", type=float, default=0.12)
    parser.add_argument("--sls_iou_warm_epochs", type=int, default=5)
    parser.add_argument("--sls_location_weight", type=float, default=0.25)
    parser.add_argument("--sdm_iou_weight", type=float, default=0.0)
    parser.add_argument("--sdm_warm_epochs", type=int, default=5)
    parser.add_argument("--sdm_delta", type=float, default=0.5)
    parser.add_argument("--sdm_with_distance", action="store_true")
    parser.add_argument("--sdm_static", dest="sdm_dynamic", action="store_false")
    parser.set_defaults(sdm_dynamic=True)
    parser.add_argument("--bg_hf_weight", type=float, default=0.22)
    parser.add_argument("--bg_pred_weight", type=float, default=0.06)
    parser.add_argument("--target_dilate", type=int, default=9)
    parser.add_argument("--teacher_weight", type=float, default=0.0)
    parser.add_argument("--teacher_bg_weight", type=float, default=1.0)
    parser.add_argument("--teacher_pos_weight", type=float, default=0.35)
    parser.add_argument("--teacher_miss_weight", type=float, default=0.25)
    parser.add_argument("--teacher_logit_weight", type=float, default=0.0)
    parser.add_argument("--teacher_bg_threshold", type=float, default=0.08)
    parser.add_argument("--teacher_pos_threshold", type=float, default=0.35)
    parser.add_argument("--teacher_margin", type=float, default=0.015)
    parser.add_argument("--teacher_hf_boost", type=float, default=1.0)
    parser.add_argument("--teacher_target_dilate", type=int, default=11)
    parser.add_argument("--pseudo_weight", type=float, default=0.0)
    parser.add_argument("--pseudo_bce_weight", type=float, default=1.0)
    parser.add_argument("--pseudo_logit_weight", type=float, default=0.05)
    parser.add_argument("--pseudo_iou_weight", type=float, default=0.25)
    parser.add_argument("--pseudo_pos_weight", type=float, default=1.5)
    parser.add_argument("--pseudo_bg_weight", type=float, default=0.8)
    parser.add_argument("--pseudo_pos_threshold", type=float, default=0.35)
    parser.add_argument("--pseudo_bg_threshold", type=float, default=0.06)
    parser.add_argument("--pseudo_hf_boost", type=float, default=0.8)
    parser.add_argument("--pseudo_target_dilate", type=int, default=9)
    parser.add_argument("--pseudo_rim_weight", type=float, default=0.0)
    parser.add_argument("--pseudo_rim_low", type=float, default=0.12)
    parser.add_argument("--pseudo_rim_high", type=float, default=0.45)
    parser.add_argument("--pseudo_rim_target", type=float, default=0.32)
    parser.add_argument("--base_anchor_weight", type=float, default=0.0)
    parser.add_argument("--base_anchor_bg_weight", type=float, default=1.0)
    parser.add_argument("--base_anchor_pos_weight", type=float, default=0.45)
    parser.add_argument("--base_anchor_logit_weight", type=float, default=0.02)
    parser.add_argument("--base_anchor_bg_threshold", type=float, default=0.08)
    parser.add_argument("--base_anchor_pos_threshold", type=float, default=0.35)
    parser.add_argument("--base_anchor_margin", type=float, default=0.006)
    parser.add_argument("--base_anchor_hf_boost", type=float, default=1.0)
    parser.add_argument("--base_anchor_target_dilate", type=int, default=11)
    parser.add_argument("--expert_gate_oracle_weight", type=float, default=0.0)
    parser.add_argument("--expert_gate_sparsity_weight", type=float, default=0.0)
    parser.add_argument("--expert_gate_oracle_margin", type=float, default=0.020)
    parser.add_argument("--expert_gate_min_disagree", type=float, default=0.020)
    parser.add_argument("--expert_gate_target_dilate", type=int, default=7)
    parser.add_argument("--expert_gate_pos_boost", type=float, default=2.0)
    parser.add_argument("--expert_gate_disagree_boost", type=float, default=1.4)
    parser.add_argument("--expert_gate_hf_boost", type=float, default=0.8)
    parser.add_argument("--tcd_gate_weight", type=float, default=0.0)
    parser.add_argument("--tcd_target_gate_weight", type=float, default=1.0)
    parser.add_argument("--tcd_clutter_gate_weight", type=float, default=0.35)
    parser.add_argument("--tcd_anti_target_weight", type=float, default=0.65)
    parser.add_argument("--tcd_target_hf_boost", type=float, default=1.0)
    parser.add_argument("--tcd_target_dilate", type=int, default=9)
    parser.add_argument("--thermal_spot_weight", type=float, default=0.0)
    parser.add_argument("--thermal_spot_sigma_scale", type=float, default=0.75)
    parser.add_argument("--thermal_spot_sigma_min", type=float, default=1.1)
    parser.add_argument("--thermal_spot_sigma_max", type=float, default=4.5)
    parser.add_argument("--thermal_spot_guard", type=int, default=11)
    parser.add_argument("--thermal_spot_core_boost", type=float, default=1.5)
    parser.add_argument("--thermal_spot_margin", type=float, default=0.10)
    parser.add_argument("--thermal_spot_contrast_weight", type=float, default=0.25)
    parser.add_argument("--thermal_spot_scale_weight", type=float, default=0.08)
    parser.add_argument("--diffusion_prior_weight", type=float, default=0.0)
    parser.add_argument("--diffusion_bg_weight", type=float, default=0.35)
    parser.add_argument("--diffusion_align_weight", type=float, default=0.04)
    parser.add_argument("--diffusion_conductance_weight", type=float, default=0.0)
    parser.add_argument("--diffusion_pos_boost", type=float, default=0.8)
    parser.add_argument("--diffusion_bg_boost", type=float, default=0.7)
    parser.add_argument("--diffusion_target_dilate", type=int, default=9)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--begin_test", type=int, default=20)
    parser.add_argument("--every_test", type=int, default=5)
    parser.add_argument("--every_print", type=int, default=10)
    parser.add_argument("--save_final", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--eval_before_train", action="store_true")
    args = parser.parse_args()

    seed_pytorch(args.seed)
    torch.backends.cudnn.benchmark = True
    dataset_dir = Path(args.dataset_dir)
    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints" / args.dataset_name
    log_dir = run_dir / "tensorboard" / args.dataset_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    text_log = run_dir / f"{args.dataset_name}_{args.method_name}.log"

    train_dataset_names = parse_dataset_list(args.train_datasets, args.dataset_name)
    pseudo_dir = Path(args.pseudo_dir) if args.pseudo_dir else None
    copy_paste_prob_by_dataset = parse_dataset_float_overrides(args.copy_paste_prob_by_dataset)
    copy_paste_max_targets_by_dataset = parse_dataset_int_overrides(args.copy_paste_max_targets_by_dataset)
    train_sets: list[Dataset] = []
    train_set_sizes: list[tuple[str, int, int]] = []
    for train_dataset_name in train_dataset_names:
        repeat = max(1, args.target_repeat if train_dataset_name == args.dataset_name else 1)
        if args.image_preprocess == "official_mshnet":
            dataset = OfficialMSHNetTrainSet(
                dataset_dir=dataset_dir,
                dataset_name=train_dataset_name,
                patch_size=args.patch_size,
                pos_prob=args.pos_prob,
                base_size=args.official_base_size,
                scale_aug=args.official_scale_aug,
                blur_prob=args.official_blur_prob,
            )
        else:
            dataset_copy_paste_prob = copy_paste_prob_by_dataset.get(train_dataset_name, args.copy_paste_prob)
            dataset_copy_paste_max_targets = copy_paste_max_targets_by_dataset.get(
                train_dataset_name,
                args.copy_paste_max_targets,
            )
            dataset = FullIRSTDTrainSet(
                dataset_dir=dataset_dir,
                dataset_name=train_dataset_name,
                patch_size=args.patch_size,
                pos_prob=args.pos_prob,
                noise_std=args.noise_std,
                gamma_prob=args.gamma_prob,
                intensity_jitter=args.intensity_jitter,
                bgm_prob=args.bgm_prob,
                bgm_region_size=args.bgm_region_size,
                bgm_mask_ratio=args.bgm_mask_ratio,
                paired_bgm=args.paired_bgm,
                pseudo_dir=pseudo_dir,
                copy_paste_prob=dataset_copy_paste_prob,
                copy_paste_max_targets=dataset_copy_paste_max_targets,
                copy_paste_min_area=args.copy_paste_min_area,
                copy_paste_max_area=args.copy_paste_max_area,
                copy_paste_scale_min=args.copy_paste_scale_min,
                copy_paste_scale_max=args.copy_paste_scale_max,
                copy_paste_alpha_min=args.copy_paste_alpha_min,
                copy_paste_alpha_max=args.copy_paste_alpha_max,
                copy_paste_contrast_jitter=args.copy_paste_contrast_jitter,
                copy_paste_guard=args.copy_paste_guard,
                copy_paste_pseudo_value=args.copy_paste_pseudo_value,
            )
        train_sets.extend([dataset] * repeat)
        train_set_sizes.append((train_dataset_name, len(dataset), repeat))
    train_set: Dataset
    if len(train_sets) == 1:
        train_set = train_sets[0]
    else:
        train_set = ConcatDataset(train_sets)
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.threads,
        pin_memory=True,
        drop_last=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(args.model, args.sls_iou_warm_epochs, args.base_checkpoint).to(device)
    if isinstance(model, SCTRefiner):
        model.initialize_refiner()
    elif isinstance(model, SCTExpertGateRefiner):
        model.initialize_refiner()
    else:
        model.apply(weights_init_kaiming)
    if isinstance(model, SCTExpertGateRefiner):
        if not args.resume_model_checkpoint:
            raise ValueError("--resume_model_checkpoint is required as primary checkpoint for sct_expert_gate_refiner")
        model.load_primary_checkpoint(Path(args.resume_model_checkpoint))
        model.load_expert_checkpoint(
            args.expert_model,
            args.expert_base_checkpoint or args.base_checkpoint,
            args.expert_checkpoint,
        )
    elif args.resume_model_checkpoint:
        load_model_checkpoint(model, Path(args.resume_model_checkpoint))
    if isinstance(model, SCTLocalContrastAnomalyRefiner):
        model.freeze_inherited_refiner()
    if isinstance(model, SCTTargetClutterDecoupleRefiner):
        model.freeze_inherited_refiner()
    if isinstance(model, SCTSpectrumAnomalyRefiner):
        model.freeze_inherited_refiner()
    if isinstance(model, SCTPromptMorphRefiner):
        model.freeze_inherited_refiner()
    if isinstance(model, SCTDiffusionPriorRefiner):
        model.freeze_inherited_refiner()
    if isinstance(model, SCTLPSMRefiner):
        model.freeze_inherited_refiner()
    teacher_model = make_teacher_model(args, device)

    if args.eval_only:
        metrics = run_evaluate(model, dataset_dir, args, device)
        print(
            f"eval_only pixAcc={metrics['pixAcc']:.6f} mIoU={metrics['mIoU']:.6f} "
            f"Pd={metrics['Pd']:.6f} Fa={metrics['Fa']:.10f}",
            flush=True,
        )
        return

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found.")
    optimizer_params = trainable_params
    if args.base_lr_scale != 1.0:
        base_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(("base.", "adapter.")):
                base_params.append(param)
            else:
                other_params.append(param)
        if base_params and other_params:
            optimizer_params = [
                {"params": other_params, "lr": args.lr},
                {"params": base_params, "lr": args.lr * args.base_lr_scale},
            ]
    if args.optimizer == "adagrad":
        optimizer = torch.optim.Adagrad(optimizer_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs, args.min_lr_ratio)
    writer = SummaryWriter(str(log_dir))

    best_miou = -1.0
    best_epoch = -1
    loss_window: list[float] = []

    with text_log.open("w") as log:
        log.write(
            f"method={args.method_name} model={args.model} dataset={args.dataset_name} "
            f"train_datasets={','.join(train_dataset_names)} image_preprocess={args.image_preprocess} "
            f"train_set_sizes={train_set_sizes} "
            f"train_size={len(train_set)} args={vars(args)}\n"
        )
        print(f"{args.dataset_name}\t{args.method_name}\t{args.model}", flush=True)
        print(
            f"train_datasets={','.join(train_dataset_names)} train_size={len(train_set)} "
            f"target_repeat={args.target_repeat} batch_size={args.batch_size} run_dir={run_dir}",
            flush=True,
        )
        if args.eval_before_train:
            metrics = run_evaluate(model, dataset_dir, args, device)
            best_miou = metrics["mIoU"]
            best_epoch = 0
            save_checkpoint(ckpt_dir / "best_mIoU.pth.tar", model, 0, metrics, args)
            msg = (
                f"best epoch=0 pixAcc={metrics['pixAcc']:.6f} "
                f"mIoU={metrics['mIoU']:.6f} Pd={metrics['Pd']:.6f} Fa={metrics['Fa']:.10f}"
            )
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
        for epoch in range(1, args.epochs + 1):
            model.train()
            for batch in loader:
                pseudo = None
                if len(batch) == 3:
                    image, mask, pseudo = batch
                else:
                    image, mask = batch
                if image.ndim == 5:
                    bsz, views, channels, height, width = image.shape
                    image = image.view(bsz * views, channels, height, width)
                    mask = mask.view(bsz * views, mask.shape[2], height, width)
                    if pseudo is not None:
                        pseudo = pseudo.view(bsz * views, pseudo.shape[2], height, width)
                elif image.ndim == 4 and mask.ndim == 4 and image.shape[1] == mask.shape[1] and image.shape[1] != 1:
                    bsz, views, height, width = image.shape
                    image = image.view(bsz * views, 1, height, width)
                    mask = mask.view(bsz * views, 1, height, width)
                    if pseudo is not None:
                        pseudo = pseudo.view(bsz * views, 1, height, width)
                image = image.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                if pseudo is not None:
                    pseudo = pseudo.to(device, non_blocking=True)
                preds = model(image, epoch=epoch)
                base_pred = getattr(model, "_last_base_pred", None)
                expert_pred = getattr(model, "_last_expert_pred", None)
                expert_gate_prob = getattr(model, "_last_expert_gate_prob", None)
                tcd_target_gate = getattr(model, "_last_tcd_target_gate", None)
                tcd_clutter_gate = getattr(model, "_last_tcd_clutter_gate", None)
                diffusion_prior = getattr(model, "_last_diffusion_prior", None)
                diffusion_conductance = getattr(model, "_last_diffusion_conductance", None)
                teacher_pred = None
                if teacher_model is not None:
                    with torch.no_grad():
                        teacher_pred = teacher_model(image, epoch=10**9)
                        if isinstance(teacher_pred, (tuple, list)):
                            teacher_pred = teacher_pred[-1]
                loss = total_loss(
                    preds,
                    mask,
                    image,
                    args,
                    epoch,
                    teacher_pred=teacher_pred,
                    pseudo=pseudo,
                    base_pred=base_pred,
                    expert_pred=expert_pred,
                    expert_gate_prob=expert_gate_prob,
                    tcd_target_gate=tcd_target_gate,
                    tcd_clutter_gate=tcd_clutter_gate,
                    diffusion_prior=diffusion_prior,
                    diffusion_conductance=diffusion_conductance,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_window.append(float(loss.detach().cpu()))
            scheduler.step()

            if epoch % args.every_print == 0:
                avg_loss = float(np.mean(loss_window)) if loss_window else 0.0
                loss_window = []
                lr_now = optimizer.param_groups[0]["lr"]
                line = f"{time.strftime('%b %d %H:%M:%S')} Epoch---{epoch}, total_loss---{avg_loss:.6f}, lr---{lr_now:.8f},"
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
                writer.add_scalar("loss", avg_loss, epoch)
                writer.add_scalar("lr", lr_now, epoch)

            if epoch >= args.begin_test and epoch % args.every_test == 0:
                metrics = run_evaluate(model, dataset_dir, args, device)
                writer.add_scalar("mIoU", metrics["mIoU"], epoch)
                writer.add_scalar("Pd", metrics["Pd"], epoch)
                writer.add_scalar("Fa", metrics["Fa"], epoch)
                eval_msg = (
                    f"eval epoch={epoch} pixAcc={metrics['pixAcc']:.6f} "
                    f"mIoU={metrics['mIoU']:.6f} Pd={metrics['Pd']:.6f} Fa={metrics['Fa']:.10f}"
                )
                print(eval_msg, flush=True)
                log.write(eval_msg + "\n")
                log.flush()
                if metrics["mIoU"] > best_miou:
                    best_miou = metrics["mIoU"]
                    best_epoch = epoch
                    save_checkpoint(ckpt_dir / "best_mIoU.pth.tar", model, epoch, metrics, args)
                    msg = (
                        f"best epoch={epoch} pixAcc={metrics['pixAcc']:.6f} "
                        f"mIoU={metrics['mIoU']:.6f} Pd={metrics['Pd']:.6f} Fa={metrics['Fa']:.10f}"
                    )
                    print(msg, flush=True)
                    log.write(msg + "\n")
                    log.flush()

        final_metrics = run_evaluate(model, dataset_dir, args, device)
        if args.save_final:
            save_checkpoint(ckpt_dir / "final.pth.tar", model, args.epochs, final_metrics, args)
        log.write(
            "final "
            f"best_epoch={best_epoch} best_mIoU={best_miou * 100:.2f} "
            f"last_mIoU={final_metrics['mIoU'] * 100:.2f} "
            f"last_Pd={final_metrics['Pd'] * 100:.2f} last_Fa={final_metrics['Fa'] * 1e6:.2f}\n"
        )
        writer.close()


if __name__ == "__main__":
    main()
