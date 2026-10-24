#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))
sys.path.insert(0, str(ROOT / "scripts"))

from metrics import PD_FA, mIoU  # noqa: E402
from train_hybrid_methods import load_model_checkpoint, make_model  # noqa: E402
from utils import Normalized, PadImg, get_img_norm_cfg  # noqa: E402

MSHNET_MODEL = ROOT / "external" / "MSHNet" / "model" / "MSHNet.py"
APCONV_MODEL = ROOT / "external" / "PConv-SDloss-Data" / "model" / "APConv.py"
BGM_MODEL = ROOT / "external" / "BGM" / "model" / "BasicUNet" / "model_BasicUNet.py"
RPCANET_ROOT = ROOT / "external" / "RPCANet_hf"


def parse_model_spec(spec: str) -> tuple[str, str, str, str]:
    parts = spec.split("|")
    if len(parts) != 4:
        raise ValueError("Model spec must be name|model|base_checkpoint|checkpoint")
    return parts[0], parts[1], parts[2], parts[3]


def load_symbol(path: Path, module_name: str, symbol: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {symbol} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, symbol)


MSHNetCore = load_symbol(MSHNET_MODEL, "mshnet_component_fusion", "MSHNet")
MSHNetChannelAttention = load_symbol(MSHNET_MODEL, "mshnet_component_fusion_ca", "ChannelAttention")
MSHNetSpatialAttention = load_symbol(MSHNET_MODEL, "mshnet_component_fusion_sa", "SpatialAttention")
OfficialPConv = load_symbol(APCONV_MODEL, "pconv_component_fusion", "PConv")
BGMBasicUNetPlus = load_symbol(BGM_MODEL, "bgm_component_fusion", "BasicUNet_plus")


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
        return self.relu(out + residual)


def parse_official_pconv_spec(spec: str) -> dict[str, object]:
    parts = spec.split("|")
    if len(parts) != 7:
        raise ValueError("Official PConv spec must be name|variant|weight|preprocess|threshold|warm|base_size")
    name, variant, weight, preprocess, threshold, warm, base_size = parts
    if variant not in {"mshnet", "p43"}:
        raise ValueError(f"Unsupported official PConv variant: {variant}")
    if preprocess not in {"sctransnet", "official_mshnet"}:
        raise ValueError(f"Unsupported official PConv preprocess: {preprocess}")
    return {
        "name": name,
        "variant": variant,
        "weight": Path(weight),
        "preprocess": preprocess,
        "threshold": float(threshold),
        "warm": warm.strip().lower() in {"1", "true", "yes", "warm"},
        "base_size": int(base_size),
    }


def load_official_pconv_models(specs: list[dict[str, object]], device: torch.device) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for spec in specs:
        if spec["variant"] == "p43":
            model = MSHNetCore(3, block=P43FirstBlockResNet).to(device)
        else:
            model = MSHNetCore(3).to(device)
        load_model_checkpoint(model, Path(spec["weight"]))
        model.eval()
        out.append({**spec, "model": model})
    return out


def parse_official_bgm_spec(spec: str) -> dict[str, object]:
    parts = spec.split("|")
    if len(parts) != 4:
        raise ValueError("Official BGM spec must be name|norm_dataset|weight|threshold")
    name, norm_dataset, weight, threshold = parts
    return {
        "name": name,
        "norm_dataset": norm_dataset,
        "weight": Path(weight),
        "threshold": float(threshold),
    }


def normalize_bgm_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("model."):
            key = key[len("model.") :]
        normalized[key] = value
    return normalized


def load_official_bgm_models(specs: list[dict[str, object]], device: torch.device) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for spec in specs:
        model = BGMBasicUNetPlus().to(device)
        ckpt = torch.load(Path(spec["weight"]), map_location="cpu")
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(normalize_bgm_state_dict(state), strict=True)
        model.eval()
        out.append({**spec, "model": model})
    return out


def parse_rpcanet_spec(spec: str) -> dict[str, object]:
    parts = spec.split("|")
    if len(parts) != 5:
        raise ValueError("RPCANet spec must be name|model|weight|threshold|base_size")
    name, model_name, weight, threshold, base_size = parts
    if model_name not in {"rpcanet", "rpcanet_pp", "rpcanet_pp_s3", "rpcanet_pp_s9"}:
        raise ValueError(f"Unsupported RPCANet model: {model_name}")
    return {
        "name": name,
        "model_name": model_name,
        "weight": Path(weight),
        "threshold": float(threshold),
        "base_size": int(base_size),
    }


def load_rpcanet_models(specs: list[dict[str, object]], device: torch.device) -> list[dict[str, object]]:
    if specs:
        sys.path.insert(0, str(RPCANET_ROOT))
        from models import get_model  # type: ignore
    out: list[dict[str, object]] = []
    for spec in specs:
        model = get_model(str(spec["model_name"])).to(device).eval()
        state = torch.load(Path(spec["weight"]), map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        out.append({**spec, "model": model})
    return out


class FullSplitSet(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        train_dataset_name: str,
        dataset_name: str,
        split: str,
        max_samples: int = 0,
    ) -> None:
        self.dataset_dir = dataset_dir / dataset_name
        self.dataset_name = dataset_name
        self.split = split
        self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, str(dataset_dir))
        index = self.dataset_dir / "img_idx" / f"{split}_{dataset_name}.txt"
        self.names = index.read_text(encoding="utf-8").splitlines()
        if max_samples > 0:
            self.names = self.names[:max_samples]

    def __len__(self) -> int:
        return len(self.names)

    def _open_pair(self, name: str) -> tuple[Image.Image, Image.Image]:
        for ext in (".png", ".bmp", ".jpg"):
            image_path = self.dataset_dir / "images" / f"{name}{ext}"
            mask_path = self.dataset_dir / "masks" / f"{name}{ext}"
            if image_path.exists() and mask_path.exists():
                return Image.open(image_path).convert("I"), Image.open(mask_path)
        raise FileNotFoundError(name)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        image_pil, mask_pil = self._open_pair(name)
        image = Normalized(np.array(image_pil, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask_pil, dtype=np.float32) / 255.0
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        height, width = image.shape
        image = PadImg(image)
        mask = PadImg(mask)
        image = torch.from_numpy(np.ascontiguousarray(image[np.newaxis, :])).float()
        mask = torch.from_numpy(np.ascontiguousarray(mask[np.newaxis, :])).float()
        return image, mask, [height, width], name


@torch.no_grad()
def build_models(specs: list[tuple[str, str, str, str]], device: torch.device) -> list[tuple[str, torch.nn.Module]]:
    models: list[tuple[str, torch.nn.Module]] = []
    for name, model_name, base_checkpoint, checkpoint in specs:
        model = make_model(model_name, warm_epoch=1, base_checkpoint=base_checkpoint).to(device)
        if hasattr(model, "initialize_refiner"):
            model.initialize_refiner()
        load_model_checkpoint(model, Path(checkpoint))
        if hasattr(model, "freeze_inherited_refiner"):
            model.freeze_inherited_refiner()
        model.eval()
        models.append((name, model))
    return models


def open_raw_pair(dataset_dir: Path, dataset_name: str, name: str) -> tuple[Image.Image, Image.Image]:
    root = dataset_dir / dataset_name
    for ext in (".png", ".bmp", ".jpg"):
        image_path = root / "images" / f"{name}{ext}"
        mask_path = root / "masks" / f"{name}{ext}"
        if image_path.exists() and mask_path.exists():
            return Image.open(image_path).convert("RGB"), Image.open(mask_path)
    raise FileNotFoundError(name)


def official_pconv_pred_map(
    official: dict[str, object],
    dataset_dir: Path,
    dataset_name: str,
    name: str,
    sc_image: torch.Tensor,
    height: int,
    width: int,
    device: torch.device,
) -> np.ndarray:
    model = official["model"]
    warm = bool(official["warm"])
    if official["preprocess"] == "official_mshnet":
        raw_image, _mask = open_raw_pair(dataset_dir, dataset_name, name)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image = raw_image.resize((int(official["base_size"]), int(official["base_size"])), Image.BILINEAR)
        image_tensor = transform(image).unsqueeze(0).to(device)
        _masks, logits = model(image_tensor, warm)
        pred = torch.sigmoid(logits)
        pred = F.interpolate(pred, size=(height, width), mode="bilinear", align_corners=False)
    else:
        image_tensor = sc_image.repeat(1, 3, 1, 1)
        _masks, logits = model(image_tensor, warm)
        pred = torch.sigmoid(logits)[:, :, :height, :width]
    prob = pred[0, 0].detach().cpu().numpy().astype(np.float32)
    return (prob > float(official["threshold"])).astype(np.float32)


def official_bgm_pred_map(
    official: dict[str, object],
    dataset_dir: Path,
    dataset_name: str,
    sample_name: str,
    height: int,
    width: int,
    device: torch.device,
) -> np.ndarray:
    dataset_path = dataset_dir / dataset_name
    raw_image = None
    for ext in (".png", ".bmp", ".jpg"):
        image_path = dataset_path / "images" / f"{sample_name}{ext}"
        if image_path.exists():
            raw_image = Image.open(image_path).convert("I")
            break
    if raw_image is None:
        raise FileNotFoundError(sample_name)
    cfg = get_img_norm_cfg(str(official["norm_dataset"]), str(dataset_dir))
    image_np = Normalized(np.array(raw_image, dtype=np.float32), cfg)
    image_np = PadImg(image_np)
    image_tensor = torch.from_numpy(np.ascontiguousarray(image_np[None, None])).float().to(device)
    model = official["model"]
    pred = model(image_tensor)
    prob = pred[0, 0, :height, :width].detach().cpu().numpy().astype(np.float32)
    return (prob > float(official["threshold"])).astype(np.float32)


def rpcanet_pred_map(
    official: dict[str, object],
    dataset_dir: Path,
    dataset_name: str,
    sample_name: str,
    height: int,
    width: int,
    device: torch.device,
) -> np.ndarray:
    dataset_path = dataset_dir / dataset_name
    raw_image = None
    for ext in (".png", ".bmp", ".jpg"):
        image_path = dataset_path / "images" / f"{sample_name}{ext}"
        if image_path.exists():
            raw_image = Image.open(image_path).convert("L")
            break
    if raw_image is None:
        raise FileNotFoundError(sample_name)
    image_np = np.array(raw_image, dtype=np.float32) / 255.0
    resized = cv2.resize(
        image_np,
        (int(official["base_size"]), int(official["base_size"])),
        interpolation=cv2.INTER_LINEAR,
    )
    image_tensor = torch.from_numpy(resized[None, None].astype(np.float32)).to(device)
    pred = official["model"](image_tensor)
    if isinstance(pred, (tuple, list)):
        pred = pred[-1]
    prob = torch.sigmoid(pred)
    prob = F.interpolate(prob, size=(height, width), mode="bilinear", align_corners=False)
    prob_np = prob[0, 0].detach().cpu().numpy().astype(np.float32)
    return (prob_np > float(official["threshold"])).astype(np.float32)


@torch.no_grad()
def collect_samples(
    models: list[tuple[str, torch.nn.Module]],
    official_pconv_models: list[dict[str, object]],
    official_bgm_models: list[dict[str, object]],
    rpcanet_models: list[dict[str, object]],
    dataset_dir: Path,
    dataset_name: str,
    split: str,
    max_samples: int,
    device: torch.device,
    handcrafted: bool = False,
    handcrafted_full: bool = False,
    handcrafted_topk: int = 32,
    contrast_candidates: bool = False,
    contrast_topk: int = 24,
    contrast_threshold: float = 0.78,
    background_model: dict[str, np.ndarray] | None = None,
    background_as_candidate: bool = False,
    background_topk: int = 16,
    background_threshold: float = 0.88,
    prototype_model: dict[str, np.ndarray] | None = None,
    prototype_topk: int = 24,
    prototype_threshold: float = 0.72,
    prototype_teacher_suppress: float = 0.35,
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
        sample_name = name[0]
        for official in official_pconv_models:
            preds[str(official["name"])] = official_pconv_pred_map(
                official,
                dataset_dir,
                dataset_name,
                sample_name,
                image,
                height,
                width,
                device,
            )
        for official in official_bgm_models:
            preds[str(official["name"])] = official_bgm_pred_map(
                official,
                dataset_dir,
                dataset_name,
                sample_name,
                height,
                width,
                device,
            )
        for official in rpcanet_models:
            preds[str(official["name"])] = rpcanet_pred_map(
                official,
                dataset_dir,
                dataset_name,
                sample_name,
                height,
                width,
                device,
            )
        img_np = image[0, 0, :height, :width].detach().cpu().numpy().astype(np.float32)
        if handcrafted:
            hand_maps = handcrafted_candidate_maps(img_np, topk=handcrafted_topk)
            if handcrafted_full:
                preds.update(hand_maps)
            else:
                preds["hand_sparse"] = hand_maps["hand_sparse"]
        feature_maps: dict[str, np.ndarray] = {}
        if contrast_candidates:
            contrast_maps = contrast_candidate_maps(
                img_np,
                topk=contrast_topk,
                threshold=contrast_threshold,
            )
            preds["contrast_sparse"] = contrast_maps["contrast_sparse"]
            feature_maps["contrast_score"] = contrast_maps["contrast_score"]
        if background_model is not None:
            bg_score = background_anomaly_score_map(img_np, background_model)
            feature_maps["bg_normality"] = bg_score
        if background_model is not None and background_as_candidate:
            preds["bg_anomaly"] = background_anomaly_candidate_map(
                img_np,
                background_model,
                score=feature_maps["bg_normality"],
                topk=background_topk,
                threshold=background_threshold,
            )
        if prototype_model is not None:
            proto_score = prototype_score_map(img_np, prototype_model)
            feature_maps["target_proto"] = proto_score
            teacher_max = np.stack(list(preds.values()), axis=0).max(axis=0) if preds else None
            preds["target_proto"] = prototype_candidate_map(
                proto_score,
                image=img_np,
                teacher_max=teacher_max,
                topk=prototype_topk,
                threshold=prototype_threshold,
                teacher_suppress=prototype_teacher_suppress,
            )
        mask_np = mask[0, 0, :height, :width].numpy().astype(np.float32)
        samples.append(
            {
                "name": name[0],
                "image": img_np,
                "mask": mask_np,
                "size": [height, width],
                "preds": preds,
                "feature_maps": feature_maps,
            }
        )
        if (idx + 1) % 100 == 0:
            print(f"collected {split} {idx + 1}/{len(dataset)}", flush=True)
    return samples


def collect_fit_samples_multi(
    models: list[tuple[str, torch.nn.Module]],
    official_pconv_models: list[dict[str, object]],
    official_bgm_models: list[dict[str, object]],
    rpcanet_models: list[dict[str, object]],
    dataset_dir: Path,
    fit_datasets: list[str],
    split: str,
    max_samples_per_dataset: int,
    device: torch.device,
    handcrafted: bool = False,
    handcrafted_full: bool = False,
    handcrafted_topk: int = 32,
    contrast_candidates: bool = False,
    contrast_topk: int = 24,
    contrast_threshold: float = 0.78,
    background_model: dict[str, np.ndarray] | None = None,
    background_as_candidate: bool = False,
    background_topk: int = 16,
    background_threshold: float = 0.88,
    prototype_model: dict[str, np.ndarray] | None = None,
    prototype_topk: int = 24,
    prototype_threshold: float = 0.72,
    prototype_teacher_suppress: float = 0.35,
) -> list[dict[str, object]]:
    samples: list[dict[str, object]] = []
    for fit_dataset in fit_datasets:
        samples.extend(
            collect_samples(
                models,
                official_pconv_models,
                official_bgm_models,
                rpcanet_models,
                dataset_dir,
                fit_dataset,
                split,
                max_samples_per_dataset,
                device,
                handcrafted=handcrafted,
                handcrafted_full=handcrafted_full,
                handcrafted_topk=handcrafted_topk,
                contrast_candidates=contrast_candidates,
                contrast_topk=contrast_topk,
                contrast_threshold=contrast_threshold,
                background_model=background_model,
                background_as_candidate=background_as_candidate,
                background_topk=background_topk,
                background_threshold=background_threshold,
                prototype_model=prototype_model,
                prototype_topk=prototype_topk,
                prototype_threshold=prototype_threshold,
                prototype_teacher_suppress=prototype_teacher_suppress,
            )
        )
    return samples


def local_stats(image: np.ndarray, label_mask: np.ndarray, bbox: tuple[int, int, int, int], pad: int = 5) -> tuple[float, float, float]:
    y0, x0, y1, x1 = bbox
    yy0 = max(0, y0 - pad)
    xx0 = max(0, x0 - pad)
    yy1 = min(image.shape[0], y1 + pad)
    xx1 = min(image.shape[1], x1 + pad)
    roi = image[yy0:yy1, xx0:xx1]
    comp = label_mask[y0:y1, x0:x1]
    fg = image[y0:y1, x0:x1][comp]
    ring_mask = np.ones((yy1 - yy0, xx1 - xx0), dtype=bool)
    ring_mask[y0 - yy0 : y1 - yy0, x0 - xx0 : x1 - xx0] = False
    ring = roi[ring_mask]
    fg_mean = float(fg.mean()) if fg.size else 0.0
    ring_mean = float(ring.mean()) if ring.size else 0.0
    ring_std = float(ring.std()) if ring.size else 0.0
    return fg_mean, fg_mean - ring_mean, (fg_mean - ring_mean) / (ring_std + 1e-6)


def component_features(
    sample: dict[str, object],
    names: list[str],
    threshold: float,
    min_seed_area: int,
    handcrafted_seed_threshold: float,
    max_components: int = 0,
) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    preds = sample["preds"]
    image = sample["image"]
    mask = sample["mask"] > 0
    stack = np.stack([preds[name] for name in names], axis=0)
    seed_maps = []
    for name in names:
        seed_th = handcrafted_seed_threshold if name.startswith("hand_") else threshold
        seed_maps.append(preds[name] > seed_th)
    union = np.stack(seed_maps, axis=0).any(axis=0)
    num_labels, label_img, stats, centroids = cv2.connectedComponentsWithStats(union.astype(np.uint8), 8)
    rows: list[dict[str, object]] = []
    features: list[list[float]] = []
    labels: list[int] = []
    priorities: list[float] = []
    lap = cv2.Laplacian(image, cv2.CV_32F, ksize=3)
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_seed_area:
            continue
        x0 = int(stats[label_id, cv2.CC_STAT_LEFT])
        y0 = int(stats[label_id, cv2.CC_STAT_TOP])
        width_i = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height_i = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        x1 = x0 + width_i
        y1 = y0 + height_i
        comp_mask = label_img == label_id
        comp_gt = mask & comp_mask
        comp_area = float(area)
        gt_overlap = int(comp_gt.sum())
        keep = 1 if gt_overlap > 0 else 0
        probs = stack[:, comp_mask]
        model_means = probs.mean(axis=1)
        model_max = probs.max(axis=1)
        model_votes = (probs > threshold).mean(axis=1)
        mean_prob = float(probs.mean())
        max_prob = float(probs.max())
        min_prob = float(probs.min())
        std_prob = float(probs.std())
        vote_models = float(((stack[:, comp_mask] > threshold).any(axis=1)).sum())
        fg_mean, local_delta, local_z = local_stats(image, comp_mask, (y0, x0, y1, x1))
        lap_mean = float(np.abs(lap[comp_mask]).mean())
        fmap_values: list[float] = []
        for fmap in sample.get("feature_maps", {}).values():
            fmap_fg_mean, fmap_delta, fmap_z = local_stats(fmap, comp_mask, (y0, x0, y1, x1))
            fmap_values.extend(
                [
                    fmap_fg_mean,
                    float(fmap[comp_mask].max()) if comp_mask.any() else 0.0,
                    fmap_delta,
                    fmap_z,
                ]
            )
        height = float(height_i)
        width = float(width_i)
        fill = comp_area / max(height * width, 1.0)
        cx, cy = centroids[label_id]
        center_y = abs(cy / max(mask.shape[0] - 1, 1) - 0.5)
        center_x = abs(cx / max(mask.shape[1] - 1, 1) - 0.5)
        feat = [
            math.log1p(comp_area),
            math.log1p(height),
            math.log1p(width),
            fill,
            mean_prob,
            max_prob,
            min_prob,
            std_prob,
            vote_models / max(len(names), 1),
            fg_mean,
            local_delta,
            local_z,
            lap_mean,
            center_y,
            center_x,
        ]
        feat.extend(model_means.astype(float).tolist())
        feat.extend(model_max.astype(float).tolist())
        feat.extend(model_votes.astype(float).tolist())
        feat.extend(fmap_values)
        rows.append({"component": comp_mask, "bbox": [int(y0), int(x0), int(y1), int(x1)], "gt_overlap": gt_overlap})
        features.append(feat)
        labels.append(keep)
        priorities.append(max_prob + 0.15 * local_z + 0.05 * math.log1p(comp_area))
    if not features:
        extra_dim = 4 * len(sample.get("feature_maps", {}))
        return rows, np.zeros((0, 15 + 3 * len(names) + extra_dim), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    if max_components > 0 and len(features) > max_components:
        keep_idx = np.argsort(np.asarray(priorities, dtype=np.float32))[::-1][:max_components]
        rows = [rows[int(i)] for i in keep_idx]
        features = [features[int(i)] for i in keep_idx]
        labels = [labels[int(i)] for i in keep_idx]
    return rows, np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def robust01(x: np.ndarray, low: float = 1.0, high: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(x, [low, high])
    if hi <= lo + 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def handcrafted_candidate_maps(image: np.ndarray, topk: int = 32) -> dict[str, np.ndarray]:
    src = image.astype(np.float32)
    src01 = robust01(src, 0.5, 99.5)
    maps: dict[str, np.ndarray] = {}
    for k in (5, 9, 13):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        maps[f"tophat{k}"] = robust01(cv2.morphologyEx(src01, cv2.MORPH_TOPHAT, kernel))
        maps[f"blackhat{k}"] = robust01(cv2.morphologyEx(src01, cv2.MORPH_BLACKHAT, kernel))
        mean = cv2.blur(src01, (k, k))
        sq_mean = cv2.blur(src01 * src01, (k, k))
        std = np.sqrt(np.maximum(sq_mean - mean * mean, 0.0))
        maps[f"localz{k}"] = robust01((src01 - mean) / (std + 1e-3), 50.0, 99.7)
    for sigma in (1.0, 1.6, 2.4):
        blur = cv2.GaussianBlur(src01, (0, 0), sigmaX=sigma, sigmaY=sigma)
        log = cv2.Laplacian(blur, cv2.CV_32F, ksize=3)
        maps[f"log_pos{sigma:g}"] = robust01(log, 50.0, 99.7)
        maps[f"log_abs{sigma:g}"] = robust01(np.abs(log), 50.0, 99.7)
    maps["hand_max"] = np.stack(list(maps.values()), axis=0).max(axis=0)
    seed = maps["hand_max"] > 0.85
    num_labels, label_img, stats, centroids = cv2.connectedComponentsWithStats(seed.astype(np.uint8), 8)
    candidates: list[tuple[float, int]] = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        if not (2 <= area <= 500 and width <= 48 and height <= 48):
            continue
        comp = label_img == label_id
        score = float(maps["hand_max"][comp].mean() + 0.5 * maps["hand_max"][comp].max())
        candidates.append((score, label_id))
    sparse = np.zeros_like(seed, dtype=np.uint8)
    for _score, label_id in sorted(candidates, reverse=True)[: max(0, topk)]:
        sparse[label_img == label_id] = 1
    sparse = cv2.dilate(sparse, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    maps["hand_sparse"] = sparse.astype(np.float32)
    return maps


def spectral_residual_map(src01: np.ndarray) -> np.ndarray:
    small = src01.astype(np.float32)
    # Downsample large padded images so the FFT cue stays cheap during full-data scans.
    max_side = max(small.shape)
    scale = 1.0
    if max_side > 256:
        scale = 256.0 / float(max_side)
        small = cv2.resize(small, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    spectrum = np.fft.fft2(small)
    log_amp = np.log(np.abs(spectrum) + 1e-6)
    phase = np.angle(spectrum)
    avg = cv2.blur(log_amp.astype(np.float32), (3, 3))
    residual = log_amp - avg
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
    sal = cv2.GaussianBlur(sal.astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0)
    if scale != 1.0:
        sal = cv2.resize(sal, (src01.shape[1], src01.shape[0]), interpolation=cv2.INTER_LINEAR)
    return robust01(sal, 60.0, 99.8)


def contrast_candidate_maps(
    image: np.ndarray,
    topk: int = 24,
    threshold: float = 0.78,
) -> dict[str, np.ndarray]:
    src01 = robust01(image.astype(np.float32), 0.5, 99.5)
    cues: list[np.ndarray] = []
    for inner, outer in ((3, 9), (5, 15), (7, 21), (9, 31)):
        inner_mean = cv2.blur(src01, (inner, inner))
        outer_mean = cv2.blur(src01, (outer, outer))
        sq_mean = cv2.blur(src01 * src01, (outer, outer))
        outer_std = np.sqrt(np.maximum(sq_mean - outer_mean * outer_mean, 0.0))
        center_surround = (inner_mean - outer_mean) / (outer_std + 1e-3)
        cues.append(robust01(center_surround, 55.0, 99.8))
    for k in (5, 9, 13, 17):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        cues.append(robust01(cv2.morphologyEx(src01, cv2.MORPH_TOPHAT, kernel), 55.0, 99.8))
    for sigma in (0.8, 1.2, 1.8, 2.6):
        small = cv2.GaussianBlur(src01, (0, 0), sigmaX=sigma, sigmaY=sigma)
        large = cv2.GaussianBlur(src01, (0, 0), sigmaX=sigma * 2.5, sigmaY=sigma * 2.5)
        cues.append(robust01(small - large, 55.0, 99.8))
    cues.append(spectral_residual_map(src01))
    score = np.stack(cues, axis=0).max(axis=0)
    score = cv2.GaussianBlur(score.astype(np.float32), (0, 0), sigmaX=0.6, sigmaY=0.6)
    score = robust01(score, 50.0, 99.7)
    seed = score > threshold
    num_labels, label_img, stats, _centroids = cv2.connectedComponentsWithStats(seed.astype(np.uint8), 8)
    candidates: list[tuple[float, int]] = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        if not (1 <= area <= 380 and width <= 42 and height <= 42):
            continue
        comp = label_img == label_id
        fill = area / float(max(width * height, 1))
        compact_penalty = abs(math.log(max(width / float(max(height, 1)), 1e-3)))
        comp_score = float(score[comp].mean() + 0.9 * score[comp].max() + 0.08 * fill - 0.03 * compact_penalty)
        candidates.append((comp_score, label_id))
    sparse = np.zeros_like(seed, dtype=np.uint8)
    for _score, label_id in sorted(candidates, reverse=True)[: max(0, topk)]:
        sparse[label_img == label_id] = 1
    sparse = cv2.dilate(sparse, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return {"contrast_score": score.astype(np.float32), "contrast_sparse": sparse.astype(np.float32)}


def dense_background_features(image: np.ndarray) -> np.ndarray:
    src01 = robust01(image.astype(np.float32), 0.5, 99.5)
    local_feats = []
    for k in (5, 9, 15):
        mean = cv2.blur(src01, (k, k))
        sq_mean = cv2.blur(src01 * src01, (k, k))
        std = np.sqrt(np.maximum(sq_mean - mean * mean, 0.0))
        local_feats.append((src01 - mean) / (std + 1e-3))
    kernel9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat9 = cv2.morphologyEx(src01, cv2.MORPH_TOPHAT, kernel9)
    tophat15 = cv2.morphologyEx(src01, cv2.MORPH_TOPHAT, kernel15)
    blur1 = cv2.GaussianBlur(src01, (0, 0), sigmaX=1.0, sigmaY=1.0)
    blur3 = cv2.GaussianBlur(src01, (0, 0), sigmaX=3.0, sigmaY=3.0)
    dog = blur1 - blur3
    lap = np.abs(cv2.Laplacian(blur1, cv2.CV_32F, ksize=3))
    grad_x = cv2.Sobel(src01, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(src01, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    feats = [src01, tophat9, tophat15, dog, lap, grad, *local_feats]
    return np.stack(feats, axis=-1).astype(np.float32)


def fit_background_normality_model(
    dataset_dir: Path,
    fit_datasets: list[str],
    split: str,
    max_samples_per_dataset: int,
    sample_pixels_per_image: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260613)
    chunks: list[np.ndarray] = []
    for dataset_name in fit_datasets:
        dataset = FullSplitSet(dataset_dir, dataset_name, dataset_name, split, max_samples=max_samples_per_dataset)
        for idx in range(len(dataset)):
            image_t, mask_t, size, _name = dataset[idx]
            height, width = int(size[0]), int(size[1])
            image = image_t[0, :height, :width].numpy().astype(np.float32)
            mask = mask_t[0, :height, :width].numpy() > 0
            guard = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), 1) > 0
            bg_idx = np.flatnonzero(~guard.reshape(-1))
            if bg_idx.size == 0:
                continue
            if bg_idx.size > sample_pixels_per_image:
                bg_idx = rng.choice(bg_idx, size=sample_pixels_per_image, replace=False)
            feats = dense_background_features(image).reshape(-1, 9)[bg_idx]
            chunks.append(feats.astype(np.float32))
    if not chunks:
        raise RuntimeError("No background pixels collected for background normality model.")
    x = np.concatenate(chunks, axis=0)
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std[std < 1e-4] = 1.0
    return {"mean": mean, "std": std}


def background_anomaly_score_map(
    image: np.ndarray,
    model: dict[str, np.ndarray],
) -> np.ndarray:
    feats = dense_background_features(image)
    z = (feats - model["mean"].reshape(1, 1, -1)) / model["std"].reshape(1, 1, -1)
    # Positive local contrast is emphasized so dark texture anomalies do not dominate.
    mahal = np.sqrt(np.mean(np.square(np.clip(z, -8.0, 8.0)), axis=-1))
    bright = np.maximum(feats[..., 6], 0.0) + 0.75 * np.maximum(feats[..., 7], 0.0) + 0.5 * np.maximum(feats[..., 8], 0.0)
    score = robust01(mahal, 55.0, 99.8) * robust01(bright, 55.0, 99.8)
    return score.astype(np.float32)


def background_anomaly_candidate_map(
    image: np.ndarray,
    model: dict[str, np.ndarray],
    score: np.ndarray | None = None,
    topk: int = 16,
    threshold: float = 0.88,
) -> np.ndarray:
    if score is None:
        score = background_anomaly_score_map(image, model)
    seed = score > threshold
    num_labels, label_img, stats, _centroids = cv2.connectedComponentsWithStats(seed.astype(np.uint8), 8)
    candidates: list[tuple[float, int]] = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        if not (1 <= area <= 350 and width <= 40 and height <= 40):
            continue
        comp = label_img == label_id
        compact = area / float(max(width * height, 1))
        comp_score = float(score[comp].mean() + 0.75 * score[comp].max() + 0.10 * compact)
        candidates.append((comp_score, label_id))
    sparse = np.zeros_like(seed, dtype=np.uint8)
    for _score, label_id in sorted(candidates, reverse=True)[: max(0, topk)]:
        sparse[label_img == label_id] = 1
    sparse = cv2.dilate(sparse, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return sparse.astype(np.float32)


def dense_target_prototype_features(image: np.ndarray) -> np.ndarray:
    src01 = robust01(image.astype(np.float32), 0.5, 99.5)
    feats = [src01]
    for k in (3, 5, 9, 15):
        mean = cv2.blur(src01, (k, k))
        sq_mean = cv2.blur(src01 * src01, (k, k))
        std = np.sqrt(np.maximum(sq_mean - mean * mean, 0.0))
        feats.append(src01 - mean)
        feats.append((src01 - mean) / (std + 1e-3))
    for k in (5, 9, 13, 17):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        feats.append(cv2.morphologyEx(src01, cv2.MORPH_TOPHAT, kernel))
    for sigma in (0.8, 1.2, 1.8, 2.6):
        small = cv2.GaussianBlur(src01, (0, 0), sigmaX=sigma, sigmaY=sigma)
        large = cv2.GaussianBlur(src01, (0, 0), sigmaX=sigma * 2.2, sigmaY=sigma * 2.2)
        feats.append(small - large)
        feats.append(np.abs(cv2.Laplacian(small, cv2.CV_32F, ksize=3)))
    grad_x = cv2.Sobel(src01, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(src01, cv2.CV_32F, 0, 1, ksize=3)
    feats.append(np.sqrt(grad_x * grad_x + grad_y * grad_y))
    return np.stack(feats, axis=-1).astype(np.float32)


def fit_target_prototype_model(
    dataset_dir: Path,
    fit_datasets: list[str],
    split: str,
    max_samples_per_dataset: int,
    positive_dilate: int,
    background_ring: int,
    hard_negative_quantile: float,
    max_pos_pixels: int,
    max_neg_pixels: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260613)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for dataset_name in fit_datasets:
        dataset = FullSplitSet(dataset_dir, dataset_name, dataset_name, split, max_samples=max_samples_per_dataset)
        for idx in range(len(dataset)):
            image_t, mask_t, size, _name = dataset[idx]
            height, width = int(size[0]), int(size[1])
            image = image_t[0, :height, :width].numpy().astype(np.float32)
            mask = mask_t[0, :height, :width].numpy() > 0
            if not mask.any():
                continue
            feats = dense_target_prototype_features(image).reshape(-1, 22)
            pos_mask = mask.astype(np.uint8)
            if positive_dilate > 0:
                k = positive_dilate + (1 - positive_dilate % 2)
                pos_mask = cv2.dilate(pos_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), 1)
            pos_idx = np.flatnonzero(pos_mask.reshape(-1) > 0)
            if pos_idx.size > max_pos_pixels:
                pos_idx = rng.choice(pos_idx, size=max_pos_pixels, replace=False)

            guard_k = background_ring + (1 - background_ring % 2)
            guard = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (guard_k, guard_k)), 1) > 0
            bg_candidates = np.flatnonzero(~guard.reshape(-1))
            if bg_candidates.size == 0:
                continue
            # Hard negatives are bright/local-contrast pixels outside the target guard.
            hardness = feats[bg_candidates, 2] + 0.7 * feats[bg_candidates, 9] + 0.4 * feats[bg_candidates, 17]
            cutoff = np.percentile(hardness, hard_negative_quantile)
            hard_idx = bg_candidates[hardness >= cutoff]
            if hard_idx.size < max_neg_pixels // 3:
                hard_idx = bg_candidates
            neg_size = min(max_neg_pixels, hard_idx.size)
            neg_idx = rng.choice(hard_idx, size=neg_size, replace=False)

            xs.append(feats[pos_idx].astype(np.float32))
            ys.append(np.ones((pos_idx.size,), dtype=np.float32))
            xs.append(feats[neg_idx].astype(np.float32))
            ys.append(np.zeros((neg_idx.size,), dtype=np.float32))
    if not xs:
        raise RuntimeError("No target prototype samples collected.")
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    mean, std = standardize_fit(x)
    xb = np.concatenate([standardize_apply(x, mean, std), np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
    pos = max(float(y.sum()), 1.0)
    neg = max(float((1.0 - y).sum()), 1.0)
    weights = np.where(y > 0, min(neg / pos, 80.0), 1.0).astype(np.float32)
    coef = np.zeros((xb.shape[1],), dtype=np.float32)
    lr = 0.05
    l2 = 0.002
    for _ in range(1200):
        pred = sigmoid(xb @ coef)
        grad = (xb.T @ ((pred - y) * weights)) / max(float(weights.sum()), 1.0)
        grad[:-1] += l2 * coef[:-1]
        coef -= lr * grad.astype(np.float32)
    return {"coef": coef.astype(np.float32), "mean": mean, "std": std}


def prototype_score_map(image: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    feats = dense_target_prototype_features(image)
    flat = feats.reshape(-1, feats.shape[-1])
    xs = standardize_apply(flat, model["mean"], model["std"])
    xb = np.concatenate([xs, np.ones((xs.shape[0], 1), dtype=np.float32)], axis=1)
    score = sigmoid(xb @ model["coef"]).reshape(image.shape)
    return robust01(score.astype(np.float32), 1.0, 99.8)


def prototype_candidate_map(
    score: np.ndarray,
    image: np.ndarray | None = None,
    teacher_max: np.ndarray | None = None,
    topk: int = 24,
    threshold: float = 0.72,
    teacher_suppress: float = 0.35,
) -> np.ndarray:
    gated_score = score.copy()
    if teacher_max is not None:
        gated_score = gated_score * (teacher_max < teacher_suppress).astype(np.float32)
    if image is not None:
        src01 = robust01(image.astype(np.float32), 0.5, 99.5)
        mean = cv2.blur(src01, (9, 9))
        sq_mean = cv2.blur(src01 * src01, (9, 9))
        std = np.sqrt(np.maximum(sq_mean - mean * mean, 0.0))
        local_z = robust01((src01 - mean) / (std + 1e-3), 55.0, 99.8)
        gated_score = gated_score * (0.35 + 0.65 * local_z)
    seed = gated_score > threshold
    num_labels, label_img, stats, _centroids = cv2.connectedComponentsWithStats(seed.astype(np.uint8), 8)
    candidates: list[tuple[float, int]] = []
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        if not (1 <= area <= 450 and width <= 44 and height <= 44):
            continue
        comp = label_img == label_id
        fill = area / float(max(width * height, 1))
        score_i = float(gated_score[comp].mean() + 0.8 * gated_score[comp].max() + 0.08 * fill)
        candidates.append((score_i, label_id))
    sparse = np.zeros_like(seed, dtype=np.uint8)
    for _score, label_id in sorted(candidates, reverse=True)[: max(0, topk)]:
        sparse[label_img == label_id] = 1
    sparse = cv2.dilate(sparse, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return sparse.astype(np.float32)


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float, steps: int, lr: float) -> dict[str, object]:
    mean, std = standardize_fit(x)
    xs = standardize_apply(x, mean, std)
    xb = np.concatenate([xs, np.ones((xs.shape[0], 1), dtype=np.float32)], axis=1)
    pos = max(float(y.sum()), 1.0)
    neg = max(float((1 - y).sum()), 1.0)
    weights = np.where(y > 0, neg / pos, 1.0).astype(np.float32)
    coef = np.zeros((xb.shape[1],), dtype=np.float32)
    for _ in range(steps):
        pred = sigmoid(xb @ coef)
        grad = (xb.T @ ((pred - y) * weights)) / max(float(weights.sum()), 1.0)
        grad[:-1] += l2 * coef[:-1]
        coef -= lr * grad.astype(np.float32)
    return {"coef": coef.astype(np.float32), "mean": mean, "std": std}


def predict_logistic(model: dict[str, object], x: np.ndarray) -> np.ndarray:
    if x.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    xs = standardize_apply(x, model["mean"], model["std"])
    xb = np.concatenate([xs, np.ones((xs.shape[0], 1), dtype=np.float32)], axis=1)
    return sigmoid(xb @ model["coef"]).astype(np.float32)


def render_components(
    sample: dict[str, object],
    names: list[str],
    components: list[dict[str, object]],
    scores: np.ndarray,
    keep_threshold: float,
    fill_mode: str,
) -> np.ndarray:
    preds = sample["preds"]
    out = np.zeros_like(sample["mask"], dtype=np.float32)
    avg = np.stack([preds[name] for name in names], axis=0).mean(axis=0)
    maxp = np.stack([preds[name] for name in names], axis=0).max(axis=0)
    for comp, score in zip(components, scores):
        if float(score) < keep_threshold:
            continue
        mask = comp["component"]
        if fill_mode == "avg":
            out[mask] = np.maximum(out[mask], avg[mask])
        elif fill_mode == "max":
            out[mask] = np.maximum(out[mask], maxp[mask])
        elif fill_mode == "binary":
            out[mask] = 1.0
        else:
            raise ValueError(fill_mode)
    return out


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


def baseline_maps(samples: list[dict[str, object]], names: list[str], mode: str) -> list[np.ndarray]:
    maps = []
    for sample in samples:
        stack = np.stack([sample["preds"][name] for name in names], axis=0)
        if mode in names:
            maps.append(sample["preds"][mode])
        elif mode == "avg":
            maps.append(stack.mean(axis=0))
        elif mode == "max":
            maps.append(stack.max(axis=0))
        elif mode == "min":
            maps.append(stack.min(axis=0))
        else:
            raise ValueError(mode)
    return maps


def build_component_cache(
    samples: list[dict[str, object]],
    names: list[str],
    threshold: float,
    min_seed_area: int,
    handcrafted_seed_threshold: float,
    max_components_per_image: int,
):
    all_components = []
    xs = []
    ys = []
    feature_dim = 15 + 3 * len(names)
    for sample in samples:
        comps, x, y = component_features(
            sample,
            names,
            threshold,
            min_seed_area,
            handcrafted_seed_threshold,
            max_components=max_components_per_image,
        )
        all_components.append((sample, comps, x, y))
        if x.shape[1] > feature_dim:
            feature_dim = x.shape[1]
        if x.shape[0]:
            xs.append(x)
            ys.append(y)
    if not xs:
        return all_components, np.zeros((0, feature_dim), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return all_components, np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def render_cache(cache, names: list[str], learner: dict[str, object] | None, keep_threshold: float, fill_mode: str, oracle: bool):
    maps = []
    total_kept = 0
    total_components = 0
    for sample, comps, x, y in cache:
        if oracle:
            scores = y.astype(np.float32)
        elif learner is None:
            scores = np.ones((len(comps),), dtype=np.float32)
        else:
            scores = predict_logistic(learner, x)
        maps.append(render_components(sample, names, comps, scores, keep_threshold, fill_mode))
        total_kept += int((scores >= keep_threshold).sum())
        total_components += int(len(comps))
    return maps, {"kept_components": total_kept, "total_components": total_components}


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn component-level teacher fusion for IRSTD predictions.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--model_spec", action="append", required=True)
    parser.add_argument(
        "--official_pconv_mshnet_spec",
        action="append",
        default=[],
        help="name|variant|weight|preprocess|threshold|warm|base_size",
    )
    parser.add_argument(
        "--official_bgm_spec",
        action="append",
        default=[],
        help="name|norm_dataset|weight|threshold",
    )
    parser.add_argument(
        "--rpcanet_spec",
        action="append",
        default=[],
        help="name|model|weight|threshold|base_size",
    )
    parser.add_argument("--handcrafted_candidates", action="store_true")
    parser.add_argument("--handcrafted_full", action="store_true")
    parser.add_argument("--handcrafted_topk", type=int, default=32)
    parser.add_argument("--contrast_candidates", action="store_true")
    parser.add_argument("--contrast_topk", type=int, default=24)
    parser.add_argument("--contrast_threshold", type=float, default=0.78)
    parser.add_argument("--background_normality_candidates", action="store_true")
    parser.add_argument("--background_as_candidate", action="store_true")
    parser.add_argument("--background_topk", type=int, default=16)
    parser.add_argument("--background_threshold", type=float, default=0.88)
    parser.add_argument("--background_sample_pixels", type=int, default=2048)
    parser.add_argument("--target_prototype_candidates", action="store_true")
    parser.add_argument("--prototype_topk", type=int, default=24)
    parser.add_argument("--prototype_threshold", type=float, default=0.72)
    parser.add_argument("--prototype_teacher_suppress", type=float, default=0.35)
    parser.add_argument("--prototype_positive_dilate", type=int, default=3)
    parser.add_argument("--prototype_background_ring", type=int, default=17)
    parser.add_argument("--prototype_hard_negative_quantile", type=float, default=97.0)
    parser.add_argument("--prototype_max_pos_pixels", type=int, default=256)
    parser.add_argument("--prototype_max_neg_pixels", type=int, default=512)
    parser.add_argument("--prototype_max_fit_samples", type=int, default=-1)
    parser.add_argument("--fit_datasets", default="")
    parser.add_argument("--fit_split", default="train")
    parser.add_argument("--eval_split", default="test")
    parser.add_argument("--seed_thresholds", default="0.35,0.40,0.45,0.50")
    parser.add_argument("--keep_thresholds", default="0.20,0.30,0.40,0.50,0.60")
    parser.add_argument("--output_thresholds", default="0.35,0.40,0.45,0.50")
    parser.add_argument("--min_seed_area", type=int, default=1)
    parser.add_argument("--max_components_per_image", type=int, default=0)
    parser.add_argument("--fill_modes", default="avg,max,binary")
    parser.add_argument("--handcrafted_seed_threshold", type=float, default=0.85)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_fit_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    specs = [parse_model_spec(spec) for spec in args.model_spec]
    names = [spec[0] for spec in specs]
    models = build_models(specs, device)
    official_pconv_specs = [parse_official_pconv_spec(spec) for spec in args.official_pconv_mshnet_spec]
    official_pconv_models = load_official_pconv_models(official_pconv_specs, device)
    official_bgm_specs = [parse_official_bgm_spec(spec) for spec in args.official_bgm_spec]
    official_bgm_models = load_official_bgm_models(official_bgm_specs, device)
    rpcanet_specs = [parse_rpcanet_spec(spec) for spec in args.rpcanet_spec]
    rpcanet_models = load_rpcanet_models(rpcanet_specs, device)
    names = names + [str(spec["name"]) for spec in official_pconv_specs]
    names = names + [str(spec["name"]) for spec in official_bgm_specs]
    names = names + [str(spec["name"]) for spec in rpcanet_specs]
    if args.handcrafted_candidates:
        if args.handcrafted_full:
            names = names + [
                "tophat5",
                "tophat9",
                "tophat13",
                "blackhat5",
                "blackhat9",
                "blackhat13",
                "localz5",
                "localz9",
                "localz13",
                "log_pos1",
                "log_pos1.6",
                "log_pos2.4",
                "log_abs1",
                "log_abs1.6",
                "log_abs2.4",
                "hand_max",
            ]
        else:
            names = names + ["hand_sparse"]
    if args.contrast_candidates:
        names = names + ["contrast_sparse"]
    fit_datasets = [item.strip() for item in args.fit_datasets.split(",") if item.strip()]
    background_model = None
    if args.background_normality_candidates:
        bg_fit_datasets = fit_datasets or [args.dataset_name]
        print(
            f"fitting background normality model on {','.join(bg_fit_datasets)} {args.fit_split}",
            flush=True,
        )
        background_model = fit_background_normality_model(
            Path(args.dataset_dir),
            bg_fit_datasets,
            args.fit_split,
            args.max_fit_samples,
            args.background_sample_pixels,
        )
        if args.background_as_candidate:
            names = names + ["bg_anomaly"]
    prototype_model = None
    if args.target_prototype_candidates:
        proto_fit_datasets = fit_datasets or [args.dataset_name]
        print(
            f"fitting target prototype model on {','.join(proto_fit_datasets)} {args.fit_split}",
            flush=True,
        )
        prototype_model = fit_target_prototype_model(
            Path(args.dataset_dir),
            proto_fit_datasets,
            args.fit_split,
            args.max_fit_samples if args.prototype_max_fit_samples < 0 else args.prototype_max_fit_samples,
            args.prototype_positive_dilate,
            args.prototype_background_ring,
            args.prototype_hard_negative_quantile,
            args.prototype_max_pos_pixels,
            args.prototype_max_neg_pixels,
        )
        names = names + ["target_proto"]
    if fit_datasets:
        fit_samples = collect_fit_samples_multi(
            models,
            official_pconv_models,
            official_bgm_models,
            rpcanet_models,
            Path(args.dataset_dir),
            fit_datasets,
            args.fit_split,
            args.max_fit_samples,
            device,
            handcrafted=args.handcrafted_candidates,
            handcrafted_full=args.handcrafted_full,
            handcrafted_topk=args.handcrafted_topk,
            contrast_candidates=args.contrast_candidates,
            contrast_topk=args.contrast_topk,
            contrast_threshold=args.contrast_threshold,
            background_model=background_model,
            background_as_candidate=args.background_as_candidate,
            background_topk=args.background_topk,
            background_threshold=args.background_threshold,
            prototype_model=prototype_model,
            prototype_topk=args.prototype_topk,
            prototype_threshold=args.prototype_threshold,
            prototype_teacher_suppress=args.prototype_teacher_suppress,
        )
    else:
        fit_samples = collect_samples(
            models,
            official_pconv_models,
            official_bgm_models,
            rpcanet_models,
            Path(args.dataset_dir),
            args.dataset_name,
            args.fit_split,
            args.max_fit_samples,
            device,
            handcrafted=args.handcrafted_candidates,
            handcrafted_full=args.handcrafted_full,
            handcrafted_topk=args.handcrafted_topk,
            contrast_candidates=args.contrast_candidates,
            contrast_topk=args.contrast_topk,
            contrast_threshold=args.contrast_threshold,
            background_model=background_model,
            background_as_candidate=args.background_as_candidate,
            background_topk=args.background_topk,
            background_threshold=args.background_threshold,
            prototype_model=prototype_model,
            prototype_topk=args.prototype_topk,
            prototype_threshold=args.prototype_threshold,
            prototype_teacher_suppress=args.prototype_teacher_suppress,
        )
    eval_samples = collect_samples(
        models,
        official_pconv_models,
        official_bgm_models,
        rpcanet_models,
        Path(args.dataset_dir),
        args.dataset_name,
        args.eval_split,
        args.max_eval_samples,
        device,
        handcrafted=args.handcrafted_candidates,
        handcrafted_full=args.handcrafted_full,
        handcrafted_topk=args.handcrafted_topk,
        contrast_candidates=args.contrast_candidates,
        contrast_topk=args.contrast_topk,
        contrast_threshold=args.contrast_threshold,
        background_model=background_model,
        background_as_candidate=args.background_as_candidate,
        background_topk=args.background_topk,
        background_threshold=args.background_threshold,
        prototype_model=prototype_model,
        prototype_topk=args.prototype_topk,
        prototype_threshold=args.prototype_threshold,
        prototype_teacher_suppress=args.prototype_teacher_suppress,
    )

    seed_thresholds = [float(x) for x in args.seed_thresholds.split(",") if x.strip()]
    keep_thresholds = [float(x) for x in args.keep_thresholds.split(",") if x.strip()]
    output_thresholds = [float(x) for x in args.output_thresholds.split(",") if x.strip()]
    fill_modes = [x.strip() for x in args.fill_modes.split(",") if x.strip()]
    for fill_mode in fill_modes:
        if fill_mode not in {"avg", "max", "binary"}:
            raise ValueError(f"Unsupported fill mode: {fill_mode}")
    rows: list[dict[str, object]] = []

    for mode in [*names, "avg", "max", "min"]:
        for out_th in output_thresholds:
            metrics = metric_from_maps(eval_samples, baseline_maps(eval_samples, names, mode), out_th)
            rows.append({"kind": "baseline", "mode": mode, "output_threshold": out_th, **metrics})

    for seed_th in seed_thresholds:
        fit_cache, x_train, y_train = build_component_cache(
            fit_samples,
            names,
            seed_th,
            args.min_seed_area,
            args.handcrafted_seed_threshold,
            args.max_components_per_image,
        )
        eval_cache, x_eval, y_eval = build_component_cache(
            eval_samples,
            names,
            seed_th,
            args.min_seed_area,
            args.handcrafted_seed_threshold,
            args.max_components_per_image,
        )
        if x_train.shape[0] == 0:
            continue
        learner = fit_logistic(x_train, y_train, args.l2, args.steps, args.lr)
        train_scores = predict_logistic(learner, x_train)
        eval_scores = predict_logistic(learner, x_eval)
        for fill_mode in fill_modes:
            oracle_maps, oracle_count = render_cache(eval_cache, names, None, 0.5, fill_mode, oracle=True)
            oracle_metrics = metric_from_maps(eval_samples, oracle_maps, 0.5 if fill_mode == "binary" else seed_th)
            rows.append(
                {
                    "kind": "test_oracle_not_valid",
                    "seed_threshold": seed_th,
                    "fill_mode": fill_mode,
                    "keep_threshold": 0.5,
                    "output_threshold": 0.5 if fill_mode == "binary" else seed_th,
                    "train_components": int(x_train.shape[0]),
                    "train_positive": int(y_train.sum()),
                    "eval_components": int(x_eval.shape[0]),
                    "eval_positive": int(y_eval.sum()),
                    "train_score_pos_mean": float(train_scores[y_train > 0].mean()) if (y_train > 0).any() else 0.0,
                    "train_score_neg_mean": float(train_scores[y_train == 0].mean()) if (y_train == 0).any() else 0.0,
                    "eval_score_pos_mean": float(eval_scores[y_eval > 0].mean()) if (y_eval > 0).any() else 0.0,
                    "eval_score_neg_mean": float(eval_scores[y_eval == 0].mean()) if (y_eval == 0).any() else 0.0,
                    **oracle_count,
                    **oracle_metrics,
                }
            )
            for keep_th in keep_thresholds:
                maps, counts = render_cache(eval_cache, names, learner, keep_th, fill_mode, oracle=False)
                out_ths = [0.5] if fill_mode == "binary" else output_thresholds
                for out_th in out_ths:
                    metrics = metric_from_maps(eval_samples, maps, out_th)
                    rows.append(
                        {
                            "kind": "train_fit_eval",
                            "seed_threshold": seed_th,
                            "fill_mode": fill_mode,
                            "keep_threshold": keep_th,
                            "output_threshold": out_th,
                            "train_components": int(x_train.shape[0]),
                            "train_positive": int(y_train.sum()),
                            "eval_components": int(x_eval.shape[0]),
                            "eval_positive": int(y_eval.sum()),
                            "train_score_pos_mean": float(train_scores[y_train > 0].mean()) if (y_train > 0).any() else 0.0,
                            "train_score_neg_mean": float(train_scores[y_train == 0].mean()) if (y_train == 0).any() else 0.0,
                            "eval_score_pos_mean": float(eval_scores[y_eval > 0].mean()) if (y_eval > 0).any() else 0.0,
                            "eval_score_neg_mean": float(eval_scores[y_eval == 0].mean()) if (y_eval == 0).any() else 0.0,
                            **counts,
                            **metrics,
                        }
                    )

    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best_valid = next((row for row in rows if row["kind"] != "test_oracle_not_valid"), rows[0])
    best_oracle = next((row for row in rows if row["kind"] == "test_oracle_not_valid"), None)
    print(
        "best_valid "
        f"kind={best_valid['kind']} mode={best_valid.get('mode', '')} "
        f"seed={best_valid.get('seed_threshold', '')} fill={best_valid.get('fill_mode', '')} "
        f"keep={best_valid.get('keep_threshold', '')} out={best_valid.get('output_threshold', '')} "
        f"mIoU={best_valid['mIoU'] * 100:.2f} nIoU={best_valid['nIoU'] * 100:.2f} "
        f"Pd={best_valid['Pd'] * 100:.2f} Fa={best_valid['Fa'] * 1e6:.2f}",
        flush=True,
    )
    if best_oracle is not None:
        print(
            "best_oracle_not_valid "
            f"seed={best_oracle.get('seed_threshold', '')} fill={best_oracle.get('fill_mode', '')} "
            f"mIoU={best_oracle['mIoU'] * 100:.2f} nIoU={best_oracle['nIoU'] * 100:.2f} "
            f"Pd={best_oracle['Pd'] * 100:.2f} Fa={best_oracle['Fa'] * 1e6:.2f}",
            flush=True,
        )
    for row in rows[:20]:
        print(
            f"{row['kind']} mode={row.get('mode', '')} seed={row.get('seed_threshold', '')} "
            f"fill={row.get('fill_mode', '')} keep={row.get('keep_threshold', '')} "
            f"out={row.get('output_threshold', '')} mIoU={row['mIoU'] * 100:.2f} "
            f"Pd={row['Pd'] * 100:.2f} Fa={row['Fa'] * 1e6:.2f}",
            flush=True,
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "dataset": args.dataset_name,
                "fit_datasets": fit_datasets or [args.dataset_name],
                "fit_split": args.fit_split,
                "eval_split": args.eval_split,
                "models": names,
                "official_pconv_mshnet_specs": args.official_pconv_mshnet_spec,
                "official_bgm_specs": args.official_bgm_spec,
                "rpcanet_specs": args.rpcanet_spec,
                "handcrafted_candidates": bool(args.handcrafted_candidates),
                "handcrafted_full": bool(args.handcrafted_full),
                "handcrafted_topk": int(args.handcrafted_topk),
                "contrast_candidates": bool(args.contrast_candidates),
                "contrast_topk": int(args.contrast_topk),
                "contrast_threshold": float(args.contrast_threshold),
                "background_normality_candidates": bool(args.background_normality_candidates),
                "background_as_candidate": bool(args.background_as_candidate),
                "background_topk": int(args.background_topk),
                "background_threshold": float(args.background_threshold),
                "background_sample_pixels": int(args.background_sample_pixels),
                "target_prototype_candidates": bool(args.target_prototype_candidates),
                "prototype_topk": int(args.prototype_topk),
                "prototype_threshold": float(args.prototype_threshold),
                "prototype_teacher_suppress": float(args.prototype_teacher_suppress),
                "prototype_positive_dilate": int(args.prototype_positive_dilate),
                "prototype_background_ring": int(args.prototype_background_ring),
                "prototype_hard_negative_quantile": float(args.prototype_hard_negative_quantile),
                "prototype_max_fit_samples": int(args.prototype_max_fit_samples),
                "max_components_per_image": int(args.max_components_per_image),
                "fill_modes": fill_modes,
                "model_specs": args.model_spec,
                "best_valid": best_valid,
                "best_oracle_not_valid": best_oracle,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
