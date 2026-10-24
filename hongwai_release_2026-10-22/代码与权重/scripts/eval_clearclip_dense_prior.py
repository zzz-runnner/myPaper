#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "ClearCLIP"))

from eval_component_fusion_learner import FullSplitSet, metric_from_maps  # noqa: E402
from gem.gem import create_gem_model, get_gem_img_transform  # noqa: E402


TARGET_PROMPTS = [
    "infrared small target",
    "tiny thermal target",
    "small bright infrared object",
    "dim point target",
]

BACKGROUND_PROMPTS = [
    "infrared background",
    "cloud clutter",
    "sea clutter",
    "sensor noise",
    "building edge",
]


def minmax(x: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(x, 1.0))
    hi = float(np.percentile(x, 99.0))
    if hi <= lo + 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def open_raw_image(dataset_dir: Path, dataset_name: str, name: str) -> Image.Image:
    root = dataset_dir / dataset_name / "images"
    for ext in (".png", ".bmp", ".jpg"):
        path = root / f"{name}{ext}"
        if path.exists():
            return Image.open(path).convert("RGB")
    raise FileNotFoundError(name)


@torch.no_grad()
def gem_logits(model: torch.nn.Module, image: torch.Tensor, prompts: list[str]) -> torch.Tensor:
    patch_feat = model.model.visual(image)
    patch_feat = F.normalize(patch_feat, dim=-1)
    text_embeddings = model.encode_text(prompts)
    logits = patch_feat @ text_embeddings.transpose(-1, -2)
    grid_h = image.shape[-2] // model.patch_size
    grid_w = image.shape[-1] // model.patch_size
    logits = rearrange(logits, "b (h w) c -> b c h w", h=grid_h, w=grid_w)
    logits = F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
    return model.min_max(logits)


@torch.no_grad()
def predict_maps(
    model: torch.nn.Module,
    transform: transforms.Compose,
    dataset_dir: Path,
    dataset_name: str,
    split: str,
    max_samples: int,
    device: torch.device,
    score_mode: str,
) -> tuple[list[dict[str, object]], list[np.ndarray]]:
    dataset = FullSplitSet(dataset_dir, dataset_name, dataset_name, split, max_samples=max_samples)
    samples: list[dict[str, object]] = []
    maps: list[np.ndarray] = []
    prompts = TARGET_PROMPTS + BACKGROUND_PROMPTS
    target_count = len(TARGET_PROMPTS)

    model.eval()
    for idx, name in enumerate(dataset.names):
        raw_image = open_raw_image(dataset_dir, dataset_name, name)
        height, width = raw_image.height, raw_image.width
        image = transform(raw_image).unsqueeze(0).to(device)
        logits = gem_logits(model, image, prompts)[0].detach()
        logits = F.interpolate(
            logits.unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0]
        target = logits[:target_count]
        background = logits[target_count:]
        if score_mode == "maxdiff":
            score = target.max(dim=0).values - background.max(dim=0).values
        elif score_mode == "meandiff":
            score = target.mean(dim=0) - background.mean(dim=0)
        elif score_mode == "targetmax":
            score = target.max(dim=0).values
        else:
            raise ValueError(f"Unsupported score mode: {score_mode}")
        score_np = minmax(score.cpu().numpy().astype(np.float32))

        _image_tensor, mask_tensor, size, _ = dataset[dataset.names.index(name)]
        h = int(size[0])
        w = int(size[1])
        mask_np = mask_tensor[0, :h, :w].numpy().astype(np.float32)
        samples.append({"name": name, "size": [h, w], "mask": mask_np, "preds": {"clearclip": score_np}})
        maps.append(score_np)
        if (idx + 1) % 20 == 0:
            print(f"{dataset_name} {split}: {idx + 1}/{len(dataset)}", flush=True)
    return samples, maps


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ClearCLIP/GEM dense text prior on IRSTD datasets.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", default="NUAA-SIRST")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model_name", default="ViT-B/16")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--gem_depth", type=int, default=7)
    parser.add_argument("--score_mode", choices=["maxdiff", "meandiff", "targetmax"], default="maxdiff")
    parser.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache_dir", default=str(ROOT / "external" / "ClearCLIP" / "weights_cache"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    model = create_gem_model(
        args.model_name,
        pretrained=args.pretrained,
        gem_depth=args.gem_depth,
        device=device,
        cache_dir=str(cache_dir),
        force_image_size=args.img_size,
    )
    transform = get_gem_img_transform((args.img_size, args.img_size))
    samples, maps = predict_maps(
        model,
        transform,
        Path(args.dataset_dir),
        args.dataset_name,
        args.split,
        args.max_samples,
        device,
        args.score_mode,
    )
    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    rows = [{"threshold": th, **metric_from_maps(samples, maps, th)} for th in thresholds]
    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    out = {
        "dataset": args.dataset_name,
        "split": args.split,
        "max_samples": args.max_samples,
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "img_size": args.img_size,
        "gem_depth": args.gem_depth,
        "score_mode": args.score_mode,
        "target_prompts": TARGET_PROMPTS,
        "background_prompts": BACKGROUND_PROMPTS,
        "best": rows[0],
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    best = rows[0]
    print(
        f"best th={best['threshold']} mIoU={best['mIoU'] * 100:.4f} "
        f"nIoU={best['nIoU'] * 100:.4f} Pd={best['Pd'] * 100:.4f} Fa={best['Fa'] * 1e6:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
