#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SCTRANSNET = ROOT / "external" / "SCTransNet"
sys.path.insert(0, str(SCTRANSNET))
sys.path.insert(0, str(ROOT / "scripts"))

from utils import Normalized, PadImg, get_img_norm_cfg, seed_pytorch  # noqa: E402
from train_hybrid_methods import make_model, load_model_checkpoint, weights_init_kaiming, SCTRefiner  # noqa: E402


def parse_teacher_spec(spec: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in spec.split(","):
        if not item.strip():
            continue
        key, value = item.split("=", 1)
        values[key.strip()] = value.strip()
    required = {"name", "model", "checkpoint"}
    missing = required - set(values)
    if missing:
        raise ValueError(f"Teacher spec missing {sorted(missing)}: {spec}")
    return values


def open_image(dataset_dir: Path, name: str) -> Image.Image:
    for ext in (".png", ".bmp", ".jpg"):
        path = dataset_dir / "images" / f"{name}{ext}"
        if path.exists():
            return Image.open(path).convert("I")
    raise FileNotFoundError(name)


class FullImageDataset(Dataset):
    def __init__(self, dataset_root: Path, dataset_name: str, split: str) -> None:
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_root / dataset_name
        self.img_norm_cfg = get_img_norm_cfg(dataset_name, str(dataset_root))
        with (self.dataset_dir / "img_idx" / f"{split}_{dataset_name}.txt").open("r") as f:
            self.names = f.read().splitlines()

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        image = np.array(open_image(self.dataset_dir, name), dtype=np.float32)
        h, w = image.shape
        image = PadImg(Normalized(image, self.img_norm_cfg))
        image = torch.from_numpy(np.ascontiguousarray(image[np.newaxis, :])).float()
        return image, torch.tensor([h, w], dtype=torch.long), name


def load_teacher(spec: dict[str, str], device: torch.device) -> torch.nn.Module:
    model = make_model(spec["model"], warm_epoch=1, base_checkpoint=spec.get("base_checkpoint", "")).to(device)
    if isinstance(model, SCTRefiner):
        model.initialize_refiner()
    else:
        model.apply(weights_init_kaiming)
    load_model_checkpoint(model, Path(spec["checkpoint"]))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@torch.no_grad()
def predict(model: torch.nn.Module, image: torch.Tensor, h: int, w: int, use_tta: bool) -> torch.Tensor:
    def run(x: torch.Tensor) -> torch.Tensor:
        pred = model(x, epoch=10**9)
        if isinstance(pred, (tuple, list)):
            pred = pred[-1]
        return pred

    preds = [run(image)]
    if use_tta:
        preds.append(torch.flip(run(torch.flip(image, dims=[-1])), dims=[-1]))
        preds.append(torch.flip(run(torch.flip(image, dims=[-2])), dims=[-2]))
        preds.append(run(image.transpose(-1, -2)).transpose(-1, -2))
    pred = torch.stack(preds, dim=0).mean(dim=0)
    return pred[:, :, :h, :w].clamp(0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate offline teacher consensus pseudo labels.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--datasets", required=True, help="Comma-separated dataset names.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--teacher", action="append", required=True, help="name=...,model=...,checkpoint=...,base_checkpoint=...")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use_tta", action="store_true")
    parser.add_argument("--consensus", choices=["mean", "max"], default="mean")
    parser.add_argument("--save_uint8", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0, help="Optional per-dataset cap for smoke tests; 0 means full split.")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    seed_pytorch(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset_root = Path(args.dataset_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    teacher_specs = [parse_teacher_spec(spec) for spec in args.teacher]
    teachers = [(spec["name"], load_teacher(spec, device)) for spec in teacher_specs]

    summary = {
        "dataset_dir": str(dataset_root),
        "datasets": [item.strip() for item in args.datasets.split(",") if item.strip()],
        "split": args.split,
        "out_dir": str(out_root),
        "teachers": teacher_specs,
        "use_tta": bool(args.use_tta),
        "consensus": args.consensus,
        "max_samples": args.max_samples,
        "samples": {},
    }

    for dataset_name in summary["datasets"]:
        dataset = FullImageDataset(dataset_root, dataset_name, args.split)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        dataset_out = out_root / dataset_name / args.split
        dataset_out.mkdir(parents=True, exist_ok=True)
        sample_count = min(len(dataset), args.max_samples) if args.max_samples > 0 else len(dataset)
        summary["samples"][dataset_name] = sample_count
        for sample_idx, (image, size, name_batch) in enumerate(loader):
            if args.max_samples > 0 and sample_idx >= args.max_samples:
                break
            image = image.to(device, non_blocking=True)
            h, w = int(size[0, 0]), int(size[0, 1])
            preds = []
            per_teacher: dict[str, float] = {}
            for teacher_name, teacher in teachers:
                pred = predict(teacher, image, h, w, args.use_tta)
                preds.append(pred)
                per_teacher[teacher_name] = float(pred.mean().detach().cpu())
            stacked = torch.cat(preds, dim=1)
            if args.consensus == "max":
                pseudo = stacked.max(dim=1, keepdim=True).values
            else:
                pseudo = stacked.mean(dim=1, keepdim=True)
            pseudo_np = pseudo[0, 0].detach().cpu().numpy()
            name = name_batch[0]
            if args.save_uint8:
                Image.fromarray(np.clip(pseudo_np * 255.0, 0, 255).astype(np.uint8)).save(dataset_out / f"{name}.png")
            else:
                np.savez_compressed(dataset_out / f"{name}.npz", pseudo=pseudo_np.astype(np.float16))
            with (dataset_out / f"{name}.json").open("w") as f:
                json.dump({"name": name, "shape": [h, w], "teacher_mean": per_teacher}, f, indent=2)

    with (out_root / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
