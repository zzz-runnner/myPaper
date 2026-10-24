#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
IRSAM = ROOT / "external" / "IRSAM"
sys.path.insert(0, str(IRSAM))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))

from eval_component_fusion_learner import metric_from_maps  # noqa: E402
from segment_anything_training.build_IRSAM import build_sam_IRSAM  # noqa: E402


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def open_pair(root: Path, name: str):
    for ext in (".png", ".bmp", ".jpg"):
        image = root / "images" / f"{name}{ext}"
        mask = root / "masks" / f"{name}{ext}"
        if image.exists() and mask.exists():
            return Image.open(image).convert("RGB"), Image.open(mask)
    raise FileNotFoundError(name)


class IRSAMTrainSet(Dataset):
    def __init__(self, dataset_dir: Path, dataset_name: str, size: int = 512) -> None:
        self.root = dataset_dir / dataset_name
        self.dataset_name = dataset_name
        self.size = size
        self.names = (self.root / "img_idx" / f"train_{dataset_name}.txt").read_text(encoding="utf-8").splitlines()

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        image_pil, mask_pil = open_pair(self.root, self.names[idx])
        image = np.array(image_pil.resize((self.size, self.size), Image.BILINEAR), dtype=np.uint8)
        mask = np.array(mask_pil.resize((self.size, self.size), Image.NEAREST), dtype=np.float32)
        if mask.ndim > 2:
            mask = mask[:, :, 0]
        mask = (mask > 127).astype(np.float32)
        if np.random.rand() < 0.5:
            image = image[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
        if np.random.rand() < 0.5:
            image = image[::-1, :].copy()
            mask = mask[::-1, :].copy()
        return torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(mask[None, :, :])


class IRSAMTestSet(Dataset):
    def __init__(self, dataset_dir: Path, dataset_name: str, size: int = 512) -> None:
        self.root = dataset_dir / dataset_name
        self.dataset_name = dataset_name
        self.size = size
        self.names = (self.root / "img_idx" / f"test_{dataset_name}.txt").read_text(encoding="utf-8").splitlines()

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        image_pil, mask_pil = open_pair(self.root, self.names[idx])
        image = np.array(image_pil.resize((self.size, self.size), Image.BILINEAR), dtype=np.uint8)
        mask = np.array(mask_pil.resize((self.size, self.size), Image.NEAREST), dtype=np.float32)
        if mask.ndim > 2:
            mask = mask[:, :, 0]
        mask = (mask > 127).astype(np.float32)
        return torch.from_numpy(image).permute(2, 0, 1), torch.from_numpy(mask[None, :, :]), self.names[idx]


def set_trainable(model: torch.nn.Module, mode: str) -> None:
    for p in model.parameters():
        p.requires_grad = False
    if mode == "decoder":
        modules = [model.mask_decoder]
    elif mode == "decoder_prompt":
        modules = [model.mask_decoder, model.prompt_encoder]
    elif mode == "all":
        modules = [model]
    else:
        raise ValueError(mode)
    for module in modules:
        for p in module.parameters():
            p.requires_grad = True


def forward_batch(model, images: torch.Tensor):
    batched = [
        {"image": image, "original_size": (int(image.shape[1]), int(image.shape[2]))}
        for image in images
    ]
    masks, edges = model(batched)
    return masks, edges


def loss_fn(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * inter + 1.0) / (denom + 1.0))
    return bce + dice.mean()


@torch.no_grad()
def evaluate(model, dataset_dir: Path, dataset_name: str, device: torch.device, threshold: float, max_samples: int = 0):
    dataset = IRSAMTestSet(dataset_dir, dataset_name)
    if max_samples > 0:
        dataset.names = dataset.names[:max_samples]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    model.eval()
    samples = []
    maps = []
    for image, mask, name in loader:
        image = image.to(device)
        logits, _edges = forward_batch(model, image)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy().astype(np.float32)
        gt = mask[0, 0].numpy().astype(np.float32)
        samples.append({"name": str(name[0]), "mask": gt, "size": [int(gt.shape[0]), int(gt.shape[1])]})
        maps.append(prob)
    return metric_from_maps(samples, maps, threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train IRSAM structure on full multi-dataset IRSTD.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--train_datasets", default="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3")
    parser.add_argument("--eval_datasets", default="IRSTD-1K,NUAA-SIRST")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--trainable", default="decoder", choices=["decoder", "decoder_prompt", "all"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--eval_max_samples", type=int, default=0)
    parser.add_argument("--max_train_samples_per_dataset", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default=str(ROOT / "runs" / "irsam_multidata"))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dataset_dir = Path(args.dataset_dir)
    train_sets = []
    for name in parse_csv(args.train_datasets):
        dataset = IRSAMTrainSet(dataset_dir, name)
        if args.max_train_samples_per_dataset > 0:
            dataset.names = dataset.names[: args.max_train_samples_per_dataset]
        train_sets.append(dataset)
    train_loader = DataLoader(
        ConcatDataset(train_sets),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    model = build_sam_IRSAM(checkpoint=args.checkpoint or None).to(device)
    set_trainable(model, args.trainable)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    run_dir = Path(args.output_dir) / f"irsam_{args.trainable}"
    run_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best = None
    with (run_dir / "train.log").open("w", encoding="utf-8") as log:
        log.write(json.dumps(vars(args), indent=2) + "\n")
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            start = time.time()
            for images, masks in train_loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                logits, edges = forward_batch(model, images)
                loss = loss_fn(logits, masks)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            row = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else 0.0, "seconds": time.time() - start}
            if epoch % args.eval_interval == 0 or epoch == args.epochs:
                evals = {
                    name: evaluate(model, dataset_dir, name, device, args.threshold, args.eval_max_samples)
                    for name in parse_csv(args.eval_datasets)
                }
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
