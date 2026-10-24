#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
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
LRR = ROOT / "external" / "LRRNet"
sys.path.insert(0, str(LRR))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))

from models.LRRNet import UNet  # noqa: E402
from utils import Normalized, get_img_norm_cfg, random_crop  # noqa: E402
from eval_component_fusion_learner import metric_from_maps  # noqa: E402


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def open_pair(root: Path, name: str) -> tuple[Image.Image, Image.Image]:
    for ext in (".png", ".bmp", ".jpg"):
        image = root / "images" / f"{name}{ext}"
        mask = root / "masks" / f"{name}{ext}"
        if image.exists() and mask.exists():
            return Image.open(image).convert("I"), Image.open(mask)
    raise FileNotFoundError(name)


class LRRTrainSet(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        dataset_name: str,
        patch_size: int,
        max_samples: int = 0,
        pos_prob: float = 0.65,
    ) -> None:
        self.root = dataset_dir / dataset_name
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.patch_size = patch_size
        self.pos_prob = pos_prob
        self.names = (self.root / "img_idx" / f"train_{dataset_name}.txt").read_text(encoding="utf-8").splitlines()
        if max_samples > 0:
            self.names = self.names[:max_samples]
        self.norm = get_img_norm_cfg(dataset_name, str(dataset_dir))

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        image_pil, mask_pil = open_pair(self.root, self.names[idx])
        image = Normalized(np.asarray(image_pil, dtype=np.float32), self.norm)
        mask = np.asarray(mask_pil, dtype=np.float32) / 255.0
        if mask.ndim > 2:
            mask = mask[:, :, 0]
        image, mask = random_crop(image, mask, self.patch_size, pos_prob=self.pos_prob)
        if random.random() < 0.5:
            image = image[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
        if random.random() < 0.5:
            image = image[::-1, :].copy()
            mask = mask[::-1, :].copy()
        if random.random() < 0.5:
            image = image.T.copy()
            mask = mask.T.copy()
        return (
            torch.from_numpy(np.ascontiguousarray(image[None, :, :])).float(),
            torch.from_numpy(np.ascontiguousarray(mask[None, :, :])).float(),
        )


class LRRTestSet(Dataset):
    def __init__(self, dataset_dir: Path, train_dataset_name: str, dataset_name: str, max_samples: int = 0) -> None:
        self.root = dataset_dir / dataset_name
        self.dataset_name = dataset_name
        self.names = (self.root / "img_idx" / f"test_{dataset_name}.txt").read_text(encoding="utf-8").splitlines()
        if max_samples > 0:
            self.names = self.names[:max_samples]
        self.norm = get_img_norm_cfg(train_dataset_name, str(dataset_dir))

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        image_pil, mask_pil = open_pair(self.root, self.names[idx])
        image = Normalized(np.asarray(image_pil, dtype=np.float32), self.norm)
        mask = np.asarray(mask_pil, dtype=np.float32) / 255.0
        if mask.ndim > 2:
            mask = mask[:, :, 0]
        height, width = image.shape
        pad_h = (16 - height % 16) % 16
        pad_w = (16 - width % 16) % 16
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="constant")
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
        return (
            torch.from_numpy(np.ascontiguousarray(image[None, :, :])).float(),
            torch.from_numpy(np.ascontiguousarray(mask[None, :, :])).float(),
            [height, width],
            self.names[idx],
        )


def soft_iou_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    return 1.0 - ((inter + 1.0) / (union + 1.0)).mean()


def soft_iou_l1_loss(logits: torch.Tensor, target: torch.Tensor, lambda_iou: float = 0.8, lambda_l1: float = 0.2) -> torch.Tensor:
    sparse = (logits > 0).float().mean()
    return lambda_iou * soft_iou_loss(logits, target) + lambda_l1 * sparse


def lrr_loss(
    recon: torch.Tensor,
    target_logits: torch.Tensor,
    image: torch.Tensor,
    mask: torch.Tensor,
    mse_weight: float,
    bce_weight: float,
    pos_weight: float,
) -> dict[str, torch.Tensor]:
    loss_iou = soft_iou_loss(target_logits, mask)
    loss_ioul1 = soft_iou_l1_loss(target_logits, mask)
    loss_mse = F.mse_loss(recon, image)
    if bce_weight > 0:
        bce = F.binary_cross_entropy_with_logits(
            target_logits,
            mask,
            pos_weight=torch.tensor([pos_weight], dtype=target_logits.dtype, device=target_logits.device).view(1, 1, 1, 1),
        )
    else:
        bce = torch.zeros((), dtype=target_logits.dtype, device=target_logits.device)
    total = loss_iou + loss_ioul1 + mse_weight * loss_mse + bce_weight * bce
    return {"total": total, "iou": loss_iou, "ioul1": loss_ioul1, "mse": loss_mse, "bce": bce}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset_dir: Path,
    norm_dataset: str,
    dataset_name: str,
    device: torch.device,
    thresholds: list[float],
    max_samples: int,
) -> dict[str, object]:
    model.eval()
    dataset = LRRTestSet(dataset_dir, norm_dataset, dataset_name, max_samples=max_samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    samples = []
    maps = []
    for image, mask, size, name in loader:
        image = image.to(device)
        _recon, target_logits = model(image)
        prob = torch.sigmoid(target_logits)
        height, width = int(size[0]), int(size[1])
        pred = prob[0, 0, :height, :width].detach().cpu().numpy().astype(np.float32)
        gt = mask[0, 0, :height, :width].numpy().astype(np.float32)
        samples.append({"name": str(name[0]), "mask": gt, "size": [height, width]})
        maps.append(pred)
    rows = [{"threshold": th, **metric_from_maps(samples, maps, th)} for th in thresholds]
    rows.sort(key=lambda r: r["mIoU"], reverse=True)
    return {"best": rows[0], "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or finetune LRRNet on the local full multi-dataset IRSTD pool.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--train_datasets", default="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3")
    parser.add_argument("--eval_datasets", default="IRSTD-1K,NUAA-SIRST")
    parser.add_argument("--norm_dataset", default="NUAA-SIRST")
    parser.add_argument("--init_checkpoint", default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eta_min", type=float, default=1e-6)
    parser.add_argument("--mse_weight", type=float, default=0.1)
    parser.add_argument("--bce_weight", type=float, default=0.15)
    parser.add_argument("--pos_weight", type=float, default=30.0)
    parser.add_argument("--pos_crop_prob", type=float, default=0.65)
    parser.add_argument("--thresholds", default="0.20,0.30,0.40,0.50,0.60,0.70,0.80")
    parser.add_argument("--eval_interval", type=int, default=2)
    parser.add_argument("--eval_max_samples", type=int, default=0)
    parser.add_argument("--max_train_samples_per_dataset", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default=str(ROOT / "runs" / "lrrnet_multidata"))
    args = parser.parse_args()

    seed_everything(args.seed)
    dataset_dir = Path(args.dataset_dir)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    train_sets = [
        LRRTrainSet(dataset_dir, name, args.patch_size, args.max_train_samples_per_dataset, pos_prob=args.pos_crop_prob)
        for name in parse_csv(args.train_datasets)
    ]
    loader = DataLoader(
        ConcatDataset(train_sets),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )
    model = UNet(img_channels=1, base_channels=16, channel_mults=(1, 2, 4, 8), num_res_blocks=2, dropout=0.2, norm="gn", num_groups=16)
    if args.init_checkpoint:
        state = torch.load(args.init_checkpoint, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.eta_min)
    thresholds = parse_floats(args.thresholds)
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best = None
    with (run_dir / "train.log").open("w", encoding="utf-8") as log:
        log.write(json.dumps(vars(args), indent=2) + "\n")
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            model.train()
            losses: list[dict[str, float]] = []
            for images, masks in loader:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                recon, logits = model(images)
                loss_items = lrr_loss(recon, logits, images, masks, args.mse_weight, args.bce_weight, args.pos_weight)
                optimizer.zero_grad(set_to_none=True)
                loss_items["total"].backward()
                optimizer.step()
                losses.append({key: float(value.detach().cpu()) for key, value in loss_items.items()})
            scheduler.step()
            row = {
                "epoch": epoch,
                "lr": float(scheduler.get_last_lr()[0]),
                "seconds": time.time() - start,
            }
            if losses:
                row.update({f"loss_{key}": float(np.mean([item[key] for item in losses])) for key in losses[0]})
            if epoch % args.eval_interval == 0 or epoch == args.epochs:
                evals = {
                    name: evaluate(model, dataset_dir, args.norm_dataset, name, device, thresholds, args.eval_max_samples)
                    for name in parse_csv(args.eval_datasets)
                }
                row["eval"] = {name: value["best"] for name, value in evals.items()}
                mean_miou = float(np.mean([value["best"]["mIoU"] for value in evals.values()]))
                if best is None or mean_miou > best["mean_mIoU"]:
                    best = {"epoch": epoch, "mean_mIoU": mean_miou, "eval": row["eval"]}
                    torch.save({"epoch": epoch, "state_dict": model.state_dict(), "best": best}, run_dir / "best.pth.tar")
            history.append(row)
            line = json.dumps(row, sort_keys=True)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
    (run_dir / "summary.json").write_text(json.dumps({"args": vars(args), "best": best, "history": history}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
