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
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "PAL"))
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))

from eval_component_fusion_learner import metric_from_maps  # noqa: E402
from utilts import access_model  # noqa: E402
from utils import Normalized, PadImg, get_img_norm_cfg  # noqa: E402


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def open_pair(dataset_dir: Path, dataset_name: str, name: str) -> tuple[Image.Image, Image.Image]:
    root = dataset_dir / dataset_name
    for ext in (".png", ".bmp", ".jpg"):
        image_path = root / "images" / f"{name}{ext}"
        mask_path = root / "masks" / f"{name}{ext}"
        if image_path.exists() and mask_path.exists():
            return Image.open(image_path).convert("I"), Image.open(mask_path)
    raise FileNotFoundError(f"{dataset_name}/{name}")


class PALTrainSet(Dataset):
    def __init__(self, dataset_dir: Path, norm_dataset: str, dataset_name: str, patch_size: int, target_repeat: int, seed: int) -> None:
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.patch_size = patch_size
        self.rng = random.Random(seed)
        self.img_norm_cfg = get_img_norm_cfg(norm_dataset, str(dataset_dir))
        index = dataset_dir / dataset_name / "img_idx" / f"train_{dataset_name}.txt"
        base_names = [line.strip() for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]
        repeat = target_repeat if dataset_name == norm_dataset else 1
        self.names = base_names * max(1, repeat)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        image_pil, mask_pil = open_pair(self.dataset_dir, self.dataset_name, name)
        image = Normalized(np.array(image_pil, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask_pil, dtype=np.float32) / 255.0
        if mask.ndim > 2:
            mask = mask[:, :, 0]
        h, w = image.shape
        if h < self.patch_size or w < self.patch_size:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="constant")
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
            h, w = image.shape
        for _ in range(40):
            y0 = self.rng.randint(0, h - self.patch_size)
            x0 = self.rng.randint(0, w - self.patch_size)
            m = mask[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
            if self.rng.random() > 0.35 or m.sum() > 0:
                break
        image = image[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
        mask = mask[y0 : y0 + self.patch_size, x0 : x0 + self.patch_size]
        if self.rng.random() < 0.5:
            image = image[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
        if self.rng.random() < 0.5:
            image = image[::-1, :].copy()
            mask = mask[::-1, :].copy()
        image3 = np.repeat(image[None, :, :], 3, axis=0)
        return torch.from_numpy(np.ascontiguousarray(image3)).float(), torch.from_numpy(np.ascontiguousarray(mask[None])).float()


class PALTestSet(Dataset):
    def __init__(self, dataset_dir: Path, norm_dataset: str, dataset_name: str, split: str, max_samples: int = 0) -> None:
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.img_norm_cfg = get_img_norm_cfg(norm_dataset, str(dataset_dir))
        index = dataset_dir / dataset_name / "img_idx" / f"{split}_{dataset_name}.txt"
        self.names = [line.strip() for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]
        if max_samples > 0:
            self.names = self.names[:max_samples]

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        image_pil, mask_pil = open_pair(self.dataset_dir, self.dataset_name, name)
        image = Normalized(np.array(image_pil, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask_pil, dtype=np.float32) / 255.0
        if mask.ndim > 2:
            mask = mask[:, :, 0]
        h, w = image.shape
        image = PadImg(image)
        image3 = np.repeat(image[None, :, :], 3, axis=0)
        return torch.from_numpy(np.ascontiguousarray(image3)).float(), mask.astype(np.float32), [h, w], name


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PAL MSDA-Net on local full-mask multi-dataset IRSTD splits.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", default="NUAA-SIRST")
    parser.add_argument("--train_datasets", default="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3")
    parser.add_argument("--model_name", default="MSDA")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--patch_size", type=int, default=512)
    parser.add_argument("--target_repeat", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--pos_weight", type=float, default=25.0)
    parser.add_argument("--thresholds", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    parser.add_argument("--eval_interval", type=int, default=2)
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
    train_sets = [
        PALTrainSet(dataset_dir, args.dataset_name, name, args.patch_size, args.target_repeat, args.seed + i)
        for i, name in enumerate(parse_csv(args.train_datasets))
    ]
    train_loader = DataLoader(ConcatDataset(train_sets), batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    test_set = PALTestSet(dataset_dir, args.dataset_name, args.dataset_name, "test", args.eval_max_samples)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = access_model(args.model_name)().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    pos_weight = torch.tensor([args.pos_weight], dtype=torch.float32, device=device).view(1, 1, 1, 1)
    thresholds = [float(x) for x in parse_csv(args.thresholds)]
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    best = None
    history = []

    with (run_dir / "train.log").open("w", encoding="utf-8") as log:
        log.write(json.dumps(vars(args), indent=2) + "\n")
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            model.train()
            losses = []
            for image, gt in train_loader:
                image = image.to(device, non_blocking=True)
                gt = gt.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                    logits = model(image)
                    if isinstance(logits, (tuple, list)):
                        logits = logits[-1]
                    bce = F.binary_cross_entropy_with_logits(logits, gt, pos_weight=pos_weight)
                    loss = bce + dice_loss(logits, gt)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
            sched.step()
            row = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else 0.0, "lr": float(sched.get_last_lr()[0]), "seconds": time.time() - start}
            if epoch % args.eval_interval == 0 or epoch == args.epochs:
                model.eval()
                samples = []
                maps = []
                with torch.no_grad():
                    for image, mask, size, name in test_loader:
                        image = image.to(device)
                        logits = model(image)
                        if isinstance(logits, (tuple, list)):
                            logits = logits[-1]
                        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
                        h, w = int(size[0]), int(size[1])
                        maps.append(prob[:h, :w].astype(np.float32))
                        samples.append({"name": str(name[0]), "mask": mask.numpy()[0].astype(np.float32), "size": [h, w]})
                rows = [{"threshold": th, **metric_from_maps(samples, maps, th)} for th in thresholds]
                rows.sort(key=lambda r: r["mIoU"], reverse=True)
                row["eval"] = {args.dataset_name: rows[0]}
                if best is None or rows[0]["mIoU"] > best["mIoU"]:
                    best = {"epoch": epoch, **rows[0]}
                    torch.save({"epoch": epoch, "state_dict": model.state_dict(), "best": best}, run_dir / "best.pth.tar")
            history.append(row)
            line = json.dumps(row, sort_keys=True)
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
    (run_dir / "summary.json").write_text(json.dumps({"args": vars(args), "best": best, "history": history}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
