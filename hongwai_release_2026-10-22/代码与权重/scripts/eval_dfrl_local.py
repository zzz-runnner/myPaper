#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
DFRL = ROOT / "external" / "DFRL"
sys.path.insert(0, str(DFRL))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))

from eval_component_fusion_learner import metric_from_maps  # noqa: E402
from model.DNANet.model_DNANet import DNANet, Res_CBAM_block  # noqa: E402
from model.PCSDL.MSHNet import MSHNet  # noqa: E402
from model.RDIAN.segmentation import RDIAN  # noqa: E402


class SplitSet(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        dataset_name: str,
        split: str,
        channels: int,
        input_size: int,
        max_samples: int,
    ) -> None:
        self.root = dataset_dir / dataset_name
        self.channels = channels
        self.input_size = input_size
        names = (self.root / "img_idx" / f"{split}_{dataset_name}.txt").read_text(encoding="utf-8").splitlines()
        self.names = names[:max_samples] if max_samples > 0 else names

    def __len__(self) -> int:
        return len(self.names)

    def _open_pair(self, name: str) -> tuple[Image.Image, Image.Image]:
        for ext in (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"):
            image_path = self.root / "images" / f"{name}{ext}"
            mask_path = self.root / "masks" / f"{name}{ext}"
            if image_path.exists() and mask_path.exists():
                mode = "RGB" if self.channels == 3 else "L"
                return Image.open(image_path).convert(mode), Image.open(mask_path).convert("L")
        raise FileNotFoundError(name)

    def __getitem__(self, index: int):
        name = self.names[index]
        image, mask = self._open_pair(name)
        width, height = image.size
        image_np = np.asarray(image.resize((self.input_size, self.input_size), Image.BICUBIC), dtype=np.float32) / 255.0
        if self.channels == 1:
            image_np = image_np[None]
        else:
            image_np = image_np.transpose(2, 0, 1)
        mask_np = np.asarray(mask, dtype=np.float32) / 255.0
        return torch.from_numpy(image_np), torch.from_numpy(mask_np), torch.tensor([height, width]), name


def build_model(name: str) -> tuple[torch.nn.Module, int]:
    if name == "dnanet":
        model = DNANet(
            num_classes=1,
            input_channels=3,
            block=Res_CBAM_block,
            num_blocks=[2, 2, 2, 2],
            nb_filter=[16, 32, 64, 128, 256],
            deep_supervision=True,
        )
        return model, 3
    if name == "pcsd":
        return MSHNet(3), 3
    if name == "rdian":
        return RDIAN(), 1
    raise ValueError(name)


def default_checkpoint(name: str) -> Path:
    if name == "dnanet":
        return DFRL / "pth" / "DNANet" / "Best_DANNet.pth"
    if name == "pcsd":
        return DFRL / "pth" / "PCSDL" / "Best_PCSDL.pth"
    if name == "rdian":
        return DFRL / "pth" / "RDIAN" / "Best_RDIAN.pth"
    raise ValueError(name)


def extract_logits(model_name: str, output):
    if model_name == "pcsd":
        return output[1]
    if isinstance(output, (list, tuple)):
        return output[-1]
    return output


@torch.no_grad()
def predict_dataset(
    model_name: str,
    model: torch.nn.Module,
    channels: int,
    dataset_dir: Path,
    dataset_name: str,
    split: str,
    input_size: int,
    max_samples: int,
    device: torch.device,
) -> list[dict[str, object]]:
    dataset = SplitSet(dataset_dir, dataset_name, split, channels, input_size, max_samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    samples: list[dict[str, object]] = []
    model.eval()
    for idx, (image, mask, size, name) in enumerate(loader):
        height, width = int(size[0, 0]), int(size[0, 1])
        image = image.to(device)
        if model_name == "pcsd":
            logits = extract_logits(model_name, model(image, False))
        else:
            logits = extract_logits(model_name, model(image))
        prob = torch.sigmoid(logits)
        prob = F.interpolate(prob, size=(height, width), mode="bilinear", align_corners=False)
        prob_np = prob[0, 0].detach().cpu().numpy().astype(np.float32)
        mask_np = mask[0].numpy().astype(np.float32) > 0
        if prob_np.shape != mask_np.shape:
            prob_np = cv2.resize(prob_np, (mask_np.shape[1], mask_np.shape[0]), interpolation=cv2.INTER_LINEAR)
        samples.append({"name": name[0], "size": [height, width], "mask": mask_np.astype(np.float32), "preds": {model_name: prob_np}})
        if (idx + 1) % 50 == 0:
            print(f"{model_name} {dataset_name} {split}: {idx + 1}/{len(dataset)}", flush=True)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DFRL official checkpoints on local prepared datasets.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--datasets", default="IRSTD-1K,NUAA-SIRST")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", choices=["dnanet", "pcsd", "rdian"], required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model, channels = build_model(args.model)
    checkpoint = Path(args.checkpoint) if args.checkpoint else default_checkpoint(args.model)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    results = []
    for dataset_name in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        samples = predict_dataset(
            args.model,
            model,
            channels,
            Path(args.dataset_dir),
            dataset_name,
            args.split,
            args.input_size,
            args.max_samples,
            device,
        )
        maps = [sample["preds"][args.model] for sample in samples]
        rows = [{"threshold": threshold, **metric_from_maps(samples, maps, threshold)} for threshold in thresholds]
        rows.sort(key=lambda row: row["mIoU"], reverse=True)
        best = rows[0]
        print(
            f"{args.model} {dataset_name}: best th={best['threshold']} "
            f"mIoU={best['mIoU'] * 100:.2f} nIoU={best['nIoU'] * 100:.2f} "
            f"Pd={best['Pd'] * 100:.2f} Fa={best['Fa'] * 1e6:.2f}",
            flush=True,
        )
        results.append({"dataset": dataset_name, "split": args.split, "best": best, "rows": rows})

    Path(args.output).write_text(
        json.dumps(
            {
                "model": args.model,
                "checkpoint": str(checkpoint),
                "input_size": args.input_size,
                "thresholds": thresholds,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
