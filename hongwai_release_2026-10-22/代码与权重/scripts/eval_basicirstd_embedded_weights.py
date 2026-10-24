#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import importlib
import types
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
BASIC = ROOT / "external" / "BasicIRSTD"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))

from eval_component_fusion_learner import metric_from_maps  # noqa: E402


def load_symbol(path: Path, module_name: str, symbol: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, symbol)


class SplitSet(Dataset):
    def __init__(self, dataset_dir: Path, dataset_name: str, split: str, max_samples: int) -> None:
        self.root = dataset_dir / dataset_name
        names = (self.root / "img_idx" / f"{split}_{dataset_name}.txt").read_text(encoding="utf-8").splitlines()
        self.names = names[:max_samples] if max_samples > 0 else names

    def __len__(self) -> int:
        return len(self.names)

    def _open_pair(self, name: str) -> tuple[Image.Image, Image.Image]:
        for ext in (".png", ".bmp", ".jpg"):
            image_path = self.root / "images" / f"{name}{ext}"
            mask_path = self.root / "masks" / f"{name}{ext}"
            if image_path.exists() and mask_path.exists():
                return Image.open(image_path).convert("L"), Image.open(mask_path).convert("L")
        raise FileNotFoundError(name)

    def __getitem__(self, index: int):
        name = self.names[index]
        image, mask = self._open_pair(name)
        width, height = image.size
        image_np = np.asarray(image, dtype=np.float32)
        mask_np = np.asarray(mask, dtype=np.float32) / 255.0
        return image_np, mask_np, torch.tensor([height, width]), name


def build_model(name: str) -> torch.nn.Module:
    if name == "rdian":
        sys.path.insert(0, str(BASIC / "model" / "RDIAN-main"))
        RDIAN = importlib.import_module("model.segmentation").RDIAN
        return RDIAN()
    if name == "istdu":
        package_name = "basic_istdu_model"
        package = types.ModuleType(package_name)
        package.__path__ = [str(BASIC / "model" / "ISTDU-Net-main" / "model")]
        sys.modules[package_name] = package
        Network = importlib.import_module(f"{package_name}.mModel").Network
        return Network()
    raise ValueError(name)


def default_checkpoint(name: str) -> Path:
    if name == "rdian":
        return BASIC / "model" / "RDIAN-main" / "params" / "best.pth"
    if name == "istdu":
        return BASIC / "model" / "ISTDU-Net-main" / "save_pth" / "ISTDU_Net" / "best.pth"
    raise ValueError(name)


def normalize_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        out[key] = value
    return out


def make_input(model_name: str, image_np: np.ndarray, device: torch.device) -> torch.Tensor:
    if model_name == "istdu":
        image = TF.to_tensor(Image.fromarray(image_np.astype(np.uint8)))
        return image.unsqueeze(0).to(device)
    image01 = image_np.astype(np.float32) / 255.0
    return torch.from_numpy(image01[None, None]).float().to(device)


def pad_to_multiple(inp: torch.Tensor, multiple: int = 32) -> torch.Tensor:
    height, width = inp.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return inp
    return F.pad(inp, (0, pad_w, 0, pad_h), mode="replicate")


def extract_prob(model_name: str, output) -> torch.Tensor:
    if model_name == "istdu":
        if isinstance(output, (tuple, list)):
            return output[-1]
        return output
    return torch.sigmoid(output)


@torch.no_grad()
def predict_dataset(
    model_name: str,
    model: torch.nn.Module,
    dataset_dir: Path,
    dataset_name: str,
    split: str,
    max_samples: int,
    device: torch.device,
) -> list[dict[str, object]]:
    dataset = SplitSet(dataset_dir, dataset_name, split, max_samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    samples: list[dict[str, object]] = []
    model.eval()
    for idx, (image, mask, size, name) in enumerate(loader):
        height, width = int(size[0, 0]), int(size[0, 1])
        image_np = image[0].numpy().astype(np.float32)
        inp = make_input(model_name, image_np, device)
        inp = pad_to_multiple(inp, 32)
        prob = extract_prob(model_name, model(inp))
        if prob.shape[-2:] != (height, width):
            prob = F.interpolate(prob, size=(height, width), mode="bilinear", align_corners=False)
        prob_np = prob[0, 0].detach().cpu().numpy().astype(np.float32)
        mask_np = mask[0].numpy().astype(np.float32)
        samples.append({"name": name[0], "size": [height, width], "mask": mask_np, "preds": {model_name: prob_np}})
        if (idx + 1) % 50 == 0:
            print(f"{model_name} {dataset_name} {split}: {idx + 1}/{len(dataset)}", flush=True)
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate embedded BasicIRSTD weights on local prepared datasets.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--datasets", default="IRSTD-1K,NUAA-SIRST")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", choices=["rdian", "istdu"], required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = build_model(args.model)
    checkpoint = Path(args.checkpoint) if args.checkpoint else default_checkpoint(args.model)
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(normalize_state_dict(state), strict=True)
    model.to(device).eval()

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    results = []
    for dataset_name in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        samples = predict_dataset(args.model, model, Path(args.dataset_dir), dataset_name, args.split, args.max_samples, device)
        maps = [sample["preds"][args.model] for sample in samples]
        rows = [{"threshold": th, **metric_from_maps(samples, maps, th)} for th in thresholds]
        rows.sort(key=lambda row: row["mIoU"], reverse=True)
        best = rows[0]
        print(
            f"{args.model} {dataset_name}: th={best['threshold']} "
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
                "thresholds": thresholds,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
