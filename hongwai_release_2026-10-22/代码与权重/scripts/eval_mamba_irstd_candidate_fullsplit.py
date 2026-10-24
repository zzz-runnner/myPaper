#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import site
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_component_fusion_learner import metric_from_maps  # noqa: E402
from eval_dccs_hf_fullsplit import selective_scan_torch_ref  # noqa: E402


def install_fake_mamba() -> None:
    mamba = types.ModuleType("mamba_ssm")
    ops = types.ModuleType("mamba_ssm.ops")
    iface = types.ModuleType("mamba_ssm.ops.selective_scan_interface")
    iface.selective_scan_fn = selective_scan_torch_ref
    iface.selective_scan_ref = selective_scan_torch_ref
    mamba.ops = ops
    ops.selective_scan_interface = iface
    sys.modules["mamba_ssm"] = mamba
    sys.modules["mamba_ssm.ops"] = ops
    sys.modules["mamba_ssm.ops.selective_scan_interface"] = iface


def install_official_mamba_ops_only() -> None:
    """Load Mamba ops without importing mamba_ssm.__init__ language-model extras."""
    root = None
    for site_dir in site.getsitepackages():
        candidate = Path(site_dir) / "mamba_ssm"
        if candidate.exists():
            root = candidate
            break
    if root is None:
        raise ImportError("mamba_ssm package not found")
    pkg = types.ModuleType("mamba_ssm")
    pkg.__path__ = [str(root)]
    ops = types.ModuleType("mamba_ssm.ops")
    ops.__path__ = [str(root / "ops")]
    utils = types.ModuleType("mamba_ssm.utils")
    utils.__path__ = [str(root / "utils")]
    sys.modules["mamba_ssm"] = pkg
    sys.modules["mamba_ssm.ops"] = ops
    sys.modules["mamba_ssm.utils"] = utils
    for module_name, module_path in [
        ("mamba_ssm.utils.torch", root / "utils" / "torch.py"),
        ("mamba_ssm.ops.selective_scan_interface", root / "ops" / "selective_scan_interface.py"),
    ]:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {module_name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)


class HaarDWTForward(torch.nn.Module):
    def __init__(self, J: int = 1, mode: str = "zero", wave: str = "haar") -> None:
        super().__init__()
        if J != 1 or wave != "haar":
            raise ValueError("Only J=1 haar fallback is implemented")

    def forward(self, x: torch.Tensor):
        if x.shape[-2] % 2 != 0 or x.shape[-1] % 2 != 0:
            x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2))
        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]
        yl = (x00 + x01 + x10 + x11) * 0.5
        lh = (x00 - x01 + x10 - x11) * 0.5
        hl = (x00 + x01 - x10 - x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return yl, (torch.stack([lh, hl, hh], dim=2),)


class HaarDWTInverse(torch.nn.Module):
    def __init__(self, mode: str = "zero", wave: str = "haar") -> None:
        super().__init__()
        if wave != "haar":
            raise ValueError("Only haar fallback is implemented")

    def forward(self, coeffs):
        yl, yh = coeffs
        lh, hl, hh = yh[0][:, :, 0], yh[0][:, :, 1], yh[0][:, :, 2]
        b, c, h, w = yl.shape
        out = yl.new_zeros((b, c, h * 2, w * 2))
        out[:, :, 0::2, 0::2] = (yl + lh + hl + hh) * 0.5
        out[:, :, 0::2, 1::2] = (yl - lh + hl - hh) * 0.5
        out[:, :, 1::2, 0::2] = (yl + lh - hl - hh) * 0.5
        out[:, :, 1::2, 1::2] = (yl - lh - hl + hh) * 0.5
        return out


def install_fake_wavelets() -> None:
    wavelets = types.ModuleType("pytorch_wavelets")
    wavelets.DWTForward = HaarDWTForward
    wavelets.DWTInverse = HaarDWTInverse
    sys.modules["pytorch_wavelets"] = wavelets


class SplitSet(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        dataset_name: str,
        split: str,
        base_size: int,
        max_samples: int = 0,
        index_file: str = "",
    ) -> None:
        self.root = dataset_dir / dataset_name
        if index_file:
            self.names = Path(index_file).read_text(encoding="utf-8").splitlines()
        else:
            self.names = (self.root / "img_idx" / f"{split}_{dataset_name}.txt").read_text(encoding="utf-8").splitlines()
        if max_samples > 0:
            self.names = self.names[:max_samples]
        self.base_size = base_size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.names)

    def _find(self, folder: str, name: str) -> Path:
        for ext in (".png", ".bmp", ".jpg"):
            path = self.root / folder / f"{name}{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(name)

    def __getitem__(self, idx: int):
        name = self.names[idx]
        image = Image.open(self._find("images", name)).convert("RGB")
        mask = Image.open(self._find("masks", name)).convert("L")
        orig_w, orig_h = image.size
        x = self.transform(image.resize((self.base_size, self.base_size), Image.BILINEAR))
        mask_np = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)
        return x, mask_np, torch.tensor([orig_h, orig_w], dtype=torch.int32), name


def parse_thresholds(text: str) -> list[float]:
    if ":" in text:
        start, stop, step = [float(x) for x in text.split(":")]
        vals = []
        cur = start
        while cur <= stop + 1e-9:
            vals.append(round(cur, 6))
            cur += step
        return vals
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def min_area_maps(maps: list[np.ndarray], threshold: float, min_area: int) -> list[np.ndarray]:
    out = []
    for prob in maps:
        binary = (prob > threshold).astype(np.uint8)
        if min_area > 1:
            num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
            keep = np.zeros_like(binary)
            for label in range(1, num):
                if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
                    keep[labels == label] = 1
            binary = keep
        out.append(binary.astype(np.float32))
    return out


def load_model(model_name: str, weight: Path, device: torch.device, wavelet_backend: str = "auto", mamba_backend: str = "fake"):
    if mamba_backend == "fake":
        install_fake_mamba()
    elif mamba_backend == "official":
        install_official_mamba_ops_only()
    elif mamba_backend == "auto":
        try:
            install_official_mamba_ops_only()
        except Exception:
            install_fake_mamba()
    else:
        raise ValueError(mamba_backend)
    if model_name == "hfmnet":
        sys.path.insert(0, str(ROOT / "external" / "HFMNet"))
        module = importlib.import_module("model.HFMNet")
        cls = module.HFMNet
    elif model_name == "dfawarenet":
        if wavelet_backend == "fake":
            install_fake_wavelets()
        elif wavelet_backend == "auto":
            try:
                importlib.import_module("pytorch_wavelets")
            except ImportError:
                install_fake_wavelets()
        elif wavelet_backend != "official":
            raise ValueError(wavelet_backend)
        sys.path.insert(0, str(ROOT / "external" / "DFAwareNet"))
        module = importlib.import_module("model.DFAwareNet")
        cls = module.DFAwareNet
    else:
        raise ValueError(model_name)
    model = cls(in_channels=3, out_channels=1, deep_supervision=False).to(device).eval()
    ckpt = torch.load(weight, map_location="cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    clean = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        if wavelet_backend == "fake" and (".dwt." in key or ".idwt." in key):
            continue
        clean[key] = value
    model.load_state_dict(clean, strict=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HFMNet/DFAwareNet with a torch selective-scan fallback.")
    parser.add_argument("--model", choices=["hfmnet", "dfawarenet"], required=True)
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--weight", required=True)
    parser.add_argument("--base_size", type=int, default=256)
    parser.add_argument("--thresholds", default="0.05:0.95:0.025")
    parser.add_argument("--min_areas", default="1,2,3,4,6,8")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--index_file", default="")
    parser.add_argument("--wavelet_backend", default="auto", choices=["auto", "official", "fake"])
    parser.add_argument("--mamba_backend", default="fake", choices=["auto", "official", "fake"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--save_maps", default="")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = load_model(args.model, Path(args.weight), device, args.wavelet_backend, args.mamba_backend)
    dataset = SplitSet(Path(args.dataset_dir), args.dataset_name, args.split, args.base_size, args.max_samples, args.index_file)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    samples = []
    maps = []
    with torch.no_grad():
        for image, mask, size, name in loader:
            image = image.to(device)
            logits = model(image)
            if isinstance(logits, (list, tuple)):
                logits = logits[-1]
            h, w = int(size[0, 0]), int(size[0, 1])
            prob = torch.sigmoid(logits)
            prob = F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)
            maps.append(prob[0, 0].detach().float().cpu().numpy().astype(np.float32))
            samples.append({"name": str(name[0]), "mask": mask.numpy()[0].astype(np.float32), "size": [h, w]})

    rows = []
    for th in parse_thresholds(args.thresholds):
        rows.append({"kind": "threshold", "threshold": th, **metric_from_maps(samples, maps, th)})
        for area in [int(x) for x in args.min_areas.split(",") if x.strip()]:
            if area <= 1:
                continue
            filtered = min_area_maps(maps, th, area)
            rows.append({"kind": "component", "threshold": th, "min_area": area, **metric_from_maps(samples, filtered, 0.5)})
    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best = rows[0]
    print(
        f"best {args.model} {args.dataset_name} kind={best['kind']} th={best.get('threshold')} "
        f"area={best.get('min_area', 1)} mIoU={best['mIoU']*100:.4f} nIoU={best['nIoU']*100:.4f} "
        f"Pd={best['Pd']*100:.4f} Fa={best['Fa']*1e6:.4f}",
        flush=True,
    )
    Path(args.output).write_text(json.dumps({"args": vars(args), "best": best, "rows": rows[:200]}, indent=2), encoding="utf-8")
    if args.save_maps:
        np.savez_compressed(
            args.save_maps,
            names=np.asarray([s["name"] for s in samples], dtype=object),
            sizes=np.asarray([s["size"] for s in samples], dtype=np.int32),
            masks=np.asarray([s["mask"] for s in samples], dtype=object),
            maps_candidate=np.asarray(maps, dtype=object),
        )


if __name__ == "__main__":
    main()
