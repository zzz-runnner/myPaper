#!/usr/bin/env python
from __future__ import annotations

import argparse
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
DCCS_ROOT = ROOT / "external" / "DCCS_hf"
sys.path.insert(0, str(DCCS_ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_component_fusion_learner import metric_from_maps  # noqa: E402


def install_official_mamba_ops_only() -> None:
    """Load Mamba selective-scan ops without importing mamba_ssm language-model extras."""
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


class ZeroVSSM(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        b, _c, h, w = x.shape
        dev, dtype = x.device, x.dtype
        return [
            torch.zeros(b, h, w, 16, device=dev, dtype=dtype),
            torch.zeros(b, h // 2, w // 2, 32, device=dev, dtype=dtype),
            torch.zeros(b, h // 4, w // 4, 64, device=dev, dtype=dtype),
            torch.zeros(b, h // 8, w // 8, 128, device=dev, dtype=dtype),
        ]


def selective_scan_torch_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
):
    del z
    if return_last_state:
        raise NotImplementedError("DCCS eval does not need return_last_state")
    batch, channels, length = u.shape
    n_state = A.shape[1]
    groups = B.shape[1]
    if channels % groups != 0:
        raise ValueError(f"channels {channels} not divisible by groups {groups}")
    per_group = channels // groups
    if delta_bias is not None:
        delta = delta + delta_bias.view(1, -1, 1)
    if delta_softplus:
        delta = torch.nn.functional.softplus(delta)
    b_group = torch.repeat_interleave(B, per_group, dim=1)
    c_group = torch.repeat_interleave(C, per_group, dim=1)
    a_t = torch.exp(delta[..., None] * A.view(1, channels, 1, n_state))
    q_t = delta[..., None] * b_group.permute(0, 1, 3, 2) * u[..., None]

    # Vectorized scan for x_t = a_t * x_{t-1} + q_t. The small epsilon avoids
    # division by exact zero when long negative-A products underflow.
    prefix = torch.cumprod(a_t.clamp_min(1e-30), dim=2)
    state = prefix * torch.cumsum(q_t / prefix.clamp_min(1e-30), dim=2)
    y = (state * c_group.permute(0, 1, 3, 2)).sum(dim=-1)
    if D is not None:
        y = y + u * D.view(1, channels, 1)
    return y


class SplitSet(Dataset):
    def __init__(self, dataset_dir: Path, dataset_name: str, split: str, base_size: int, max_samples: int = 0) -> None:
        self.root = dataset_dir / dataset_name
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate local DCCS_hf weights on full local splits.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--weight", required=True)
    parser.add_argument("--base_size", type=int, default=256)
    parser.add_argument("--mode", choices=["official", "torch_ref", "zero_vssm"], default="official")
    parser.add_argument("--thresholds", default="0.05:0.95:0.025")
    parser.add_argument("--min_areas", default="1,2,3,4,6,8")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--save_maps", default="")
    args = parser.parse_args()

    if args.mode == "official":
        install_official_mamba_ops_only()
    import model.auxiliary as dccs_aux  # noqa: PLC0415
    from model.DCCS import DCCS  # noqa: PLC0415
    if args.mode == "official":
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref  # noqa: PLC0415

        dccs_aux.selective_scan_fn = selective_scan_fn
        dccs_aux.selective_scan_ref = selective_scan_ref

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = DCCS(3).to(device).eval()
    state = torch.load(args.weight, map_location="cpu")
    model.load_state_dict(state, strict=True)
    if args.mode == "zero_vssm":
        model.VSSM = ZeroVSSM().to(device)
    elif args.mode == "torch_ref":
        dccs_aux.selective_scan_fn = selective_scan_torch_ref
        dccs_aux.selective_scan_ref = selective_scan_torch_ref

    dataset = SplitSet(Path(args.dataset_dir), args.dataset_name, args.split, args.base_size, args.max_samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    samples = []
    maps = []
    with torch.no_grad():
        for image, mask, size, name in loader:
            image = image.to(device)
            _aux, logits = model(image, False)
            prob = torch.sigmoid(logits)
            h, w = int(size[0, 0]), int(size[0, 1])
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
        f"best {args.dataset_name} {args.mode} kind={best['kind']} th={best.get('threshold')} "
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
            maps_dccs=np.asarray(maps, dtype=object),
        )


if __name__ == "__main__":
    main()
