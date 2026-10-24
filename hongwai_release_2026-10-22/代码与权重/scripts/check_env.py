#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCTRANSNET = ROOT / "external" / "SCTransNet"
sys.path.insert(0, str(SCTRANSNET))


def main() -> None:
    print(f"project_root: {ROOT}")
    print(f"python: {sys.executable}")
    print(f"torch: {torch.__version__}")
    print(f"torch_cuda: {torch.version.cuda}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"gpu_count: {torch.cuda.device_count()}")
    for idx in range(torch.cuda.device_count()):
        print(f"gpu_{idx}: {torch.cuda.get_device_name(idx)}")

    import cv2  # noqa: F401
    import einops  # noqa: F401
    import ml_collections  # noqa: F401
    import skimage  # noqa: F401
    import tensorboard  # noqa: F401
    import thop  # noqa: F401

    from model.SCTransNet import SCTransNet
    import model.Config as config

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    net = SCTransNet(config.get_SCTrans_config(), mode="test", deepsuper=True).to(device).eval()
    x = torch.randn(1, 1, 256, 256, device=device)
    with torch.no_grad():
        y = net(x)
    shape = tuple(y.shape) if hasattr(y, "shape") else [tuple(t.shape) for t in y]
    print(f"forward_shape: {shape}")
    if device.type == "cuda":
        mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        print(f"forward_peak_mem_mb: {mem_mb:.1f}")
    print("env_check: ok")


if __name__ == "__main__":
    main()
