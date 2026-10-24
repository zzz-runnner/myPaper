#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image


def find_image(path_no_ext: Path) -> Path | None:
    for suffix in (".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"):
        candidate = path_no_ext.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def check_split(dataset_root: Path, dataset_name: str, split: str) -> tuple[int, int]:
    idx_file = dataset_root / "img_idx" / f"{split}_{dataset_name}.txt"
    if not idx_file.exists():
        raise FileNotFoundError(f"missing index file: {idx_file}")

    names = [line.strip() for line in idx_file.read_text().splitlines() if line.strip()]
    missing = 0
    size_mismatch = 0

    for name in names:
        image_path = find_image(dataset_root / "images" / name)
        mask_path = find_image(dataset_root / "masks" / name)
        if image_path is None or mask_path is None:
            missing += 1
            print(f"missing_pair[{split}]: {name}")
            continue
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                size_mismatch += 1
                print(f"size_mismatch[{split}]: {name} image={image.size} mask={mask.size}")

    print(f"{split}: {len(names)} samples, missing={missing}, size_mismatch={size_mismatch}")
    return missing, size_mismatch


def main() -> None:
    parser = argparse.ArgumentParser(description="Check SCTransNet dataset layout.")
    parser.add_argument("dataset_name", help="For example: SIRST3, NUDT-SIRST, IRSTD-1K, NUAA-SIRST")
    parser.add_argument(
        "--dataset_dir",
        default=os.environ.get("IRSTD_DATASET_DIR", "/data1/irstd/datasets"),
        help="Parent directory containing dataset folders.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_dir) / args.dataset_name
    print(f"dataset_root: {dataset_root}")
    if not dataset_root.exists():
        raise FileNotFoundError(f"dataset folder does not exist: {dataset_root}")

    for subdir in ("images", "masks", "img_idx"):
        path = dataset_root / subdir
        if not path.exists():
            raise FileNotFoundError(f"missing required folder: {path}")

    total_missing = 0
    total_mismatch = 0
    for split in ("train", "test"):
        missing, mismatch = check_split(dataset_root, args.dataset_name, split)
        total_missing += missing
        total_mismatch += mismatch

    if total_missing or total_mismatch:
        raise SystemExit("dataset_check: failed")
    print("dataset_check: ok")


if __name__ == "__main__":
    main()
