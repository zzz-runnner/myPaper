#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def read_split_from_zip(zf: zipfile.ZipFile, member: str) -> list[str]:
    raw = zf.read(member).decode("utf-8", errors="ignore")
    names: list[str] = []
    for line in raw.splitlines():
        value = line.strip()
        if not value:
            continue
        names.append(Path(value).stem)
    return names


def save_png_from_member(zf: zipfile.ZipFile, member: str, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as fp:
        image = Image.open(fp).convert(mode)
        image.save(dst)


def binarize_mask(path: Path) -> None:
    mask = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    Image.fromarray(binary, mode="L").save(path)


def reset_dataset(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)
    (root / "img_idx").mkdir(parents=True)


def write_names(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SIRST-UAVB into this project's IRSTD dataset layout.")
    parser.add_argument("--zip_path", default="/home/cx/hongwai/dataset/SIRST-UAVB/SIRST-UAVB_download")
    parser.add_argument("--out_root", default="/home/cx/hongwai/irstd-lab/datasets")
    parser.add_argument("--dataset_name", default="SIRST-UAVB")
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    out = Path(args.out_root) / args.dataset_name
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    reset_dataset(out)

    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        image_members = {
            Path(member).stem: member
            for member in members
            if member.startswith("SIRST-UAVB/images/all/")
            and member.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            and "/._" not in member
        }
        mask_members = {
            Path(member).stem: member
            for member in members
            if member.startswith("SIRST-UAVB/masks/")
            and member.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            and "/._" not in member
        }
        train = read_split_from_zip(zf, "SIRST-UAVB/train.txt")
        test = read_split_from_zip(zf, "SIRST-UAVB/val.txt")

        names = sorted(set(image_members) & set(mask_members), key=lambda value: int(value) if value.isdigit() else value)
        missing_train = sorted(set(train) - set(names), key=lambda value: int(value) if value.isdigit() else value)
        missing_test = sorted(set(test) - set(names), key=lambda value: int(value) if value.isdigit() else value)
        if missing_train or missing_test:
            raise RuntimeError(f"Missing split entries: train={missing_train[:5]} test={missing_test[:5]}")

        for idx, name in enumerate(names, start=1):
            image_path = out / "images" / f"{name}.png"
            mask_path = out / "masks" / f"{name}.png"
            save_png_from_member(zf, image_members[name], image_path, "L")
            save_png_from_member(zf, mask_members[name], mask_path, "L")
            with Image.open(image_path) as image, Image.open(mask_path) as mask:
                if image.size != mask.size:
                    resized = mask.resize(image.size, Image.Resampling.NEAREST)
                    resized.save(mask_path)
            binarize_mask(mask_path)
            if idx % 500 == 0:
                print(f"converted {idx}/{len(names)}")

    write_names(out / "img_idx" / f"train_{args.dataset_name}.txt", train)
    write_names(out / "img_idx" / f"test_{args.dataset_name}.txt", test)
    print(f"{args.dataset_name}: images={len(names)} train={len(train)} test={len(test)} out={out}")


if __name__ == "__main__":
    main()
