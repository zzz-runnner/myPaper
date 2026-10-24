#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCTRANS_SPLIT = PROJECT_ROOT / "external" / "SCTransNet" / "datasets" / "SIRST3" / "img_idx"


def read_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def write_names(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + "\n")


def reset_dataset(root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    (root / "images").mkdir(parents=True)
    (root / "masks").mkdir(parents=True)
    (root / "img_idx").mkdir(parents=True)


def copy_zip_member(zip_path: Path, member: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as src, dst.open("wb") as out:
        shutil.copyfileobj(src, out)


def stem_from_member(member: str) -> str:
    return Path(member).stem


def prepare_sirst(raw: Path, out_root: Path, sirst3_train: list[str], sirst3_test: list[str]) -> None:
    dst = out_root / "NUAA-SIRST"
    reset_dataset(dst)

    images_zip = raw / "images.zip"
    masks_zip = raw / "masks.zip"
    if not images_zip.exists() or not masks_zip.exists():
        raise FileNotFoundError("SIRST-v1 needs images.zip and masks.zip")

    image_members: dict[str, str] = {}
    with zipfile.ZipFile(images_zip) as zf:
        for member in zf.namelist():
            if member.startswith("images/") and member.endswith(".png") and "/._" not in member:
                image_members[stem_from_member(member)] = member

    mask_members: dict[str, str] = {}
    with zipfile.ZipFile(masks_zip) as zf:
        for member in zf.namelist():
            if member.startswith("masks/") and member.endswith("_pixels0.png") and "/._" not in member:
                name = Path(member).stem.replace("_pixels0", "")
                mask_members[name] = member

    names = sorted(set(image_members) & set(mask_members), key=lambda s: int(s.split("_")[1]))
    for name in names:
        image_path = dst / "images" / f"{name}.png"
        mask_path = dst / "masks" / f"{name}.png"
        copy_zip_member(images_zip, image_members[name], image_path)
        copy_zip_member(masks_zip, mask_members[name], mask_path)
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                resized = mask.resize(image.size, Image.Resampling.NEAREST)
                resized.save(mask_path)
                print(f"NUAA-SIRST: resized mask {name} from {mask.size} to {image.size}")

    train = [name for name in sirst3_train if name in image_members]
    test = [name for name in sirst3_test if name in image_members]
    write_names(dst / "img_idx" / "train_NUAA-SIRST.txt", train)
    write_names(dst / "img_idx" / "test_NUAA-SIRST.txt", test)
    print(f"NUAA-SIRST: images={len(names)} train={len(train)} test={len(test)}")


def prepare_irstd(raw: Path, out_root: Path) -> None:
    dst = out_root / "IRSTD-1K"
    reset_dataset(dst)

    zip_path = raw / "IRSTD-1k.zip"
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        image_members = {
            stem_from_member(member): member
            for member in zf.namelist()
            if member.startswith("IRSTD-1k/IRSTD1k_Img/") and member.endswith(".png")
        }
        mask_members = {
            stem_from_member(member): member
            for member in zf.namelist()
            if member.startswith("IRSTD-1k/IRSTD1k_Label/") and member.endswith(".png")
        }
        names = sorted(set(image_members) & set(mask_members), key=lambda s: int(s.replace("XDU", "")))
        for name in names:
            copy_zip_member(zip_path, image_members[name], dst / "images" / f"{name}.png")
            copy_zip_member(zip_path, mask_members[name], dst / "masks" / f"{name}.png")

        train = [line.strip() for line in zf.read("IRSTD-1k/trainval.txt").decode().splitlines() if line.strip()]
        test = [line.strip() for line in zf.read("IRSTD-1k/test.txt").decode().splitlines() if line.strip()]

    write_names(dst / "img_idx" / "train_IRSTD-1K.txt", train)
    write_names(dst / "img_idx" / "test_IRSTD-1K.txt", test)
    print(f"IRSTD-1K: images={len(names)} train={len(train)} test={len(test)}")


def prepare_nudt(raw: Path, out_root: Path, sirst3_train: list[str], sirst3_test: list[str]) -> None:
    dst = out_root / "NUDT-SIRST"
    reset_dataset(dst)

    zip_path = raw / "NUDT-SIRST.zip"
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        image_members = {
            stem_from_member(member): member
            for member in zf.namelist()
            if member.startswith("images/") and member.endswith(".png")
        }
        mask_members = {
            stem_from_member(member): member
            for member in zf.namelist()
            if member.startswith("masks/") and member.endswith(".png")
        }
        names = sorted(set(image_members) & set(mask_members))
        for name in names:
            copy_zip_member(zip_path, image_members[name], dst / "images" / f"{name}.png")
            copy_zip_member(zip_path, mask_members[name], dst / "masks" / f"{name}.png")

    train = [name for name in sirst3_train if name in image_members]
    test = [name for name in sirst3_test if name in image_members]
    write_names(dst / "img_idx" / "train_NUDT-SIRST.txt", train)
    write_names(dst / "img_idx" / "test_NUDT-SIRST.txt", test)
    print(f"NUDT-SIRST: images={len(names)} train={len(train)} test={len(test)}")


def prepare_sirst3(out_root: Path, sirst3_train: list[str], sirst3_test: list[str]) -> None:
    dst = out_root / "SIRST3"
    reset_dataset(dst)

    sources = [out_root / "NUAA-SIRST", out_root / "IRSTD-1K", out_root / "NUDT-SIRST"]
    count = 0
    for source in sources:
        for image in (source / "images").glob("*.png"):
            mask = source / "masks" / image.name
            if not mask.exists():
                raise FileNotFoundError(mask)
            shutil.copy2(image, dst / "images" / image.name)
            shutil.copy2(mask, dst / "masks" / image.name)
            count += 1

    write_names(dst / "img_idx" / "train_SIRST3.txt", sirst3_train)
    write_names(dst / "img_idx" / "test_SIRST3.txt", sirst3_test)
    print(f"SIRST3: images={count} train={len(sirst3_train)} test={len(sirst3_test)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare uploaded IRSTD datasets for SCTransNet.")
    parser.add_argument("--raw_dir", default="/home/cx/hongwai/dataset")
    parser.add_argument("--out_dir", default="/home/cx/hongwai/irstd-lab/datasets")
    args = parser.parse_args()

    raw = Path(args.raw_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    sirst3_train = read_names(SCTRANS_SPLIT / "train_SIRST3.txt")
    sirst3_test = read_names(SCTRANS_SPLIT / "test_SIRST3.txt")

    prepare_sirst(raw, out_root, sirst3_train, sirst3_test)
    prepare_irstd(raw, out_root)
    prepare_nudt(raw, out_root, sirst3_train, sirst3_test)
    prepare_sirst3(out_root, sirst3_train, sirst3_test)


if __name__ == "__main__":
    main()
