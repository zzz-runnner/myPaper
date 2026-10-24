#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = ("IRSTD-1K", "NUAA-SIRST", "NUDT-SIRST", "SIRST3")


@dataclass(frozen=True)
class Sample:
    dataset: str
    name: str
    image: Path
    mask: Path
    split: str


def read_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, copy: bool = False) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def load_samples(dataset_root: Path, dataset_names: list[str], target_datasets: set[str]) -> list[Sample]:
    samples: list[Sample] = []
    seen: set[tuple[str, str]] = set()
    for dataset in dataset_names:
        root = dataset_root / dataset
        split_root = root / "img_idx"
        for split in ("train", "test"):
            split_path = split_root / f"{split}_{dataset}.txt"
            if not split_path.exists():
                raise FileNotFoundError(split_path)
            mapped_split = "val" if dataset in target_datasets and split == "test" else "train"
            if dataset not in target_datasets and split == "test":
                continue
            for name in read_names(split_path):
                key = (dataset, name)
                if key in seen:
                    continue
                seen.add(key)
                image = root / "images" / f"{name}.png"
                mask = root / "masks" / f"{name}.png"
                if not image.exists():
                    matches = sorted((root / "images").glob(f"{name}.*"))
                    if not matches:
                        raise FileNotFoundError(image)
                    image = matches[0]
                if not mask.exists():
                    matches = sorted((root / "masks").glob(f"{name}.*"))
                    if not matches:
                        raise FileNotFoundError(mask)
                    mask = matches[0]
                samples.append(Sample(dataset=dataset, name=name, image=image, mask=mask, split=mapped_split))
    return samples


def connected_boxes(mask_bin: np.ndarray, min_area: int) -> list[tuple[int, int, int, int]]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin.astype(np.uint8), connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        boxes.append((int(x), int(y), int(x + w - 1), int(y + h - 1)))
    return boxes


def mask_to_boxes(mask_path: Path, min_area: int, mode: str) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    mask = np.array(Image.open(mask_path).convert("L"))
    mask_bin = mask > 0
    if not mask_bin.any():
        return [], mask_bin.astype(np.uint8)

    if mode == "whole":
        ys, xs = np.where(mask_bin)
        boxes = [(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))]
    elif mode == "components":
        boxes = connected_boxes(mask_bin, min_area=min_area)
        if not boxes:
            ys, xs = np.where(mask_bin)
            boxes = [(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))]
    else:
        raise ValueError(f"unknown bbox mode: {mode}")
    return boxes, mask_bin.astype(np.uint8)


def parse_repeat_factors(spec: str) -> dict[str, int]:
    factors: dict[str, int] = {}
    if not spec.strip():
        return factors
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"repeat factor must be DATASET:N, got {item!r}")
        dataset, value = item.split(":", 1)
        factor = int(value)
        if factor < 1:
            raise ValueError(f"repeat factor for {dataset!r} must be >= 1")
        factors[dataset.strip()] = factor
    return factors


def write_yolo_label(path: Path, boxes: list[tuple[int, int, int, int]], width: int, height: int) -> None:
    lines: list[str] = []
    for x1, y1, x2, y2 in boxes:
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))
        bw = max(1, x2 - x1 + 1)
        bh = max(1, y2 - y1 + 1)
        cx = (x1 + x2 + 1) / 2 / width
        cy = (y1 + y2 + 1) / 2 / height
        lines.append(f"0 {cx:.8f} {cy:.8f} {bw / width:.8f} {bh / height:.8f}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def prepare(args: argparse.Namespace) -> dict[str, object]:
    dataset_root = Path(args.dataset_root)
    out_root = Path(args.out_root)
    target_datasets = {d.strip() for d in args.target_datasets.split(",") if d.strip()}
    dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    repeat_factors = parse_repeat_factors(args.train_repeat_factors)

    reset_dir(out_root)
    for rel in (
        "images/train",
        "images/val",
        "labels/train",
        "labels/val",
        "segimages/train",
        "segimages/val",
        "seglabels/train",
        "seglabels/val",
    ):
        (out_root / rel).mkdir(parents=True, exist_ok=True)

    samples = load_samples(dataset_root, dataset_names, target_datasets)
    stats: dict[str, object] = {
        "out_root": str(out_root),
        "bbox_mode": args.bbox_mode,
        "target_datasets": sorted(target_datasets),
        "datasets": dataset_names,
        "train_repeat_factors": repeat_factors,
        "splits": defaultdict(int),
        "datasets_by_split": defaultdict(lambda: defaultdict(int)),
        "boxes_by_split": defaultdict(int),
        "empty_masks": [],
    }

    for sample in samples:
        with Image.open(sample.image) as image:
            width, height = image.size
        boxes, mask_bin = mask_to_boxes(sample.mask, min_area=args.min_area, mode=args.bbox_mode)
        if not boxes:
            stats["empty_masks"].append(f"{sample.dataset}/{sample.name}")  # type: ignore[index]

        repeat = repeat_factors.get(sample.dataset, 1) if sample.split == "train" else 1
        for rep in range(repeat):
            suffix = f"__r{rep}" if repeat > 1 else ""
            out_name = f"{sample.dataset}__{sample.name}{suffix}.png"

            link_or_copy(sample.image, out_root / "images" / sample.split / out_name, copy=args.copy_images)
            link_or_copy(sample.image, out_root / "segimages" / sample.split / out_name, copy=args.copy_images)
            Image.fromarray(mask_bin.astype(np.uint8)).save(out_root / "seglabels" / sample.split / out_name)
            write_yolo_label(out_root / "labels" / sample.split / f"{Path(out_name).stem}.txt", boxes, width, height)

            stats["splits"][sample.split] += 1  # type: ignore[index]
            stats["datasets_by_split"][sample.split][sample.dataset] += 1  # type: ignore[index]
            stats["boxes_by_split"][sample.split] += len(boxes)  # type: ignore[index]

    yaml_path = out_root / "mtmlnet_full.yaml"
    yaml_data = {
        "train": str(out_root / "images" / "train"),
        "val": str(out_root / "images" / "val"),
        "test": str(out_root / "images" / "val"),
        "segtrain": str(out_root),
        "segval": str(out_root),
        "segtest": str(out_root),
        "nc": 1,
        "names": ["target"],
    }
    yaml_path.write_text(yaml.safe_dump(yaml_data, sort_keys=False, allow_unicode=True))
    stats["yaml"] = str(yaml_path)

    serializable = json.loads(json.dumps(stats, default=lambda x: dict(x)))
    (out_root / "summary.json").write_text(json.dumps(serializable, indent=2, ensure_ascii=False) + "\n")
    return serializable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert IRSTD masks into MTMLNet det+seg data.")
    parser.add_argument("--dataset_root", default=str(PROJECT_ROOT / "datasets"))
    parser.add_argument("--out_root", default="/data1/irstd_mtmlnet_full")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--target_datasets", default="IRSTD-1K,NUAA-SIRST")
    parser.add_argument("--bbox_mode", choices=["whole", "components"], default="components")
    parser.add_argument("--min_area", type=int, default=1)
    parser.add_argument(
        "--train_repeat_factors",
        default="",
        help="Comma-separated DATASET:N factors applied to train samples only, e.g. IRSTD-1K:2,NUAA-SIRST:4.",
    )
    parser.add_argument("--copy_images", action="store_true")
    return parser.parse_args()


def main() -> None:
    stats = prepare(parse_args())
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
