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
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_component_fusion_learner import (  # noqa: E402
    FullSplitSet,
    build_models,
    load_official_pconv_models,
    metric_from_maps,
    open_raw_pair,
    parse_model_spec,
    parse_official_pconv_spec,
)


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


@torch.no_grad()
def pconv_prob_map(official: dict[str, object], dataset_dir: Path, dataset_name: str, name: str, sc_image: torch.Tensor, height: int, width: int, device: torch.device) -> np.ndarray:
    model = official["model"]
    warm = bool(official["warm"])
    if official["preprocess"] == "official_mshnet":
        raw_image, _mask = open_raw_pair(dataset_dir, dataset_name, name)
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image = raw_image.resize((int(official["base_size"]), int(official["base_size"])), Image.BILINEAR)
        image_tensor = transform(image).unsqueeze(0).to(device)
        _masks, logits = model(image_tensor, warm)
        pred = torch.sigmoid(logits)
        pred = F.interpolate(pred, size=(height, width), mode="bilinear", align_corners=False)
    else:
        image_tensor = sc_image.repeat(1, 3, 1, 1)
        _masks, logits = model(image_tensor, warm)
        pred = torch.sigmoid(logits)[:, :, :height, :width]
    return pred[0, 0].detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def collect_samples(models, official, dataset_dir: Path, dataset_name: str, split: str, max_samples: int, device: torch.device):
    dataset = FullSplitSet(dataset_dir, dataset_name, dataset_name, split, max_samples=max_samples)
    loader = DataLoader(dataset, batch_size=1, num_workers=1, shuffle=False)
    samples = []
    for idx, (image, mask, size, name) in enumerate(loader):
        image = image.to(device)
        height = int(size[0])
        width = int(size[1])
        teacher_maps = []
        for _model_name, model in models:
            pred = model(image, epoch=10**9)
            if isinstance(pred, (tuple, list)):
                pred = pred[-1]
            teacher_maps.append(pred[0, 0, :height, :width].detach().cpu().numpy().astype(np.float32))
        teacher_max = np.stack(teacher_maps, axis=0).max(axis=0)
        pconv = pconv_prob_map(official, dataset_dir, dataset_name, name[0], image, height, width, device)
        image_np = image[0, 0, :height, :width].detach().cpu().numpy().astype(np.float32)
        mask_np = mask[0, 0, :height, :width].numpy().astype(np.float32)
        samples.append(
            {
                "name": name[0],
                "image": image_np,
                "mask": mask_np,
                "size": [height, width],
                "teacher": teacher_max,
                "pconv": pconv,
            }
        )
        if (idx + 1) % 100 == 0:
            print(f"collected {dataset_name} {split} {idx + 1}/{len(dataset)}", flush=True)
    return samples


def collect_fit_samples(models, official, dataset_dir: Path, datasets: list[str], split: str, max_samples: int, device: torch.device):
    all_samples = []
    for dataset_name in datasets:
        all_samples.extend(collect_samples(models, official, dataset_dir, dataset_name, split, max_samples, device))
    return all_samples


def local_z_map(image: np.ndarray) -> np.ndarray:
    src = image.astype(np.float32)
    mean = cv2.blur(src, (9, 9))
    sq_mean = cv2.blur(src * src, (9, 9))
    std = np.sqrt(np.maximum(sq_mean - mean * mean, 0.0))
    return (src - mean) / (std + 1e-6)


def accepted_component_mask(sample: dict[str, object], params: dict[str, float]) -> np.ndarray:
    pconv = sample["pconv"]
    teacher = sample["teacher"]
    image = sample["image"]
    seed = pconv >= params["seed"]
    num_labels, label_img, stats, _centroids = cv2.connectedComponentsWithStats(seed.astype(np.uint8), 8)
    keep = np.zeros_like(seed, dtype=bool)
    zmap = local_z_map(image)
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        if area < params["min_area"] or area > params["max_area"]:
            continue
        if max(width, height) > params["max_side"]:
            continue
        fill = area / float(max(width * height, 1))
        if fill < params["min_fill"]:
            continue
        comp = label_img == label_id
        if float(zmap[comp].mean()) < params["min_z"]:
            continue
        if float(teacher[comp].mean()) > params["max_teacher_mean"]:
            continue
        keep[comp] = True
    return keep


def render(samples: list[dict[str, object]], params: dict[str, float]) -> list[np.ndarray]:
    maps = []
    for sample in samples:
        out = sample["teacher"].copy()
        keep = accepted_component_mask(sample, params)
        if params["fill"] == "binary":
            out[keep] = np.maximum(out[keep], 1.0)
        else:
            out[keep] = np.maximum(out[keep], sample["pconv"][keep])
        maps.append(out.astype(np.float32))
    return maps


def metric(samples: list[dict[str, object]], maps: list[np.ndarray], threshold: float) -> dict[str, float]:
    wrapped = [
        {"mask": sample["mask"], "size": sample["size"]}
        for sample in samples
    ]
    return metric_from_maps(wrapped, maps, threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search a train-fit gate for official PConv candidates.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--fit_datasets", default="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3")
    parser.add_argument("--model_spec", action="append", required=True)
    parser.add_argument("--official_pconv_mshnet_spec", required=True)
    parser.add_argument("--seed_thresholds", default="0.35,0.45,0.55,0.60")
    parser.add_argument("--max_areas", default="16,32,64,128")
    parser.add_argument("--max_sides", default="8,12,18,28")
    parser.add_argument("--min_fills", default="0.15,0.25,0.35")
    parser.add_argument("--min_zs", default="-1.0,0.0,0.35,0.70")
    parser.add_argument("--max_teacher_means", default="1.1,0.45,0.25,0.12")
    parser.add_argument("--output_thresholds", default="0.45,0.50")
    parser.add_argument("--max_fit_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    models = build_models([parse_model_spec(spec) for spec in args.model_spec], device)
    official = load_official_pconv_models([parse_official_pconv_spec(args.official_pconv_mshnet_spec)], device)[0]
    fit_datasets = [x.strip() for x in args.fit_datasets.split(",") if x.strip()]
    fit_samples = collect_fit_samples(models, official, Path(args.dataset_dir), fit_datasets, "train", args.max_fit_samples, device)
    eval_samples = collect_samples(models, official, Path(args.dataset_dir), args.dataset_name, "test", args.max_eval_samples, device)

    seeds = parse_float_list(args.seed_thresholds)
    max_areas = parse_int_list(args.max_areas)
    max_sides = parse_int_list(args.max_sides)
    min_fills = parse_float_list(args.min_fills)
    min_zs = parse_float_list(args.min_zs)
    max_teacher_means = parse_float_list(args.max_teacher_means)
    output_thresholds = parse_float_list(args.output_thresholds)

    rows = []
    teacher_maps = [sample["teacher"] for sample in eval_samples]
    for out_th in output_thresholds:
        rows.append({"kind": "teacher_baseline", "output_threshold": out_th, **metric(eval_samples, teacher_maps, out_th)})
    max_maps = [np.maximum(sample["teacher"], sample["pconv"]) for sample in eval_samples]
    for out_th in output_thresholds:
        rows.append({"kind": "teacher_pconv_max", "output_threshold": out_th, **metric(eval_samples, max_maps, out_th)})

    best_fit = None
    best_fit_score = -1.0
    for seed in seeds:
        for max_area in max_areas:
            for max_side in max_sides:
                for min_fill in min_fills:
                    for min_z in min_zs:
                        for max_teacher_mean in max_teacher_means:
                            params = {
                                "seed": seed,
                                "min_area": 1.0,
                                "max_area": float(max_area),
                                "max_side": float(max_side),
                                "min_fill": min_fill,
                                "min_z": min_z,
                                "max_teacher_mean": max_teacher_mean,
                                "fill": "prob",
                            }
                            maps = render(fit_samples, params)
                            for out_th in output_thresholds:
                                m = metric(fit_samples, maps, out_th)
                                if m["mIoU"] > best_fit_score:
                                    best_fit_score = m["mIoU"]
                                    best_fit = {**params, "output_threshold": out_th, "fit_metrics": m}
    assert best_fit is not None
    eval_maps = render(eval_samples, best_fit)
    eval_metrics = metric(eval_samples, eval_maps, float(best_fit["output_threshold"]))
    rows.append({"kind": "train_fit_gate_eval", **best_fit, **eval_metrics})

    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    print("best_fit", best_fit, flush=True)
    print(
        f"best_eval kind={rows[0]['kind']} mIoU={rows[0]['mIoU']*100:.2f} "
        f"Pd={rows[0]['Pd']*100:.2f} Fa={rows[0]['Fa']*1e6:.2f}",
        flush=True,
    )
    for row in rows[:20]:
        print(
            f"{row['kind']} out={row.get('output_threshold')} mIoU={row['mIoU']*100:.2f} "
            f"Pd={row['Pd']*100:.2f} Fa={row['Fa']*1e6:.2f}",
            flush=True,
        )

    Path(args.output).write_text(
        json.dumps(
            {
                "dataset": args.dataset_name,
                "fit_datasets": fit_datasets,
                "official_pconv_mshnet_spec": args.official_pconv_mshnet_spec,
                "best_fit": best_fit,
                "best_eval": rows[0],
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
