#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))

from eval_component_fusion_learner import metric_from_maps  # noqa: E402
from inmemory_cached_weight_search import load_cache_dense  # noqa: E402


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def cache_dataset_name(path: Path) -> str:
    return path.name.split("__", 1)[0]


def stable_bucket(name: str, seed: str) -> float:
    raw = hashlib.sha1(f"{seed}:{name}".encode("utf-8")).hexdigest()[:8]
    return int(raw, 16) / float(0xFFFFFFFF)


def load_split_cache(path: Path, dataset_dir: Path):
    samples, names, preds, masks = load_cache_dense(path)
    dataset = cache_dataset_name(path)
    for sample in samples:
        sample["dataset"] = dataset
        sample["cache_id"] = f"{dataset}:{sample['name']}"
    images = [load_raw_image(dataset_dir, dataset, str(sample["name"]), tuple(sample["size"])) for sample in samples]
    return samples, names, preds, masks, images


def load_many(paths: list[str], dataset_dir: Path):
    all_samples = []
    all_preds = []
    all_masks = []
    all_images = []
    model_names = None
    for raw in paths:
        samples, names, preds, masks, images = load_split_cache(Path(raw), dataset_dir)
        if model_names is None:
            model_names = names
        elif names != model_names:
            raise ValueError(f"model names mismatch for {raw}: {names} != {model_names}")
        all_samples.extend(samples)
        all_preds.extend(preds)
        all_masks.extend(masks)
        all_images.extend(images)
    if model_names is None:
        raise ValueError("no caches")
    return all_samples, model_names, all_preds, all_masks, all_images


def load_raw_image(dataset_dir: Path, dataset: str, name: str, size: tuple[int, int]) -> np.ndarray:
    root = dataset_dir / dataset / "images"
    for ext in (".png", ".bmp", ".jpg"):
        path = root / f"{name}{ext}"
        if path.exists():
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                break
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image.astype(np.float32)
            if image.shape[:2] != size:
                image = cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
            lo, hi = np.percentile(image, [1.0, 99.5])
            image = (image - lo) / max(float(hi - lo), 1e-6)
            return np.clip(image, 0.0, 1.0).astype(np.float32)
    raise FileNotFoundError(f"{dataset}/{name}")


def base_map(stack: np.ndarray, names: list[str], mode: str) -> np.ndarray:
    if mode == "avg":
        return stack.mean(axis=0).astype(np.float32)
    if mode == "max":
        return stack.max(axis=0).astype(np.float32)
    if mode in names:
        return stack[names.index(mode)].astype(np.float32)
    parts = [x.strip() for x in mode.split("+") if x.strip()]
    if parts and all(part in names for part in parts):
        return stack[[names.index(part) for part in parts]].mean(axis=0).astype(np.float32)
    raise ValueError(f"unknown base mode {mode}; names={names}")


def image_feature_maps(image: np.ndarray) -> dict[str, np.ndarray]:
    blur7 = cv2.GaussianBlur(image, (0, 0), 1.2)
    blur21 = cv2.GaussianBlur(image, (0, 0), 4.0)
    local_mean = cv2.blur(image, (17, 17))
    local_sq = cv2.blur(image * image, (17, 17))
    local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 1e-6))
    z = (image - local_mean) / (local_std + 1e-6)
    dog = blur7 - blur21
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    lap = np.abs(cv2.Laplacian(image, cv2.CV_32F, ksize=3))
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    return {
        "z": z.astype(np.float32),
        "dog": dog.astype(np.float32),
        "tophat": tophat.astype(np.float32),
        "lap": lap.astype(np.float32),
        "grad": grad.astype(np.float32),
    }


def ring_mask(component: np.ndarray, radius: int = 7) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius, radius))
    dilated = cv2.dilate(component.astype(np.uint8), kernel, iterations=1).astype(bool)
    return dilated & ~component


def safe_mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def build_component_table(samples, preds, masks, images, names, base_mode: str, seed_th: float, out_th: float):
    rows = []
    comp_refs = []
    bases = []
    base_inter = []
    base_union = []
    for sample_idx, (sample, stack, mask, image) in enumerate(zip(samples, preds, masks, images)):
        gt = mask.astype(bool)
        pred = base_map(stack, names, base_mode)
        bases.append(pred.copy())
        pred_bin = pred > out_th
        base_inter.append(float((pred_bin & gt).sum()))
        base_union.append(float((pred_bin | gt).sum()))
        feats = image_feature_maps(image)
        seed = pred > seed_th
        num_labels, label_img, stats, _ = cv2.connectedComponentsWithStats(seed.astype(np.uint8), 8)
        for label_id in range(1, num_labels):
            comp = label_img == label_id
            active = comp & pred_bin
            if not active.any():
                continue
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            width = int(stats[label_id, cv2.CC_STAT_WIDTH])
            height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            ring = ring_mask(comp)
            comp_vals = image[comp]
            ring_vals = image[ring]
            model_means = stack[:, comp].mean(axis=1)
            tp_removed = float((active & gt).sum())
            fp_removed = float((active & ~gt).sum())
            rows.append(
                {
                    "sample_idx": float(sample_idx),
                    "area": float(area),
                    "max_side": float(max(width, height)),
                    "fill": float(area / max(width * height, 1)),
                    "base_mean": float(pred[comp].mean()),
                    "base_max": float(pred[comp].max()),
                    "model_min": float(model_means.min()),
                    "model_std": float(model_means.std()),
                    "intensity": safe_mean(comp_vals),
                    "ring_contrast": safe_mean(comp_vals) - safe_mean(ring_vals),
                    "z_mean": safe_mean(feats["z"][comp]),
                    "dog_mean": safe_mean(feats["dog"][comp]),
                    "tophat_mean": safe_mean(feats["tophat"][comp]),
                    "lap_mean": safe_mean(feats["lap"][comp]),
                    "grad_mean": safe_mean(feats["grad"][comp]),
                    "tp_removed": tp_removed,
                    "fp_removed": fp_removed,
                }
            )
            comp_refs.append((sample_idx, comp))
    if rows:
        table = {key: np.asarray([row[key] for row in rows], dtype=np.float32) for key in rows[0]}
        table["sample_idx"] = table["sample_idx"].astype(np.int32)
    else:
        keys = [
            "sample_idx",
            "area",
            "max_side",
            "fill",
            "base_mean",
            "base_max",
            "model_min",
            "model_std",
            "intensity",
            "ring_contrast",
            "z_mean",
            "dog_mean",
            "tophat_mean",
            "lap_mean",
            "grad_mean",
            "tp_removed",
            "fp_removed",
        ]
        table = {key: np.zeros((0,), dtype=np.float32) for key in keys}
        table["sample_idx"] = table["sample_idx"].astype(np.int32)
    return {
        "base_maps": bases,
        "base_inter": np.asarray(base_inter, dtype=np.float32),
        "base_union": np.asarray(base_union, dtype=np.float32),
        "table": table,
        "comp_refs": comp_refs,
    }


def keep_mask(table: dict[str, np.ndarray], params: dict[str, float]) -> np.ndarray:
    if table["area"].size == 0:
        return np.zeros((0,), dtype=bool)
    if params.get("noop", 0.0) > 0.5:
        return np.ones_like(table["area"], dtype=bool)
    return (
        (table["area"] >= params["min_area"])
        & (table["area"] <= params["max_area"])
        & (table["max_side"] <= params["max_side"])
        & (table["fill"] >= params["min_fill"])
        & (table["base_mean"] >= params["min_base_mean"])
        & (table["base_max"] >= params["min_base_max"])
        & (table["model_min"] >= params["min_model_min"])
        & (table["model_std"] <= params["max_model_std"])
        & (table["ring_contrast"] >= params["min_ring_contrast"])
        & (table["z_mean"] >= params["min_z"])
        & (table["dog_mean"] >= params["min_dog"])
        & (table["tophat_mean"] >= params["min_tophat"])
    )


def table_niou(cache: dict[str, object], keep: np.ndarray) -> float:
    table = cache["table"]
    remove = ~keep
    n = len(cache["base_inter"])
    lost_tp = np.zeros((n,), dtype=np.float32)
    lost_fp = np.zeros((n,), dtype=np.float32)
    if remove.size:
        np.add.at(lost_tp, table["sample_idx"][remove], table["tp_removed"][remove])
        np.add.at(lost_fp, table["sample_idx"][remove], table["fp_removed"][remove])
    inter = cache["base_inter"] - lost_tp
    union = cache["base_union"] - lost_fp
    return float(np.mean(inter / (union + np.spacing(1))))


def render_maps(cache: dict[str, object], keep: np.ndarray) -> list[np.ndarray]:
    maps = [x.copy() for x in cache["base_maps"]]
    for comp_idx in np.flatnonzero(~keep):
        sample_idx, comp = cache["comp_refs"][int(comp_idx)]
        maps[sample_idx][comp] = 0.0
    return [x.astype(np.float32) for x in maps]


def candidate_params(args: argparse.Namespace):
    for min_area in parse_int_list(args.min_areas):
        for max_area in parse_int_list(args.max_areas):
            for max_side in parse_int_list(args.max_sides):
                for min_fill in parse_float_list(args.min_fills):
                    for min_base_mean in parse_float_list(args.min_base_means):
                        for min_base_max in parse_float_list(args.min_base_maxes):
                            for min_model_min in parse_float_list(args.min_model_mins):
                                for max_model_std in parse_float_list(args.max_model_stds):
                                    for min_ring_contrast in parse_float_list(args.min_ring_contrasts):
                                        for min_z in parse_float_list(args.min_zs):
                                            for min_dog in parse_float_list(args.min_dogs):
                                                for min_tophat in parse_float_list(args.min_tophats):
                                                    yield {
                                                        "min_area": float(min_area),
                                                        "max_area": float(max_area),
                                                        "max_side": float(max_side),
                                                        "min_fill": float(min_fill),
                                                        "min_base_mean": float(min_base_mean),
                                                        "min_base_max": float(min_base_max),
                                                        "min_model_min": float(min_model_min),
                                                        "max_model_std": float(max_model_std),
                                                        "min_ring_contrast": float(min_ring_contrast),
                                                        "min_z": float(min_z),
                                                        "min_dog": float(min_dog),
                                                        "min_tophat": float(min_tophat),
                                                    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-selected anomaly-prior suppression on cached IRSTD predictions.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--train_cache", action="append", required=True)
    parser.add_argument("--test_cache", required=True)
    parser.add_argument("--base_modes", default="avg")
    parser.add_argument("--seed_thresholds", default="0.45,0.50")
    parser.add_argument("--output_thresholds", default="0.45,0.50,0.55")
    parser.add_argument("--val_fraction", type=float, default=0.25)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--split_seed", default="adaptive-anomaly-20260614")
    parser.add_argument("--min_areas", default="1,2")
    parser.add_argument("--max_areas", default="64,128,512")
    parser.add_argument("--max_sides", default="24,40,64")
    parser.add_argument("--min_fills", default="0.00,0.15,0.30")
    parser.add_argument("--min_base_means", default="0.00,0.35,0.50")
    parser.add_argument("--min_base_maxes", default="0.00,0.55,0.70")
    parser.add_argument("--min_model_mins", default="0.00,0.20,0.35")
    parser.add_argument("--max_model_stds", default="1.00,0.20,0.08")
    parser.add_argument("--min_ring_contrasts", default="-1.00,0.00,0.03,0.06")
    parser.add_argument("--min_zs", default="-9.00,0.00,0.50")
    parser.add_argument("--min_dogs", default="-1.00,0.00,0.02")
    parser.add_argument("--min_tophats", default="-1.00,0.00,0.02")
    parser.add_argument("--val_topk", type=int, default=24)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    train_samples, names, train_preds, train_masks, train_images = load_many(args.train_cache, dataset_dir)
    test_samples, test_names, test_preds, test_masks, test_images = load_many([args.test_cache], dataset_dir)
    if test_names != names:
        raise ValueError(f"model names mismatch: {test_names} != {names}")

    val_keep = [stable_bucket(str(sample["cache_id"]), args.split_seed) < args.val_fraction for sample in train_samples]
    val_samples = [sample for sample, keep in zip(train_samples, val_keep) if keep]
    val_preds = [pred for pred, keep in zip(train_preds, val_keep) if keep]
    val_masks = [mask for mask, keep in zip(train_masks, val_keep) if keep]
    val_images = [image for image, keep in zip(train_images, val_keep) if keep]
    if args.max_val_samples > 0:
        val_samples = val_samples[: args.max_val_samples]
        val_preds = val_preds[: args.max_val_samples]
        val_masks = val_masks[: args.max_val_samples]
        val_images = val_images[: args.max_val_samples]
    if args.max_test_samples > 0:
        test_samples = test_samples[: args.max_test_samples]
        test_preds = test_preds[: args.max_test_samples]
        test_masks = test_masks[: args.max_test_samples]
        test_images = test_images[: args.max_test_samples]

    val_rows = []
    for base_mode in [x.strip() for x in args.base_modes.split(",") if x.strip()]:
        for seed_th in parse_float_list(args.seed_thresholds):
            for out_th in parse_float_list(args.output_thresholds):
                cache = build_component_table(val_samples, val_preds, val_masks, val_images, names, base_mode, seed_th, out_th)
                for params in candidate_params(args):
                    keep = keep_mask(cache["table"], params)
                    val_rows.append(
                        {
                            "base_mode": base_mode,
                            "seed_threshold": seed_th,
                            "output_threshold": out_th,
                            **params,
                            "val_fast_nIoU": table_niou(cache, keep),
                            "val_components": int(cache["table"]["area"].size),
                            "val_kept": int(keep.sum()),
                        }
                    )
                keep = np.ones_like(cache["table"]["area"], dtype=bool)
                val_rows.append(
                    {
                        "base_mode": base_mode,
                        "seed_threshold": seed_th,
                        "output_threshold": out_th,
                        "noop": 1.0,
                        "min_area": 0.0,
                        "max_area": 1e9,
                        "max_side": 1e9,
                        "min_fill": 0.0,
                        "min_base_mean": 0.0,
                        "min_base_max": 0.0,
                        "min_model_min": 0.0,
                        "max_model_std": 1e9,
                        "min_ring_contrast": -1e9,
                        "min_z": -1e9,
                        "min_dog": -1e9,
                        "min_tophat": -1e9,
                        "val_fast_nIoU": table_niou(cache, keep),
                        "val_components": int(cache["table"]["area"].size),
                        "val_kept": int(keep.sum()),
                    }
                )
    val_rows.sort(key=lambda row: row["val_fast_nIoU"], reverse=True)

    test_rows = []
    cache_by_key = {}
    for row in val_rows[: max(int(args.val_topk), 1)]:
        key = (row["base_mode"], float(row["seed_threshold"]), float(row["output_threshold"]))
        if key not in cache_by_key:
            cache_by_key[key] = build_component_table(
                test_samples, test_preds, test_masks, test_images, names, key[0], key[1], key[2]
            )
        cache = cache_by_key[key]
        params = {
            key: float(row[key])
            for key in [
                "min_area",
                "max_area",
                "max_side",
                "min_fill",
                "min_base_mean",
                "min_base_max",
                "min_model_min",
                "max_model_std",
                "min_ring_contrast",
                "min_z",
                "min_dog",
                "min_tophat",
            ]
        }
        params["noop"] = float(row.get("noop", 0.0))
        keep = keep_mask(cache["table"], params)
        maps = render_maps(cache, keep)
        test_rows.append(
            {
                **row,
                "test_components": int(cache["table"]["area"].size),
                "test_kept": int(keep.sum()),
                **{f"test_{k}": v for k, v in metric_from_maps(test_samples, maps, float(row["output_threshold"])).items()},
            }
        )
    selected = test_rows[0]
    best_ref = sorted(test_rows, key=lambda row: row["test_mIoU"], reverse=True)[0]
    print(
        f"selected base={selected['base_mode']} seed={selected['seed_threshold']} out={selected['output_threshold']} "
        f"val_nIoU={selected['val_fast_nIoU']*100:.2f} test_mIoU={selected['test_mIoU']*100:.2f} "
        f"Pd={selected['test_Pd']*100:.2f} Fa={selected['test_Fa']*1e6:.2f}",
        flush=True,
    )
    print(
        f"best_test_in_val_topk base={best_ref['base_mode']} seed={best_ref['seed_threshold']} out={best_ref['output_threshold']} "
        f"test_mIoU={best_ref['test_mIoU']*100:.2f} Pd={best_ref['test_Pd']*100:.2f} Fa={best_ref['test_Fa']*1e6:.2f}",
        flush=True,
    )
    Path(args.output).write_text(
        json.dumps(
            {
                "train_caches": args.train_cache,
                "test_cache": args.test_cache,
                "models": names,
                "val_fraction": args.val_fraction,
                "split_seed": args.split_seed,
                "train_samples": len(train_samples),
                "val_samples": len(val_samples),
                "test_samples": len(test_samples),
                "selected_by_val_fast_nIoU": selected,
                "best_test_in_val_topk_reference": best_ref,
                "val_rows": val_rows,
                "test_rows": test_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
