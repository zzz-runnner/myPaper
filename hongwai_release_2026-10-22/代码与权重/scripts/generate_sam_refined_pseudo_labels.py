#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCTRANSNET = ROOT / "external" / "SCTransNet"
sys.path.insert(0, str(SCTRANSNET))
sys.path.insert(0, str(ROOT / "scripts"))

from segment_anything import SamPredictor, sam_model_registry  # noqa: E402
from train_hybrid_methods import SCTRefiner, load_model_checkpoint, make_model, weights_init_kaiming  # noqa: E402
from utils import Normalized, PadImg, get_img_norm_cfg, seed_pytorch  # noqa: E402


def parse_teacher_spec(spec: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for item in spec.split(","):
        if not item.strip():
            continue
        key, value = item.split("=", 1)
        values[key.strip()] = value.strip()
    missing = {"name", "model", "checkpoint"} - set(values)
    if missing:
        raise ValueError(f"Teacher spec missing {sorted(missing)}: {spec}")
    return values


def find_image_path(dataset_dir: Path, name: str) -> Path:
    for ext in (".png", ".bmp", ".jpg", ".jpeg"):
        path = dataset_dir / "images" / f"{name}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing image for {name} in {dataset_dir / 'images'}")


def find_mask_path(dataset_dir: Path, name: str) -> Path:
    for ext in (".png", ".bmp", ".jpg", ".jpeg"):
        path = dataset_dir / "masks" / f"{name}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing mask for {name} in {dataset_dir / 'masks'}")


def read_names(dataset_dir: Path, dataset_name: str, split: str) -> list[str]:
    with (dataset_dir / dataset_name / "img_idx" / f"{split}_{dataset_name}.txt").open("r") as f:
        return f.read().splitlines()


def read_gray(path: Path) -> np.ndarray:
    image = Image.open(path)
    array = np.array(image)
    if array.ndim == 3:
        array = np.array(image.convert("L"))
    return array.astype(np.float32)


def read_mask(path: Path) -> np.ndarray:
    mask = np.array(Image.open(path), dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask > 127).astype(np.uint8)


def gray_to_sam_rgb(gray: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(gray, [1.0, 99.5])
    if hi <= lo + 1e-6:
        lo = float(gray.min())
        hi = float(gray.max())
    if hi <= lo + 1e-6:
        img8 = np.zeros_like(gray, dtype=np.uint8)
    else:
        img8 = np.clip((gray - lo) / (hi - lo), 0.0, 1.0)
        img8 = (img8 * 255.0 + 0.5).astype(np.uint8)
    return np.repeat(img8[:, :, None], 3, axis=2)


def prepare_teacher_tensor(gray: np.ndarray, norm_cfg: dict[str, float], device: torch.device) -> torch.Tensor:
    normalized = Normalized(gray.astype(np.float32), norm_cfg)
    padded = PadImg(normalized)
    tensor = torch.from_numpy(np.ascontiguousarray(padded[None, None, :, :])).float()
    return tensor.to(device, non_blocking=True)


def load_teacher(spec: dict[str, str], device: torch.device) -> torch.nn.Module:
    model = make_model(spec["model"], warm_epoch=1, base_checkpoint=spec.get("base_checkpoint", "")).to(device)
    if isinstance(model, SCTRefiner):
        model.initialize_refiner()
    else:
        model.apply(weights_init_kaiming)
    load_model_checkpoint(model, Path(spec["checkpoint"]))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@torch.no_grad()
def predict_teacher(model: torch.nn.Module, image: torch.Tensor, h: int, w: int, use_tta: bool) -> torch.Tensor:
    def run(x: torch.Tensor) -> torch.Tensor:
        pred = model(x, epoch=10**9)
        if isinstance(pred, (tuple, list)):
            pred = pred[-1]
        return pred

    preds = [run(image)]
    if use_tta:
        preds.append(torch.flip(run(torch.flip(image, dims=[-1])), dims=[-1]))
        preds.append(torch.flip(run(torch.flip(image, dims=[-2])), dims=[-2]))
        preds.append(run(image.transpose(-1, -2)).transpose(-1, -2))
    pred = torch.stack(preds, dim=0).mean(dim=0)
    return pred[:, :, :h, :w].clamp(0.0, 1.0)


def consensus_prediction(
    teachers: list[tuple[str, torch.nn.Module]],
    image: torch.Tensor,
    h: int,
    w: int,
    use_tta: bool,
    consensus: str,
) -> tuple[np.ndarray, dict[str, float]]:
    preds = []
    means: dict[str, float] = {}
    for name, teacher in teachers:
        pred = predict_teacher(teacher, image, h, w, use_tta)
        preds.append(pred)
        means[name] = float(pred.mean().detach().cpu())
    stacked = torch.cat(preds, dim=1)
    if consensus == "max":
        pseudo = stacked.max(dim=1, keepdim=True).values
    else:
        pseudo = stacked.mean(dim=1, keepdim=True)
    return pseudo[0, 0].detach().cpu().numpy().astype(np.float32), means


def extract_components(prob: np.ndarray, args: argparse.Namespace) -> list[dict[str, object]]:
    binary = prob >= args.teacher_threshold
    if int(binary.sum()) == 0 and float(prob.max()) >= args.fallback_peak_threshold:
        relaxed = max(args.teacher_threshold * 0.55, float(prob.max()) * args.fallback_peak_ratio)
        binary = prob >= relaxed
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    components: list[dict[str, object]] = []
    h, w = prob.shape
    max_abs_area = max(1, int(args.max_component_area_ratio * h * w))
    for label_idx in range(1, num_labels):
        x, y, bw, bh, area = stats[label_idx]
        area = int(area)
        if area < args.min_component_area:
            continue
        if area > min(args.max_component_area, max_abs_area):
            continue
        comp_mask = labels == label_idx
        ys, xs = np.where(comp_mask)
        if len(xs) == 0:
            continue
        comp_prob = prob[comp_mask]
        max_pos = int(np.argmax(comp_prob))
        point = [int(xs[max_pos]), int(ys[max_pos])]
        score = float(comp_prob[max_pos])
        mean_score = float(comp_prob.mean())
        pad = int(args.box_pad)
        box = [
            max(0, int(x) - pad),
            max(0, int(y) - pad),
            min(w - 1, int(x + bw - 1) + pad),
            min(h - 1, int(y + bh - 1) + pad),
        ]
        components.append(
            {
                "label": label_idx,
                "mask": comp_mask,
                "box": box,
                "point": point,
                "area": area,
                "score": score,
                "mean_score": mean_score,
            }
        )
    components.sort(key=lambda item: (float(item["score"]), int(item["area"])), reverse=True)
    return components[: args.max_prompts]


def dilate_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return mask.astype(bool)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


def bbox_mask(shape: tuple[int, int], box: list[int]) -> np.ndarray:
    h, w = shape
    x1, y1, x2, y2 = box
    out = np.zeros((h, w), dtype=bool)
    out[max(0, y1) : min(h, y2 + 1), max(0, x1) : min(w, x2 + 1)] = True
    return out


def make_prompt_points(comp: dict[str, object], teacher_prob: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    positive = [comp["point"]]
    labels = [1]
    if args.negative_points <= 0:
        return np.array(positive, dtype=np.float32), np.array(labels, dtype=np.int32)
    h, w = teacher_prob.shape
    x1, y1, x2, y2 = [int(v) for v in comp["box"]]
    cx, cy = [int(v) for v in comp["point"]]
    candidates = [
        (x1, cy),
        (x2, cy),
        (cx, y1),
        (cx, y2),
        (x1, y1),
        (x1, y2),
        (x2, y1),
        (x2, y2),
    ]
    ranked: list[tuple[float, tuple[int, int]]] = []
    for x, y in candidates:
        x = min(max(0, x), w - 1)
        y = min(max(0, y), h - 1)
        ranked.append((float(teacher_prob[y, x]), (x, y)))
    ranked.sort(key=lambda item: item[0])
    for _, point in ranked[: args.negative_points]:
        positive.append([point[0], point[1]])
        labels.append(0)
    return np.array(positive, dtype=np.float32), np.array(labels, dtype=np.int32)


def select_sam_mask(
    masks: np.ndarray,
    sam_scores: np.ndarray,
    teacher_prob: np.ndarray,
    component_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, dict[str, float]]:
    h, w = teacher_prob.shape
    image_area = float(h * w)
    comp_area = max(1.0, float(component_mask.sum()))
    best_mask: np.ndarray | None = None
    best_meta: dict[str, float] = {"score": -1.0}
    for idx, candidate in enumerate(masks):
        mask = candidate.astype(bool)
        area = float(mask.sum())
        if area < args.min_sam_area:
            continue
        area_limit = min(args.max_sam_area_ratio * image_area, max(args.min_sam_area, comp_area * args.max_sam_area_mult))
        if area > area_limit:
            continue
        overlap = float(np.logical_and(mask, component_mask).sum())
        cover = overlap / comp_area
        teacher_mean = float(teacher_prob[mask].mean()) if area > 0 else 0.0
        teacher_mass = float(teacher_prob[mask].sum() / max(1.0, area))
        if cover < args.min_component_cover:
            continue
        if teacher_mass < args.min_mask_teacher_mean:
            continue
        area_penalty = area / max(1.0, area_limit)
        score = float(sam_scores[idx]) + 0.85 * cover + 0.45 * teacher_mean - 0.20 * area_penalty
        if score > best_meta["score"]:
            best_mask = mask
            best_meta = {
                "score": score,
                "sam_score": float(sam_scores[idx]),
                "area": area,
                "area_limit": float(area_limit),
                "cover": cover,
                "teacher_mean": teacher_mean,
                "teacher_mass": teacher_mass,
            }
    return best_mask, best_meta


def refine_with_sam(
    predictor: SamPredictor,
    rgb: np.ndarray,
    teacher_prob: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object]]:
    predictor.set_image(rgb)
    refined = teacher_prob.copy()
    sam_soft = np.zeros_like(teacher_prob, dtype=np.float32)
    components = extract_components(teacher_prob, args)
    accepted: list[dict[str, float]] = []
    rejected = 0
    for comp in components:
        point, labels = make_prompt_points(comp, teacher_prob, args)
        box = np.array(comp["box"], dtype=np.float32)
        masks, scores, _ = predictor.predict(
            point_coords=point,
            point_labels=labels,
            box=box,
            multimask_output=True,
        )
        best_mask, meta = select_sam_mask(masks, scores, teacher_prob, comp["mask"], args)
        if best_mask is None:
            rejected += 1
            continue
        guard = dilate_mask(comp["mask"], args.sam_guard_dilate) & bbox_mask(teacher_prob.shape, comp["box"])
        local_mask = best_mask & guard
        if int(local_mask.sum()) < args.min_sam_area:
            rejected += 1
            continue
        confidence = np.clip(
            args.sam_conf_base
            + args.sam_conf_score_weight * meta["sam_score"]
            + args.sam_conf_cover_weight * meta["cover"]
            + args.sam_conf_teacher_weight * meta["teacher_mean"],
            args.sam_conf_min,
            args.sam_conf_max,
        )
        sam_soft[local_mask] = np.maximum(sam_soft[local_mask], float(confidence))
        meta = dict(meta)
        meta["local_area"] = float(local_mask.sum())
        meta["confidence"] = float(confidence)
        meta["component_area"] = float(comp["area"])
        meta["component_score"] = float(comp["score"])
        accepted.append(meta)

    if args.fusion == "max":
        refined = np.maximum(refined, sam_soft)
    elif args.fusion == "blend":
        refined = np.maximum(refined * args.teacher_keep, sam_soft)
    elif args.fusion == "soft_rim":
        rim = (sam_soft > 0) & (teacher_prob < args.teacher_core_threshold)
        soft_rim = args.sam_rim_value + args.sam_rim_teacher_scale * teacher_prob
        soft_rim = np.clip(soft_rim, args.sam_rim_min, args.sam_rim_max)
        refined[rim] = np.maximum(refined[rim], soft_rim[rim])
    else:
        raise ValueError(f"Unknown fusion mode: {args.fusion}")
    refined = np.clip(refined, 0.0, 1.0)
    return refined, {
        "components": len(components),
        "accepted": len(accepted),
        "rejected": rejected,
        "accepted_meta": accepted,
        "teacher_pixels_05": int((teacher_prob >= 0.5).sum()),
        "sam_pixels": int((sam_soft > 0).sum()),
        "refined_pixels_05": int((refined >= 0.5).sum()),
    }


def update_quality(metrics: dict[str, int], prefix: str, pred: np.ndarray, gt: np.ndarray, threshold: float) -> None:
    binary = pred >= threshold
    gt_bool = gt.astype(bool)
    metrics[f"{prefix}_inter"] += int(np.logical_and(binary, gt_bool).sum())
    metrics[f"{prefix}_union"] += int(np.logical_or(binary, gt_bool).sum())
    metrics[f"{prefix}_pred"] += int(binary.sum())
    metrics[f"{prefix}_gt"] += int(gt_bool.sum())


def make_quality_metrics() -> dict[str, int]:
    return {
        "teacher_inter": 0,
        "teacher_union": 0,
        "teacher_pred": 0,
        "teacher_gt": 0,
        "refined_inter": 0,
        "refined_union": 0,
        "refined_pred": 0,
        "refined_gt": 0,
    }


def finalize_quality(metrics: dict[str, int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for prefix in ("teacher", "refined"):
        inter = metrics[f"{prefix}_inter"]
        union = metrics[f"{prefix}_union"]
        pred = metrics[f"{prefix}_pred"]
        gt = metrics[f"{prefix}_gt"]
        out[f"{prefix}_mIoU"] = 100.0 * inter / max(1, union)
        out[f"{prefix}_precision"] = 100.0 * inter / max(1, pred)
        out[f"{prefix}_recall"] = 100.0 * inter / max(1, gt)
        out[f"{prefix}_pred_pixels"] = float(pred)
    out["gt_pixels"] = float(metrics["teacher_gt"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate conservative SAM-refined teacher pseudo labels.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--datasets", required=True, help="Comma-separated dataset names.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--teacher", action="append", required=True, help="name=...,model=...,checkpoint=...,base_checkpoint=...")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sam_checkpoint", default=str(ROOT / "pretrained" / "sam" / "sam_vit_b_01ec64.pth"))
    parser.add_argument("--sam_model_type", default="vit_b")
    parser.add_argument("--use_tta", action="store_true")
    parser.add_argument("--consensus", choices=["mean", "max"], default="mean")
    parser.add_argument("--fusion", choices=["max", "blend", "soft_rim"], default="max")
    parser.add_argument("--teacher_keep", type=float, default=0.92)
    parser.add_argument("--teacher_core_threshold", type=float, default=0.50)
    parser.add_argument("--sam_rim_value", type=float, default=0.18)
    parser.add_argument("--sam_rim_teacher_scale", type=float, default=0.55)
    parser.add_argument("--sam_rim_min", type=float, default=0.20)
    parser.add_argument("--sam_rim_max", type=float, default=0.42)
    parser.add_argument("--teacher_threshold", type=float, default=0.35)
    parser.add_argument("--fallback_peak_threshold", type=float, default=0.22)
    parser.add_argument("--fallback_peak_ratio", type=float, default=0.70)
    parser.add_argument("--min_component_area", type=int, default=1)
    parser.add_argument("--max_component_area", type=int, default=600)
    parser.add_argument("--max_component_area_ratio", type=float, default=0.025)
    parser.add_argument("--max_prompts", type=int, default=12)
    parser.add_argument("--box_pad", type=int, default=8)
    parser.add_argument("--negative_points", type=int, default=4)
    parser.add_argument("--sam_guard_dilate", type=int, default=13)
    parser.add_argument("--min_sam_area", type=int, default=2)
    parser.add_argument("--max_sam_area_mult", type=float, default=14.0)
    parser.add_argument("--max_sam_area_ratio", type=float, default=0.020)
    parser.add_argument("--min_component_cover", type=float, default=0.10)
    parser.add_argument("--min_mask_teacher_mean", type=float, default=0.06)
    parser.add_argument("--sam_conf_base", type=float, default=0.35)
    parser.add_argument("--sam_conf_score_weight", type=float, default=0.22)
    parser.add_argument("--sam_conf_cover_weight", type=float, default=0.18)
    parser.add_argument("--sam_conf_teacher_weight", type=float, default=0.20)
    parser.add_argument("--sam_conf_min", type=float, default=0.55)
    parser.add_argument("--sam_conf_max", type=float, default=0.92)
    parser.add_argument("--eval_threshold", type=float, default=0.5)
    parser.add_argument("--eval_thresholds", default="", help="Optional comma-separated quality thresholds.")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard_index must be within [0, num_shards)")

    seed_pytorch(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset_root = Path(args.dataset_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    teacher_specs = [parse_teacher_spec(spec) for spec in args.teacher]
    teachers = [(spec["name"], load_teacher(spec, device)) for spec in teacher_specs]
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint).to(device)
    sam.eval()
    predictor = SamPredictor(sam)

    dataset_names = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if args.eval_thresholds.strip():
        eval_thresholds = [float(item) for item in args.eval_thresholds.split(",") if item.strip()]
    else:
        eval_thresholds = [float(args.eval_threshold)]
    summary: dict[str, object] = {
        "dataset_dir": str(dataset_root),
        "datasets": dataset_names,
        "split": args.split,
        "out_dir": str(out_root),
        "teachers": teacher_specs,
        "sam_checkpoint": args.sam_checkpoint,
        "sam_model_type": args.sam_model_type,
        "device": str(device),
        "use_tta": bool(args.use_tta),
        "consensus": args.consensus,
        "fusion": args.fusion,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "max_samples": args.max_samples,
        "samples": {},
        "quality": {},
    }

    for dataset_name in dataset_names:
        dataset_dir = dataset_root / dataset_name
        names = read_names(dataset_root, dataset_name, args.split)
        if args.max_samples > 0:
            names = names[: args.max_samples]
        shard_names = [name for idx, name in enumerate(names) if idx % args.num_shards == args.shard_index]
        norm_cfg = get_img_norm_cfg(dataset_name, str(dataset_root))
        dataset_out = out_root / dataset_name / args.split
        dataset_out.mkdir(parents=True, exist_ok=True)
        quality_by_threshold = {f"{threshold:.3f}": make_quality_metrics() for threshold in eval_thresholds}
        accepted_total = 0
        components_total = 0
        processed = 0
        for local_idx, name in enumerate(shard_names):
            out_npz = dataset_out / f"{name}.npz"
            out_json = dataset_out / f"{name}.json"
            if args.skip_existing and out_npz.exists():
                continue
            image_path = find_image_path(dataset_dir, name)
            gray = read_gray(image_path)
            h, w = gray.shape
            image_tensor = prepare_teacher_tensor(gray, norm_cfg, device)
            teacher_prob, teacher_means = consensus_prediction(
                teachers,
                image_tensor,
                h,
                w,
                args.use_tta,
                args.consensus,
            )
            refined, sample_meta = refine_with_sam(predictor, gray_to_sam_rgb(gray), teacher_prob, args)
            np.savez_compressed(out_npz, pseudo=refined.astype(np.float16))
            mask = read_mask(find_mask_path(dataset_dir, name))
            for threshold in eval_thresholds:
                metrics = quality_by_threshold[f"{threshold:.3f}"]
                update_quality(metrics, "teacher", teacher_prob, mask, threshold)
                update_quality(metrics, "refined", refined, mask, threshold)
            sample_meta.update(
                {
                    "name": name,
                    "shape": [h, w],
                    "teacher_mean": teacher_means,
                    "teacher_max": float(teacher_prob.max()),
                    "refined_max": float(refined.max()),
                }
            )
            with out_json.open("w") as f:
                json.dump(sample_meta, f, indent=2)
            accepted_total += int(sample_meta["accepted"])
            components_total += int(sample_meta["components"])
            processed += 1
            if processed % 25 == 0 or local_idx == len(shard_names) - 1:
                print(
                    f"{dataset_name} shard {args.shard_index}/{args.num_shards}: "
                    f"{processed}/{len(shard_names)} accepted={accepted_total}/{max(1, components_total)}",
                    flush=True,
                )

        summary["samples"][dataset_name] = {
            "total_before_shard": len(names),
            "processed_by_this_shard": processed,
            "accepted_components": accepted_total,
            "candidate_components": components_total,
        }
        summary["quality"][dataset_name] = {
            threshold: finalize_quality(metrics) for threshold, metrics in quality_by_threshold.items()
        }

    summary_path = out_root / (
        "summary.json" if args.num_shards == 1 else f"summary_shard{args.shard_index:02d}_of_{args.num_shards:02d}.json"
    )
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["quality"], indent=2), flush=True)


if __name__ == "__main__":
    main()
