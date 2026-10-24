#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
ANOMALY_ROOT = ROOT / "external" / "AnomalyCLIP"
sys.path.insert(0, str(ANOMALY_ROOT))
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))

import AnomalyCLIP_lib  # noqa: E402
from prompt_ensemble import AnomalyCLIP_PromptLearner  # noqa: E402
from metrics import PD_FA, mIoU  # noqa: E402

_utils_spec = importlib.util.spec_from_file_location("anomalyclip_utils", ANOMALY_ROOT / "utils.py")
if _utils_spec is None or _utils_spec.loader is None:
    raise ImportError("Could not load AnomalyCLIP utils.py")
_anomaly_utils = importlib.util.module_from_spec(_utils_spec)
_utils_spec.loader.exec_module(_anomaly_utils)
get_transform = _anomaly_utils.get_transform


def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_pair(dataset_dir: Path, dataset: str, name: str) -> tuple[Path, Path]:
    root = dataset_dir / dataset
    for ext in (".png", ".bmp", ".jpg"):
        image = root / "images" / f"{name}{ext}"
        mask = root / "masks" / f"{name}{ext}"
        if image.exists() and mask.exists():
            return image, mask
    raise FileNotFoundError(name)


def load_model(args: argparse.Namespace, device: str):
    params = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx,
    }
    model, _ = AnomalyCLIP_lib.load(args.clip_model, device=device, design_details=params)
    model.eval()
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), params)
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=args.dpam_layer)
    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return model, text_features


@torch.no_grad()
def anomaly_map_for_image(model, text_features, image_path: Path, preprocess, args: argparse.Namespace, device: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    image = preprocess(img).reshape(1, 3, args.image_size, args.image_size).to(device)
    _image_features, patch_features = model.encode_image(image, args.features_list, DPAM_layer=args.dpam_layer)
    maps = []
    for idx, patch_feature in enumerate(patch_features):
        if idx >= args.feature_map_layer:
            patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
            similarity, _ = AnomalyCLIP_lib.compute_similarity(patch_feature, text_features[0])
            similarity_map = AnomalyCLIP_lib.get_similarity_map(similarity[:, 1:, :], args.image_size)
            anomaly = (similarity_map[..., 1] + 1 - similarity_map[..., 0]) / 2.0
            maps.append(anomaly)
    anomaly_map = torch.stack(maps).sum(dim=0)
    anomaly_map = anomaly_map[0].detach().cpu().numpy()
    anomaly_map = gaussian_filter(anomaly_map, sigma=args.sigma)
    anomaly_map = cv2.resize(anomaly_map.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    lo, hi = np.percentile(anomaly_map, [1.0, 99.0])
    anomaly_map = (anomaly_map - lo) / max(hi - lo, 1e-6)
    return np.clip(anomaly_map, 0.0, 1.0).astype(np.float32)


def metric_from_maps(maps: list[np.ndarray], masks: list[np.ndarray], sizes: list[tuple[int, int]], threshold: float) -> dict[str, float]:
    iou_metric = mIoU()
    pdfa_metric = PD_FA()
    sample_ious = []
    for score, mask, size in zip(maps, masks, sizes):
        pred = score >= threshold
        gt = mask > 0
        inter = int((pred & gt).sum())
        union = int((pred | gt).sum())
        sample_ious.append(inter / float(union + np.spacing(1)))
        pred_t = torch.from_numpy(pred.astype(np.float32)).view(1, 1, *pred.shape)
        gt_t = torch.from_numpy(gt.astype(np.float32)).view(1, 1, *gt.shape)
        iou_metric.update(pred_t, gt_t)
        pdfa_metric.update(pred_t[0, 0], gt_t[0, 0], list(size))
    pix_acc, miou = iou_metric.get()
    return {
        "pixAcc": float(pix_acc),
        "mIoU": float(miou),
        "nIoU": float(np.mean(sample_ious)),
        "Pd": float(pdfa_metric.PD) / float(max(pdfa_metric.target, 1)),
        "Fa": float(pdfa_metric.dismatch_pixel) / float(max(pdfa_metric.all_pixel, 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AnomalyCLIP anomaly maps on local IRSTD datasets.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--checkpoint_path", default=str(ANOMALY_ROOT / "checkpoints" / "9_12_4_multiscale" / "epoch_15.pth"))
    parser.add_argument("--clip_model", default="ViT-L/14@336px")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--feature_map_layer", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n_ctx", type=int, default=12)
    parser.add_argument("--t_n_ctx", type=int, default=4)
    parser.add_argument("--dpam_layer", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=4.0)
    parser.add_argument("--thresholds", default="0.70,0.75,0.80,0.85,0.90,0.95")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    setup_seed(args.seed)
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    model, text_features = load_model(args, device)
    preprocess, _target_transform = get_transform(args)

    index = Path(args.dataset_dir) / args.dataset_name / "img_idx" / f"{args.split}_{args.dataset_name}.txt"
    names = [line.strip() for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_samples > 0:
        names = names[: args.max_samples]

    maps: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    sizes: list[tuple[int, int]] = []
    for idx, name in enumerate(names, start=1):
        image_path, mask_path = find_pair(Path(args.dataset_dir), args.dataset_name, name)
        score = anomaly_map_for_image(model, text_features, image_path, preprocess, args, device)
        mask = np.array(Image.open(mask_path), dtype=np.float32)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        maps.append(score)
        masks.append((mask > 0).astype(np.uint8))
        sizes.append(score.shape)
        if idx % 20 == 0 or idx == len(names):
            print(f"processed {idx}/{len(names)}", flush=True)

    rows = []
    for threshold in [float(x) for x in args.thresholds.split(",") if x.strip()]:
        rows.append({"threshold": threshold, **metric_from_maps(maps, masks, sizes, threshold)})
    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    out = {
        "dataset": args.dataset_name,
        "split": args.split,
        "num_images": len(names),
        "checkpoint": str(Path(args.checkpoint_path).resolve()),
        "best": rows[0] if rows else None,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    if rows:
        best = rows[0]
        print(
            f"best th={best['threshold']} mIoU={best['mIoU'] * 100:.2f} "
            f"Pd={best['Pd'] * 100:.2f} Fa={best['Fa'] * 1e6:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
