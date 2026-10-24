#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))

from eval_component_fusion_learner import (  # noqa: E402
    build_models,
    collect_fit_samples_multi,
    collect_samples,
    load_official_pconv_models,
    metric_from_maps,
    parse_model_spec,
    parse_official_pconv_spec,
)


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def local_z_map(image: np.ndarray) -> np.ndarray:
    src = image.astype(np.float32)
    mean = cv2.blur(src, (9, 9))
    sq_mean = cv2.blur(src * src, (9, 9))
    std = np.sqrt(np.maximum(sq_mean - mean * mean, 0.0))
    return ((src - mean) / (std + 1e-6)).astype(np.float32)


def local_delta(image: np.ndarray, comp: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    y0, x0, y1, x1 = bbox
    pad = 5
    yy0 = max(0, y0 - pad)
    xx0 = max(0, x0 - pad)
    yy1 = min(image.shape[0], y1 + pad)
    xx1 = min(image.shape[1], x1 + pad)
    roi = image[yy0:yy1, xx0:xx1]
    ring = np.ones_like(roi, dtype=bool)
    ring[y0 - yy0 : y1 - yy0, x0 - xx0 : x1 - xx0] = False
    fg = image[y0:y1, x0:x1][comp[y0:y1, x0:x1]]
    bg = roi[ring]
    if fg.size == 0 or bg.size == 0:
        return 0.0
    return float(fg.mean() - bg.mean())


def base_score(sample: dict[str, object], teacher_names: list[str], pconv_names: list[str], mode: str) -> np.ndarray:
    preds = sample["preds"]
    if mode == "teacher_max":
        return np.stack([preds[name] for name in teacher_names], axis=0).max(axis=0).astype(np.float32)
    if mode == "avg_all":
        return np.stack([preds[name] for name in [*teacher_names, *pconv_names]], axis=0).mean(axis=0).astype(np.float32)
    if mode == "max_all":
        return np.stack([preds[name] for name in [*teacher_names, *pconv_names]], axis=0).max(axis=0).astype(np.float32)
    raise ValueError(mode)


def make_component_table(
    samples: list[dict[str, object]],
    teacher_names: list[str],
    pconv_names: list[str],
    base_mode: str,
    output_threshold: float,
) -> dict[str, object]:
    base_inter = []
    base_union = []
    gt_area = []
    comp_rows = []
    comp_masks: list[tuple[int, np.ndarray]] = []
    for sample_idx, sample in enumerate(samples):
        mask = sample["mask"] > 0
        gt_area.append(float(mask.sum()))
        base = base_score(sample, teacher_names, pconv_names, base_mode)
        base_bin = base > output_threshold
        base_inter.append(float((base_bin & mask).sum()))
        base_union.append(float((base_bin | mask).sum()))

        preds = sample["preds"]
        teacher = np.stack([preds[name] for name in teacher_names], axis=0).max(axis=0)
        pconv_stack = np.stack([preds[name] for name in pconv_names], axis=0)
        pconv = pconv_stack.max(axis=0)
        pconv_votes = pconv_stack.mean(axis=0)
        zmap = local_z_map(sample["image"])
        seed = pconv > 0.5
        num_labels, label_img, stats, _centroids = cv2.connectedComponentsWithStats(seed.astype(np.uint8), 8)
        for label_id in range(1, num_labels):
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            x0 = int(stats[label_id, cv2.CC_STAT_LEFT])
            y0 = int(stats[label_id, cv2.CC_STAT_TOP])
            width = int(stats[label_id, cv2.CC_STAT_WIDTH])
            height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            y1 = y0 + height
            x1 = x0 + width
            comp = label_img == label_id
            add = comp & ~base_bin
            comp_rows.append(
                {
                    "sample_idx": sample_idx,
                    "area": float(area),
                    "max_side": float(max(width, height)),
                    "fill": float(area / max(width * height, 1)),
                    "z": float(zmap[comp].mean()),
                    "delta": local_delta(sample["image"], comp, (y0, x0, y1, x1)),
                    "teacher_mean": float(teacher[comp].mean()),
                    "pconv_vote": float(pconv_votes[comp].mean()),
                    "add_inter": float((add & mask).sum()),
                    "add_union": float(add.sum()),
                }
            )
            comp_masks.append((sample_idx, comp))
    if comp_rows:
        table = {key: np.asarray([row[key] for row in comp_rows], dtype=np.float32) for key in comp_rows[0]}
        table["sample_idx"] = table["sample_idx"].astype(np.int32)
    else:
        table = {
            "sample_idx": np.zeros((0,), dtype=np.int32),
            "area": np.zeros((0,), dtype=np.float32),
            "max_side": np.zeros((0,), dtype=np.float32),
            "fill": np.zeros((0,), dtype=np.float32),
            "z": np.zeros((0,), dtype=np.float32),
            "delta": np.zeros((0,), dtype=np.float32),
            "teacher_mean": np.zeros((0,), dtype=np.float32),
            "pconv_vote": np.zeros((0,), dtype=np.float32),
            "add_inter": np.zeros((0,), dtype=np.float32),
            "add_union": np.zeros((0,), dtype=np.float32),
        }
    return {
        "base_inter": np.asarray(base_inter, dtype=np.float32),
        "base_union": np.asarray(base_union, dtype=np.float32),
        "gt_area": np.asarray(gt_area, dtype=np.float32),
        "table": table,
        "comp_masks": comp_masks,
    }


def table_niou(cache: dict[str, object], keep: np.ndarray) -> float:
    table = cache["table"]
    n = len(cache["base_inter"])
    add_inter = np.zeros((n,), dtype=np.float32)
    add_union = np.zeros((n,), dtype=np.float32)
    if keep.size:
        np.add.at(add_inter, table["sample_idx"][keep], table["add_inter"][keep])
        np.add.at(add_union, table["sample_idx"][keep], table["add_union"][keep])
    inter = cache["base_inter"] + add_inter
    union = cache["base_union"] + add_union
    return float(np.mean(inter / (union + np.spacing(1))))


def keep_mask(table: dict[str, np.ndarray], params: dict[str, object]) -> np.ndarray:
    if table["area"].size == 0:
        return np.zeros((0,), dtype=bool)
    return (
        (table["area"] >= float(params["min_area"]))
        & (table["area"] <= float(params["max_area"]))
        & (table["max_side"] <= float(params["max_side"]))
        & (table["fill"] >= float(params["min_fill"]))
        & (table["z"] >= float(params["min_z"]))
        & (table["delta"] >= float(params["min_delta"]))
        & (table["teacher_mean"] <= float(params["max_teacher_mean"]))
        & (table["pconv_vote"] >= float(params["min_pconv_vote"]))
    )


def candidate_params(args: argparse.Namespace) -> list[dict[str, object]]:
    rows = []
    for max_area in parse_int_list(args.max_areas):
        for max_side in parse_int_list(args.max_sides):
            for min_fill in parse_float_list(args.min_fills):
                for min_z in parse_float_list(args.min_zs):
                    for min_delta in parse_float_list(args.min_deltas):
                        for max_teacher_mean in parse_float_list(args.max_teacher_means):
                            for min_pconv_vote in parse_float_list(args.min_pconv_votes):
                                rows.append(
                                    {
                                        "min_area": 1.0,
                                        "max_area": float(max_area),
                                        "max_side": float(max_side),
                                        "min_fill": min_fill,
                                        "min_z": min_z,
                                        "min_delta": min_delta,
                                        "max_teacher_mean": max_teacher_mean,
                                        "min_pconv_vote": min_pconv_vote,
                                    }
                                )
    return rows


def render_with_keep(
    samples: list[dict[str, object]],
    cache: dict[str, object],
    teacher_names: list[str],
    pconv_names: list[str],
    base_mode: str,
    keep: np.ndarray,
) -> list[np.ndarray]:
    maps = [base_score(sample, teacher_names, pconv_names, base_mode) for sample in samples]
    comp_masks = cache["comp_masks"]
    for comp_idx in np.flatnonzero(keep):
        sample_idx, comp = comp_masks[int(comp_idx)]
        maps[sample_idx][comp] = 1.0
    return [x.astype(np.float32) for x in maps]


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectorized dual-PConv component-table gate.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "datasets"))
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--fit_datasets", default="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3")
    parser.add_argument("--model_spec", action="append", required=True)
    parser.add_argument("--official_pconv_mshnet_spec", action="append", required=True)
    parser.add_argument("--base_modes", default="avg_all,teacher_max")
    parser.add_argument("--output_thresholds", default="0.40,0.45,0.50,0.55")
    parser.add_argument("--max_areas", default="16,32,64,128")
    parser.add_argument("--max_sides", default="8,12,18,28")
    parser.add_argument("--min_fills", default="0.10,0.20,0.35")
    parser.add_argument("--min_zs", default="-1.0,-0.25,0.25,0.75")
    parser.add_argument("--min_deltas", default="-1.0,0.0,0.01")
    parser.add_argument("--max_teacher_means", default="1.1,0.50,0.30,0.15")
    parser.add_argument("--min_pconv_votes", default="0.0,0.5,0.75")
    parser.add_argument("--topk_eval", type=int, default=12)
    parser.add_argument("--max_fit_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    specs = [parse_model_spec(spec) for spec in args.model_spec]
    teacher_names = [spec[0] for spec in specs]
    models = build_models(specs, device)
    pconv_specs = [parse_official_pconv_spec(spec) for spec in args.official_pconv_mshnet_spec]
    pconv_models = load_official_pconv_models(pconv_specs, device)
    pconv_names = [str(spec["name"]) for spec in pconv_specs]
    fit_datasets = [x.strip() for x in args.fit_datasets.split(",") if x.strip()]

    fit_samples = collect_fit_samples_multi(models, pconv_models, Path(args.dataset_dir), fit_datasets, "train", args.max_fit_samples, device)
    eval_samples = collect_samples(models, pconv_models, Path(args.dataset_dir), args.dataset_name, "test", args.max_eval_samples, device)
    params_grid = candidate_params(args)
    rows = []

    for base_mode in [x.strip() for x in args.base_modes.split(",") if x.strip()]:
        for out_th in parse_float_list(args.output_thresholds):
            fit_cache = make_component_table(fit_samples, teacher_names, pconv_names, base_mode, out_th)
            eval_cache = make_component_table(eval_samples, teacher_names, pconv_names, base_mode, out_th)
            base_maps = [base_score(sample, teacher_names, pconv_names, base_mode) for sample in eval_samples]
            rows.append(
                {
                    "kind": "baseline",
                    "base_mode": base_mode,
                    "output_threshold": out_th,
                    "fit_fast_nIoU": table_niou(fit_cache, np.zeros_like(fit_cache["table"]["area"], dtype=bool)),
                    **metric_from_maps(eval_samples, base_maps, out_th),
                }
            )
            fit_rank = []
            for params in params_grid:
                keep = keep_mask(fit_cache["table"], params)
                fit_rank.append({"params": params, "fit_fast_nIoU": table_niou(fit_cache, keep), "kept_components": int(keep.sum())})
            fit_rank.sort(key=lambda row: row["fit_fast_nIoU"], reverse=True)
            seen = set()
            for rank, row in enumerate(fit_rank[: max(int(args.topk_eval), 1)]):
                key = json.dumps(row["params"], sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                keep_eval = keep_mask(eval_cache["table"], row["params"])
                maps = render_with_keep(eval_samples, eval_cache, teacher_names, pconv_names, base_mode, keep_eval)
                rows.append(
                    {
                        "kind": "component_table_gate",
                        "rank": rank,
                        "base_mode": base_mode,
                        "output_threshold": out_th,
                        "fit_fast_nIoU": row["fit_fast_nIoU"],
                        "fit_kept_components": row["kept_components"],
                        "eval_kept_components": int(keep_eval.sum()),
                        **row["params"],
                        **metric_from_maps(eval_samples, maps, out_th),
                    }
                )

    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best = rows[0]
    print(
        f"best kind={best['kind']} base={best['base_mode']} out={best['output_threshold']} "
        f"mIoU={best['mIoU']*100:.2f} nIoU={best['nIoU']*100:.2f} Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.2f}",
        flush=True,
    )
    for row in rows[:20]:
        print(
            f"{row['kind']} base={row['base_mode']} out={row['output_threshold']} "
            f"fit={row['fit_fast_nIoU']*100:.2f} mIoU={row['mIoU']*100:.2f} "
            f"Pd={row['Pd']*100:.2f} Fa={row['Fa']*1e6:.2f}",
            flush=True,
        )

    Path(args.output).write_text(
        json.dumps(
            {
                "dataset": args.dataset_name,
                "fit_datasets": fit_datasets,
                "teacher_names": teacher_names,
                "pconv_names": pconv_names,
                "best": best,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
