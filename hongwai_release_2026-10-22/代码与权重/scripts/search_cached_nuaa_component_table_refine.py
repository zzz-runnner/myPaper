#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_cached_prediction_fusion import load_cache  # noqa: E402
from eval_component_fusion_learner import metric_from_maps  # noqa: E402


@dataclass(frozen=True)
class Rule:
    base_mode: str
    threshold: float
    min_area: int
    max_area: int
    max_side: int
    min_mean: float
    min_peak: float
    min_support: int
    min_bgm: float
    min_rpcanet: float


def parse_csv(value: str, cast):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def load_cache_limited(path: Path, max_samples: int = 0) -> tuple[list[dict[str, object]], list[str]]:
    samples, names = load_cache(path)
    if max_samples > 0:
        samples = samples[:max_samples]
    return samples, names


def load_many(paths: list[Path], max_samples_per_cache: int = 0) -> tuple[list[dict[str, object]], list[str]]:
    out: list[dict[str, object]] = []
    names_ref: list[str] | None = None
    for path in paths:
        samples, names = load_cache_limited(path, max_samples_per_cache)
        if names_ref is None:
            names_ref = names
        elif names_ref != names:
            raise ValueError(f"cache name mismatch: {path}")
        out.extend(samples)
    if names_ref is None:
        raise ValueError("no caches")
    return out, names_ref


def component_features(sample: dict[str, object], base_mode: str, threshold: float) -> list[dict[str, object]]:
    preds = sample["preds"]
    base = preds[base_mode].astype(np.float32)
    binary = (base >= threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    support_names = [name for name in ("tpc_a", "tpc_b", "lca", "bgm_nuaa", "rpcanet_sirst_s3") if name in preds]
    support_stack = [(preds[name].astype(np.float32) >= 0.5).astype(np.uint8) for name in support_names]
    support = np.sum(np.stack(support_stack, axis=0), axis=0) if support_stack else np.zeros_like(binary)
    bgm = preds.get("bgm_nuaa", base).astype(np.float32)
    rpcanet = preds.get("rpcanet_sirst_s3", base).astype(np.float32)
    comps: list[dict[str, object]] = []
    for label_id in range(1, num_labels):
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        mask = labels[y : y + h, x : x + w] == label_id
        vals = base[y : y + h, x : x + w][mask]
        comps.append(
            {
                "bbox": [x, y, w, h],
                "mask": mask,
                "area": area,
                "side": max(w, h),
                "mean": float(vals.mean()) if vals.size else 0.0,
                "peak": float(vals.max()) if vals.size else 0.0,
                "support": int(support[y : y + h, x : x + w][mask].max()) if vals.size else 0,
                "bgm": float(bgm[y : y + h, x : x + w][mask].max()) if vals.size else 0.0,
                "rpcanet": float(rpcanet[y : y + h, x : x + w][mask].max()) if vals.size else 0.0,
            }
        )
    return comps


def build_tables(samples: list[dict[str, object]], base_modes: list[str], thresholds: list[float]) -> dict[tuple[str, float], list[list[dict[str, object]]]]:
    tables: dict[tuple[str, float], list[list[dict[str, object]]]] = {}
    for mode in base_modes:
        for threshold in thresholds:
            key = (mode, threshold)
            tables[key] = [component_features(sample, mode, threshold) for sample in samples]
            print(f"built table {mode} th={threshold} samples={len(samples)}", flush=True)
    return tables


def keep(comp: dict[str, object], rule: Rule) -> bool:
    area = int(comp["area"])
    if area < rule.min_area:
        return False
    if rule.max_area > 0 and area > rule.max_area:
        return False
    if rule.max_side > 0 and int(comp["side"]) > rule.max_side:
        return False
    return (
        float(comp["mean"]) >= rule.min_mean
        and float(comp["peak"]) >= rule.min_peak
        and int(comp["support"]) >= rule.min_support
        and float(comp["bgm"]) >= rule.min_bgm
        and float(comp["rpcanet"]) >= rule.min_rpcanet
    )


def render_from_table(samples: list[dict[str, object]], table: list[list[dict[str, object]]], rule: Rule) -> list[np.ndarray]:
    maps: list[np.ndarray] = []
    for sample, comps in zip(samples, table):
        h, w = sample["mask"].shape
        out = np.zeros((h, w), dtype=np.float32)
        for comp in comps:
            if not keep(comp, rule):
                continue
            x, y, bw, bh = [int(v) for v in comp["bbox"]]
            mask = comp["mask"]
            out[y : y + bh, x : x + bw][mask] = 1.0
        maps.append(out)
    return maps


def fast_metrics(samples: list[dict[str, object]], table: list[list[dict[str, object]]], rule: Rule) -> dict[str, float]:
    inter = 0.0
    pred_sum = 0.0
    gt_sum = 0.0
    kept_components = 0
    for sample, comps in zip(samples, table):
        gt = sample["mask"].astype(bool)
        pred = np.zeros(gt.shape, dtype=bool)
        for comp in comps:
            if not keep(comp, rule):
                continue
            kept_components += 1
            x, y, bw, bh = [int(v) for v in comp["bbox"]]
            pred[y : y + bh, x : x + bw][comp["mask"]] = True
        inter += float(np.logical_and(pred, gt).sum())
        pred_sum += float(pred.sum())
        gt_sum += float(gt.sum())
    union = pred_sum + gt_sum - inter
    iou = inter / union if union > 0 else 1.0
    precision = inter / pred_sum if pred_sum > 0 else 0.0
    recall = inter / gt_sum if gt_sum > 0 else 0.0
    fp_pixels = max(pred_sum - inter, 0.0)
    total_pixels = float(sum(sample["mask"].size for sample in samples))
    return {
        "fast_iou": iou,
        "fast_precision": precision,
        "fast_recall": recall,
        "fast_fp_per_million": fp_pixels / max(total_pixels, 1.0) * 1e6,
        "kept_components": float(kept_components),
    }


def score(metrics: dict[str, float], fa_penalty: float, pd_bonus: float) -> float:
    return float(metrics["mIoU"]) - fa_penalty * float(metrics["Fa"]) * 1e6 + pd_bonus * float(metrics["Pd"])


def fast_score(metrics: dict[str, float], fp_penalty: float, recall_bonus: float) -> float:
    return float(metrics["fast_iou"]) - fp_penalty * float(metrics["fast_fp_per_million"]) + recall_bonus * float(metrics["fast_recall"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast cached component-table refinement for NUAA strong fusion.")
    parser.add_argument("--fit_cache", action="append", required=True)
    parser.add_argument("--test_cache", required=True)
    parser.add_argument("--base_modes", default="weighted_raw,strong_fusion")
    parser.add_argument("--thresholds", default="0.46,0.48,0.50,0.52,0.54,0.56")
    parser.add_argument("--min_areas", default="1,2,3,4,5")
    parser.add_argument("--max_areas", default="0,64,128")
    parser.add_argument("--max_sides", default="0,16,24,32")
    parser.add_argument("--min_means", default="0.0,0.40,0.45,0.50")
    parser.add_argument("--min_peaks", default="0.0,0.52,0.56,0.60")
    parser.add_argument("--min_supports", default="0,1,2")
    parser.add_argument("--min_bgms", default="0.0,0.25,0.30")
    parser.add_argument("--min_rpcanets", default="0.0,0.30,0.40")
    parser.add_argument("--max_fit_samples_per_cache", type=int, default=0)
    parser.add_argument("--top_fit", type=int, default=48)
    parser.add_argument("--fast_top", type=int, default=256)
    parser.add_argument("--fa_penalty", type=float, default=0.0000015)
    parser.add_argument("--pd_bonus", type=float, default=0.0)
    parser.add_argument("--fast_fp_penalty", type=float, default=0.0000005)
    parser.add_argument("--fast_recall_bonus", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base_modes = parse_csv(args.base_modes, str)
    thresholds = parse_csv(args.thresholds, float)
    fit_samples, fit_names = load_many([Path(p) for p in args.fit_cache], args.max_fit_samples_per_cache)
    test_samples, test_names = load_cache_limited(Path(args.test_cache), 0)
    if fit_names != test_names:
        raise ValueError("fit/test names mismatch")

    fit_tables = build_tables(fit_samples, base_modes, thresholds)
    test_tables = build_tables(test_samples, base_modes, thresholds)

    rules: list[Rule] = []
    for base_mode in base_modes:
        for threshold in thresholds:
            for min_area in parse_csv(args.min_areas, int):
                for max_area in parse_csv(args.max_areas, int):
                    for max_side in parse_csv(args.max_sides, int):
                        for min_mean in parse_csv(args.min_means, float):
                            for min_peak in parse_csv(args.min_peaks, float):
                                for min_support in parse_csv(args.min_supports, int):
                                    for min_bgm in parse_csv(args.min_bgms, float):
                                        for min_rpcanet in parse_csv(args.min_rpcanets, float):
                                            rules.append(Rule(base_mode, threshold, min_area, max_area, max_side, min_mean, min_peak, min_support, min_bgm, min_rpcanet))

    fast_rows = []
    for idx, rule in enumerate(rules, 1):
        metrics = fast_metrics(fit_samples, fit_tables[(rule.base_mode, rule.threshold)], rule)
        fast_rows.append({"rule": asdict(rule), **metrics, "fast_score": fast_score(metrics, args.fast_fp_penalty, args.fast_recall_bonus)})
        if idx % 1000 == 0:
            print(f"scored {idx}/{len(rules)}", flush=True)
    fast_rows.sort(key=lambda row: row["fast_score"], reverse=True)

    fit_rows = []
    for row in fast_rows[: args.fast_top]:
        rule = Rule(**row["rule"])
        maps = render_from_table(fit_samples, fit_tables[(rule.base_mode, rule.threshold)], rule)
        metrics = metric_from_maps(fit_samples, maps, 0.5)
        fit_rows.append({"rule": asdict(rule), "fast": {k: row[k] for k in ("fast_iou", "fast_precision", "fast_recall", "fast_fp_per_million", "kept_components", "fast_score")}, **metrics, "score": score(metrics, args.fa_penalty, args.pd_bonus)})
    fit_rows.sort(key=lambda row: row["score"], reverse=True)

    eval_rows = []
    for row in fit_rows[: args.top_fit]:
        rule = Rule(**row["rule"])
        maps = render_from_table(test_samples, test_tables[(rule.base_mode, rule.threshold)], rule)
        metrics = metric_from_maps(test_samples, maps, 0.5)
        eval_rows.append({"rule": asdict(rule), "fit": {k: row[k] for k in ("mIoU", "nIoU", "Pd", "Fa", "score")}, "test": metrics})
    eval_rows.sort(key=lambda row: row["test"]["mIoU"], reverse=True)

    baseline = []
    for mode in base_modes:
        for threshold in thresholds:
            maps = [sample["preds"][mode].astype(np.float32) for sample in test_samples]
            baseline.append({"mode": mode, "threshold": threshold, **metric_from_maps(test_samples, maps, threshold)})
    baseline.sort(key=lambda row: row["mIoU"], reverse=True)

    payload = {
        "num_fit_samples": len(fit_samples),
        "num_test_samples": len(test_samples),
        "num_rules": len(rules),
        "baseline_best": baseline[0],
        "selected_by_fit": eval_rows[0] if eval_rows else None,
        "eval_rows": eval_rows,
        "top_fit": fit_rows[: args.top_fit],
        "top_fast": fast_rows[: min(args.fast_top, 64)],
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if eval_rows:
        t = eval_rows[0]["test"]
        print(f"best selected-test mIoU={t['mIoU']*100:.4f} Pd={t['Pd']*100:.2f} Fa={t['Fa']*1e6:.4f}", flush=True)
    print(f"baseline mIoU={baseline[0]['mIoU']*100:.4f}", flush=True)


if __name__ == "__main__":
    main()
