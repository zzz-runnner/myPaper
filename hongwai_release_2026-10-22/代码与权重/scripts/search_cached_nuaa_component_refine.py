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
    min_contrast: float


def parse_csv(value: str, cast):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def load_many(paths: list[Path], max_samples_per_cache: int = 0) -> tuple[list[dict[str, object]], list[str]]:
    all_samples: list[dict[str, object]] = []
    names_ref: list[str] | None = None
    for path in paths:
        samples, names = load_cache(path)
        if names_ref is None:
            names_ref = names
        elif names_ref != names:
            raise ValueError(f"name mismatch in {path}: {names} != {names_ref}")
        if max_samples_per_cache > 0:
            samples = samples[:max_samples_per_cache]
        all_samples.extend(samples)
    if names_ref is None:
        raise ValueError("no caches")
    return all_samples, names_ref


def image_contrast_feature(prob: np.ndarray) -> np.ndarray:
    # Probability-local contrast is cheap and cache-only; it catches isolated bright target components.
    blur = cv2.blur(prob.astype(np.float32), (15, 15))
    return np.clip(prob - blur, 0.0, 1.0).astype(np.float32)


def render_rule(samples: list[dict[str, object]], rule: Rule) -> list[np.ndarray]:
    out_maps: list[np.ndarray] = []
    for sample in samples:
        preds = sample["preds"]
        base = preds[rule.base_mode].astype(np.float32)
        binary = (base >= rule.threshold).astype(np.uint8)
        support_stack = []
        for name in ("tpc_a", "tpc_b", "lca", "bgm_nuaa", "rpcanet_sirst_s3"):
            if name in preds:
                support_stack.append((preds[name].astype(np.float32) >= 0.5).astype(np.uint8))
        support = np.sum(np.stack(support_stack, axis=0), axis=0) if support_stack else np.zeros_like(binary)
        bgm = preds.get("bgm_nuaa", base).astype(np.float32)
        rpcanet = preds.get("rpcanet_sirst_s3", base).astype(np.float32)
        contrast = image_contrast_feature(base)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        kept = np.zeros_like(binary)
        for label_id in range(1, num_labels):
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            w = int(stats[label_id, cv2.CC_STAT_WIDTH])
            h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            if area < rule.min_area:
                continue
            if rule.max_area > 0 and area > rule.max_area:
                continue
            if rule.max_side > 0 and max(w, h) > rule.max_side:
                continue
            mask = labels[y : y + h, x : x + w] == label_id
            base_crop = base[y : y + h, x : x + w][mask]
            if float(base_crop.mean()) < rule.min_mean:
                continue
            if float(base_crop.max()) < rule.min_peak:
                continue
            if int(support[y : y + h, x : x + w][mask].max()) < rule.min_support:
                continue
            if float(bgm[y : y + h, x : x + w][mask].max()) < rule.min_bgm:
                continue
            if float(rpcanet[y : y + h, x : x + w][mask].max()) < rule.min_rpcanet:
                continue
            if float(contrast[y : y + h, x : x + w][mask].mean()) < rule.min_contrast:
                continue
            kept[labels == label_id] = 1
        out_maps.append(kept.astype(np.float32))
    return out_maps


def score(metrics: dict[str, float], fa_penalty: float, pd_bonus: float) -> float:
    return float(metrics["mIoU"]) - fa_penalty * float(metrics["Fa"]) * 1e6 + pd_bonus * float(metrics["Pd"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Cached train-selected component refinement for NUAA strong fusion.")
    parser.add_argument("--fit_cache", action="append", required=True)
    parser.add_argument("--test_cache", required=True)
    parser.add_argument("--base_modes", default="weighted_raw,strong_fusion")
    parser.add_argument("--thresholds", default="0.40,0.42,0.44,0.46,0.48,0.50,0.52,0.54,0.56")
    parser.add_argument("--min_areas", default="1,2,3,4,5,6,8")
    parser.add_argument("--max_areas", default="0,64,128,256")
    parser.add_argument("--max_sides", default="0,16,24,32,48")
    parser.add_argument("--min_means", default="0.0,0.35,0.40,0.45,0.50")
    parser.add_argument("--min_peaks", default="0.0,0.50,0.55,0.60,0.65")
    parser.add_argument("--min_supports", default="0,1,2,3")
    parser.add_argument("--min_bgms", default="0.0,0.20,0.25,0.30,0.35")
    parser.add_argument("--min_rpcanets", default="0.0,0.20,0.30,0.40,0.50")
    parser.add_argument("--min_contrasts", default="0.0,0.02,0.04,0.06")
    parser.add_argument("--max_fit_samples_per_cache", type=int, default=0)
    parser.add_argument("--fa_penalty", type=float, default=0.0000015)
    parser.add_argument("--pd_bonus", type=float, default=0.0)
    parser.add_argument("--top_fit", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    fit_samples, names = load_many([Path(x) for x in args.fit_cache], args.max_fit_samples_per_cache)
    test_samples, test_names = load_cache(Path(args.test_cache))
    if names != test_names:
        raise ValueError("fit/test cache names differ")

    rules: list[Rule] = []
    for base_mode in parse_csv(args.base_modes, str):
        for threshold in parse_csv(args.thresholds, float):
            for min_area in parse_csv(args.min_areas, int):
                for max_area in parse_csv(args.max_areas, int):
                    for max_side in parse_csv(args.max_sides, int):
                        for min_mean in parse_csv(args.min_means, float):
                            for min_peak in parse_csv(args.min_peaks, float):
                                for min_support in parse_csv(args.min_supports, int):
                                    for min_bgm in parse_csv(args.min_bgms, float):
                                        for min_rpcanet in parse_csv(args.min_rpcanets, float):
                                            for min_contrast in parse_csv(args.min_contrasts, float):
                                                rules.append(
                                                    Rule(
                                                        base_mode,
                                                        threshold,
                                                        min_area,
                                                        max_area,
                                                        max_side,
                                                        min_mean,
                                                        min_peak,
                                                        min_support,
                                                        min_bgm,
                                                        min_rpcanet,
                                                        min_contrast,
                                                    )
                                                )

    fit_rows = []
    for idx, rule in enumerate(rules, 1):
        maps = render_rule(fit_samples, rule)
        metrics = metric_from_maps(fit_samples, maps, 0.5)
        fit_rows.append({"rule": asdict(rule), **metrics, "score": score(metrics, args.fa_penalty, args.pd_bonus)})
        if idx % 500 == 0:
            print(f"scored {idx}/{len(rules)}", flush=True)
    fit_rows.sort(key=lambda row: row["score"], reverse=True)

    eval_rows = []
    seen = set()
    for row in fit_rows[: max(1, args.top_fit)]:
        rule = Rule(**row["rule"])
        key = tuple(asdict(rule).items())
        if key in seen:
            continue
        seen.add(key)
        maps = render_rule(test_samples, rule)
        metrics = metric_from_maps(test_samples, maps, 0.5)
        eval_rows.append(
            {
                "rule": asdict(rule),
                "fit": {k: row[k] for k in ("mIoU", "nIoU", "Pd", "Fa", "score")},
                "test": metrics,
            }
        )
    eval_rows.sort(key=lambda row: row["test"]["mIoU"], reverse=True)

    baseline_rows = []
    for mode in parse_csv(args.base_modes, str):
        for threshold in parse_csv(args.thresholds, float):
            maps = [sample["preds"][mode].astype(np.float32) for sample in test_samples]
            baseline_rows.append({"mode": mode, "threshold": threshold, **metric_from_maps(test_samples, maps, threshold)})
    baseline_rows.sort(key=lambda row: row["mIoU"], reverse=True)

    payload = {
        "fit_caches": args.fit_cache,
        "test_cache": args.test_cache,
        "num_fit_samples": len(fit_samples),
        "num_test_samples": len(test_samples),
        "num_rules": len(rules),
        "baseline_best": baseline_rows[0],
        "selected": eval_rows[0] if eval_rows else None,
        "eval_rows": eval_rows,
        "top_fit": fit_rows[: args.top_fit],
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if eval_rows:
        best = eval_rows[0]["test"]
        print(f"best test mIoU={best['mIoU']*100:.4f} Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.4f}", flush=True)
    print(f"baseline test mIoU={baseline_rows[0]['mIoU']*100:.4f}", flush=True)


if __name__ == "__main__":
    main()
