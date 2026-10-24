#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "SCTransNet"))

from eval_component_fusion_learner import metric_from_maps  # noqa: E402


def parse_float_list(value: str) -> list[float]:
    return [float(x) for x in value.split(",") if x.strip()]


def load_cache(path: Path) -> tuple[list[dict[str, object]], list[str]]:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta_json"]))
    names = json.loads(str(data["names_json"]))
    samples: list[dict[str, object]] = []
    for idx, item in enumerate(meta):
        height, width = [int(x) for x in item["size"]]
        preds = {name: data[f"pred_{idx}_{name}"].astype(np.float32)[:height, :width] for name in names}
        mask = data[f"mask_{idx}"].astype(np.float32)[:height, :width]
        samples.append({"name": item["name"], "size": [height, width], "mask": mask, "preds": preds})
    return samples, names


def render(samples: list[dict[str, object]], names: list[str], mode: str) -> list[np.ndarray]:
    maps = []
    for sample in samples:
        if mode in names:
            maps.append(sample["preds"][mode].astype(np.float32))
            continue
        stack = np.stack([sample["preds"][name] for name in names], axis=0)
        if mode == "avg":
            maps.append(stack.mean(axis=0).astype(np.float32))
        elif mode == "max":
            maps.append(stack.max(axis=0).astype(np.float32))
        elif mode == "min":
            maps.append(stack.min(axis=0).astype(np.float32))
        else:
            raise ValueError(mode)
    return maps


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate simple fusion modes from a prediction cache.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--modes", default="avg,max")
    parser.add_argument("--thresholds", default="0.35,0.40,0.45,0.50")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    samples, names = load_cache(Path(args.cache))
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    thresholds = parse_float_list(args.thresholds)
    rows = []
    for mode in modes:
        maps = render(samples, names, mode)
        for threshold in thresholds:
            rows.append({"mode": mode, "threshold": threshold, **metric_from_maps(samples, maps, threshold)})
    rows.sort(key=lambda row: row["mIoU"], reverse=True)
    best = rows[0]
    print(
        f"best mode={best['mode']} th={best['threshold']} mIoU={best['mIoU']*100:.2f} "
        f"Pd={best['Pd']*100:.2f} Fa={best['Fa']*1e6:.2f}",
        flush=True,
    )
    Path(args.output).write_text(
        json.dumps({"cache": args.cache, "models": names, "best": best, "rows": rows}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
