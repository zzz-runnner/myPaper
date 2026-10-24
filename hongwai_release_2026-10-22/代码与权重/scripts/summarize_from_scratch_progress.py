#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


EPOCH_RE = re.compile(r"Epoch---(\d+), total_loss---([0-9.eE+-]+), lr---([0-9.eE+-]+)")
BEST_RE = re.compile(r"the best model epoch.*?_(\d+)")
NUMBER_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")

README_TARGETS = {
    "NUAA-SIRST": {"name": "SIRST-v1 / NUAA-SIRST", "miou": 77.50, "pd": 96.95, "fa": 13.92},
    "NUDT-SIRST": {"name": "NUDT-SIRST", "miou": 94.09, "pd": 98.62, "fa": 4.29},
    "IRSTD-1K": {"name": "IRSTD-1K", "miou": 68.03, "pd": 93.27, "fa": 10.74},
}


@dataclass
class Progress:
    dataset: str
    latest_epoch: int | None = None
    latest_loss: float | None = None
    best_epoch: int | None = None
    best_miou: float | None = None
    best_pd: float | None = None
    best_fa: float | None = None


def metric_pair(line: str) -> tuple[float, float] | None:
    values = NUMBER_RE.findall(line)
    if len(values) < 2:
        return None
    return float(values[0]), float(values[-1])


def dataset_from_log(path: Path) -> str:
    name = path.name
    for suffix in ("_gpu0.log", "_gpu1.log", "_gpu2.log", "_gpu3.log", ".log"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def parse_log(path: Path) -> Progress:
    progress = Progress(dataset=dataset_from_log(path))
    lines = path.read_text(errors="ignore").splitlines()
    pending_best_epoch = None

    for line in lines:
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            progress.latest_epoch = int(epoch_match.group(1))
            progress.latest_loss = float(epoch_match.group(2))
            continue

        best_match = BEST_RE.search(line)
        if best_match:
            pending_best_epoch = int(best_match.group(1))
            progress.best_epoch = pending_best_epoch
            continue

        if line.startswith("pixAcc, mIoU:"):
            pair = metric_pair(line)
            if pair and pending_best_epoch is not None:
                progress.best_miou = pair[1] * 100
            continue

        if line.startswith("PD, FA:"):
            pair = metric_pair(line)
            if pair and pending_best_epoch is not None:
                progress.best_pd = pair[0] * 100
                progress.best_fa = pair[1] * 1_000_000
                pending_best_epoch = None

    return progress


def fmt(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def delta(current: float | None, target: float | None) -> str:
    if current is None or target is None:
        return "-"
    return f"{current - target:+.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize active SCTransNet baseline progress.")
    parser.add_argument("--run_root", default="/home/cx/hongwai/irstd-lab/runs/baseline_4gpu")
    args = parser.parse_args()

    log_dir = Path(args.run_root) / "logs"
    progresses = {p.dataset: p for p in [parse_log(path) for path in sorted(log_dir.glob("*.log"))]}

    order = ["SIRST3", "NUDT-SIRST", "IRSTD-1K", "NUAA-SIRST"]
    print("| Dataset | Latest epoch | Latest loss | Best epoch | Best mIoU | README mIoU | Delta | Best Pd | README Pd | Best Fa | README Fa |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for dataset in order:
        progress = progresses.get(dataset, Progress(dataset=dataset))
        target = README_TARGETS.get(dataset)
        target_miou = target["miou"] if target else None
        target_pd = target["pd"] if target else None
        target_fa = target["fa"] if target else None
        display_name = target["name"] if target else dataset
        print(
            f"| {display_name} | {fmt(progress.latest_epoch)} | {fmt(progress.latest_loss, 6)} | "
            f"{fmt(progress.best_epoch)} | {fmt(progress.best_miou)} | {fmt(target_miou)} | "
            f"{delta(progress.best_miou, target_miou)} | {fmt(progress.best_pd)} | {fmt(target_pd)} | "
            f"{fmt(progress.best_fa)} | {fmt(target_fa)} |"
        )

    print()
    print("SIRST3 has no single README target in the authors' table; after that run finishes, its best checkpoint is evaluated on SIRST-v1, NUDT-SIRST, and IRSTD-1K for the paper-style comparison.")


if __name__ == "__main__":
    main()
