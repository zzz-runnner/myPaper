#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path


EPOCH_RE = re.compile(r"Epoch---(\d+), total_loss---([0-9.eE+-]+), lr---([0-9.eE+-]+)")
BEST_RE = re.compile(r"the best model epoch.*?_(\d+)")


def parse_number(text: str) -> float:
    matches = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", text)
    if not matches:
        raise ValueError(text)
    return float(matches[-1])


def parse_metric_pair(line: str) -> tuple[float, float] | None:
    values = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", line)
    if len(values) < 2:
        return None
    return float(values[0]), float(values[-1])


def parse_file(path: Path) -> None:
    latest_epoch = None
    latest_loss = None
    best_epoch = None
    best_miou = None
    best_pd = None
    best_fa = None
    pending_best_epoch = None
    lines = path.read_text(errors="ignore").splitlines()

    for idx, line in enumerate(lines):
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            latest_epoch = int(epoch_match.group(1))
            latest_loss = float(epoch_match.group(2))
        best_match = BEST_RE.search(line)
        if best_match:
            pending_best_epoch = int(best_match.group(1))
            best_epoch = pending_best_epoch
            continue
        if line.startswith("pixAcc, mIoU:"):
            pair = parse_metric_pair(line)
            if pair and pending_best_epoch is not None:
                best_miou = pair[1]
            continue
        if line.startswith("PD, FA:"):
            pair = parse_metric_pair(line)
            if pair and pending_best_epoch is not None:
                best_pd = pair[0]
                best_fa = pair[1]
                pending_best_epoch = None

    if latest_epoch is None and best_epoch is None:
        return

    print(path)
    print(f"  latest_epoch={latest_epoch} latest_loss={latest_loss}")
    if best_epoch is not None:
        miou = f"{best_miou * 100:.2f}" if best_miou is not None else "unknown"
        pd = f"{best_pd * 100:.2f}" if best_pd is not None else "unknown"
        fa = f"{best_fa * 1e6:.2f}" if best_fa is not None else "unknown"
        print("  best " f"epoch={best_epoch} " f"mIoU={miou} " f"Pd={pd} " f"Fa={fa}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse SCTransNet training text logs.")
    parser.add_argument("--run_root", default="/home/cx/hongwai/irstd-lab/runs/baseline_4gpu")
    parser.add_argument("--include_console_logs", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    for path in sorted(run_root.glob("*/checkpoints/*_SCTransNet_*.txt")):
        parse_file(path)
    if args.include_console_logs:
        for path in sorted((run_root / "logs").glob("*.log")):
            parse_file(path)


if __name__ == "__main__":
    main()
