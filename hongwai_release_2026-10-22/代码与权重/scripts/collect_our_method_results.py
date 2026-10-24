#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README_TARGETS = {
    "NUAA-SIRST": 77.50,
    "IRSTD-1K": 68.03,
}


def parse_best(log_text: str) -> tuple[int | None, float | None, float | None, float | None]:
    best_epoch = None
    best_miou = None
    best_pd = None
    best_fa = None
    pending = None
    for line in log_text.splitlines():
        m = re.search(r"the best model epoch\s+\t?_(\d+)", line)
        if not m:
            m = re.search(r"save the best model epoch .*? _(\d+)", line)
        if m:
            pending = int(m.group(1))
            best_epoch = pending
            continue
        if pending is not None and line.startswith("pixAcc, mIoU:"):
            nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", line)
            if len(nums) >= 2:
                best_miou = float(nums[-1]) * 100.0
            continue
        if pending is not None and line.startswith("PD, FA:"):
            nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", line)
            if len(nums) >= 2:
                best_pd = float(nums[0]) * 100.0
                best_fa = float(nums[-1]) * 1e6
                pending = None
    return best_epoch, best_miou, best_pd, best_fa


def parse_latest(log_text: str) -> tuple[int | None, float | None]:
    latest_epoch = None
    latest_loss = None
    for line in log_text.splitlines():
        m = re.search(r"Epoch---(\d+), total_loss---([0-9.eE+-]+)", line)
        if m:
            latest_epoch = int(m.group(1))
            latest_loss = float(m.group(2))
    return latest_epoch, latest_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", default=str(ROOT / "runs" / "our_method_4gpu"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "our_method_results.md"))
    args = parser.parse_args()

    run_root = Path(args.run_root)
    rows = []
    for log in sorted((run_root / "logs").glob("*.log")):
        text = log.read_text(errors="ignore")
        latest_epoch, latest_loss = parse_latest(text)
        best_epoch, best_miou, best_pd, best_fa = parse_best(text)
        parts = log.stem.split("_")
        dataset = "_".join(parts[:-2]) if len(parts) >= 3 else log.stem
        variant = parts[-2] if len(parts) >= 3 else "-"
        gpu = parts[-1].replace("gpu", "") if len(parts) >= 3 else "-"
        target = README_TARGETS.get(dataset)
        rows.append((dataset, variant, gpu, latest_epoch, latest_loss, best_epoch, best_miou, target, best_pd, best_fa, log))

    lines = ["# Our Method Results", ""]
    lines.append(f"Run root: `{run_root}`")
    lines.append("")
    lines.append("| Dataset | Variant | GPU | Latest epoch | Latest loss | Best epoch | Best mIoU | README mIoU | Delta | Best Pd | Best Fa |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for dataset, variant, gpu, latest_epoch, latest_loss, best_epoch, best_miou, target, best_pd, best_fa, _ in rows:
        delta = best_miou - target if best_miou is not None and target is not None else None
        lines.append(
            "| "
            f"{dataset} | {variant} | {gpu} | "
            f"{latest_epoch if latest_epoch is not None else '-'} | "
            f"{latest_loss:.6f}" if latest_loss is not None else f"| {dataset} | {variant} | {gpu} | - | -"
        )
        lines[-1] += f" | {best_epoch if best_epoch is not None else '-'}"
        lines[-1] += f" | {best_miou:.2f}" if best_miou is not None else " | -"
        lines[-1] += f" | {target:.2f}" if target is not None else " | -"
        lines[-1] += f" | {delta:+.2f}" if delta is not None else " | -"
        lines[-1] += f" | {best_pd:.2f}" if best_pd is not None else " | -"
        lines[-1] += f" | {best_fa:.2f}" if best_fa is not None else " | -"
        lines[-1] += " |"

    lines.append("")
    lines.append("## Log Files")
    lines.append("")
    for *_, log in rows:
        lines.append(f"- `{log}`")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    print(output)


if __name__ == "__main__":
    main()
