#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
from pathlib import Path

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception:  # pragma: no cover - optional report enhancement
    EventAccumulator = None


README_MIOU = {"IRSTD-1K": 68.03, "NUAA-SIRST": 77.50, "NUDT-SIRST": 94.09}
DATASETS = tuple(README_MIOU.keys()) + ("SIRST3",)


def read_tensorboard_latest(run_dir: Path, dataset: str) -> dict[str, float | int] | None:
    if EventAccumulator is None:
        return None
    tb_dir = run_dir / "tensorboard" / dataset
    event_files = sorted(tb_dir.glob("events.out.tfevents*"), key=lambda path: path.stat().st_mtime)
    if not event_files:
        return None
    latest: dict[str, float | int] = {}
    for event_file in event_files:
        try:
            accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            accumulator.Reload()
        except Exception:
            continue
        for tag, out_key, scale in (("mIoU", "mIoU", 100.0), ("Pd", "Pd", 100.0), ("Fa", "Fa", 1e6), ("loss", "loss", 1.0)):
            if tag not in accumulator.Tags().get("scalars", []):
                continue
            values = accumulator.Scalars(tag)
            if not values:
                continue
            value = values[-1]
            latest[f"{out_key}_epoch"] = int(value.step)
            latest[out_key] = float(value.value) * scale
    return latest or None


def parse_log(path: Path) -> dict[str, object] | None:
    text = path.read_text(errors="ignore")
    dataset_pat = "|".join(re.escape(dataset) for dataset in DATASETS)
    header = re.search(rf"^({dataset_pat})\s+(\S+)\s+(\S+)", text, flags=re.MULTILINE)
    if not header:
        return None
    dataset, method, model = header.groups()
    latest_epoch = None
    latest_loss = None
    for match in re.finditer(r"Epoch---(\d+), total_loss---([0-9.]+)", text):
        latest_epoch = int(match.group(1))
        latest_loss = float(match.group(2))

    best = None
    for match in re.finditer(
        r"best epoch=(\d+) pixAcc=([0-9.]+) mIoU=([0-9.]+) Pd=([0-9.]+) Fa=([0-9.]+)",
        text,
    ):
        epoch = int(match.group(1))
        miou = float(match.group(3)) * 100.0
        row = {
            "best_epoch": epoch,
            "pixAcc": float(match.group(2)) * 100.0,
            "mIoU": miou,
            "Pd": float(match.group(4)) * 100.0,
            "Fa": float(match.group(5)) * 1e6,
        }
        if best is None or row["mIoU"] > best["mIoU"]:
            best = row

    latest_eval = None
    for match in re.finditer(
        r"eval epoch=(\d+) pixAcc=([0-9.]+) mIoU=([0-9.]+) Pd=([0-9.]+) Fa=([0-9.]+)",
        text,
    ):
        latest_eval = {
            "eval_epoch": int(match.group(1)),
            "pixAcc": float(match.group(2)) * 100.0,
            "mIoU": float(match.group(3)) * 100.0,
            "Pd": float(match.group(4)) * 100.0,
            "Fa": float(match.group(5)) * 1e6,
        }
    run_dir = path.parent.parent / path.stem
    tb_latest = read_tensorboard_latest(run_dir, dataset)
    if tb_latest and "mIoU" in tb_latest:
        latest_eval = {
            "eval_epoch": int(tb_latest.get("mIoU_epoch", tb_latest.get("Pd_epoch", -1))),
            "pixAcc": None,
            "mIoU": float(tb_latest["mIoU"]),
            "Pd": None if "Pd" not in tb_latest else float(tb_latest["Pd"]),
            "Fa": None if "Fa" not in tb_latest else float(tb_latest["Fa"]),
        }

    return {
        "dataset": dataset,
        "method": method,
        "model": model,
        "latest_epoch": latest_epoch,
        "latest_loss": latest_loss,
        "latest_eval": latest_eval,
        "best": best,
        "log": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", default="/home/cx/hongwai/irstd-lab/runs/hybrid_4gpu")
    parser.add_argument("--output", default="/home/cx/hongwai/irstd-lab/reports/hybrid_results.md")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    rows = []
    for log in sorted((run_root / "logs").glob("*.log")):
        parsed = parse_log(log)
        if parsed:
            rows.append(parsed)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Hybrid Method Results", "", f"Run root: `{run_root}`", ""]
    lines.append("| Dataset | Method | Model | Latest epoch | Latest loss | Latest eval | Latest eval mIoU | Best epoch | Best mIoU | README mIoU | Delta | Best Pd | Best Fa |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(rows, key=lambda r: (r["dataset"], r["method"])):
        best = row["best"] or {}
        latest_eval = row["latest_eval"] or {}
        dataset = str(row["dataset"])
        readme = README_MIOU.get(dataset)
        miou = best.get("mIoU")
        delta = miou - readme if miou is not None and readme is not None else None
        lines.append(
            "| "
            + " | ".join(
                [
                    dataset,
                    str(row["method"]),
                    str(row["model"]),
                    "" if row["latest_epoch"] is None else str(row["latest_epoch"]),
                    "" if row["latest_loss"] is None else f"{row['latest_loss']:.6f}",
                    "" if not latest_eval else str(latest_eval["eval_epoch"]),
                    "" if not latest_eval else f"{latest_eval['mIoU']:.2f}",
                    "" if not best else str(best["best_epoch"]),
                    "" if miou is None else f"{miou:.2f}",
                    "" if readme is None else f"{readme:.2f}",
                    "" if delta is None else f"{delta:+.2f}",
                    "" if not best else f"{best['Pd']:.2f}",
                    "" if not best else f"{best['Fa']:.2f}",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Logs", ""])
    for row in rows:
        lines.append(f"- `{row['log']}`")
    out.write_text("\n".join(lines) + "\n")
    print(out)


if __name__ == "__main__":
    main()
