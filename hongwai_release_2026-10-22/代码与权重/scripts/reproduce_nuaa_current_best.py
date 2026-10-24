#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def load_bgm_maps(cache_path: Path) -> list[np.ndarray]:
    data = np.load(cache_path, allow_pickle=True)
    maps: list[np.ndarray] = []
    idx = 0
    while f"pred_{idx}_bgm_nuaa" in data.files:
        maps.append(np.asarray(data[f"pred_{idx}_bgm_nuaa"], dtype=np.float32))
        idx += 1
    if not maps:
        raise RuntimeError(f"No pred_i_bgm_nuaa maps found in {cache_path}")
    return maps


def fuse_bgm_negative(base_npz: Path, cache_path: Path, output_npz: Path) -> None:
    base = np.load(base_npz, allow_pickle=True)
    bgm_maps = load_bgm_maps(cache_path)
    s703_maps = [np.asarray(x, dtype=np.float32) for x in list(base["maps_no_tta"])]
    if len(s703_maps) != len(bgm_maps):
        raise RuntimeError(f"Map count mismatch: {len(s703_maps)} s703 maps vs {len(bgm_maps)} BGM maps")
    fused = []
    for s703, bgm in zip(s703_maps, bgm_maps):
        if s703.shape != bgm.shape:
            raise RuntimeError(f"Shape mismatch: s703={s703.shape} bgm={bgm.shape}")
        fused.append(np.clip(0.96 * s703 - 0.04 * bgm, 0.0, 1.0) ** 0.94)
    np.savez_compressed(
        output_npz,
        names=base["names"],
        sizes=base["sizes"],
        masks=base["masks"],
        maps_best=np.asarray(fused, dtype=object),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the current best NUAA-SIRST s703+BGM+component-dropk recipe.")
    parser.add_argument("--python", default=str(ROOT / "envs" / "irstd" / "bin" / "python"))
    parser.add_argument("--dataset_name", default="NUAA-SIRST")
    parser.add_argument("--train_cache", default=str(ROOT / "reports" / "fusion_cache_bgm_rpcanet_train_a" / "NUAA-SIRST__train.npz"))
    parser.add_argument("--test_cache", default=str(ROOT / "reports" / "fusion_cache_bgm_rpcanet_train_b_test" / "NUAA-SIRST__test.npz"))
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "runs" / "strong_fusion_hfmlk_nuaa_4gpu" / "nuaa_strong_fusion_s703" / "best.pth.tar"),
    )
    parser.add_argument("--device_train", default="cuda:0")
    parser.add_argument("--device_test", default="cuda:1")
    parser.add_argument("--output", default=str(ROOT / "reports" / "nuaa_current_best_repro_eval.json"))
    parser.add_argument("--report", default=str(ROOT / "reports" / "nuaa_current_best_repro_report.md"))
    parser.add_argument("--keep_tmp", action="store_true")
    args = parser.parse_args()

    py = args.python
    tmp_train_base = ROOT / "reports" / "tmp_nuaa_current_best_repro_train_base.npz"
    tmp_test_base = ROOT / "reports" / "tmp_nuaa_current_best_repro_test_base.npz"
    tmp_train_eval = ROOT / "reports" / "tmp_nuaa_current_best_repro_train_base_eval.json"
    tmp_test_eval = ROOT / "reports" / "tmp_nuaa_current_best_repro_test_base_eval.json"
    tmp_train_fused = ROOT / "reports" / "tmp_nuaa_current_best_repro_train_fused.npz"
    tmp_test_fused = ROOT / "reports" / "tmp_nuaa_current_best_repro_test_fused.npz"
    output = Path(args.output)
    report = Path(args.report)

    cleanup_paths = [tmp_train_base, tmp_test_base, tmp_train_eval, tmp_test_eval, tmp_train_fused, tmp_test_fused]
    try:
        for split, cache, device, eval_json, maps_npz in (
            ("train", args.train_cache, args.device_train, tmp_train_eval, tmp_train_base),
            ("test", args.test_cache, args.device_test, tmp_test_eval, tmp_test_base),
        ):
            print(f"Exporting {split} s703 no-TTA maps...", flush=True)
            run(
                [
                    py,
                    "scripts/eval_hfm_lk_refiner_tta.py",
                    "--dataset_name",
                    args.dataset_name,
                    "--cache",
                    cache,
                    "--checkpoint",
                    args.checkpoint,
                    "--teacher_mode",
                    "strong_fusion",
                    "--base_channels",
                    "48",
                    "--residual_scale",
                    "0.32",
                    "--thresholds",
                    "0.472",
                    "--tta",
                    "none",
                    "--device",
                    device,
                    "--output",
                    str(eval_json),
                    "--save_maps",
                    str(maps_npz),
                ]
            )

        print("Fusing BGM-negative maps...", flush=True)
        fuse_bgm_negative(tmp_train_base, Path(args.train_cache), tmp_train_fused)
        fuse_bgm_negative(tmp_test_base, Path(args.test_cache), tmp_test_fused)

        run(
            [
                py,
                "scripts/npz_component_stump_dropk_sweep.py",
                "--train_npz",
                str(tmp_train_fused),
                "--test_npz",
                str(tmp_test_fused),
                "--map_key",
                "maps_best",
                "--seed_thresholds",
                "0.472",
                "--output_thresholds",
                "0.456,0.46,0.464,0.468,0.472",
                "--pos_weight_scales",
                "0.5",
                "--rounds_list",
                "40",
                "--drop_counts",
                "0:8",
                "--lr",
                "0.25",
                "--bins",
                "11",
                "--max_components_per_image",
                "96",
                "--output",
                str(output),
            ]
        )

        result = json.loads(output.read_text(encoding="utf-8"))
        best = result["best_valid"]
        report.write_text(
            "\n".join(
                [
                    "# NUAA Current Best Reproduction",
                    "",
                    "Generated by `scripts/reproduce_nuaa_current_best.py`.",
                    "",
                    "## Recipe",
                    "",
                    "- checkpoint: `runs/strong_fusion_hfmlk_nuaa_4gpu/nuaa_strong_fusion_s703/best.pth.tar`",
                    "- no-TTA s703 map",
                    "- fusion: `clip(0.96 * s703 - 0.04 * bgm_nuaa, 0, 1) ** 0.94`",
                    "- component seed threshold: `0.472`",
                    "- boosted-stump selector: `pos_weight_scale=0.5`, `rounds=40`, `bins=11`",
                    "- drop lowest-scored `4` components",
                    "- output threshold: `0.456`",
                    "",
                    "## Best Reproduced Row",
                    "",
                    "```json",
                    json.dumps(best, indent=2),
                    "```",
                    "",
                    "## Status",
                    "",
                    f"- reproduced mIoU: `{best['mIoU'] * 100:.4f}`",
                    "- NUAA +5 target: `82.6400`",
                    f"- remaining gap: `{82.64 - best['mIoU'] * 100:.4f}`",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            f"reproduced best mIoU={best['mIoU'] * 100:.4f} "
            f"nIoU={best['nIoU'] * 100:.4f} Pd={best['Pd'] * 100:.4f} Fa={best['Fa'] * 1e6:.4f}",
            flush=True,
        )
    finally:
        if not args.keep_tmp:
            for path in cleanup_paths:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass


if __name__ == "__main__":
    main()
