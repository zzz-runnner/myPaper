#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
RUNNER="$ROOT/scripts/run_hfmlk_aug_official_mamba_teacher_one.sh"
OUT="$ROOT/runs/hfmlk_aug_official_mamba_teacher_nuaa_4gpu"
mkdir -p "$OUT"

launch() {
  local gpu="$1"; shift
  local name="$1"; shift
  nohup bash "$RUNNER" "$gpu" "$name" "$@" > "$OUT/$name.log" 2>&1 &
  echo "$!" > "$OUT/$name.pid"
}

launch 0 aug_s971 20260619 4e-5 0.06 0.14 0.35 0.006 --uncertainty_weight 0.25
launch 1 aug_s972 20260620 5e-5 0.08 0.12 0.50 0.010 --miss_weight 0.20
launch 2 aug_fp_s973 20260621 5e-5 0.10 0.10 0.40 0.008 --fp_topk_weight 0.015 --pos_peak_weight 0.02
launch 3 aug_focus_s974 20260622 4e-5 0.06 0.16 0.55 0.012 --focus_patch_prob 0.55 --miss_weight 0.30

cat "$OUT"/*.pid
