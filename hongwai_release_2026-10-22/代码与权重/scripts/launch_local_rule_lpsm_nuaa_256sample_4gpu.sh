#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
RUN_ONE="$ROOT/scripts/run_local_rule_lpsm_one.sh"
OUT="$ROOT/runs/local_rule_lpsm_nuaa_256sample_4gpu"
LOG="$OUT/logs"
SESSION="local_rule_lpsm_256"

mkdir -p "$LOG"
tmux kill-session -t "$SESSION" 2>/dev/null || true

launch() {
  local gpu="$1"; shift
  local name="$1"; shift
  local seed="$1"; shift
  local residual="$1"; shift
  local anchor="$1"; shift
  local lr="$1"; shift
  local hard_clutter="$1"; shift
  local miss="$1"; shift
  local uncertainty="$1"; shift
  rm -rf "$OUT/$name"
  tmux new-window -t "$SESSION" -n "$name" \
    "cd '$ROOT' && '$RUN_ONE' '$gpu' '$name' '$seed' 'e2,mid' '$residual' '$anchor' '$lr' '$hard_clutter' '$miss' '$uncertainty' '$OUT' --base_channels 64 --batch_size 3 --epochs 3 --max_train_samples_per_cache 256 --thresholds '0.498,0.499,0.500,0.501,0.502,0.503,0.504' 2>&1 | tee '$LOG/$name.log'"
}

tmux new-session -d -s "$SESSION" -n s1331 \
  "cd '$ROOT' && '$RUN_ONE' '0' 's1331_r0055_a018_s71' '20260681' 'e2,mid' '0.055' '0.18' '0.000034' '0.80' '1.00' '0.15' '$OUT' --base_channels 64 --batch_size 3 --epochs 3 --max_train_samples_per_cache 256 --thresholds '0.498,0.499,0.500,0.501,0.502,0.503,0.504' 2>&1 | tee '$LOG/s1331_r0055_a018_s71.log'"

launch 1 s1332_r0055_a018_s72 20260682 0.055 0.18 0.000034 0.80 1.00 0.15
launch 2 s1333_r0058_a017_s73 20260683 0.058 0.17 0.000032 0.85 1.05 0.14
launch 3 s1334_r0052_a019_s74 20260684 0.052 0.19 0.000036 0.75 0.95 0.15

tmux ls | grep "$SESSION" || true
