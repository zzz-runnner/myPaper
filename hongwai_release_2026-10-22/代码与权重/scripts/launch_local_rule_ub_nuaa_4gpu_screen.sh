#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
RUN_ONE="$ROOT/scripts/run_local_rule_ub_one.sh"
OUT="$ROOT/runs/local_rule_ub_nuaa_4gpu_screen"
LOG="$OUT/logs"
SESSION="local_rule_ub_nuaa"

mkdir -p "$LOG"
tmux kill-session -t "$SESSION" 2>/dev/null || true

launch() {
  local gpu="$1"; shift
  local name="$1"; shift
  local seed="$1"; shift
  local residual="$1"; shift
  local anchor="$1"; shift
  local lr="$1"; shift
  local uncertainty="$1"; shift
  local boundary="$1"; shift
  local false_positive="$1"; shift
  local pos_weight="$1"; shift
  local run_dir="$OUT/$name"
  local log_file="$LOG/$name.log"
  rm -rf "$run_dir"
  tmux new-window -t "$SESSION" -n "$name" \
    "cd '$ROOT' && '$RUN_ONE' '$gpu' '$name' '$seed' '$residual' '$anchor' '$lr' '$uncertainty' '$boundary' '$false_positive' '$pos_weight' '$OUT' 2>&1 | tee '$log_file'"
}

tmux new-session -d -s "$SESSION" -n s1101 \
  "cd '$ROOT' && '$RUN_ONE' '0' 's1101_r006_a004_ub15' '20260631' '0.060' '0.040' '0.000080' '0.60' '1.50' '0.80' '25.0' '$OUT' 2>&1 | tee '$LOG/s1101_r006_a004_ub15.log'"

launch 1 s1102_r008_a005_ub20 20260632 0.080 0.050 0.000070 0.80 2.00 1.00 28.0
launch 2 s1103_r010_a006_ub25 20260633 0.100 0.060 0.000060 1.00 2.50 1.20 30.0
launch 3 s1104_r012_a004_ub30 20260634 0.120 0.040 0.000050 1.20 3.00 1.50 32.0

tmux ls | grep "$SESSION" || true
