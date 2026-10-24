#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
RUN_ONE="$ROOT/scripts/run_local_rule_hfmlk_one.sh"
OUT="$ROOT/runs/local_rule_hfmlk_nuaa_4gpu_screen"
LOG="$OUT/logs"
SESSION="local_rule_hfmlk_nuaa"

mkdir -p "$LOG"
tmux kill-session -t "$SESSION" 2>/dev/null || true

launch() {
  local gpu="$1"; shift
  local name="$1"; shift
  local seed="$1"; shift
  local residual="$1"; shift
  local anchor="$1"; shift
  local lr="$1"; shift
  local focus="$1"; shift
  local extra=("$@")
  local run_dir="$OUT/$name"
  local log_file="$LOG/$name.log"
  rm -rf "$run_dir"
  tmux new-window -t "$SESSION" -n "$name" \
    "cd '$ROOT' && '$RUN_ONE' '$gpu' '$name' '$seed' '$residual' '$anchor' '$lr' '$focus' '$OUT' ${extra[*]} 2>&1 | tee '$log_file'"
}

tmux new-session -d -s "$SESSION" -n s1011 \
  "cd '$ROOT' && '$RUN_ONE' '0' 's1011_r004_a020' '20260621' '0.040' '0.20' '0.000030' '0.45' '$OUT' 2>&1 | tee '$LOG/s1011_r004_a020.log'"

launch 1 s1012_r006_a018 20260622 0.060 0.18 0.000035 0.50
launch 2 s1013_r003_a024 20260623 0.030 0.24 0.000028 0.40 --miss_weight 0.10
launch 3 s1014_r005_a022 20260624 0.050 0.22 0.000032 0.55 --uncertainty_weight 0.20

tmux ls | grep "$SESSION" || true
