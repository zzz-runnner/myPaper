#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
RUN_ONE="$ROOT/scripts/run_local_rule_dhif_one.sh"
OUT="$ROOT/runs/local_rule_dhif_nuaa_4gpu_screen"
LOG="$OUT/logs"
SESSION="local_rule_dhif_nuaa"

mkdir -p "$LOG"
tmux kill-session -t "$SESSION" 2>/dev/null || true

launch() {
  local gpu="$1"; shift
  local name="$1"; shift
  local seed="$1"; shift
  local stages="$1"; shift
  local residual="$1"; shift
  local anchor="$1"; shift
  local lr="$1"; shift
  local hard_clutter="$1"; shift
  local miss="$1"; shift
  local uncertainty="$1"; shift
  local log_file="$LOG/$name.log"
  rm -rf "$OUT/$name"
  tmux new-window -t "$SESSION" -n "$name" \
    "cd '$ROOT' && '$RUN_ONE' '$gpu' '$name' '$seed' '$stages' '$residual' '$anchor' '$lr' '$hard_clutter' '$miss' '$uncertainty' '$OUT' 2>&1 | tee '$log_file'"
}

tmux new-session -d -s "$SESSION" -n s1201 \
  "cd '$ROOT' && '$RUN_ONE' '0' 's1201_e1_r004_hc04_m08' '20260641' 'e1' '0.040' '0.18' '0.000040' '0.40' '0.80' '0.10' '$OUT' 2>&1 | tee '$LOG/s1201_e1_r004_hc04_m08.log'"

launch 1 s1202_e12_r005_hc06_m10 20260642 e1,e2 0.050 0.20 0.000035 0.60 1.00 0.15
launch 2 s1203_e123_r004_hc08_m06 20260643 e1,e2,e3 0.040 0.22 0.000030 0.80 0.60 0.20
launch 3 s1204_e2_r006_hc10_m12 20260644 e2 0.060 0.18 0.000032 1.00 1.20 0.10

tmux ls | grep "$SESSION" || true
