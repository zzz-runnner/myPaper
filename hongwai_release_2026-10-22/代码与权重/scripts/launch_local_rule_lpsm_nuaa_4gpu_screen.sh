#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
RUN_ONE="$ROOT/scripts/run_local_rule_lpsm_one.sh"
OUT="$ROOT/runs/local_rule_lpsm_nuaa_4gpu_screen"
LOG="$OUT/logs"
SESSION="local_rule_lpsm_nuaa"

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
  local extra=("$@")
  rm -rf "$OUT/$name"
  tmux new-window -t "$SESSION" -n "$name" \
    "cd '$ROOT' && '$RUN_ONE' '$gpu' '$name' '$seed' '$stages' '$residual' '$anchor' '$lr' '$hard_clutter' '$miss' '$uncertainty' '$OUT' ${extra[*]} 2>&1 | tee '$LOG/$name.log'"
}

tmux new-session -d -s "$SESSION" -n s1301 \
  "cd '$ROOT' && '$RUN_ONE' '0' 's1301_e2e3m_r004' '20260651' 'e2,e3,mid' '0.040' '0.20' '0.000040' '0.50' '0.80' '0.10' '$OUT' 2>&1 | tee '$LOG/s1301_e2e3m_r004.log'"

launch 1 s1302_e3m_r005 20260652 e3,mid 0.050 0.20 0.000035 0.70 0.80 0.15
launch 2 s1303_e2m_r006 20260653 e2,mid 0.060 0.18 0.000032 0.80 1.00 0.10
launch 3 s1304_mid_r004 20260654 mid 0.040 0.22 0.000045 0.40 1.20 0.20 --base_channels 64

tmux ls | grep "$SESSION" || true
