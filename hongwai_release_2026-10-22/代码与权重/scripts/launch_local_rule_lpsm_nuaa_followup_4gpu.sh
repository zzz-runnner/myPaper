#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
RUN_ONE="$ROOT/scripts/run_local_rule_lpsm_one.sh"
OUT="$ROOT/runs/local_rule_lpsm_nuaa_followup_4gpu"
LOG="$OUT/logs"
SESSION="local_rule_lpsm_follow"

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

tmux new-session -d -s "$SESSION" -n s1311 \
  "cd '$ROOT' && '$RUN_ONE' '0' 's1311_e2m_r0065_a016' '20260661' 'e2,mid' '0.065' '0.16' '0.000030' '0.90' '1.10' '0.12' '$OUT' --base_channels 64 --batch_size 3 --epochs 4 --thresholds '0.496,0.498,0.500,0.501,0.502,0.503,0.504,0.506' 2>&1 | tee '$LOG/s1311_e2m_r0065_a016.log'"

launch 1 s1312_e2m_r007_a014 20260662 e2,mid 0.070 0.14 0.000026 1.00 1.20 0.10 --base_channels 64 --batch_size 3 --epochs 4 --thresholds "0.496,0.498,0.500,0.501,0.502,0.503,0.504,0.506"
launch 2 s1313_e2m_r0055_a018 20260663 e2,mid 0.055 0.18 0.000034 0.80 1.00 0.15 --base_channels 64 --batch_size 3 --epochs 4 --thresholds "0.496,0.498,0.500,0.501,0.502,0.503,0.504,0.506"
launch 3 s1314_e2m_e3_r006 20260664 e2,e3,mid 0.060 0.16 0.000028 0.90 1.10 0.12 --base_channels 56 --batch_size 4 --epochs 4 --thresholds "0.496,0.498,0.500,0.501,0.502,0.503,0.504,0.506"

tmux ls | grep "$SESSION" || true
