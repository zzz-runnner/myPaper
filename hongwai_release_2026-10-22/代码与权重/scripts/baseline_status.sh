#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${IRSTD_RUN_ROOT:-$ROOT/runs/baseline_4gpu}"
SESSION="${IRSTD_TMUX_SESSION:-irstd_baseline4}"

echo "== tmux =="
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux list-windows -t "$SESSION"
else
  echo "tmux session not found: $SESSION"
fi

echo
echo "== train processes =="
ps -eo pid,ppid,stat,etime,cmd | grep -E 'train.py --dataset_dir' | grep -v grep || true

echo
echo "== latest log lines =="
for log in "$RUN_ROOT"/logs/*.log; do
  [[ -e "$log" ]] || continue
  echo "--- $(basename "$log") ---"
  tail -n 12 "$log"
done

echo
echo "== gpu =="
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
