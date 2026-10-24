#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${IRSTD_RUN_ROOT:-$ROOT/runs/baseline_4gpu}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"
LOG="$ROOT/reports/wait_and_collect.log"

mkdir -p "$ROOT/reports"

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] waiting for baseline training to finish" >> "$LOG"

while true; do
  if ! ps -eo cmd | grep -E 'train.py --dataset_dir .*/runs/baseline_4gpu' | grep -v grep >/dev/null; then
    break
  fi
  "$ROOT/envs/irstd/bin/python" "$ROOT/scripts/parse_training_logs.py" \
    --run_root "$RUN_ROOT" \
    --include_console_logs >> "$LOG" 2>&1 || true
  sleep "$SLEEP_SECONDS"
done

echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] training finished, collecting results" >> "$LOG"
IRSTD_RUN_ROOT="$RUN_ROOT" bash "$ROOT/scripts/collect_baseline_results.sh" >> "$LOG" 2>&1
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] done" >> "$LOG"
