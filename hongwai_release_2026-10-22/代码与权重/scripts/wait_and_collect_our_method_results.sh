#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${IRSTD_RUN_ROOT:-$ROOT/runs/our_method_4gpu}"
LOG="$ROOT/reports/wait_and_collect_our_method.log"
mkdir -p "$ROOT/reports"

echo "[$(date -u '+%F %T UTC')] waiting for our-method training processes" | tee -a "$LOG"
while true; do
  if ! pgrep -af "$ROOT/scripts/train_our_method.py.*$RUN_ROOT" >/dev/null; then
    break
  fi
  sleep 300
done

echo "[$(date -u '+%F %T UTC')] training finished, collecting our-method results" | tee -a "$LOG"
"$ROOT/envs/irstd/bin/python" "$ROOT/scripts/collect_our_method_results.py" \
  --run_root "$RUN_ROOT" \
  --output "$ROOT/reports/our_method_results.md" | tee -a "$LOG"
echo "[$(date -u '+%F %T UTC')] done" | tee -a "$LOG"
