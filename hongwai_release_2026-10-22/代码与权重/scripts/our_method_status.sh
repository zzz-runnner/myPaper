#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${IRSTD_RUN_ROOT:-$ROOT/runs/our_method_4gpu}"

echo "== GPU =="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits

echo
echo "== Training Progress =="
"$ROOT/envs/irstd/bin/python" "$ROOT/scripts/parse_training_logs.py" --run_root "$RUN_ROOT" --include_console_logs || true

echo
echo "== Logs =="
if [[ -d "$RUN_ROOT/logs" ]]; then
  for log in "$RUN_ROOT"/logs/*.log; do
    [[ -e "$log" ]] || continue
    echo "--- $log"
    tail -n 8 "$log"
  done
fi
