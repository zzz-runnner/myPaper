#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_DIR="${IRSTD_DATASET_DIR:-$ROOT/datasets}"
RUN_ROOT="${IRSTD_RUN_ROOT:-$ROOT/runs/baseline_4gpu}"
SESSION="${IRSTD_TMUX_SESSION:-irstd_baseline4}"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PATCH_SIZE="${PATCH_SIZE:-256}"
THREADS="${THREADS:-4}"
BEGIN_TEST="${BEGIN_TEST:-500}"
EVERY_TEST="${EVERY_TEST:-1}"
EVERY_PRINT="${EVERY_PRINT:-10}"
EVERY_SAVE_PTH="${EVERY_SAVE_PTH:-1000}"

mkdir -p "$RUN_ROOT/logs"

make_cmd() {
  local dataset="$1"
  local gpu="$2"
  local log="$RUN_ROOT/logs/${dataset}_gpu${gpu}.log"
  printf 'cd %q && IRSTD_DATASET_DIR=%q IRSTD_RUN_DIR=%q GPU=%q EPOCHS=%q BATCH_SIZE=%q PATCH_SIZE=%q THREADS=%q BEGIN_TEST=%q EVERY_TEST=%q EVERY_PRINT=%q EVERY_SAVE_PTH=%q PYTHONUNBUFFERED=1 bash %q %q >%q 2>&1; echo finished $?; sleep 30' \
    "$ROOT" \
    "$DATASET_DIR" \
    "$RUN_ROOT/$dataset" \
    "$gpu" \
    "$EPOCHS" \
    "$BATCH_SIZE" \
    "$PATCH_SIZE" \
    "$THREADS" \
    "$BEGIN_TEST" \
    "$EVERY_TEST" \
    "$EVERY_PRINT" \
    "$EVERY_SAVE_PTH" \
    "$ROOT/scripts/train_sctransnet.sh" \
    "$dataset" \
    "$log"
}

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  tmux list-windows -t "$SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" -n "SIRST3" "$(make_cmd SIRST3 0)"
tmux new-window -t "$SESSION" -n "NUDT-SIRST" "$(make_cmd NUDT-SIRST 1)"
tmux new-window -t "$SESSION" -n "IRSTD-1K" "$(make_cmd IRSTD-1K 2)"
tmux new-window -t "$SESSION" -n "NUAA-SIRST" "$(make_cmd NUAA-SIRST 3)"

echo "created tmux session: $SESSION"
tmux list-windows -t "$SESSION"
