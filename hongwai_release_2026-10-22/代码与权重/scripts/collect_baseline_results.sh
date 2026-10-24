#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${IRSTD_RUN_ROOT:-$ROOT/runs/baseline_4gpu}"
REPORT_DIR="$ROOT/reports"
PYTHON="$ROOT/envs/irstd/bin/python"
DATASET_DIR="$ROOT/datasets"
SKIP_EVAL="${SKIP_EVAL:-0}"
SIRST3_EVAL_DEVICE="${SIRST3_EVAL_DEVICE:-cuda:0}"
NUDT_EVAL_DEVICE="${NUDT_EVAL_DEVICE:-cuda:0}"
IRSTD_EVAL_DEVICE="${IRSTD_EVAL_DEVICE:-cuda:0}"
NUAA_EVAL_DEVICE="${NUAA_EVAL_DEVICE:-cuda:0}"

mkdir -p "$REPORT_DIR"

SUMMARY_TXT="$REPORT_DIR/from_scratch_training_log_summary.txt"
PROGRESS_MD="$REPORT_DIR/from_scratch_progress_table.md"
SIRST3_EVAL_TXT="$REPORT_DIR/from_scratch_sirst3_best_eval.txt"
SINGLE_EVAL_TXT="$REPORT_DIR/from_scratch_single_dataset_best_eval.txt"
FINAL_MD="$REPORT_DIR/from_scratch_baseline_results.md"

"$PYTHON" "$ROOT/scripts/parse_training_logs.py" \
  --run_root "$RUN_ROOT" \
  --include_console_logs > "$SUMMARY_TXT"
"$PYTHON" "$ROOT/scripts/summarize_from_scratch_progress.py" \
  --run_root "$RUN_ROOT" > "$PROGRESS_MD"

find_best() {
  local dataset="$1"
  find "$RUN_ROOT/$dataset/checkpoints" -type f -name '*_best.pth.tar' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

SIRST3_BEST="$(find_best SIRST3 || true)"
NUDT_BEST="$(find_best NUDT-SIRST || true)"
IRSTD_BEST="$(find_best IRSTD-1K || true)"
NUAA_BEST="$(find_best NUAA-SIRST || true)"

{
  echo "# From-Scratch SCTransNet Baseline Results"
  echo
  echo "Generated: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo
  echo "## Training Log Summary"
  echo
  echo '```text'
  cat "$SUMMARY_TXT"
  echo '```'
  echo
  echo "## Current Best vs README"
  echo
  cat "$PROGRESS_MD"
  echo
  echo "## Best Checkpoints"
  echo
  echo "| Dataset | Best checkpoint |"
  echo "|---|---|"
  echo "| SIRST3 | ${SIRST3_BEST:-missing} |"
  echo "| NUDT-SIRST | ${NUDT_BEST:-missing} |"
  echo "| IRSTD-1K | ${IRSTD_BEST:-missing} |"
  echo "| NUAA-SIRST | ${NUAA_BEST:-missing} |"
} > "$FINAL_MD"

if [[ "$SKIP_EVAL" == "1" ]]; then
  {
    echo
    echo "## Evaluation"
    echo
    echo "Skipped because SKIP_EVAL=1."
  } >> "$FINAL_MD"
  echo "$FINAL_MD"
  exit 0
fi

if [[ -n "${SIRST3_BEST:-}" ]]; then
  "$PYTHON" "$ROOT/scripts/eval_checkpoint.py" \
    --checkpoint "$SIRST3_BEST" \
    --dataset_dir "$DATASET_DIR" \
    --train_dataset_name SIRST3 \
    --test_dataset_names NUAA-SIRST NUDT-SIRST IRSTD-1K \
    --device "$SIRST3_EVAL_DEVICE" > "$SIRST3_EVAL_TXT"
  {
    echo
    echo "## SIRST3 Best Checkpoint Cross-Dataset Evaluation"
    echo
    echo '```text'
    cat "$SIRST3_EVAL_TXT"
    echo '```'
  } >> "$FINAL_MD"
fi

: > "$SINGLE_EVAL_TXT"
if [[ -n "${NUDT_BEST:-}" ]]; then
  "$PYTHON" "$ROOT/scripts/eval_checkpoint.py" \
    --checkpoint "$NUDT_BEST" \
    --dataset_dir "$DATASET_DIR" \
    --train_dataset_name NUDT-SIRST \
    --test_dataset_names NUDT-SIRST \
    --device "$NUDT_EVAL_DEVICE" >> "$SINGLE_EVAL_TXT"
fi
if [[ -n "${IRSTD_BEST:-}" ]]; then
  "$PYTHON" "$ROOT/scripts/eval_checkpoint.py" \
    --checkpoint "$IRSTD_BEST" \
    --dataset_dir "$DATASET_DIR" \
    --train_dataset_name IRSTD-1K \
    --test_dataset_names IRSTD-1K \
    --device "$IRSTD_EVAL_DEVICE" >> "$SINGLE_EVAL_TXT"
fi
if [[ -n "${NUAA_BEST:-}" ]]; then
  "$PYTHON" "$ROOT/scripts/eval_checkpoint.py" \
    --checkpoint "$NUAA_BEST" \
    --dataset_dir "$DATASET_DIR" \
    --train_dataset_name NUAA-SIRST \
    --test_dataset_names NUAA-SIRST \
    --device "$NUAA_EVAL_DEVICE" >> "$SINGLE_EVAL_TXT"
fi
if [[ -s "$SINGLE_EVAL_TXT" ]]; then
  {
    echo
    echo "## Single-Dataset Best Checkpoint Evaluation"
    echo
    echo '```text'
    cat "$SINGLE_EVAL_TXT"
    echo '```'
  } >> "$FINAL_MD"
fi

echo "$FINAL_MD"
