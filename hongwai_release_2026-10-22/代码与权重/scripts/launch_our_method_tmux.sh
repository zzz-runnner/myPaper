#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/envs/irstd/bin/python"
SESSION="${SESSION:-irstd_our_method}"
RUN_ROOT="${IRSTD_RUN_ROOT:-$ROOT/runs/our_method_4gpu}"
DATASET_DIR="${IRSTD_DATASET_DIR:-$ROOT/datasets}"
EPOCHS="${EPOCHS:-1000}"
BEGIN_TEST="${BEGIN_TEST:-350}"
EVERY_TEST="${EVERY_TEST:-1}"
THREADS="${THREADS:-4}"

mkdir -p "$RUN_ROOT/logs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  exit 1
fi

launch() {
  local window="$1"
  local gpu="$2"
  local dataset="$3"
  local variant="$4"
  local seed="$5"
  local batch="$6"
  local lr="$7"
  local pos_prob="$8"
  local noise="$9"
  local gamma="${10}"
  local jitter="${11}"
  local sls="${12}"
  local posw="${13}"
  local hardneg="${14}"
  local focalw="${15}"
  local rdir="$RUN_ROOT/${dataset}_${variant}_gpu${gpu}"
  local log="$RUN_ROOT/logs/${dataset}_${variant}_gpu${gpu}.log"
  local cmd="cd '$ROOT' && CUDA_VISIBLE_DEVICES=$gpu '$PYTHON' '$ROOT/scripts/train_our_method.py' --dataset_dir '$DATASET_DIR' --dataset_name '$dataset' --run_dir '$rdir' --method_name 'TCD-SCTransNet-${variant}' --epochs '$EPOCHS' --begin_test '$BEGIN_TEST' --every_test '$EVERY_TEST' --batch_size '$batch' --threads '$THREADS' --seed '$seed' --lr '$lr' --pos_prob '$pos_prob' --noise_std '$noise' --gamma_prob '$gamma' --intensity_jitter '$jitter' --sls_weight '$sls' --pos_weight '$posw' --hard_neg_weight '$hardneg' --focal_weight '$focalw' 2>&1 | tee '$log'"
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-session -d -s "$SESSION" -n "$window" "$cmd"
  else
    tmux new-window -t "$SESSION" -n "$window" "$cmd"
  fi
}

# Two full-data variants per dataset. 0/1 are IRSTD-1K, 2/3 are NUAA-SIRST.
launch "irstd_tcd_a" 0 "IRSTD-1K" "A" 43 20 0.00070 0.80 0.012 0.30 0.06 2.0 4.0 1.2 0.35
launch "irstd_tcd_b" 1 "IRSTD-1K" "B" 47 20 0.00055 0.90 0.020 0.50 0.10 2.6 5.0 1.6 0.45
launch "nuaa_tcd_a" 2 "NUAA-SIRST" "A" 43 20 0.00070 0.80 0.012 0.30 0.06 2.0 4.0 1.2 0.35
launch "nuaa_tcd_b" 3 "NUAA-SIRST" "B" 47 20 0.00055 0.90 0.020 0.50 0.10 2.6 5.0 1.6 0.45

echo "Started tmux session: $SESSION"
echo "Run root: $RUN_ROOT"
tmux ls
