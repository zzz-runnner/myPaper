#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_teacher_anchor_refiner_from_cache.py"
RUN_ROOT="$ROOT/runs/hard_clutter_anchor_4gpu"
LOG_ROOT="$ROOT/reports/hard_clutter_anchor_logs"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

launch() {
  local session="$1"
  local gpu="$2"
  local run_name="$3"
  local seed="$4"
  local residual="$5"
  local anchor="$6"
  local hard="$7"
  local miss="$8"
  local uncert="$9"
  local pos="${10}"
  local batch="${11}"
  local base="${12}"
  local lr="${13}"
  local out="$RUN_ROOT/$run_name"
  local log="$LOG_ROOT/$run_name.log"

  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" \
    "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && CUDA_VISIBLE_DEVICES=$gpu '$PY' '$SCRIPT' \
      --dataset_dir '$ROOT/datasets' \
      --train_cache_spec 'IRSTD-1K|$ROOT/reports/prediction_cache_nuaa_tpc_lca_train_a/IRSTD-1K__train.npz' \
      --train_cache_spec 'NUAA-SIRST|$ROOT/reports/prediction_cache_nuaa_tpc_lca_train_a/NUAA-SIRST__train.npz' \
      --train_cache_spec 'NUDT-SIRST|$ROOT/reports/prediction_cache_nuaa_tpc_lca_train_b_test/NUDT-SIRST__train.npz' \
      --train_cache_spec 'SIRST3|$ROOT/reports/prediction_cache_nuaa_tpc_lca_train_b_test/SIRST3__train.npz' \
      --test_cache_spec 'NUAA-SIRST|$ROOT/reports/prediction_cache_nuaa_tpc_lca_train_b_test/NUAA-SIRST__test.npz' \
      --teacher_mode avg \
      --epochs 10 \
      --batch_size '$batch' \
      --patch_size 512 \
      --base_channels '$base' \
      --lr '$lr' \
      --residual_scale '$residual' \
      --anchor_weight '$anchor' \
      --pos_weight '$pos' \
      --hard_clutter_weight '$hard' \
      --hard_clutter_threshold 0.34 \
      --miss_weight '$miss' \
      --miss_threshold 0.22 \
      --uncertainty_weight '$uncert' \
      --weighted_dice \
      --thresholds '0.30,0.34,0.38,0.42,0.46,0.50,0.54,0.58,0.62,0.66' \
      --eval_interval 2 \
      --seed '$seed' \
      --device cuda:0 \
      --output_dir '$out' \
      2>&1 | tee '$log'"
}

launch hca_nuaa_s201 0 nuaa_hard_a_s201 202606201 0.40 0.025 1.6 2.4 0.6 28 5 64 0.00022
launch hca_nuaa_s202 1 nuaa_hard_b_s202 202606202 0.35 0.020 2.2 3.0 0.9 32 5 64 0.00018
launch hca_nuaa_s203 2 nuaa_hard_c_s203 202606203 0.45 0.030 1.2 3.6 0.7 30 5 64 0.00020
launch hca_nuaa_s204 3 nuaa_hard_d_s204 202606204 0.30 0.018 2.8 2.2 1.1 34 5 64 0.00016

tmux ls
