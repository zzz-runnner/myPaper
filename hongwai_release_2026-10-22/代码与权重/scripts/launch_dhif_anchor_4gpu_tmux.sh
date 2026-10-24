#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_dhif_anchor_refiner_from_cache.py"
RUN_ROOT="$ROOT/runs/dhif_anchor_4gpu"
LOG_ROOT="$ROOT/reports/dhif_anchor_logs"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

launch() {
  local session="$1"
  local gpu="$2"
  local run_name="$3"
  local train_a="$4"
  local train_b="$5"
  local test_cache="$6"
  local test_dataset="$7"
  local stages="$8"
  local residual="$9"
  local anchor="${10}"
  local lr="${11}"
  local pos_weight="${12}"
  local extra="${13}"
  local run_dir="$RUN_ROOT/$run_name"
  local log="$LOG_ROOT/$run_name.log"

  tmux new-session -d -s "$session" \
    "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && CUDA_VISIBLE_DEVICES=$gpu '$PY' '$SCRIPT' \
      --dataset_dir '$ROOT/datasets' \
      --train_cache_spec 'IRSTD-1K|$train_a/IRSTD-1K__train.npz' \
      --train_cache_spec 'NUAA-SIRST|$train_a/NUAA-SIRST__train.npz' \
      --train_cache_spec 'NUDT-SIRST|$train_b/NUDT-SIRST__train.npz' \
      --train_cache_spec 'SIRST3|$train_b/SIRST3__train.npz' \
      --test_cache_spec '$test_dataset|$test_cache' \
      --teacher_mode avg --epochs 8 --batch_size 4 --patch_size 512 --base_channels 56 \
      --dhif_stages '$stages' --lr '$lr' --residual_scale '$residual' --anchor_weight '$anchor' \
      --pos_weight '$pos_weight' --thresholds '0.26,0.30,0.34,0.38,0.42,0.46,0.50,0.54,0.58,0.62,0.66' \
      --eval_interval 2 --seed '202606${gpu}51' --device cuda:0 --amp \
      --output_dir '$run_dir' $extra 2>&1 | tee '$log'"
}

IRSTD_A="$ROOT/reports/prediction_cache_irstd_dual_train_a"
IRSTD_B="$ROOT/reports/prediction_cache_irstd_dual_train_b_test"
NUAA_A="$ROOT/reports/prediction_cache_nuaa_tpc_lca_train_a"
NUAA_B="$ROOT/reports/prediction_cache_nuaa_tpc_lca_train_b_test"

launch dhif_irstd_s501 0 irstd_dhif_e1_s501 \
  "$IRSTD_A" "$IRSTD_B" "$IRSTD_B/IRSTD-1K__test.npz" IRSTD-1K \
  "e1" 0.35 0.020 0.00020 25 ""

launch dhif_irstd_s502 1 irstd_dhif_e12_s502 \
  "$IRSTD_A" "$IRSTD_B" "$IRSTD_B/IRSTD-1K__test.npz" IRSTD-1K \
  "e1,e2" 0.40 0.025 0.00018 25 ""

launch dhif_nuaa_s503 2 nuaa_dhif_e1_s503 \
  "$NUAA_A" "$NUAA_B" "$NUAA_B/NUAA-SIRST__test.npz" NUAA-SIRST \
  "e1" 0.35 0.020 0.00018 30 \
  "--hard_clutter_weight 1.8 --hard_clutter_threshold 0.34 --miss_weight 2.8 --miss_threshold 0.22 --uncertainty_weight 0.7 --weighted_dice"

launch dhif_nuaa_s504 3 nuaa_dhif_e12_s504 \
  "$NUAA_A" "$NUAA_B" "$NUAA_B/NUAA-SIRST__test.npz" NUAA-SIRST \
  "e1,e2" 0.45 0.030 0.00020 30 \
  "--hard_clutter_weight 1.2 --hard_clutter_threshold 0.34 --miss_weight 3.6 --miss_threshold 0.22 --uncertainty_weight 0.7 --weighted_dice"

tmux ls
