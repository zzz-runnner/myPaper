#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/cx/hongwai/irstd-lab
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hfm_lk_anchor_refiner_from_cache.py"
RUN_ROOT="$ROOT/runs/hfm_lk_anchor_4gpu"
mkdir -p "$RUN_ROOT"

COMMON_NUAA="\
 --train_cache_spec 'NUAA-SIRST|$ROOT/reports/prediction_cache_nuaa_tpc_lca_train_a/NUAA-SIRST__train.npz'\
 --train_cache_spec 'IRSTD-1K|$ROOT/reports/prediction_cache_nuaa_tpc_lca_train_a/IRSTD-1K__train.npz'\
 --train_cache_spec 'NUDT-SIRST|$ROOT/reports/prediction_cache_nuaa_tpc_lca_train_b_test/NUDT-SIRST__train.npz'\
 --train_cache_spec 'SIRST3|$ROOT/reports/prediction_cache_nuaa_tpc_lca_train_b_test/SIRST3__train.npz'\
 --test_cache_spec 'NUAA-SIRST|$ROOT/reports/prediction_cache_nuaa_tpc_lca_train_b_test/NUAA-SIRST__test.npz'\
 --teacher_mode avg --epochs 6 --patch_size 512 --eval_interval 2 --eval_max_samples 120\
 --thresholds 0.50,0.54,0.58,0.62,0.66 --amp"

COMMON_IRSTD="\
 --train_cache_spec 'IRSTD-1K|$ROOT/reports/prediction_cache_irstd_dual_train_a/IRSTD-1K__train.npz'\
 --train_cache_spec 'NUAA-SIRST|$ROOT/reports/prediction_cache_irstd_dual_train_a/NUAA-SIRST__train.npz'\
 --train_cache_spec 'NUDT-SIRST|$ROOT/reports/prediction_cache_irstd_dual_train_b_test/NUDT-SIRST__train.npz'\
 --train_cache_spec 'SIRST3|$ROOT/reports/prediction_cache_irstd_dual_train_b_test/SIRST3__train.npz'\
 --test_cache_spec 'IRSTD-1K|$ROOT/reports/prediction_cache_irstd_dual_train_b_test/IRSTD-1K__test.npz'\
 --teacher_mode avg --epochs 6 --patch_size 512 --eval_interval 2 --eval_max_samples 120\
 --thresholds 0.30,0.34,0.38,0.42,0.46 --amp"

launch() {
  local session=$1
  local gpu=$2
  local common=$3
  local extra=$4
  local out=$5
  tmux new-session -d -s "$session" \
    "cd '$ROOT' && CUDA_VISIBLE_DEVICES=$gpu '$PY' '$SCRIPT' $common $extra --device cuda:0 --output_dir '$RUN_ROOT/$out' 2>&1 | tee '$RUN_ROOT/$out.log'"
}

launch hfm_lk_irstd_a 0 "$COMMON_IRSTD" "--batch_size 2 --base_channels 48 --lr 0.00020 --residual_scale 0.35 --anchor_weight 0.025 --pos_weight 28 --seed 631" irstd_hfmlk_s631
launch hfm_lk_irstd_b 1 "$COMMON_IRSTD" "--batch_size 2 --base_channels 56 --lr 0.00016 --residual_scale 0.40 --anchor_weight 0.030 --pos_weight 30 --seed 632" irstd_hfmlk_s632
launch hfm_lk_nuaa_a 2 "$COMMON_NUAA" "--batch_size 2 --base_channels 48 --lr 0.00018 --residual_scale 0.35 --anchor_weight 0.025 --pos_weight 32 --hard_clutter_weight 1.4 --miss_weight 2.8 --uncertainty_weight 0.4 --weighted_dice --seed 633" nuaa_hfmlk_s633
launch hfm_lk_nuaa_b 3 "$COMMON_NUAA" "--batch_size 2 --base_channels 56 --lr 0.00014 --residual_scale 0.45 --anchor_weight 0.030 --pos_weight 34 --hard_clutter_weight 1.8 --miss_weight 3.2 --uncertainty_weight 0.6 --weighted_dice --seed 634" nuaa_hfmlk_s634

tmux ls | grep hfm_lk || true
