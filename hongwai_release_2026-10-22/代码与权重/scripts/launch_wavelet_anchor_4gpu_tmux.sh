#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_wavelet_anchor_refiner_from_cache.py"
RUN_ROOT="$ROOT/runs/wavelet_anchor_4gpu"
LOG_ROOT="$ROOT/reports/wavelet_anchor_logs"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"

launch_nuaa() {
  local session="$1" gpu="$2" run="$3" seed="$4" residual="$5" anchor="$6" hard="$7" miss="$8" uncert="$9" lr="${10}"
  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" \
    "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && CUDA_VISIBLE_DEVICES=$gpu '$PY' '$SCRIPT' \
      --dataset_dir '$ROOT/datasets' \
      --train_cache_spec 'IRSTD-1K|$ROOT/reports/prediction_cache_nuaa_tpc_lca_train_a/IRSTD-1K__train.npz' \
      --train_cache_spec 'NUAA-SIRST|$ROOT/reports/prediction_cache_nuaa_tpc_lca_train_a/NUAA-SIRST__train.npz' \
      --train_cache_spec 'NUDT-SIRST|$ROOT/reports/prediction_cache_nuaa_tpc_lca_train_b_test/NUDT-SIRST__train.npz' \
      --train_cache_spec 'SIRST3|$ROOT/reports/prediction_cache_nuaa_tpc_lca_train_b_test/SIRST3__train.npz' \
      --test_cache_spec 'NUAA-SIRST|$ROOT/reports/prediction_cache_nuaa_tpc_lca_train_b_test/NUAA-SIRST__test.npz' \
      --teacher_mode avg --epochs 8 --batch_size 4 --patch_size 512 --base_channels 56 --lr '$lr' \
      --residual_scale '$residual' --anchor_weight '$anchor' --pos_weight 30 \
      --hard_clutter_weight '$hard' --hard_clutter_threshold 0.34 \
      --miss_weight '$miss' --miss_threshold 0.22 --uncertainty_weight '$uncert' --weighted_dice \
      --thresholds '0.30,0.34,0.38,0.42,0.46,0.50,0.54,0.58,0.62,0.66' \
      --eval_interval 2 --seed '$seed' --device cuda:0 --amp \
      --output_dir '$RUN_ROOT/$run' 2>&1 | tee '$LOG_ROOT/$run.log'"
}

launch_irstd() {
  local session="$1" gpu="$2" run="$3" seed="$4" residual="$5" anchor="$6" lr="$7"
  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" \
    "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && CUDA_VISIBLE_DEVICES=$gpu '$PY' '$SCRIPT' \
      --dataset_dir '$ROOT/datasets' \
      --train_cache_spec 'IRSTD-1K|$ROOT/reports/prediction_cache_irstd_dual_train_a/IRSTD-1K__train.npz' \
      --train_cache_spec 'NUAA-SIRST|$ROOT/reports/prediction_cache_irstd_dual_train_a/NUAA-SIRST__train.npz' \
      --train_cache_spec 'NUDT-SIRST|$ROOT/reports/prediction_cache_irstd_dual_train_b_test/NUDT-SIRST__train.npz' \
      --train_cache_spec 'SIRST3|$ROOT/reports/prediction_cache_irstd_dual_train_b_test/SIRST3__train.npz' \
      --test_cache_spec 'IRSTD-1K|$ROOT/reports/prediction_cache_irstd_dual_train_b_test/IRSTD-1K__test.npz' \
      --teacher_mode avg --epochs 8 --batch_size 4 --patch_size 512 --base_channels 56 --lr '$lr' \
      --residual_scale '$residual' --anchor_weight '$anchor' --pos_weight 25 \
      --thresholds '0.26,0.30,0.34,0.38,0.42,0.46,0.50,0.54,0.58' \
      --eval_interval 2 --seed '$seed' --device cuda:0 --amp \
      --output_dir '$RUN_ROOT/$run' 2>&1 | tee '$LOG_ROOT/$run.log'"
}

launch_irstd wav_irstd_s401 0 irstd_wavelet_s401 202606401 0.35 0.020 0.00020
launch_irstd wav_irstd_s402 1 irstd_wavelet_s402 202606402 0.45 0.030 0.00018
launch_nuaa wav_nuaa_s403 2 nuaa_wavelet_s403 202606403 0.35 0.020 1.8 2.8 0.7 0.00018
launch_nuaa wav_nuaa_s404 3 nuaa_wavelet_s404 202606404 0.45 0.030 1.2 3.6 0.7 0.00020

tmux ls
