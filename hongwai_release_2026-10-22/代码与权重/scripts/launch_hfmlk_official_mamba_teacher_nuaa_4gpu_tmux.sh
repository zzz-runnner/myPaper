#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hfm_lk_anchor_refiner_from_cache.py"
OUT="$ROOT/runs/hfmlk_official_mamba_teacher_nuaa_4gpu"
mkdir -p "$OUT"

COMMON=(
  "$SCRIPT"
  --train_cache_spec "IRSTD-1K|$ROOT/reports/fusion_cache_bgm_rpcanet_train_a/IRSTD-1K__train.npz"
  --train_cache_spec "NUAA-SIRST|$ROOT/reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__train.npz"
  --train_cache_spec "NUDT-SIRST|$ROOT/reports/fusion_cache_bgm_rpcanet_train_b_test/NUDT-SIRST__train.npz"
  --train_cache_spec "SIRST3|$ROOT/reports/fusion_cache_bgm_rpcanet_train_b_test/SIRST3__train.npz"
  --test_cache_spec "NUAA-SIRST|$ROOT/reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__test.npz"
  --teacher_mode "official_mamba_fused,strong_fusion"
  --epochs 4
  --batch_size 3
  --patch_size 512
  --base_channels 48
  --thresholds "0.468,0.476,0.480,0.488,0.496,0.504,0.512,0.520,0.540"
  --eval_interval 1
  --amp
  --weighted_dice
  --focus_dataset NUAA-SIRST
  --focus_patch_prob 0.35
  --focus_hard_threshold 0.45
  --focus_miss_threshold 0.18
  --resume_checkpoint "$ROOT/runs/strong_fusion_hfmlk_nuaa_4gpu/nuaa_strong_fusion_s703/best.pth.tar"
)

launch() {
  local gpu="$1"; shift
  local name="$1"; shift
  local seed="$1"; shift
  local lr="$1"; shift
  local rscale="$1"; shift
  local anchor="$1"; shift
  local extra=("$@")
  local session="om_hfmlk_${name}"
  tmux kill-session -t "$session" 2>/dev/null || true
  CUDA_VISIBLE_DEVICES="$gpu" tmux new-session -d -s "$session" \
    "cd '$ROOT' && '$PY' '${COMMON[@]}' --seed '$seed' --lr '$lr' --residual_scale '$rscale' --anchor_weight '$anchor' ${extra[*]} --device cuda:0 --output_dir '$OUT/$name' 2>&1 | tee '$OUT/$name.log'"
}

launch 0 nuaa_official_mamba_s961 20260615 6e-5 0.10 0.08
launch 1 nuaa_official_mamba_s962 20260616 4e-5 0.08 0.12 --uncertainty_weight 0.4
launch 2 nuaa_official_mamba_fp_s963 20260617 5e-5 0.10 0.10 --fp_topk_weight 0.015 --pos_peak_weight 0.02
launch 3 nuaa_official_mamba_focus_s964 20260618 4e-5 0.06 0.16 --focus_patch_prob 0.50 --miss_weight 0.25

tmux ls | grep 'om_hfmlk_' || true
