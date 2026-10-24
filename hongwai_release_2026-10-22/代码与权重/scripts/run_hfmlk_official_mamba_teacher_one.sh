#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hfm_lk_anchor_refiner_from_cache.py"
OUT_ROOT="$ROOT/runs/hfmlk_official_mamba_teacher_nuaa_4gpu"

if [[ "$#" -lt 6 ]]; then
  echo "usage: $0 GPU NAME SEED LR RESIDUAL_SCALE ANCHOR_WEIGHT [extra args...]" >&2
  exit 2
fi

GPU="$1"; shift
NAME="$1"; shift
SEED="$1"; shift
LR="$1"; shift
RESIDUAL_SCALE="$1"; shift
ANCHOR_WEIGHT="$1"; shift

mkdir -p "$OUT_ROOT/$NAME"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec "$PY" "$SCRIPT" \
  --train_cache_spec "IRSTD-1K|$ROOT/reports/fusion_cache_bgm_rpcanet_train_a/IRSTD-1K__train.npz" \
  --train_cache_spec "NUAA-SIRST|$ROOT/reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__train.npz" \
  --train_cache_spec "NUDT-SIRST|$ROOT/reports/fusion_cache_bgm_rpcanet_train_b_test/NUDT-SIRST__train.npz" \
  --train_cache_spec "SIRST3|$ROOT/reports/fusion_cache_bgm_rpcanet_train_b_test/SIRST3__train.npz" \
  --test_cache_spec "NUAA-SIRST|$ROOT/reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__test.npz" \
  --teacher_mode "official_mamba_fused,strong_fusion" \
  --epochs 4 \
  --batch_size 2 \
  --patch_size 512 \
  --base_channels 48 \
  --thresholds "0.468,0.476,0.480,0.488,0.496,0.504,0.512,0.520,0.540" \
  --eval_interval 1 \
  --amp \
  --weighted_dice \
  --focus_dataset NUAA-SIRST \
  --focus_patch_prob 0.35 \
  --focus_hard_threshold 0.45 \
  --focus_miss_threshold 0.18 \
  --resume_checkpoint "$ROOT/runs/strong_fusion_hfmlk_nuaa_4gpu/nuaa_strong_fusion_s703/best.pth.tar" \
  --seed "$SEED" \
  --lr "$LR" \
  --residual_scale "$RESIDUAL_SCALE" \
  --anchor_weight "$ANCHOR_WEIGHT" \
  "$@" \
  --device cuda:0 \
  --output_dir "$OUT_ROOT/$NAME"
