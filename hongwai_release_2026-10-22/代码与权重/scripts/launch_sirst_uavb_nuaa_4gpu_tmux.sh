#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
RUN_ROOT="$ROOT/runs/sirst_uavb_nuaa_4gpu"
LOG_ROOT="$RUN_ROOT/logs"
SESSION="sirst_uavb_nuaa_4gpu"

mkdir -p "$LOG_ROOT"
tmux kill-session -t "$SESSION" 2>/dev/null || true

TRAIN_DATASETS="IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3,SIRST-UAVB"
NUAA_BASE="$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar"
NUAA_LCA="$ROOT/runs/selected_checkpoints/lca/NUAA-SIRST_SCT-LCA-B_epoch5_best.pth.tar"
NUAA_TCD="$ROOT/runs/selected_checkpoints/tcd_decouple/NUAA-SIRST_SCT-TCD-Decouple-A_epoch2_best.pth.tar"
NUAA_TPC="$ROOT/runs/tpc_dhif_teacher4/NUAA-SIRST_SCT-TPC-DHiF-A_gpu2/checkpoints/NUAA-SIRST/best_mIoU.pth.tar"

common_args=(
  --dataset_dir "$ROOT/datasets"
  --dataset_name "NUAA-SIRST"
  --train_datasets "$TRAIN_DATASETS"
  --target_repeat 5
  --epochs 80
  --begin_test 1
  --every_test 1
  --every_print 20
  --patch_size 256
  --threads 4
  --threshold 0.5
  --warmup_epochs 3
  --min_lr_ratio 0.04
  --weight_decay 0.00005
  --pos_prob 0.58
  --noise_std 0.003
  --gamma_prob 0.08
  --intensity_jitter 0.012
  --bgm_prob 0.16
  --bgm_mask_ratio 0.08
  --copy_paste_prob_by_dataset "SIRST-UAVB=0.25,NUAA-SIRST=0.08"
  --copy_paste_max_targets_by_dataset "SIRST-UAVB=2,NUAA-SIRST=1"
  --copy_paste_max_area 260
  --copy_paste_scale_min 0.70
  --copy_paste_scale_max 1.25
  --bce_weight 1.0
  --dice_weight 0.38
  --focal_weight 0.05
  --small_weight 0.65
  --pos_weight 0.86
  --hard_neg_weight 2.8
  --bg_hf_weight 0.20
  --bg_pred_weight 0.055
)

run_cmd() {
  local gpu="$1"; shift
  local name="$1"; shift
  local model="$1"; shift
  local base="$1"; shift
  local resume="$1"; shift
  local lr="$1"; shift
  local batch="$1"; shift
  local seed="$1"; shift
  local extra=("$@")
  rm -rf "$RUN_ROOT/$name"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$SCRIPT" \
    "${common_args[@]}" \
    --run_dir "$RUN_ROOT/$name" \
    --method_name "$name" \
    --model "$model" \
    --base_checkpoint "$base" \
    --lr "$lr" \
    --batch_size "$batch" \
    --seed "$seed" \
    "${extra[@]}"
}

tmux new-session -d -s "$SESSION" -n lca_uavb \
  "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 0 NUAA-UAVB-LCA sct_lca_refiner '$NUAA_BASE' '' 0.0000022 56 6101 --resume_model_checkpoint '$NUAA_LCA' --base_anchor_weight 0.18 --base_anchor_margin 0.005 --teacher_weight 0.06 --teacher_model sct_tpc_dhif_refiner --teacher_base_checkpoint '$NUAA_BASE' --teacher_checkpoint '$NUAA_TPC' 2>&1 | tee '$LOG_ROOT/NUAA-UAVB-LCA_gpu0.log'"

tmux new-window -t "$SESSION" -n tcd_uavb \
  "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 1 NUAA-UAVB-TCD sct_tcd_decouple_refiner '$NUAA_BASE' '' 0.0000020 56 6103 --resume_model_checkpoint '$NUAA_TCD' --tcd_gate_weight 0.06 --thermal_spot_weight 0.012 --base_anchor_weight 0.20 --base_anchor_margin 0.005 2>&1 | tee '$LOG_ROOT/NUAA-UAVB-TCD_gpu1.log'"

tmux new-window -t "$SESSION" -n bng_uavb \
  "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 2 NUAA-UAVB-BNG sct_bng_featup_refiner '$NUAA_BASE' '' 0.0000024 28 6105 --resume_model_checkpoint '$NUAA_LCA' --thermal_spot_weight 0.016 --base_anchor_weight 0.18 --base_anchor_margin 0.005 2>&1 | tee '$LOG_ROOT/NUAA-UAVB-BNG_gpu2.log'"

tmux new-window -t "$SESSION" -n apconv_uavb \
  "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 3 NUAA-UAVB-APConv sct_apconv_tpc_refiner '$NUAA_BASE' '' 0.0000022 32 6107 --resume_model_checkpoint '$NUAA_TPC' --teacher_weight 0.04 --teacher_model sct_lca_refiner --teacher_base_checkpoint '$NUAA_BASE' --teacher_checkpoint '$NUAA_LCA' --base_anchor_weight 0.18 --base_anchor_margin 0.005 2>&1 | tee '$LOG_ROOT/NUAA-UAVB-APConv_gpu3.log'"

echo "launched tmux session: $SESSION"
