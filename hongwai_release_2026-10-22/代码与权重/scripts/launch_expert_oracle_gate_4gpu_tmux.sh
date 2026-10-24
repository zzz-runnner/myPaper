#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/train_hybrid_methods.py"
RUN_ROOT="$ROOT/runs/expert_oracle_gate_4gpu"
LOG_DIR="$RUN_ROOT/logs"
SESSION="expert_oracle_gate_4gpu"

mkdir -p "$LOG_DIR"
tmux kill-session -t "$SESSION" 2>/dev/null || true

common_args=(
  --dataset_dir "$ROOT/datasets"
  --train_datasets "IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3"
  --model sct_expert_gate_refiner
  --epochs 24
  --batch_size 32
  --threads 4
  --weight_decay 0.00001
  --warmup_epochs 1
  --min_lr_ratio 0.30
  --begin_test 2
  --every_test 2
  --every_print 1
  --eval_before_train
  --bgm_prob 0.035
  --bgm_mask_ratio 0.026
  --noise_std 0.00020
  --gamma_prob 0.003
  --intensity_jitter 0.0010
  --bce_weight 0.70
  --dice_weight 0.25
  --focal_weight 0.018
  --sls_iou_weight 0.006
  --sls_iou_warm_epochs 1
  --bg_hf_weight 0.020
  --bg_pred_weight 0.002
  --base_anchor_bg_weight 1.0
  --base_anchor_pos_weight 0.70
  --base_anchor_logit_weight 0.075
  --base_anchor_bg_threshold 0.08
  --base_anchor_pos_threshold 0.35
  --base_anchor_margin 0.0025
  --base_anchor_hf_boost 1.4
  --base_anchor_target_dilate 11
  --expert_gate_oracle_margin 0.018
  --expert_gate_min_disagree 0.012
  --expert_gate_target_dilate 7
  --expert_gate_pos_boost 2.4
  --expert_gate_disagree_boost 1.8
  --expert_gate_hf_boost 0.9
  --thermal_spot_sigma_scale 0.70
  --thermal_spot_sigma_min 1.0
  --thermal_spot_sigma_max 4.0
  --thermal_spot_guard 11
  --thermal_spot_core_boost 1.4
  --thermal_spot_margin 0.07
  --thermal_spot_contrast_weight 0.18
  --thermal_spot_scale_weight 0.04
)

run_cmd() {
  local gpu="$1"
  local dataset="$2"
  local target_repeat="$3"
  local method="$4"
  local base_ckpt="$5"
  local primary_ckpt="$6"
  local expert_model="$7"
  local expert_base="$8"
  local expert_ckpt="$9"
  local lr="${10}"
  local seed="${11}"
  local anchor_weight="${12}"
  local oracle_weight="${13}"
  local sparse_weight="${14}"
  local thermal_weight="${15}"
  local pos_prob="${16}"
  local run_dir="$RUN_ROOT/${dataset}_${method}_gpu${gpu}"

  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$SCRIPT" "${common_args[@]}" \
    --dataset_name "$dataset" \
    --target_repeat "$target_repeat" \
    --run_dir "$run_dir" \
    --method_name "$method" \
    --base_checkpoint "$base_ckpt" \
    --resume_model_checkpoint "$primary_ckpt" \
    --expert_model "$expert_model" \
    --expert_base_checkpoint "$expert_base" \
    --expert_checkpoint "$expert_ckpt" \
    --lr "$lr" \
    --seed "$seed" \
    --base_anchor_weight "$anchor_weight" \
    --expert_gate_oracle_weight "$oracle_weight" \
    --expert_gate_sparsity_weight "$sparse_weight" \
    --thermal_spot_weight "$thermal_weight" \
    --pos_prob "$pos_prob"
}

tmux new-session -d -s "$SESSION" -n irstd_oracle_a "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 0 IRSTD-1K 6 SCT-ExpertOracleGate-A '$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar' '$ROOT/runs/sct_refiner_fixed/IRSTD-1K_SCT-Refiner-Fixed_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar' sct_tpc_dhif_refiner '$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar' '$ROOT/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-A_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar' 0.0000070 2111 2.30 0.55 0.018 0.004 0.60 2>&1 | tee '$LOG_DIR/IRSTD-1K_SCT-ExpertOracleGate-A_gpu0.log'"

tmux new-window -t "$SESSION" -n irstd_oracle_b "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 1 IRSTD-1K 7 SCT-ExpertOracleGate-B '$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar' '$ROOT/runs/sct_refiner_fixed/IRSTD-1K_SCT-Refiner-Fixed_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar' sct_tpc_dhif_refiner '$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar' '$ROOT/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-B_gpu1/checkpoints/IRSTD-1K/best_mIoU.pth.tar' 0.0000060 2113 2.60 0.65 0.022 0.004 0.62 2>&1 | tee '$LOG_DIR/IRSTD-1K_SCT-ExpertOracleGate-B_gpu1.log'"

tmux new-window -t "$SESSION" -n nuaa_oracle_a "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 2 NUAA-SIRST 5 SCT-ExpertOracleGate-A '$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar' '$ROOT/runs/sct_refiner_fixed/NUAA-SIRST_SCT-Refiner-Fixed_gpu2/checkpoints/NUAA-SIRST/best_mIoU.pth.tar' sct_tpc_dhif_refiner '$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar' '$ROOT/runs/tpc_dhif_teacher4/NUAA-SIRST_SCT-TPC-DHiF-A_gpu2/checkpoints/NUAA-SIRST/best_mIoU.pth.tar' 0.0000040 2115 3.10 0.32 0.035 0.002 0.56 2>&1 | tee '$LOG_DIR/NUAA-SIRST_SCT-ExpertOracleGate-A_gpu2.log'"

tmux new-window -t "$SESSION" -n nuaa_oracle_b "cd '$ROOT' && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && $(declare -p common_args); $(declare -f run_cmd); RUN_ROOT='$RUN_ROOT'; PY='$PY'; SCRIPT='$SCRIPT'; run_cmd 3 NUAA-SIRST 6 SCT-ExpertOracleGate-B '$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar' '$ROOT/runs/sct_refiner_fixed/NUAA-SIRST_SCT-Refiner-Fixed_gpu2/checkpoints/NUAA-SIRST/best_mIoU.pth.tar' sct_lca_refiner '$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar' '$ROOT/runs/selected_checkpoints/lca/NUAA-SIRST_SCT-LCA-B_epoch5_best.pth.tar' 0.0000035 2117 3.30 0.30 0.040 0.002 0.58 2>&1 | tee '$LOG_DIR/NUAA-SIRST_SCT-ExpertOracleGate-B_gpu3.log'"

tmux ls | grep "$SESSION"
