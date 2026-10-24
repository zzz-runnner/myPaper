#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cx/hongwai/irstd-lab"
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/generate_sam_refined_pseudo_labels.py"
LOG_DIR="$ROOT/generated/logs"
mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --datasets IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3
  --split train
  --fusion soft_rim
  --teacher_core_threshold 0.50
  --sam_rim_value 0.16
  --sam_rim_teacher_scale 0.45
  --sam_rim_min 0.18
  --sam_rim_max 0.38
  --teacher_threshold 0.40
  --box_pad 8
  --negative_points 4
  --sam_guard_dilate 9
  --max_prompts 8
  --max_sam_area_mult 5.0
  --max_sam_area_ratio 0.008
  --min_mask_teacher_mean 0.20
  --sam_conf_base 0.24
  --sam_conf_score_weight 0.13
  --sam_conf_cover_weight 0.10
  --sam_conf_teacher_weight 0.12
  --sam_conf_min 0.52
  --sam_conf_max 0.68
  --eval_thresholds 0.2,0.35,0.5,0.7
  --num_shards 2
  --skip_existing
)

launch() {
  local session="$1"
  local gpu="$2"
  local shard="$3"
  local out_dir="$4"
  shift 4
  tmux new-session -d -s "$session" \
    "cd '$ROOT' && CUDA_VISIBLE_DEVICES=$gpu '$PY' '$SCRIPT' ${COMMON_ARGS[*]} --device cuda:0 --out_dir '$out_dir' --shard_index '$shard' $* 2>&1 | tee '$LOG_DIR/${session}.log'"
}

IRSTD_OUT="$ROOT/generated/sam_refined_irstd_tpc_rim_train"
NUAA_OUT="$ROOT/generated/sam_refined_nuaa_lca_tcd_rim_train"

launch sam_pseudo_irstd_shard0 0 0 "$IRSTD_OUT" \
  --teacher 'name=tpc_a,model=sct_tpc_dhif_refiner,checkpoint=/home/cx/hongwai/irstd-lab/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-A_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar,base_checkpoint=/home/cx/hongwai/irstd-lab/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar' \
  --teacher 'name=tpc_b,model=sct_tpc_dhif_refiner,checkpoint=/home/cx/hongwai/irstd-lab/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-B_gpu1/checkpoints/IRSTD-1K/best_mIoU.pth.tar,base_checkpoint=/home/cx/hongwai/irstd-lab/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar'

launch sam_pseudo_irstd_shard1 1 1 "$IRSTD_OUT" \
  --teacher 'name=tpc_a,model=sct_tpc_dhif_refiner,checkpoint=/home/cx/hongwai/irstd-lab/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-A_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar,base_checkpoint=/home/cx/hongwai/irstd-lab/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar' \
  --teacher 'name=tpc_b,model=sct_tpc_dhif_refiner,checkpoint=/home/cx/hongwai/irstd-lab/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-B_gpu1/checkpoints/IRSTD-1K/best_mIoU.pth.tar,base_checkpoint=/home/cx/hongwai/irstd-lab/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar'

launch sam_pseudo_nuaa_shard0 2 0 "$NUAA_OUT" \
  --teacher 'name=lca,model=sct_lca_refiner,checkpoint=/home/cx/hongwai/irstd-lab/runs/selected_checkpoints/lca/NUAA-SIRST_SCT-LCA-B_epoch5_best.pth.tar,base_checkpoint=/home/cx/hongwai/irstd-lab/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar' \
  --teacher 'name=tcd,model=sct_tcd_decouple_refiner,checkpoint=/home/cx/hongwai/irstd-lab/runs/selected_checkpoints/tcd_decouple/NUAA-SIRST_SCT-TCD-Decouple-A_epoch2_best.pth.tar,base_checkpoint=/home/cx/hongwai/irstd-lab/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar'

launch sam_pseudo_nuaa_shard1 3 1 "$NUAA_OUT" \
  --teacher 'name=lca,model=sct_lca_refiner,checkpoint=/home/cx/hongwai/irstd-lab/runs/selected_checkpoints/lca/NUAA-SIRST_SCT-LCA-B_epoch5_best.pth.tar,base_checkpoint=/home/cx/hongwai/irstd-lab/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar' \
  --teacher 'name=tcd,model=sct_tcd_decouple_refiner,checkpoint=/home/cx/hongwai/irstd-lab/runs/selected_checkpoints/tcd_decouple/NUAA-SIRST_SCT-TCD-Decouple-A_epoch2_best.pth.tar,base_checkpoint=/home/cx/hongwai/irstd-lab/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar'

tmux ls
