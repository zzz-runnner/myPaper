#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/cx/hongwai/irstd-lab
PY=$ROOT/envs/irstd/bin/python
RUN_ROOT=$ROOT/runs/official_mshnet_finetune_4gpu
LOG_DIR=$RUN_ROOT/logs
TRAIN_DATASETS=IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3

mkdir -p "$LOG_DIR"

tmux new-session -d -s official_mshnet_ft4 -n gpu0 "cd $ROOT && CUDA_VISIBLE_DEVICES=0 $PY scripts/train_hybrid_methods.py \
  --dataset_name IRSTD-1K --train_datasets $TRAIN_DATASETS --target_repeat 4 \
  --run_dir $RUN_ROOT/IRSTD-1K_MSHNet-D05_gpu0 --method_name MSHNet-D05-FullData \
  --model mshnet3_sls --resume_model_checkpoint external/PConv-SDloss-Data/weight_extracted/MSHNet-IRSTD-1K-D-0.5/weight.pkl \
  --image_preprocess official_mshnet --optimizer adagrad --lr 0.006 --weight_decay 0.00001 \
  --batch_size 24 --patch_size 256 --pos_prob 0.70 --official_scale_aug --official_blur_prob 0.20 \
  --bce_weight 0.0 --dice_weight 0.0 --focal_weight 0.0 --sls_iou_weight 1.0 --sls_iou_warm_epochs 2 \
  --bg_hf_weight 0.02 --bg_pred_weight 0.01 --sdm_iou_weight 0.20 --sdm_warm_epochs 2 --sdm_with_distance \
  --epochs 80 --warmup_epochs 3 --min_lr_ratio 0.10 --threshold 0.35 \
  --begin_test 1 --every_test 2 --every_print 1 --eval_before_train --save_final \
  > $LOG_DIR/IRSTD-1K_MSHNet-D05_gpu0.log 2>&1"

tmux new-window -t official_mshnet_ft4 -n gpu1 "cd $ROOT && CUDA_VISIBLE_DEVICES=1 $PY scripts/train_hybrid_methods.py \
  --dataset_name IRSTD-1K --train_datasets $TRAIN_DATASETS --target_repeat 4 \
  --run_dir $RUN_ROOT/IRSTD-1K_MSHNet-P43_gpu1 --method_name MSHNet-P43-FullData \
  --model mshnet_p43_sls --resume_model_checkpoint external/PConv-SDloss-Data/weight_extracted/MSHNet-IRSTD-1K-P43/weight.pkl \
  --image_preprocess official_mshnet --optimizer adagrad --lr 0.006 --weight_decay 0.00001 \
  --batch_size 24 --patch_size 256 --pos_prob 0.70 --official_scale_aug --official_blur_prob 0.20 \
  --bce_weight 0.0 --dice_weight 0.0 --focal_weight 0.0 --sls_iou_weight 1.0 --sls_iou_warm_epochs 2 \
  --bg_hf_weight 0.02 --bg_pred_weight 0.01 --sdm_iou_weight 0.20 --sdm_warm_epochs 2 --sdm_with_distance \
  --epochs 80 --warmup_epochs 3 --min_lr_ratio 0.10 --threshold 0.60 \
  --begin_test 1 --every_test 2 --every_print 1 --eval_before_train --save_final \
  > $LOG_DIR/IRSTD-1K_MSHNet-P43_gpu1.log 2>&1"

tmux new-window -t official_mshnet_ft4 -n gpu2 "cd $ROOT && CUDA_VISIBLE_DEVICES=2 $PY scripts/train_hybrid_methods.py \
  --dataset_name NUAA-SIRST --train_datasets $TRAIN_DATASETS --target_repeat 8 \
  --run_dir $RUN_ROOT/NUAA-SIRST_MSHNet-D05_gpu2 --method_name MSHNet-D05-FullData \
  --model mshnet3_sls --resume_model_checkpoint external/PConv-SDloss-Data/weight_extracted/MSHNet-IRSTD-1K-D-0.5/weight.pkl \
  --image_preprocess official_mshnet --optimizer adagrad --lr 0.004 --weight_decay 0.00001 \
  --batch_size 24 --patch_size 256 --pos_prob 0.75 --official_scale_aug --official_blur_prob 0.20 \
  --bce_weight 0.0 --dice_weight 0.0 --focal_weight 0.0 --sls_iou_weight 1.0 --sls_iou_warm_epochs 2 \
  --bg_hf_weight 0.02 --bg_pred_weight 0.01 --sdm_iou_weight 0.20 --sdm_warm_epochs 2 --sdm_with_distance \
  --epochs 80 --warmup_epochs 3 --min_lr_ratio 0.10 --threshold 0.35 \
  --begin_test 1 --every_test 2 --every_print 1 --eval_before_train --save_final \
  > $LOG_DIR/NUAA-SIRST_MSHNet-D05_gpu2.log 2>&1"

tmux new-window -t official_mshnet_ft4 -n gpu3 "cd $ROOT && CUDA_VISIBLE_DEVICES=3 $PY scripts/train_hybrid_methods.py \
  --dataset_name NUAA-SIRST --train_datasets $TRAIN_DATASETS --target_repeat 8 \
  --run_dir $RUN_ROOT/NUAA-SIRST_MSHNet-P43_gpu3 --method_name MSHNet-P43-FullData \
  --model mshnet_p43_sls --resume_model_checkpoint external/PConv-SDloss-Data/weight_extracted/MSHNet-IRSTD-1K-P43/weight.pkl \
  --image_preprocess official_mshnet --optimizer adagrad --lr 0.004 --weight_decay 0.00001 \
  --batch_size 24 --patch_size 256 --pos_prob 0.75 --official_scale_aug --official_blur_prob 0.20 \
  --bce_weight 0.0 --dice_weight 0.0 --focal_weight 0.0 --sls_iou_weight 1.0 --sls_iou_warm_epochs 2 \
  --bg_hf_weight 0.02 --bg_pred_weight 0.01 --sdm_iou_weight 0.20 --sdm_warm_epochs 2 --sdm_with_distance \
  --epochs 80 --warmup_epochs 3 --min_lr_ratio 0.10 --threshold 0.60 \
  --begin_test 1 --every_test 2 --every_print 1 --eval_before_train --save_final \
  > $LOG_DIR/NUAA-SIRST_MSHNet-P43_gpu3.log 2>&1"

tmux ls | grep official_mshnet_ft4
