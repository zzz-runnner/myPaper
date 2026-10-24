#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/cx/hongwai/irstd-lab
PY=$ROOT/envs/irstd/bin/python
LOGDIR=$ROOT/reports/component_mlp_selector_logs
RUNDIR=$ROOT/reports/component_mlp_selector_runners
mkdir -p "$LOGDIR" "$RUNDIR"

FIT_DATASETS=IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3
COMMON="--dataset_dir $ROOT/datasets --fit_datasets $FIT_DATASETS --seed_thresholds 0.45 --keep_thresholds 0.30,0.50,0.70 --output_thresholds 0.45,0.50 --fill_modes avg,max --steps 250 --seeds 0 --pos_weight_scales 0.60,1.00 --max_components_per_image 80"

IRSTD_BASE="--model_spec 'baseq|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-B_gpu1/checkpoints/IRSTD-1K/best_mIoU.pth.tar' --model_spec 'refiner|sct_refiner|$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar|$ROOT/runs/sct_refiner_fixed/IRSTD-1K_SCT-Refiner-Fixed_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar'"
NUAA_TPC_LCA="--model_spec 'tpc_a|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/NUAA-SIRST_SCT-TPC-DHiF-A_gpu2/checkpoints/NUAA-SIRST/best_mIoU.pth.tar' --model_spec 'tpc_b|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/NUAA-SIRST_SCT-TPC-DHiF-B_gpu3/checkpoints/NUAA-SIRST/best_mIoU.pth.tar' --model_spec 'lca|sct_lca_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/selected_checkpoints/lca/NUAA-SIRST_SCT-LCA-B_epoch5_best.pth.tar'"
NUAA_LCA_TCD="--model_spec 'lca|sct_lca_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/selected_checkpoints/lca/NUAA-SIRST_SCT-LCA-B_epoch5_best.pth.tar' --model_spec 'tcd_a|sct_tcd_decouple_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/selected_checkpoints/tcd_decouple/NUAA-SIRST_SCT-TCD-Decouple-A_epoch2_best.pth.tar' --model_spec 'tcd_b|sct_tcd_decouple_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/selected_checkpoints/tcd_decouple/NUAA-SIRST_SCT-TCD-Decouple-B_epoch2_best.pth.tar'"

PCONV_D05="--official_pconv_mshnet_spec 'pconv_d05|mshnet|$ROOT/external/PConv-SDloss-Data/weight_extracted/MSHNet-IRSTD-1K-D-0.5/weight.pkl|official_mshnet|0.35|false|256'"
PCONV_P43="--official_pconv_mshnet_spec 'pconv_p43|p43|$ROOT/external/PConv-SDloss-Data/weight_extracted/MSHNet-IRSTD-1K-P43/weight.pkl|official_mshnet|0.60|false|256'"

launch() {
  local session=$1
  local gpu=$2
  local dataset=$3
  local output=$4
  local specs=$5
  local extra=$6
  local runner=$RUNDIR/$session.sh
  cat > "$runner" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=$gpu
$PY $ROOT/scripts/eval_component_mlp_selector.py $COMMON --dataset_name $dataset $specs $extra --device cuda:0 --output $output 2>&1 | tee "$LOGDIR/$session.log"
EOF
  chmod +x "$runner"
  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" "$runner"
}

launch component_mlp_fast_irstd_p43 0 IRSTD-1K "$ROOT/reports/component_mlp_selector_fast_irstd_p43.json" "$IRSTD_BASE" "$PCONV_P43"
launch component_mlp_fast_irstd_dual 1 IRSTD-1K "$ROOT/reports/component_mlp_selector_fast_irstd_dual_d05_p43.json" "$IRSTD_BASE" "$PCONV_D05 $PCONV_P43"
launch component_mlp_fast_nuaa_tpc_lca_p43 2 NUAA-SIRST "$ROOT/reports/component_mlp_selector_fast_nuaa_tpc_lca_p43.json" "$NUAA_TPC_LCA" "$PCONV_P43"
launch component_mlp_fast_nuaa_lca_tcd_p43 3 NUAA-SIRST "$ROOT/reports/component_mlp_selector_fast_nuaa_lca_tcd_p43.json" "$NUAA_LCA_TCD" "$PCONV_P43"

tmux ls | grep component_mlp || true
