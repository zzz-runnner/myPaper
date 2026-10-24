#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/cx/hongwai/irstd-lab
PY="$ROOT/envs/irstd/bin/python"
SCRIPT="$ROOT/scripts/eval_fixed_weighted_fusion.py"
REPORT="$ROOT/reports"
LOGDIR="$REPORT/fixed_weighted_fusion_logs"
RUNDIR="$REPORT/fixed_weighted_fusion_runners"
mkdir -p "$LOGDIR" "$RUNDIR"

COMMON_THRESH="0.35,0.40,0.45,0.50,0.55,0.60"
COMMON_GAMMAS="0.85,0.90,1.00,1.10,1.15"

write_runner() {
  local name="$1"
  local gpu="$2"
  local body="$3"
  local runner="$RUNDIR/${name}.sh"
  cat > "$runner" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=$gpu
$body
EOF
  chmod +x "$runner"
  tmux kill-session -t "$name" 2>/dev/null || true
  tmux new-session -d -s "$name" "$runner > '$LOGDIR/${name}.log' 2>&1"
}

IRSTD_BASE="--dataset_dir $ROOT/datasets --dataset_name IRSTD-1K --output_thresholds $COMMON_THRESH --gammas $COMMON_GAMMAS"
IRSTD_TEACHERS="--model_spec 'baseq|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/IRSTD-1K_SCT-TPC-DHiF-B_gpu1/checkpoints/IRSTD-1K/best_mIoU.pth.tar' --model_spec 'refiner|sct_refiner|$ROOT/runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar|$ROOT/runs/sct_refiner_fixed/IRSTD-1K_SCT-Refiner-Fixed_gpu0/checkpoints/IRSTD-1K/best_mIoU.pth.tar'"
IRSTD_D05="--official_pconv_mshnet_spec 'pconv_d05|mshnet|$ROOT/external/PConv-SDloss-Data/weight_extracted/MSHNet-IRSTD-1K-D-0.5/weight.pkl|official_mshnet|0.35|false|256'"
IRSTD_P43="--official_pconv_mshnet_spec 'pconv_p43|p43|$ROOT/external/PConv-SDloss-Data/weight_extracted/MSHNet-IRSTD-1K-P43/weight.pkl|official_mshnet|0.60|false|256'"
IRSTD_DENSE_WEIGHTS="--weight_spec 't55_d05heavy:baseq=0.275,refiner=0.275,pconv_d05=0.30,pconv_p43=0.15' --weight_spec 't55_p43heavy:baseq=0.275,refiner=0.275,pconv_d05=0.15,pconv_p43=0.30' --weight_spec 't60_d05heavy:baseq=0.30,refiner=0.30,pconv_d05=0.25,pconv_p43=0.15' --weight_spec 't60_p43heavy:baseq=0.30,refiner=0.30,pconv_d05=0.15,pconv_p43=0.25' --weight_spec 't65_d05heavy:baseq=0.325,refiner=0.325,pconv_d05=0.22,pconv_p43=0.13' --weight_spec 't65_p43heavy:baseq=0.325,refiner=0.325,pconv_d05=0.13,pconv_p43=0.22' --weight_spec 't70_d05heavy:baseq=0.35,refiner=0.35,pconv_d05=0.20,pconv_p43=0.10' --weight_spec 't70_p43heavy:baseq=0.35,refiner=0.35,pconv_d05=0.10,pconv_p43=0.20'"

NUAA_BASE="--dataset_dir $ROOT/datasets --dataset_name NUAA-SIRST --output_thresholds $COMMON_THRESH --gammas $COMMON_GAMMAS"
NUAA_P43="--official_pconv_mshnet_spec 'pconv_p43|p43|$ROOT/external/PConv-SDloss-Data/weight_extracted/MSHNet-IRSTD-1K-P43/weight.pkl|official_mshnet|0.60|false|256'"
NUAA_TPC_LCA="--model_spec 'tpc_a|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/NUAA-SIRST_SCT-TPC-DHiF-A_gpu2/checkpoints/NUAA-SIRST/best_mIoU.pth.tar' --model_spec 'tpc_b|sct_tpc_dhif_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/tpc_dhif_teacher4/NUAA-SIRST_SCT-TPC-DHiF-B_gpu3/checkpoints/NUAA-SIRST/best_mIoU.pth.tar' --model_spec 'lca|sct_lca_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/selected_checkpoints/lca/NUAA-SIRST_SCT-LCA-B_epoch5_best.pth.tar'"
NUAA_LCA_TCD="--model_spec 'lca|sct_lca_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/selected_checkpoints/lca/NUAA-SIRST_SCT-LCA-B_epoch5_best.pth.tar' --model_spec 'tcd_a|sct_tcd_decouple_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/selected_checkpoints/tcd_decouple/NUAA-SIRST_SCT-TCD-Decouple-A_epoch2_best.pth.tar' --model_spec 'tcd_b|sct_tcd_decouple_refiner|$ROOT/runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar|$ROOT/runs/selected_checkpoints/tcd_decouple/NUAA-SIRST_SCT-TCD-Decouple-B_epoch2_best.pth.tar'"
NUAA_WEIGHTS="--weight_spec 'teacher80_p43:tpc_a=0.2667,tpc_b=0.2667,lca=0.2666,pconv_p43=0.20' --weight_spec 'teacher90_p43:tpc_a=0.30,tpc_b=0.30,lca=0.30,pconv_p43=0.10'"

write_runner fixed_weighted_irstd_dual 0 "$PY $SCRIPT $IRSTD_BASE $IRSTD_TEACHERS $IRSTD_D05 $IRSTD_P43 $IRSTD_DENSE_WEIGHTS --full_metric_topk 4 --output $REPORT/fixed_weighted_irstd_dual.json"
write_runner fixed_weighted_irstd_p43 1 "$PY $SCRIPT $IRSTD_BASE $IRSTD_TEACHERS $IRSTD_P43 --output $REPORT/fixed_weighted_irstd_p43.json"
write_runner fixed_weighted_nuaa_tpc_lca_p43 2 "$PY $SCRIPT $NUAA_BASE $NUAA_TPC_LCA $NUAA_P43 $NUAA_WEIGHTS --output $REPORT/fixed_weighted_nuaa_tpc_lca_p43.json"
write_runner fixed_weighted_nuaa_lca_tcd_p43 3 "$PY $SCRIPT $NUAA_BASE $NUAA_LCA_TCD $NUAA_P43 --weight_spec 'teacher80_p43:lca=0.2667,tcd_a=0.2667,tcd_b=0.2666,pconv_p43=0.20' --weight_spec 'teacher90_p43:lca=0.30,tcd_a=0.30,tcd_b=0.30,pconv_p43=0.10' --output $REPORT/fixed_weighted_nuaa_lca_tcd_p43.json"

tmux ls
