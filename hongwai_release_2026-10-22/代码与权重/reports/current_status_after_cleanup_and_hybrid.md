# Current Status After Cleanup And Hybrid Launch

Generated: 2026-06-12 UTC

## Cleanup

- Removed bulky failed/intermediate run directories after preserving result reports and selected best checkpoints.
- Kept selected checkpoints in `/home/cx/hongwai/irstd-lab/runs/selected_checkpoints`.
- Kept summary reports in `/home/cx/hongwai/irstd-lab/reports`.
- Run storage reduced from about 27G to a few hundred MB plus the new active hybrid run.

Important preserved checkpoints:

- Baseline SCTransNet IRSTD-1K best: `runs/selected_checkpoints/baseline/IRSTD-1K_SCTransNet_500_best.pth.tar`
- Baseline SCTransNet NUAA-SIRST best: `runs/selected_checkpoints/baseline/NUAA-SIRST_SCTransNet_823_best.pth.tar`
- Baseline SCTransNet NUDT-SIRST best: `runs/selected_checkpoints/baseline/NUDT-SIRST_SCTransNet_547_best.pth.tar`
- Baseline SCTransNet SIRST3 best: `runs/selected_checkpoints/baseline/SIRST3_SCTransNet_678_best.pth.tar`
- Previous our-method IRSTD-1K best: `runs/selected_checkpoints/our_method/IRSTD-1K_TCD-SCTransNet-Q_12_best.pth.tar`
- Previous our-method NUAA-SIRST best: `runs/selected_checkpoints/our_method/NUAA-SIRST_TCD-SCTransNet-L_66_best.pth.tar`

## Previous Best Results

- Official SCTransNet evaluation matched README:
  - NUAA-SIRST: 77.64 mIoU vs README 77.50.
  - NUDT-SIRST: 94.11 mIoU vs README 94.09.
  - IRSTD-1K: 68.03 mIoU vs README 68.03.
- From-scratch SCTransNet best:
  - IRSTD-1K: 67.32.
  - NUAA-SIRST: 77.52.
  - NUDT-SIRST: 93.76.
  - SIRST3: 82.96.
- Previous SCTransNet-derived our-method best:
  - IRSTD-1K: 68.05, only +0.02 over README.
  - NUAA-SIRST: 79.10, +1.60 over README.

## New Direction

The new active work is no longer just SCTransNet training-parameter tuning.

Two full-data lines were launched:

- `BGM-BasicUNet`: official-style BGM background masking line using BasicUNet_plus and full-data default mIoU evaluation.
- `TCF-BGM`: a hybrid innovation line combining:
  - BGM/BasicUNet-style encoder-decoder.
  - FeatUp-inspired high-resolution feature up-projection to recover tiny-target evidence after downsampling.
  - Background-normality memory bank inspired by anomaly detection to suppress target-like clutter.
  - SLSIoU and high-frequency background suppression losses.

## Active Runs

tmux session: `irstd_hybrid4`

Run root: `/home/cx/hongwai/irstd-lab/runs/hybrid_4gpu`

- GPU0: IRSTD-1K, BGM-BasicUNet, 400 epochs.
- GPU1: IRSTD-1K, TCF-BGM, 400 epochs.
- GPU2: NUAA-SIRST, BGM-BasicUNet, 400 epochs.
- GPU3: NUAA-SIRST, TCF-BGM, 400 epochs.

Current result table can be regenerated with:

```bash
/home/cx/hongwai/irstd-lab/envs/irstd/bin/python /home/cx/hongwai/irstd-lab/scripts/collect_hybrid_results.py \
  --run_root /home/cx/hongwai/irstd-lab/runs/hybrid_4gpu \
  --output /home/cx/hongwai/irstd-lab/reports/hybrid_results.md
```

## Notes

- `/data1` has about 3.4T free but is not writable by user `cx` at its root, so current active output is under `/home/cx/hongwai/irstd-lab/runs`.
- Root filesystem had about 64G free after cleanup, enough for the current four active runs because only best checkpoints are saved.
- MSHNet smoke test passed, but the first four GPUs were assigned to BGM and TCF-BGM. MSHNet full-data runs should be launched as the next batch or when a GPU frees up.

## 2026-06-12 Follow-up

Added a prepared next-batch launcher for full-data MSHNet/SLS:

- `/home/cx/hongwai/irstd-lab/scripts/launch_mshnet_next_tmux.sh`

Added a new candidate model branch in `train_hybrid_methods.py`:

- `pconv_tcf`: TCF-BGM plus pinwheel/asymmetric local thermal-spot convolution blocks, inspired by recent IRSTD shape-prior methods.

A short smoke test confirmed `pconv_tcf` can train for one epoch. The smoke process and output were then removed so it would not steal GPU memory from active full-data runs.

Latest active full-data hybrid results at this checkpoint:

- IRSTD-1K / BGM-BasicUNet: best 59.37 mIoU at epoch 30.
- IRSTD-1K / TCF-BGM: best 57.46 mIoU at epoch 20.
- NUAA-SIRST / BGM-BasicUNet: best 75.63 mIoU at epoch 105.
- NUAA-SIRST / TCF-BGM: best 73.12 mIoU at epoch 70.

Current decision: keep all four active runs alive. NUAA-BGM is still improving toward the 77.50 README baseline, and IRSTD runs are still early.
