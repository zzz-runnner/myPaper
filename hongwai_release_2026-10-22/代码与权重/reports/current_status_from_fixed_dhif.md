# Current Status: From-Fixed DHiF

Generated: 2026-06-12 UTC

## Key Fix

- Added `--resume_model_checkpoint` to `train_hybrid_methods.py`.
- Added zero-gain identity initialization to `DynamicHighFreqGate`.
- This fixed the problem where inserting DHiF into a trained refiner changed predictions at epoch 0.
- `SCT-DHiF-FromFixed` now starts exactly from the preserved `SCT-Refiner-Fixed` best checkpoints, then trains the new DHiF gate.

## Current Active Runs

| GPU | Dataset | Method | Starting point | Status |
|---:|---|---|---|---|
| 0 | IRSTD-1K | SCT-DHiF-FromFixed | fixed-refiner best 68.09 | active |
| 1 | IRSTD-1K | SCT-DHiF-Balanced2 | base checkpoint 68.05 | active |
| 2 | NUAA-SIRST | SCT-DHiF-FromFixed | fixed-refiner best 79.12 | active |
| 3 | NUAA-SIRST | MSHNet-SLS | from scratch MSHNet | active |

## Latest Results

| Dataset | Method | Best epoch | Best mIoU | README | Delta |
|---|---|---:|---:|---:|---:|
| IRSTD-1K | SCT-DHiF-FromFixed | 0 | 68.09 | 68.03 | +0.06 |
| NUAA-SIRST | SCT-DHiF-FromFixed | 0 | 79.12 | 77.50 | +1.62 |
| IRSTD-1K | SCT-DHiF-Balanced2 | 0 | 68.05 | 68.03 | +0.02 |
| NUAA-SIRST | MSHNet-SLS | 235 | 76.91 | 77.50 | -0.59 |

## Stopped Or Replaced

- `SCT-DHiF-Multi` stopped because equal multi-dataset sampling underperformed fixed refiner.
- `SCT-DHiF-Balanced` on GPU0/GPU2 stopped because it still did not beat fixed refiner before FromFixed became available.
- `IRSTD-1K / MSHNet-SLS` stopped at 61.57 mIoU; its weak checkpoint was deleted, logs/report retained.

## Status Against Goal

The +5 mIoU target is not achieved. The strongest preserved results remain:

- IRSTD-1K: 68.09, only +0.06 over README 68.03.
- NUAA-SIRST: 79.12, +1.62 over README 77.50.

The active best-aligned direction is now no-loss DHiF insertion from the fixed-refiner best checkpoint plus target-balanced multi-dataset training.
