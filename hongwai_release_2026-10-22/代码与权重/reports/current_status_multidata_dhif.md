# Current Status: Multi-Dataset DHiF Refiner

Generated: 2026-06-12 UTC

## New Implementation

- Added multi-dataset training support to `scripts/train_hybrid_methods.py` via `--train_datasets`.
- Verified the four full train splits concatenate correctly:
  - IRSTD-1K: 800
  - NUAA-SIRST: 213
  - NUDT-SIRST: 663
  - SIRST3: 1676
  - Total: 3352 training samples
- Added `sct_dhif_refiner`, a new external-transfer candidate:
  - SCTransNet strong checkpoint as frozen coarse predictor.
  - PConv local target-shape branch.
  - FeatUp-style high-resolution projection.
  - Background-normality memory.
  - Dynamic high-frequency gate inspired by DHiF/SET-style target-vs-clutter frequency separation.

## Preserved Fixed-Refiner Stage

The previous fixed single-dataset refiner was stopped, not deleted, because it produced a small valid improvement:

| Dataset | Method | Best epoch | Best mIoU | README | Delta |
|---|---|---:|---:|---:|---:|
| IRSTD-1K | SCT-Refiner-Fixed | 15 | 68.09 | 68.03 | +0.06 |
| NUAA-SIRST | SCT-Refiner-Fixed | 25 | 79.12 | 77.50 | +1.62 |

Report: `reports/sct_refiner_fixed_results.md`

## Current Active Runs

| GPU | Dataset | Method | Training data | Status |
|---:|---|---|---|---|
| 0 | IRSTD-1K | SCT-DHiF-Multi | IRSTD-1K + NUAA-SIRST + NUDT-SIRST + SIRST3 | active |
| 1 | IRSTD-1K | MSHNet-SLS | IRSTD-1K | active |
| 2 | NUAA-SIRST | SCT-DHiF-Multi | IRSTD-1K + NUAA-SIRST + NUDT-SIRST + SIRST3 | active |
| 3 | NUAA-SIRST | MSHNet-SLS | NUAA-SIRST | active |

## Latest Results

| Dataset | Method | Best epoch | Best mIoU | README | Delta | Note |
|---|---|---:|---:|---:|---:|---|
| IRSTD-1K | SCT-DHiF-Multi | 0 | 68.05 | 68.03 | +0.02 | active |
| NUAA-SIRST | SCT-DHiF-Multi | 5 | 79.11 | 77.50 | +1.61 | active |
| IRSTD-1K | MSHNet-SLS | 70 | 61.51 | 68.03 | -6.52 | active |
| NUAA-SIRST | MSHNet-SLS | - | - | 77.50 | - | active, pre-first-eval |
| NUAA-SIRST | TCF-BGM | 335 | 77.52 | 77.50 | +0.02 | completed at epoch 400; not enough |

## Judgment

- The requested +5 mIoU target is still not achieved.
- The best valid progress remains small: IRSTD +0.06 and NUAA +1.62 from the fixed-refiner stage.
- `NUAA-SIRST / TCF-BGM` completed 400 epochs with only +0.02 over README, so GPU3 was reassigned to the in-domain `MSHNet-SLS` line.
- The active best-aligned experiment is now `SCT-DHiF-Multi`, because it combines the user's requested ingredients:
  - full multi-dataset training,
  - in-domain strong SCTransNet checkpoint,
  - PConv/shape prior,
  - external FeatUp-style feature recovery,
  - anomaly-style background normality,
  - DHiF/SET-style frequency target-clutter separation.
