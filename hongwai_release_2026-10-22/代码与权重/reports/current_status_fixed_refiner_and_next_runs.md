# Current Status: Fixed Refiner And Next Full Runs

Generated: 2026-06-12 UTC

## What Changed

- Stopped the broken `SCT-Refiner` runs in `runs/sct_refiner_full`.
- Diagnosed the bug: `train_hybrid_methods.py` reinitialized the whole model after loading the SCTransNet checkpoint, destroying the pretrained base.
- Fixed `SCTRefiner` so only the new residual branch is initialized.
- Added input-adapter checkpoint compatibility for IRSTD-1K `TCD-SCTransNet-Q`, so the refiner preserves the actual previous best checkpoint behavior.
- Added `--eval_only` and `--eval_before_train` to verify and log the frozen-base starting point.
- Deleted failed/debug refiner outputs after recording the cause.

## Verified Strong Starting Points

Corrected `SCT-Refiner-Fixed` now starts from the preserved best checkpoints instead of collapsing:

| Dataset | Checkpoint source | Epoch | mIoU | README | Delta |
|---|---|---:|---:|---:|---:|
| IRSTD-1K | `TCD-SCTransNet-Q_12_best` | 0 | 68.05 | 68.03 | +0.02 |
| NUAA-SIRST | `TCD-SCTransNet-L_66_best` | 0 | 79.10 | 77.50 | +1.60 |

## Current Active Runs

| GPU | Run | Method line | Run root | Status |
|---:|---|---|---|---|
| 0 | IRSTD-1K | `SCT-Refiner-Fixed` | `runs/sct_refiner_fixed/IRSTD-1K_SCT-Refiner-Fixed_gpu0` | active |
| 1 | IRSTD-1K | `MSHNet-SLS` | `runs/mshnet_irstd_gpu1/IRSTD-1K_MSHNet-SLS_gpu1` | active |
| 2 | NUAA-SIRST | `SCT-Refiner-Fixed` | `runs/sct_refiner_fixed/NUAA-SIRST_SCT-Refiner-Fixed_gpu2` | active |
| 3 | NUAA-SIRST | `TCF-BGM` | `runs/hybrid_4gpu/NUAA-SIRST_TCF-BGM_gpu3` | active |

Latest observed GPU memory after restart:

| GPU | Memory used | Note |
|---:|---:|---|
| 0 | about 7.2G / 11G | refiner batch 64 |
| 1 | about 4.2G / 11G | MSHNet-SLS batch 18 |
| 2 | about 5.5G / 11G | refiner batch 64 |
| 3 | about 10.1G / 11G | NUAA TCF-BGM batch 18 |

## Latest Results

`TCF-BGM`:

| Dataset | Best epoch | Best mIoU | README | Delta |
|---|---:|---:|---:|---:|
| IRSTD-1K | 65 | 61.41 | 68.03 | -6.62 |
| NUAA-SIRST | 300 | 77.24 | 77.50 | -0.26 |

The IRSTD-1K TCF-BGM run was stopped because it remained far below SCTransNet after 80 epochs. NUAA-SIRST TCF-BGM is still running because it is close to the README baseline.

`SCT-Refiner-Fixed`:

| Dataset | Best epoch | Best mIoU | README | Delta |
|---|---:|---:|---:|---:|
| IRSTD-1K | 0 | 68.05 | 68.03 | +0.02 |
| NUAA-SIRST | 0 | 79.10 | 77.50 | +1.60 |

Training has only just restarted after the fix, so no post-training improvement has been observed yet.

## Current Judgment

- The previous broken `SCT-Refiner` results are invalid and were deleted.
- The corrected refiner line is now valid because epoch 0 reproduces the saved strong checkpoints.
- The external-transfer line is represented by FeatUp-style high-resolution feature projection and background-normality memory inside `TCF-BGM` / `SCT-Refiner-Fixed`.
- The in-domain line is now represented by `MSHNet-SLS` on IRSTD-1K, while PConv-TCF remains prepared but not currently active.
- The target of +5 mIoU over SCTransNet on both IRSTD-1K and NUAA-SIRST is not achieved yet.
