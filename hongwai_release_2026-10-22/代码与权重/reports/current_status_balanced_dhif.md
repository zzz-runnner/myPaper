# Current Status: Balanced DHiF Runs

Generated: 2026-06-12 UTC

## Current Active Runs

| GPU | Dataset | Method | Training data | Status |
|---:|---|---|---|---|
| 0 | IRSTD-1K | SCT-DHiF-Balanced | 4 datasets, target repeat 3 | active |
| 1 | IRSTD-1K | SCT-DHiF-Balanced2 | 4 datasets, target repeat 4 | active |
| 2 | NUAA-SIRST | SCT-DHiF-Balanced | 4 datasets, target repeat 3 | active |
| 3 | NUAA-SIRST | MSHNet-SLS | NUAA-SIRST | active |

## Why The Switch Happened

- `SCT-DHiF-Multi` used all four train splits equally. It preserved the strong checkpoint start but did not beat the earlier fixed refiner:
  - IRSTD-1K: 68.06, lower than fixed refiner 68.09.
  - NUAA-SIRST: 79.11, lower than fixed refiner 79.12.
- Added `--target_repeat` so full multi-dataset training still uses all data, but target-domain samples are oversampled to reduce cross-domain dilution.
- Stopped weak `IRSTD-1K / MSHNet-SLS` at epoch 110:
  - best mIoU 61.57 vs README 68.03.
  - checkpoint removed; logs/report retained.
- Started `IRSTD-1K / SCT-DHiF-Balanced2` on GPU1 as a second IRSTD-focused configuration.

## Latest Results

| Dataset | Method | Best epoch | Best mIoU | README | Delta |
|---|---|---:|---:|---:|---:|
| IRSTD-1K | SCT-DHiF-Balanced | 0 | 68.05 | 68.03 | +0.02 |
| IRSTD-1K | SCT-DHiF-Balanced2 | 0 | 68.05 | 68.03 | +0.02 |
| NUAA-SIRST | SCT-DHiF-Balanced | 5 | 79.11 | 77.50 | +1.61 |
| NUAA-SIRST | MSHNet-SLS | 160 | 74.62 | 77.50 | -2.88 |

Best preserved valid result so far:

| Dataset | Method | Best mIoU | README | Delta |
|---|---|---:|---:|---:|
| IRSTD-1K | SCT-Refiner-Fixed | 68.09 | 68.03 | +0.06 |
| NUAA-SIRST | SCT-Refiner-Fixed | 79.12 | 77.50 | +1.62 |

## Storage

- Current `runs` size: about 645M.
- Weak MSHNet-IRSTD checkpoint was deleted after recording results.
- Reports and logs are retained as experiment evidence.

## Status Against Goal

The +5 mIoU goal is still not achieved. Current work is focused on the best-aligned route: strong SCTransNet initialization plus DHiF/SET-style frequency target-clutter filtering, background-normality memory, PConv shape prior, FeatUp-style high-resolution feature recovery, and full multi-dataset target-balanced training.
