# Official-Mamba Cross-Domain IRSTD-Weight NUAA Attempt

Generated: 2026-06-15 UTC

## Purpose

Test whether IRSTD-1K-trained official-Mamba DFAwareNet/HFMNet weights provide a new cross-domain dense source for NUAA-SIRST, complementary to the current NUAA-trained official-Mamba fusion.

## Results

Full NUAA-SIRST evaluation with official Mamba backend:

| Source weight | Split | Best kind | Threshold | Min area | mIoU | nIoU | Pd | Fa |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DFAwareNet IRSTD-1K weight | train | component | 0.30 | 3 | 62.4213 | 70.8035 | 90.3704 | 60.5889e-6 |
| DFAwareNet IRSTD-1K weight | test | component | 0.30 | 4 | 64.4541 | 69.0328 | 90.4943 | 43.1499e-6 |
| HFMNet IRSTD-1K weight | train | component | 0.30 | 2 | 64.8146 | 73.6494 | 90.3704 | 41.7158e-6 |
| HFMNet IRSTD-1K weight | test | component | 0.30 | 3 | 66.3231 | 71.6063 | 91.2548 | 36.0154e-6 |

## Decision

These cross-domain maps are far below the retained NUAA best `82.2554` and too noisy to justify fusion. Do not add them to the current main fusion pool.

The saved probability maps were deleted after evaluation to save space. Only the small eval JSON files and this report were kept.
