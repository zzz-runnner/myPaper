# DFRL External Candidate Attempt

Generated: 2026-06-14 UTC

## Purpose

NUAA-SIRST is still below the requested +5 mIoU target. This attempt tested a directly downloadable, non-SCTransNet external source:

```text
DFRL for single-domain generalization in infrared small target detection
```

The repository provides runnable test scripts and official pretrained weights for:

```text
DNANet
PCSDL / MSHNet
RDIAN
```

## Setup

Cloned:

```text
external/DFRL
```

Official checkpoints:

```text
external/DFRL/pth/DNANet/Best_DANNet.pth
external/DFRL/pth/PCSDL/Best_PCSDL.pth
external/DFRL/pth/RDIAN/Best_RDIAN.pth
```

Added local evaluator:

```text
scripts/eval_dfrl_local.py
```

The evaluator loads the official checkpoints, runs the local prepared full test splits, and sweeps thresholds without saving prediction maps.

Input details:

| Model | Input channels | Input size | Output used |
| --- | ---: | ---: | --- |
| DNANet | 3 | 256 | last deep-supervision output |
| PCSDL | 3 | 256 | final output branch |
| RDIAN | 1 | 256 | direct output |

## Full Local Test Results

| Model | Dataset | Best threshold | mIoU | nIoU | Pd | Fa |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DNANet | IRSTD-1K | 0.20 | 40.7207 | 42.3657 | 71.0438 | 130.6866e-6 |
| DNANet | NUAA-SIRST | 0.05 | 66.6145 | 66.0534 | 89.7338 | 23.8731e-6 |
| PCSDL | IRSTD-1K | 0.25 | 45.5710 | 54.7810 | 90.5724 | 107.9882e-6 |
| PCSDL | NUAA-SIRST | 0.25 | 65.1579 | 67.9249 | 97.3384 | 73.3343e-6 |
| RDIAN | IRSTD-1K | 0.70 | 39.0263 | 41.4279 | 83.1650 | 134.6152e-6 |
| RDIAN | NUAA-SIRST | 0.25 | 60.0665 | 66.0078 | 92.0152 | 82.8698e-6 |

Result JSON:

```text
reports/dfrl_dnanet_full_test.json
reports/dfrl_pcsd_full_test.json
reports/dfrl_rdian_full_test.json
```

## Interpretation

DFRL is runnable and the checkpoints load cleanly, but these official weights are far below the current local bests:

| Dataset | Current reliable best | Best DFRL result |
| --- | ---: | ---: |
| IRSTD-1K | 73.0367 | 45.5710 |
| NUAA-SIRST | 79.34 | 66.6145 |

So DFRL is not a useful dense candidate source for the current +5 objective.

This also agrees with the official result text bundled inside the DFRL repository: the provided checkpoints are not in the 80+ mIoU NUAA region.

## Cleanup

No bulky training output or prediction archive was created. Persistent additions are small:

```text
scripts/eval_dfrl_local.py
reports/dfrl_*_full_test.json
reports/dfrl_*_smoke5.json
external/DFRL
```

The DFRL repository is retained because the total size is small and it records a checked external-source negative result.

## Decision

Do not continue DFRL as a main line. The next NUAA work should still focus on a stronger pretrained dense source or new NUAA-like labeled data, because the current local candidate ceiling remains below the requested 82.64 mIoU target.
