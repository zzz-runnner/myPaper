# Train-Selected Threshold Calibration Results

Generated: 2026-06-13 UTC

## Purpose

This run checked whether the small IRSTD-1K gain from dual-PConv average fusion is supported by train-selected threshold calibration, rather than only by choosing a threshold on the test split.

Code:

```text
scripts/eval_train_threshold_calibration.py
```

The script collects full multi-dataset train predictions, selects the average-fusion output threshold by train mIoU, and evaluates that selected threshold on the target test split. It writes only small JSON/log files and no checkpoint.

## Runs

IRSTD-1K:

```text
reports/threshold_calibration_irstd_dual_pconv.json
```

Candidate set:

```text
baseq + fixed refiner + PConv-D05 + PConv-P43
```

NUAA-SIRST:

```text
reports/threshold_calibration_nuaa_tpc_lca_p43.json
```

Candidate set:

```text
TPC-A + TPC-B + LCA + PConv-P43
```

Both used the full fit pool:

```text
IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3
```

## Results

| Dataset | Selection rule | Threshold | Test mIoU | nIoU | Pd | Fa |
|---|---|---:|---:|---:|---:|---:|
| IRSTD-1K | selected by full multi-dataset train mIoU | 0.38 | 72.31 | 66.51 | 94.95 | 7.04 |
| IRSTD-1K | test sweep only, diagnostic upper reference | 0.42 | 72.49 | 66.62 | 94.95 | 5.54 |
| NUAA-SIRST | selected by full multi-dataset train mIoU | 0.60 | 79.34 | 79.83 | 96.20 | 29.02 |
| NUAA-SIRST | test sweep only | 0.60 | 79.34 | 79.83 | 96.20 | 29.02 |

## Interpretation

For IRSTD-1K, test-split threshold sweeping can reach 72.49 mIoU, but the threshold chosen by the full multi-dataset train pool gives 72.31. Therefore the 72.49 number should be treated only as a diagnostic upper reference, not as the strict train-selected result.

For NUAA-SIRST, threshold calibration does not improve over the existing reliable best. It ties 79.34, but with a higher false-alarm rate than the background-normality component result.

## Current Conservative Best

| Dataset | Conservative best valid method | mIoU |
|---|---|---:|
| IRSTD-1K | dual-PConv average fusion, output 0.40/0.45 region | 72.43 |
| NUAA-SIRST | background-normality component fusion | 79.34 |

## Decision

Threshold calibration is not enough to reach the requested +5 mIoU. It confirms that the remaining IRSTD gap is about 0.5-0.7 mIoU and that NUAA still needs a genuinely different candidate source.

No large artifacts were created.
