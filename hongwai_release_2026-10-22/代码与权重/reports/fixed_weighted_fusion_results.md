# Fixed Weighted Fusion Results

Generated: 2026-06-13 UTC

## Purpose

This diagnostic tested whether simple fixed linear weighting around the current IRSTD-1K best row could push beyond:

```text
baseq + fixed refiner + PConv-D05 + PConv-P43 average fusion, output 0.45 = 72.41 mIoU
```

It was intended as a small, no-checkpoint followup after the train-fit weighted search proved too slow.

## Code

Added/updated:

```text
scripts/eval_fixed_weighted_fusion.py
scripts/launch_fixed_weighted_fusion_4gpu_tmux.sh
```

The script evaluates fixed weighted averages over existing teacher and PConv prediction maps. It writes only small JSON files and no training checkpoints.

Because full `PD_FA`/mIoU evaluation is slow for many candidate rows, the script now supports:

```text
--fast_only
--full_metric_topk
```

`--fast_only` writes diagnostic nIoU-only rows and is not a final method metric.

## Completed Results

| Dataset | Candidate set | Metric mode | Best row | mIoU | fast nIoU | Pd | Fa |
|---|---|---|---|---:|---:|---:|---:|
| IRSTD-1K | baseq + refiner + PConv-P43 | full metric | single baseq, out 0.50, gamma 1.0 | 68.29 | 68.00 | 93.60 | 12.51 |
| NUAA-SIRST | TPC + LCA + PConv-P43 | full metric | avg all, out 0.60, gamma 1.0 | 79.34 | 79.83 | 96.20 | 29.02 |
| NUAA-SIRST | LCA + TCD + PConv-P43 | full metric | avg all, out 0.55, gamma 1.15 | 79.34 | 79.80 | 96.20 | 29.22 |
| IRSTD-1K | baseq + refiner + PConv-D05 + PConv-P43 | fast nIoU only | single baseq, out 0.45, gamma 1.15 | n/a | 68.02 | n/a | n/a |

## Interpretation

Fixed linear weighted averaging did not improve either dataset.

For IRSTD-1K, the important diagnostic is negative: even with the dual PConv candidate set, the fast nIoU ranking preferred a single SCTransNet-family teacher around 68.02 nIoU, not the PConv weighted combinations. This means the 72.41 valid result from the previous component MLP selector report is not reproduced by plain weighted averaging. The useful behavior there came from the exact average/max candidate handling in that pipeline, not from generic fixed linear calibration.

For NUAA-SIRST, fixed weighting tied the existing 79.34 best mIoU but with worse false alarms than the background-normality component result. It is not a new best.

## Current Best After This Diagnostic

| Dataset | Best valid method still standing | mIoU |
|---|---|---:|
| IRSTD-1K | baseq + fixed refiner + PConv-D05 + PConv-P43 average fusion from component MLP selector fast run | 72.41 |
| NUAA-SIRST | background-normality component fusion | 79.34 |

## Decision

Do not continue fixed weighted averaging as the main path. It did not convert the remaining IRSTD 0.62 mIoU gap and did not help NUAA.

Next useful direction:

1. For IRSTD-1K, keep PConv as an external candidate source but use non-linear/local selection rather than global linear weights.
2. For NUAA-SIRST, look for a different candidate generator; PConv-P43/D05 remains unhelpful.

No large artifacts were created. The completed outputs are small JSON/log files only.
