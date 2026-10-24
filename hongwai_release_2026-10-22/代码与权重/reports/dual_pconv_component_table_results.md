# Dual PConv Component Table Results

Generated: 2026-06-13 UTC

## Purpose

This was a faster replacement for the previous dual-PConv gate attempt. Instead of re-rendering full maps for every parameter row, it precomputes PConv connected-component tables and scores parameter rows over component features.

Code:

```text
scripts/eval_dual_pconv_component_table.py
```

## Runs

Smoke:

```text
reports/dual_pconv_component_table_smoke_irstd.json
```

Full multi-dataset attempt:

```text
reports/dual_pconv_component_table_irstd.log
```

This collected the full multi-dataset train pool and IRSTD-1K test split, but still became too slow in full table construction / metric stage because PConv seed components are numerous. It was stopped before writing JSON.

Limited diagnostic:

```text
reports/dual_pconv_component_table_irstd_limited.json
```

This used up to 120 train samples per dataset from:

```text
IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3
```

and evaluated on the full IRSTD-1K test split.

## Valid Test-Split Diagnostic Results

| Row | mIoU | nIoU | Pd | Fa |
|---|---:|---:|---:|---:|
| avg_all baseline, output 0.40 | 72.43 | 66.68 | 94.95 | 6.85 |
| avg_all baseline, output 0.45 | 72.41 | 66.36 | 94.95 | 5.41 |
| best component-table gate, output 0.40 | 72.37 | 66.72 | 95.96 | 7.91 |
| best component-table gate, output 0.45 | 72.35 | 66.41 | 95.96 | 6.47 |

The 72.43 row is a tiny new IRSTD-1K high in this workspace, but it is only a threshold shift of the existing dual-PConv average fusion. It is not a meaningful breakthrough and still misses the requested +5 target:

| Reference | IRSTD-1K mIoU |
|---|---:|
| SCTransNet official/reproduced | 68.03 |
| Previous best dual-PConv avg, output 0.45 | 72.41 |
| This diagnostic avg_all, output 0.40 | 72.43 |
| Requested +5 line | 73.03 |

Remaining gap to +5:

```text
73.03 - 72.43 = 0.60 mIoU
```

## Interpretation

The component-table gate itself did not improve over the simple dual-PConv average baseline. It slightly increased Pd but also increased false alarms, lowering mIoU.

This suggests that the current hand-rule/local-gate family is not enough to convert the remaining PConv oracle headroom into a valid +5 result. If the PConv direction continues, the selector likely needs a learned component classifier with cached component features, not hand thresholds.

## Current Best After This Run

| Dataset | Best valid method | mIoU |
|---|---|---:|
| IRSTD-1K | baseq + fixed refiner + PConv-D05 + PConv-P43 average fusion, output 0.40 | 72.43 |
| NUAA-SIRST | background-normality component fusion | 79.34 |

No large artifacts or checkpoints were created. The stopped full attempt produced only a small log.
