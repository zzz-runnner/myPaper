# Vectorized Average-Map Component Suppression Results

Generated: 2026-06-13 UTC

## Purpose

This run rewrote the previous slow average-map component suppression attempt into a vectorized component-table version. The goal was to preserve the current dual-PConv average probability map and only suppress likely false-positive connected components.

## Added Code

```text
scripts/vectorized_avg_component_suppression.py
```

The script extracts connected components from the average probability map once per seed/output threshold pair and stores component features plus the TP/FP pixels that would be removed. Rule evaluation then becomes table aggregation instead of repeated connected-component extraction.

## Inputs

```text
reports/prediction_cache_irstd_dual_train_a/IRSTD-1K__train.npz
reports/prediction_cache_irstd_dual_train_b_test/IRSTD-1K__test.npz
```

Candidate maps:

```text
baseq + fixed refiner + PConv-D05 + PConv-P43
```

## Main Suppression Run

Output:

```text
reports/vectorized_avg_component_suppression_irstd.json
```

Validation-selected row:

| Seed th | Output th | Val nIoU | Test mIoU | Pd | Fa | Test kept / components |
|---:|---:|---:|---:|---:|---:|---:|
| 0.42 | 0.40 | 69.26 | 67.05 | 93.94 | 6.24 | 317 / 321 |

Best test row inside validation top-k:

| Seed th | Output th | Test mIoU | Pd | Fa |
|---:|---:|---:|---:|---:|
| 0.42 | 0.40 | 67.06 | 93.94 | 6.17 |

This is far below the current best.

## No-Delete Sanity Check

Output:

```text
reports/vectorized_avg_component_suppression_irstd_nodelete.json
```

No-delete baseline:

| Seed th | Output th | Val nIoU | Test mIoU | Pd | Fa | Test kept / components |
|---:|---:|---:|---:|---:|---:|---:|
| 0.40 | 0.40 | 69.60 | 72.45 | 94.95 | 6.87 | 323 / 323 |

This confirms the implementation can reproduce the current dual-PConv average-fusion region. It is close to the prior 72.43/72.48 reference, with small differences due to threshold choice/cache precision.

## Interpretation

The table implementation works and is much faster than the previous repeated connected-component grid. However, simple threshold-based component suppression is not helpful: it deletes true target pixels/components while only mildly reducing false alarms.

The best behavior remains "do not suppress"; therefore this route does not close the remaining IRSTD gap.

Current best remains:

| Dataset | Valid best | mIoU |
|---|---|---:|
| IRSTD-1K | dual-PConv average fusion region | 72.43 |
| NUAA-SIRST | background-normality component fusion | 79.34 |

## Decision

Do not continue simple threshold-rule suppression. If the average-map path continues, it needs a learned calibration objective that preserves target recall, not hand threshold deletion of components.
