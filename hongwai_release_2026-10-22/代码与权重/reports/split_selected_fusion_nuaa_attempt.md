# Split-Selected NUAA TPC/LCA Fusion Attempt

Generated: 2026-06-14 UTC

## Purpose

This mirrored the IRSTD split-selection check for NUAA-SIRST. The goal was to verify whether the existing cached TPC/LCA teacher pool hides any small valid gain from train-split selection of weights and threshold.

## Data

Validation source:

```text
reports/prediction_cache_nuaa_tpc_lca_train_a/NUAA-SIRST__train.npz
```

Test source:

```text
reports/prediction_cache_nuaa_tpc_lca_train_b_test/NUAA-SIRST__test.npz
```

Model maps:

```text
tpc_a, tpc_b, lca
```

Script:

```text
scripts/search_cache_split_selected_fusion.py
```

## Result

Validation-selected row:

| Weights | Threshold | Val nIoU | Test mIoU | Test nIoU | Pd | Fa |
|---|---:|---:|---:|---:|---:|---:|
| tpc_a 1.0, tpc_b 1.0, lca 1.0 | 0.55 | 90.69 | 78.95 | 79.28 | 96.58 | 1.75e-5 |

Best test row inside validation top-20, diagnostic only:

| Weights | Threshold | Val nIoU | Test mIoU | Test nIoU | Pd | Fa |
|---|---:|---:|---:|---:|---:|---:|
| tpc_a 1.0, tpc_b 1.0, lca 1.0 | 0.50 | 90.63 | 79.11 | 79.39 | 96.58 | 1.77e-5 |

## Interpretation

This does not improve the current NUAA-SIRST best:

| Reference | NUAA-SIRST mIoU |
|---|---:|
| SCTransNet official/reproduced | 77.64 |
| TPC/LCA split-selected fusion | 78.95 |
| TPC/LCA top-20 diagnostic | 79.11 |
| Current reliable background-normality component fusion | 79.34 |
| Requested +5 target | 82.64 |

The cached TPC/LCA teacher pool is saturated. Simple threshold/weight selection does not move NUAA toward the +5 target.

No checkpoint or large artifact was created.
