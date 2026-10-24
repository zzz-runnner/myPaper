# Cached Weight Search Results

Generated: 2026-06-13 UTC

## Purpose

This round used the newly built prediction caches to test whether simple train-selected cached fusion could improve the current best results without rerunning neural models.

This is still a non-SCTransNet-parameter direction: it searches fusion weights over external/teacher candidate maps already generated from the multi-dataset cache.

## Added/Updated Code

Added:

```text
scripts/fast_cached_weight_search.py
scripts/inmemory_cached_weight_search.py
```

Updated:

```text
scripts/search_cached_train_selected_fusion.py
```

The first broad cached search over all four train caches was stopped because compressed `.npz` random access was too slow for repeated weight-grid scans. No output JSON was written from those stopped attempts. The final successful runs used `inmemory_cached_weight_search.py`, loading one target train cache and one target test cache into RAM.

## IRSTD-1K Result

Output:

```text
reports/inmemory_cached_weight_search_irstd_targettrain.json
```

Candidate set:

```text
baseq + fixed refiner + PConv-D05 + PConv-P43
```

Selection split:

```text
IRSTD-1K train cache
```

Evaluation split:

```text
IRSTD-1K test cache
```

Train-selected result:

| Selection | Weights | Threshold | Gamma | Train fast nIoU | Test mIoU | Pd | Fa |
|---|---|---:|---:|---:|---:|---:|---:|
| selected by train fast nIoU | `teacher60_pconv40` | 0.44 | 1.00 | 70.63 | 67.80 | 93.94 | 6.89 |

This is not useful; train fast nIoU selected an overconservative threshold/weight combination that performs badly on test mIoU.

Best test reference inside train-top-k:

| Reference only | Weights | Threshold | Gamma | Test mIoU | Pd | Fa |
|---|---|---:|---:|---:|---:|---:|
| best test row among train-top-k | `teacher60_pconv40` | 0.40 | 1.00 | 72.69 | 94.61 | 11.01 |

Important: 72.69 is not a strict final valid result because it is selected by looking at test rows inside the train-top-k candidate set. It is only diagnostic. It does suggest that slight reweighting may have small headroom, but the current train selection objective is not reliable enough.

Current conservative IRSTD-1K best remains:

```text
72.43 mIoU
```

## NUAA-SIRST Result

Output:

```text
reports/inmemory_cached_weight_search_nuaa_targettrain.json
```

Candidate set:

```text
TPC-A + TPC-B + LCA
```

Selection split:

```text
NUAA-SIRST train cache
```

Evaluation split:

```text
NUAA-SIRST test cache
```

Train-selected result:

| Selection | Weights | Threshold | Gamma | Train fast nIoU | Test mIoU | Pd | Fa |
|---|---|---:|---:|---:|---:|---:|---:|
| selected by train fast nIoU | `tpc_a_only` | 0.65 | 1.10 | 90.01 | 78.85 | 96.20 | 29.77 |

Best test reference inside train-top-k:

| Reference only | Weights | Threshold | Gamma | Test mIoU | Pd | Fa |
|---|---|---:|---:|---:|---:|---:|
| best test row among train-top-k | `tpc_pair` | 0.65 | 1.05 | 78.88 | 96.20 | 29.98 |

Both are below the current NUAA best:

```text
79.34 mIoU
```

## Interpretation

Cached weighted fusion did not produce a new valid best on either target dataset.

For IRSTD-1K, the diagnostic 72.69 test-reference row is above the current conservative 72.43, but it is not train-selected. The remaining issue is not just weight tuning; it is reliable validation/selection of thresholds and components.

For NUAA-SIRST, TPC/LCA weighted fusion is exhausted. It does not approach the +5 target and does not beat background-normality component fusion.

## Process and Cleanup

No GPU process remains active after this round. The stopped broad-search attempts produced no JSON output. The kept artifacts are small JSON reports and reusable scripts.

Current best remains unchanged:

| Dataset | Valid best | mIoU |
|---|---|---:|
| IRSTD-1K | dual-PConv average fusion region | 72.43 |
| NUAA-SIRST | background-normality component fusion | 79.34 |

Next useful direction is not more global weighted fusion. For IRSTD, the next selector should optimize a validation proxy closer to mIoU and false-alarm behavior, or use component-level cached features with a held-out split. For NUAA, a different candidate generator is still needed.
