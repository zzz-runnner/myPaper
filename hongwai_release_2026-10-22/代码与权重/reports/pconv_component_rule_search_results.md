# PConv Component Rule Search Results

Generated: 2026-06-13 UTC

## Purpose

This round moved away from global weight tuning and tested a component-level rule path for IRSTD-1K. The goal was to keep useful PConv-introduced candidates while suppressing false alarms, using cached predictions rather than rerunning neural models.

## Added Code

```text
scripts/inmemory_pconv_component_rule_search.py
scripts/vectorized_pconv_component_table_search.py
```

The first script directly rendered rules over connected components. It was stopped because the grid recomputed connected components for each rule and was too slow.

The second script builds vectorized component tables per base/PConv threshold, then scores rule parameters by table aggregation.

## Completed Narrow Diagnostic

Output:

```text
reports/vectorized_pconv_component_table_search_irstd_narrow.json
```

Cache inputs:

```text
reports/prediction_cache_irstd_dual_train_a/IRSTD-1K__train.npz
reports/prediction_cache_irstd_dual_train_b_test/IRSTD-1K__test.npz
```

Candidate source:

```text
baseq + fixed refiner + PConv-D05 + PConv-P43
```

Narrow grid:

```text
base_mode = avg_all
base_threshold = 0.40, 0.42
pconv_threshold = 0.45
small-component filters only
```

Validation-selected result:

| Base | Base th | PConv th | Output th | Val nIoU | Test mIoU | Pd | Fa |
|---|---:|---:|---:|---:|---:|---:|---:|
| `avg_all` | 0.40 | 0.45 | 0.40 | 65.41 | 69.80 | 95.96 | 8.71 |

Best test row inside validation top-k:

| Base | Base th | PConv th | Output th | Test mIoU | Pd | Fa |
|---|---:|---:|---:|---:|---:|---:|
| `avg_all` | 0.40 | 0.45 | 0.42 | 69.81 | 95.96 | 7.86 |

## Interpretation

This component-addition rule did not improve IRSTD-1K. It is far below the current 72.43 dual-PConv average-fusion best.

The failure mode is clear: the current best comes from continuous averaging of teacher and PConv probability maps. Turning PConv additions into sparse hard connected components breaks that calibrated probability structure and drops the score back toward the 69-70 region.

Do not continue this exact "add sparse PConv components onto base map" route. If component-level selection is revisited, it should operate as a mask/attenuation over the existing dual-PConv average map or as a learned calibration of continuous probabilities, not as binary component insertion.

## Current Best Remains

| Dataset | Valid best | mIoU |
|---|---|---:|
| IRSTD-1K | dual-PConv average fusion region | 72.43 |
| NUAA-SIRST | background-normality component fusion | 79.34 |

No large artifacts were created. The stopped broad component-rule attempt wrote no JSON.
