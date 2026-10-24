# Final common-fusion metrics table

Date: 2026-06-16

## Metric convention

F-measure is pixel-level F1 computed from the final binary prediction maps:

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 * Precision * Recall / (Precision + Recall)
```

`mIoU`, `nIoU`, `Pd`, and `Fa` use the same project evaluation helpers used in the result JSON files. `Fa` is reported as `10^-6`.

## Table

| Dataset | mIoU (%) | nIoU (%) | F-measure (%) | Pd (%) | Fa (10^-6) |
| --- | ---: | ---: | ---: | ---: | ---: |
| IRSTD-1K | 73.2142 | 67.5042 | 84.5355 | 93.6027 | 10.3433 |
| NUAA-SIRST | 82.7020 | 82.2698 | 90.5316 | 96.9582 | 9.5355 |

## Recipes

IRSTD-1K:

```text
0.876 * anchor-refiner no-TTA
0.062 * DFAwareNet official-Mamba
0.062 * HFMNet official-Mamba
gamma = 0.994
threshold = 0.400
connected-component min_area = 3
```

NUAA-SIRST:

```text
0.876 * LPSM pair
0.062 * DFAwareNet official-Mamba
0.062 * HFMNet official-Mamba
gamma = 0.994
threshold = 0.4995
connected-component min_area = 3
```

Raw JSON:

```text
reports/final_common_fusion_metrics_table.json
```
