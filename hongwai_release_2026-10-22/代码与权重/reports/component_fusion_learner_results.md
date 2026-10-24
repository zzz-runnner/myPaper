# Component Fusion Learner Results

Generated: 2026-06-13 UTC

This line is a component-level learned teacher fusion / false-alarm filtering attempt. It is not SCTransNet hyperparameter tuning. The script builds connected components from the union of strong teacher predictions, extracts per-component features such as area, shape, teacher probability statistics, local thermal contrast, Laplacian response, and model agreement, then fits a small logistic component keeper on the full training split and evaluates on the test split.

Script:

```text
scripts/eval_component_fusion_learner.py
```

Outputs:

```text
reports/component_fusion_fast_irstd_tpc.json
reports/component_fusion_fast_irstd_mixed.json
reports/component_fusion_fast_nuaa_lca_tcd.json
reports/component_fusion_fast_nuaa_tpc_lca.json
```

## Valid Train-Fit / Test-Eval Results

| Dataset | Teacher set | Best valid setting | mIoU | Pd | Fa |
|---|---|---|---:|---:|---:|
| IRSTD-1K | TPC teachers | seed 0.45, keep 0.30, fill max, output 0.50 | 69.13 | 90.24 | 7.06 |
| IRSTD-1K | TPC + fixed refiner | seed 0.45, keep 0.30, fill max, output 0.50 | 69.17 | 89.90 | 6.09 |
| NUAA-SIRST | LCA + TCD teachers | seed 0.45, keep 0.40, fill avg, output 0.50 | 79.27 | 96.58 | 16.40 |
| NUAA-SIRST | TPC + LCA teachers | seed 0.45, keep 0.40, fill avg, output 0.50 | 79.26 | 96.58 | 16.40 |

Approximate comparison to current reliable best region:

| Dataset | Previous reliable region | Best valid component-fusion result | Delta |
|---|---:|---:|---:|
| IRSTD-1K | ~68.30 | 69.17 | +0.87 |
| NUAA-SIRST | ~79.15 | 79.27 | +0.12 |

## Test Oracle Reference Only

These numbers use test-set GT to decide which predicted components to keep, so they are **not valid final method results**. They are useful only as an upper-bound diagnostic.

| Dataset | Teacher set | Oracle mIoU | Pd | Fa |
|---|---|---:|---:|---:|
| IRSTD-1K | TPC teachers | 70.32 | 93.60 | 3.70 |
| IRSTD-1K | TPC + fixed refiner | 70.40 | 93.60 | 3.66 |
| NUAA-SIRST | LCA + TCD teachers | 80.27 | 96.58 | 8.09 |
| NUAA-SIRST | TPC + LCA teachers | 80.27 | 96.58 | 8.09 |

## Interpretation

This route gives a real IRSTD gain compared with the prior plateau, mainly by reducing false alarms. It still does not achieve the requested +5 mIoU over SCTransNet. The oracle gap suggests there is about 1-2 mIoU of component-selection headroom with these teachers, not 5 points, unless we add stronger/diverse candidate generators or a better component learner trained from more data.

Bulky failed full-grid runs were stopped; no large run directory was kept. Only small logs and JSON evidence remain.
