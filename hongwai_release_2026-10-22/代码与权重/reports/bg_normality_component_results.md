# Background Normality Component Results

Generated: 2026-06-13 UTC

This branch is an external-domain transfer attempt inspired by UniVAD / AnomalyCLIP / BGM-style background normality and false-alarm suppression. It is not SCTransNet hyperparameter tuning.

Implemented in:

```text
scripts/eval_component_fusion_learner.py
scripts/launch_bg_normality_remaining_tmux.sh
```

The method fits a lightweight background-normality model from GT-background pixels over the full multi-dataset train pool:

```text
IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3
```

It extracts local intensity, top-hat, DoG/Laplacian/gradient, and local-z features, then uses the resulting background-anomaly score as extra component-level features for the teacher-fusion keeper. The final candidate components still come from strong neural teachers by default; the background-normality signal is used to judge whether a teacher component looks like target-like clutter.

## Full Valid Results

| Dataset | Teacher set | Best setting | mIoU | Pd | Fa |
|---|---|---|---:|---:|---:|
| IRSTD-1K | TPC + fixed refiner | seed 0.45, keep 0.15, fill max, out 0.50 | 69.03 | 90.24 | 6.68 |
| IRSTD-1K | TPC teachers | seed 0.40, keep 0.20, fill avg, out 0.40 | 68.93 | 88.55 | 7.23 |
| NUAA-SIRST | TPC + LCA | seed 0.45, keep 0.15, fill avg, out 0.50 | 79.34 | 96.58 | 15.92 |
| NUAA-SIRST | LCA + TCD | seed 0.45, keep 0.15, fill avg, out 0.50 | 79.34 | 96.58 | 15.92 |

## Comparison to Previous Reliable Region

| Dataset | Previous best valid | Background-normality best | Delta |
|---|---:|---:|---:|
| IRSTD-1K | 69.17 | 69.03 | -0.14 |
| NUAA-SIRST | 79.28 | 79.34 | +0.06 |

The NUAA result is a new tiny valid high in this workspace, but the gain is only about +0.06 mIoU. It is not close to the requested +5 mIoU objective.

## Oracle Reference Only

The oracle rows use test-set GT to select components and are not valid final method results.

| Dataset | Teacher set | Oracle mIoU | Pd | Fa |
|---|---|---:|---:|---:|
| IRSTD-1K | TPC + fixed refiner | 70.39 | 93.60 | 3.80 |
| IRSTD-1K | TPC teachers | 70.41 | 93.60 | 3.82 |
| NUAA-SIRST | TPC + LCA | 80.35 | 96.58 | 7.89 |
| NUAA-SIRST | LCA + TCD | 80.35 | 96.58 | 7.89 |

## Interpretation

The background-normality feature does help NUAA slightly by reducing false alarms, but it does not improve IRSTD and does not meaningfully expand the teacher oracle ceiling. This suggests the current teacher-generated component set still lacks enough new true-target candidates for a five-point jump.

This route is worth keeping as a small component feature, but it is not a sufficient main path toward +5 unless paired with a genuinely new dense candidate generator or model family. No bulky run directory was created. Only small JSON/log/report evidence remains.
