# Background-Candidate and Stump-Boost Update

Generated: 2026-06-13 UTC

## Purpose

This round continued the non-SCTransNet-parameter direction requested for the IR small-target project. It tested:

1. Background-normality as an actual candidate source, not only as component features.
2. A new dependency-free boosted decision-stump component selector, intended as a more rule-like alternative to the failed MLP selector.

Both are external-domain inspired post-hoc candidate/selector modules rather than SCTransNet training-parameter tuning.

## Current Process State

The four `bgcand_*` tmux jobs were inspected repeatedly. One completed with JSON output:

```text
reports/bg_candidate_irstd_mixed.json
```

The other three completed train/test prediction collection but did not progress into JSON output during the observation window and were stopped to avoid wasting CPU/GPU time:

```text
bgcand_irstd_dual
bgcand_nuaa_lca_tcd
bgcand_nuaa_tpc_lca
```

Logs were preserved in:

```text
reports/bg_candidate_logs/
```

No large checkpoint or bulky failed artifact was created.

## Background Candidate Result

Completed full multi-dataset fit, IRSTD-1K test:

| Dataset | Candidate set | Best valid row | mIoU | Pd | Fa |
|---|---|---|---:|---:|---:|
| IRSTD-1K | baseq + fixed refiner + background anomaly candidate | baseline baseq, out 0.50 | 68.30 | 93.60 | 12.53 |

Oracle reference only:

| Dataset | Best oracle | mIoU | Pd | Fa |
|---|---|---:|---:|---:|
| IRSTD-1K | seed 0.40, max fill | 70.08 | 93.60 | 3.72 |

Interpretation: background anomaly as a standalone candidate source did not help IRSTD-1K. It is far below the current 72.43 dual-PConv best and does not expand oracle headroom enough.

## New Stump-Boost Selector

Added script:

```text
scripts/eval_component_stump_boost_selector.py
```

This trains a small boosted-decision-stump selector over component features without adding sklearn/xgboost dependencies. The intent is to test a more rule-like non-linear selector for PConv/teacher components.

Smoke output:

```text
reports/stump_boost_smoke_irstd_dual.json
```

Smoke setting:

```text
dataset = IRSTD-1K
fit_datasets = IRSTD-1K,NUAA-SIRST
max_fit_samples = 20
max_eval_samples = 20
candidate set = baseq + fixed refiner + PConv-D05 + PConv-P43
```

Smoke result:

| Row | mIoU | Pd | Fa |
|---|---:|---:|---:|
| best valid | 77.29 | 88.24 | 40.82 |
| best stump-boost rows | did not beat baseline | - | - |
| oracle reference | 76.46 | 97.06 | 0.00 |

The smoke run validates that the script executes and writes JSON, but it does not show a useful selector improvement. On the tiny split, the selector mostly degenerates to baseline-like behavior.

## Current Best Remains Unchanged

| Dataset | Best valid method | mIoU |
|---|---|---:|
| IRSTD-1K | baseq + fixed refiner + PConv-D05 + PConv-P43 average fusion, output 0.40 | 72.43 |
| NUAA-SIRST | background-normality component fusion | 79.34 |

The +5 targets are still not reached:

| Dataset | Target line | Current best | Gap |
|---|---:|---:|---:|
| IRSTD-1K | 73.03 | 72.43 | 0.60 |
| NUAA-SIRST | about 82.64 | 79.34 | about 3.30 |

## Decision

Do not continue background-as-candidate in this exact form. It failed on the completed IRSTD run and the remaining jobs were too slow after collection.

Do not launch full stump-boost immediately. The smoke run proves the code path but does not yet justify a four-card full run. If this selector is continued, it should be improved first with cached predictions/components and a fit-selection objective closer to test mIoU/nIoU, so the selector does not collapse to baseline behavior.
