# Cached Component Stump-Boost Attempt

Generated: 2026-06-13 UTC

## Purpose

This was a follow-up to the earlier direct `eval_component_stump_boost_selector.py` smoke. The goal was to avoid rerunning neural teachers and instead train a dependency-free boosted decision-stump component selector from the existing prediction caches.

This is not SCTransNet parameter tuning. It is a post-hoc component selector over frozen multi-model predictions.

## Code Added

```text
scripts/cached_component_stump_boost.py
```

The script loads `.npz` prediction caches, builds connected-component features, trains boosted decision stumps, then renders selected components back into prediction maps.

## Smoke Run

Output:

```text
reports/cached_component_stump_boost_smoke_irstd.json
```

Settings:

```text
train cache = IRSTD-1K train, 100 samples
test cache = IRSTD-1K test, 50 samples
candidate maps = baseq + refiner + PConv-D05 + PConv-P43
seed mode = avg
seed threshold = 0.40
```

Best smoke rows:

| Row | mIoU | Pd | Fa |
|---|---:|---:|---:|
| baseline `baseq`, out 0.40 | 74.85 | 92.59 | 19.68 |
| baseline avg, out 0.42 | 72.98 | 95.06 | 2.98 |
| stump-boost avg, keep 0.20/0.40, out 0.42 | 72.98 | 95.06 | 2.98 |
| oracle avg, seed 0.40 | 73.80 | 95.06 | 0.00 |

The learned selector did not beat the baseline; it degenerated to the average-map behavior on the smoke split.

## Cache Recheck

Output:

```text
reports/cache_recheck_irstd_dual_eval.json
```

Full IRSTD-1K test cache still confirms the previous test-sweep reference:

| Mode | Threshold | mIoU | Pd | Fa |
|---|---:|---:|---:|---:|
| avg | 0.42 | 72.48 | 94.95 | 5.54 |

This remains diagnostic/test-swept rather than a stricter train-selected final row.

## Decision

Do not run the current cached stump-boost selector full-scale. The small smoke does not show selector headroom, and the component-table implementation is CPU-heavy. A full four-dataset cache run would likely spend a lot of CPU time to reproduce the existing average-fusion region.

Current reliable best remains:

| Dataset | Valid best | mIoU |
|---|---|---:|
| IRSTD-1K | dual-PConv average fusion region | 72.43 |
| NUAA-SIRST | background-normality component fusion | 79.34 |
