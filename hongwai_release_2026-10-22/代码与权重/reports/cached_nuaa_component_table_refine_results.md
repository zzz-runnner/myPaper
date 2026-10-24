# Cached NUAA Component-Table Refinement Results

Generated: 2026-06-15 UTC

## Purpose

After the strong-fusion HFM-LK refiner only improved NUAA-SIRST to `81.1502` mIoU, this test checked a cheaper targeted route:

```text
strong cached fusion
+ connected-component table
+ train-selected false-positive / support filtering
```

The goal was to see whether false-positive components from the BGM/RPCANet/TPC/LCA fusion could be removed without losing true small targets.

## Code

Added:

```text
scripts/search_cached_nuaa_component_table_refine.py
```

The script reads the corrected strong-fusion caches directly:

```text
reports/fusion_cache_bgm_rpcanet_train_a/IRSTD-1K__train.npz
reports/fusion_cache_bgm_rpcanet_train_a/NUAA-SIRST__train.npz
reports/fusion_cache_bgm_rpcanet_train_b_test/NUDT-SIRST__train.npz
reports/fusion_cache_bgm_rpcanet_train_b_test/SIRST3__train.npz
reports/fusion_cache_bgm_rpcanet_train_b_test/NUAA-SIRST__test.npz
```

It precomputes connected-component tables for `weighted_raw` and `strong_fusion` thresholds, then screens rules using train-side cached samples before evaluating only selected rows on the full NUAA-SIRST test cache.

An initial direct component-search script was also added:

```text
scripts/search_cached_nuaa_component_refine.py
```

That direct version was too slow because it recomputed/rendered components for every rule, so it was superseded by the component-table version.

## Runs

### 40 samples per train cache

Output:

```text
reports/cached_nuaa_component_table_refine_40per.json
```

Fit data:

```text
IRSTD-1K train: 40 samples
NUAA-SIRST train: 40 samples
NUDT-SIRST train: 40 samples
SIRST3 train: 40 samples
```

Full NUAA test selected result:

| method | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: |
| train-selected component table rule | 80.7370 | 81.3302 | 96.9582 | 7.2717e-6 |
| cache baseline in same run | 81.0489 | 81.6157 | 98.0989 | 7.4775e-6 |

The selected rule reduced false alarms slightly but lost too much target recall and mIoU.

### Conservative 80 samples per train cache

Output:

```text
reports/cached_nuaa_component_table_refine_conservative80.json
```

Fit data:

```text
IRSTD-1K train: 80 samples
NUAA-SIRST train: 80 samples
NUDT-SIRST train: 80 samples
SIRST3 train: 80 samples
```

This run used only near-no-op filters:

```text
min_area: 1,2
max_area: no limit
max_side: no limit
min_mean / min_peak / source support: mild thresholds only
```

Full NUAA test selected result:

| method | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: |
| conservative selected rule | 81.0269 | 81.5482 | 98.0989 | 7.4775e-6 |
| cache baseline in same run | 81.0489 | 81.6157 | 98.0989 | 7.4775e-6 |

The train-selected conservative rule reverted to the old strong-fusion region and still did not beat the raw corrected cache.

## Current Best Comparison

Current NUAA best remains:

```text
81.1502 mIoU
```

from:

```text
runs/strong_fusion_hfmlk_nuaa_4gpu/nuaa_strong_fusion_s703/best.pth.tar
```

The component-table refinement did not beat it:

```text
81.1502 - 81.0269 = +0.1233 mIoU retained advantage
```

Gap to requested NUAA target remains:

```text
82.64 - 81.1502 = 1.4898 mIoU
```

## Cleanup

Deleted smoke output:

```text
reports/cached_nuaa_component_table_refine_smoke.json
```

The stopped broad-grid attempts produced no JSON output. Retained artifacts are small:

```text
scripts/search_cached_nuaa_component_refine.py
scripts/search_cached_nuaa_component_table_refine.py
reports/cached_nuaa_component_table_refine_40per.json
reports/cached_nuaa_component_table_refine_conservative80.json
```

No checkpoint or bulky prediction archive was created by this step.

## Interpretation

This route is not a useful next main line. The current strong fusion is already tightly calibrated: component filtering can reduce some false positives, but it also removes true small targets and does not improve mIoU.

The remaining NUAA gap likely requires a genuinely new complementary dense source or additional NUAA-like training data, rather than more post-hoc component suppression on the existing six-source fusion.
