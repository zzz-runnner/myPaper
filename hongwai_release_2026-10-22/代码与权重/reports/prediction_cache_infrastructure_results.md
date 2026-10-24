# Prediction Cache Infrastructure Results

Generated: 2026-06-13 UTC

## Purpose

The previous component-fusion and selector experiments repeatedly spent most time in full-dataset teacher/PConv prediction collection. This round adds reusable compact prediction caches so future non-SCTransNet selector/fusion experiments can iterate without rerunning all neural teachers each time.

This is infrastructure for the requested cross-domain/module-combination search, not a new final method by itself.

## Added Code

```text
scripts/build_prediction_cache.py
scripts/eval_cached_prediction_fusion.py
```

`build_prediction_cache.py` stores per-image masks, candidate prediction maps, image sizes, and names in compressed `.npz` files. Prediction maps are stored as `float16`; masks are stored as `uint8`.

`eval_cached_prediction_fusion.py` reloads those caches and evaluates simple single/avg/max/min fusions to verify the cache path.

## Smoke Test

Smoke cache:

```text
reports/cache_smoke_irstd_dual/IRSTD-1K__test.npz
```

Result:

```text
5 samples, 0.72 MB
cached eval best = baseq at threshold 0.40, mIoU 75.60 on the tiny smoke split
```

This only verified read/write correctness on a tiny subset.

## Full Caches Built

Four tmux jobs were launched across the four RTX 2080 Ti GPUs:

| tmux job | Candidate set | Splits |
|---|---|---|
| `cache_irstd_a` | baseq + fixed refiner + PConv-D05 + PConv-P43 | IRSTD-1K train, NUAA-SIRST train |
| `cache_irstd_b` | baseq + fixed refiner + PConv-D05 + PConv-P43 | NUDT-SIRST train, SIRST3 train, IRSTD-1K test |
| `cache_nuaa_a` | TPC-A + TPC-B + LCA | IRSTD-1K train, NUAA-SIRST train |
| `cache_nuaa_b` | TPC-A + TPC-B + LCA | NUDT-SIRST train, SIRST3 train, NUAA-SIRST test |

Completed cache directories:

```text
reports/prediction_cache_irstd_dual_train_a
reports/prediction_cache_irstd_dual_train_b_test
reports/prediction_cache_nuaa_tpc_lca_train_a
reports/prediction_cache_nuaa_tpc_lca_train_b_test
```

Disk usage:

| Directory | Size |
|---|---:|
| `prediction_cache_irstd_dual_train_a` | 131M |
| `prediction_cache_irstd_dual_train_b_test` | 211M |
| `prediction_cache_nuaa_tpc_lca_train_a` | 357M |
| `prediction_cache_nuaa_tpc_lca_train_b_test` | 531M |

Total cache size is about 1.23 GB.

## Cache Verification Metrics

IRSTD-1K dual-PConv test cache:

```text
reports/prediction_cache_irstd_dual_test_eval.json
```

Best cached simple fusion:

| Dataset | Mode | Threshold | mIoU | Pd | Fa |
|---|---|---:|---:|---:|---:|
| IRSTD-1K | avg | 0.42 | 72.48 | 94.95 | 5.54 |

Important: this is a test-sweep verification value and should not replace the conservative 72.43 best unless the threshold is selected without test-set tuning.

NUAA-SIRST TPC/LCA test cache:

```text
reports/prediction_cache_nuaa_tpc_lca_test_eval.json
```

Best cached simple fusion:

| Dataset | Mode | Threshold | mIoU | Pd | Fa |
|---|---|---:|---:|---:|---:|
| NUAA-SIRST | tpc_b | 0.50 | 79.13 | 96.58 | 17.70 |

This verifies the cached TPC/LCA teacher region but does not beat the existing NUAA best of 79.34.

## Current Best Remains

| Dataset | Current valid best | mIoU |
|---|---|---:|
| IRSTD-1K | dual-PConv average fusion region | 72.43 |
| NUAA-SIRST | background-normality component fusion | 79.34 |

## Next Use

The caches make the next useful step much cheaper: fit and evaluate component/pixel selectors from cached predictions rather than rerunning all neural models. This is especially important for the IRSTD PConv line, where oracle headroom exists but previous selectors failed to convert it.
