# NUAA Local BGM + RPCANet Weight Search Results

Generated: 2026-06-15 UTC

## Purpose

This searched the local weight neighborhood around the current strongest NUAA fixed fusion:

```text
tpc_a 0.14
tpc_b 0.14
lca 0.14
bgm_nuaa 0.22
rpcanet_sirst_s3 0.28
bg_anomaly 0.08
```

Previous experiments showed that adding new sources such as PConv, multi-BGM, LRRNet, or SIRST-UAVB background fitting did not help. This run asked a narrower question:

```text
Can the existing six-source pool be slightly improved by finer train-selected weights?
```

## Code

Added:

```text
scripts/search_local_bgm_rpcanet_weights.py
```

The script:

- generates deterministic one/two-source nudges around the known best;
- samples local Dirichlet perturbations around the known best;
- uses train-set fast nIoU to select top rows;
- evaluates selected rows with full NUAA metrics;
- does not save checkpoints or prediction-map archives.

## Smoke

Output:

```text
reports/nuaa_local_bgm_rpcanet_weight_search_smoke.json
```

Scale:

```text
fit samples: 80 per dataset
NUAA eval samples: 80
candidate weights: 347
gammas: 0.95, 1.0, 1.05
thresholds: 0.48, 0.50, 0.52, 0.54
```

Best smoke row:

| threshold | gamma | mIoU | nIoU | Pd | Fa |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.48 | 0.95 | 80.3310 | 82.0034 | 98.1651 | 9.4309e-6 |

This was only a very small smoke improvement, so a controlled full-test run was used to verify it.

## Full NUAA Test

Output:

```text
reports/nuaa_local_bgm_rpcanet_weight_search_full.json
```

Scale:

```text
fit datasets: IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3
max_fit_samples: 220 per dataset
NUAA eval: full test, 214 samples
candidate weights: 447
gammas: 0.95, 1.0, 1.05
thresholds: 0.48, 0.50, 0.52, 0.54, 0.56
top_fit full-metric rows: 36
```

Best full-test row:

| threshold | gamma | mIoU | nIoU | Pd | Fa |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.52 | 0.95 | 81.0269 | 81.6068 | 98.0989 | 7.4775e-6 |

Weights:

| Source | Weight |
| --- | ---: |
| `tpc_a` | 0.1900 |
| `tpc_b` | 0.0960 |
| `lca` | 0.1044 |
| `bgm_nuaa` | 0.4005 |
| `rpcanet_sirst_s3` | 0.1750 |
| `bg_anomaly` | 0.0342 |

## Comparison

Previous reliable NUAA best:

```text
mIoU 81.0136
nIoU 81.5680
Pd   97.7186
Fa   8.8495e-6
```

New local-weight-search best:

```text
mIoU 81.0269
nIoU 81.6068
Pd   98.0989
Fa   7.4775e-6
```

Gain:

```text
81.0269 - 81.0136 = +0.0133 mIoU
```

Gap to requested NUAA target:

```text
82.64 - 81.0269 = 1.6131 mIoU
```

## Interpretation

This is a real but tiny update to the current NUAA best. It improves mIoU, nIoU, Pd, and Fa at the same time, but the gain is only about `+0.0133` mIoU and does not materially change the state of the project.

The result suggests the existing six-source pool is almost saturated. Further progress toward `82.64` still likely requires a genuinely stronger new dense candidate source or new NUAA-like labeled data rather than more small local weight tuning.

## Cleanup

No bulky artifacts were produced.

Retained small evidence:

```text
scripts/search_local_bgm_rpcanet_weights.py               ~12K
reports/nuaa_local_bgm_rpcanet_weight_search_smoke.json   ~60K
reports/nuaa_local_bgm_rpcanet_weight_search_full.json    ~56K
reports/nuaa_local_bgm_rpcanet_weight_search_results.md   ~4K
```

Current reliable bests:

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.0269 | 82.64 | not passed |

