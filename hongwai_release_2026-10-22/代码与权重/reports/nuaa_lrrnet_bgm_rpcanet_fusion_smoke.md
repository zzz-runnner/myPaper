# NUAA LRRNet + BGM + RPCANet Fusion Smoke

Generated: 2026-06-15 UTC

## Purpose

This tested whether LRRNet's low-rank background representation can complement the current strongest NUAA BGM+RPCANet fixed fusion.

Current fixed-fusion base:

```text
tpc_a 0.14
tpc_b 0.14
lca 0.14
bgm_nuaa 0.22
rpcanet_sirst_s3 0.28
bg_anomaly 0.08
```

LRRNet source:

```text
external/LRRNet/checkpoints/IRDST_1K_miou_0.7239356450020329.pkl
```

The test tried LRRNet as:

- small weighted blend;
- low-base target rescue;
- false-positive suppressor;
- rescue plus suppression.

## Code

Added:

```text
scripts/eval_lrrnet_bgm_rpcanet_fusion.py
```

Implementation note:

LRRNet and RPCANet both use a top-level Python package name `models`. The script loads LRRNet by file path through `importlib` to avoid polluting RPCANet imports.

## Scale

Output:

```text
reports/nuaa_lrrnet_bgm_rpcanet_fusion_smoke.json
```

Settings:

```text
fit datasets: IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3
max_fit_samples: 80 per dataset
NUAA eval samples: 80
thresholds: 0.48, 0.50, 0.52, 0.55
top_fit rows: 32
LRRNet input size: 512
```

No checkpoint or prediction archive was written.

## Result

Best eval row:

| kind | threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0.50 | 80.3145 | 81.7124 | 97.2477 | 13.3918e-6 |

Top LRRNet rescue rows were identical to the base row:

```text
max_rescue rows: same mIoU, same nIoU, same Pd, same Fa as base
```

This means LRRNet did not add a useful changed region under the train-selected rules.

## Interpretation

LRRNet is conceptually relevant for background suppression, but the released IRDST/IRSTD checkpoint does not improve the current NUAA BGM+RPCANet fusion in this smoke test.

Decision:

```text
Do not run a full 214-image NUAA test for this LRRNet fusion configuration.
Do not add LRRNet to the current NUAA fixed-fusion pool.
```

This agrees with earlier LRRNet evidence:

```text
standalone NUAA full-test best: 67.25 mIoU
multi-dataset LRRNet probe best NUAA: 70.84 mIoU
```

## Cleanup

No bulky artifacts were produced.

Retained small files:

```text
scripts/eval_lrrnet_bgm_rpcanet_fusion.py         ~12K
reports/nuaa_lrrnet_bgm_rpcanet_fusion_smoke.json ~28K
reports/nuaa_lrrnet_bgm_rpcanet_fusion_smoke.md   ~4K
```

Current reliable bests remain:

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.0136 | 82.64 | not passed |

