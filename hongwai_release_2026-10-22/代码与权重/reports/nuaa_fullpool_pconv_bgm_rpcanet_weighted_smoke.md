# NUAA Full-Pool PConv + BGM + RPCANet Weighted Smoke

Generated: 2026-06-15 UTC

## Purpose

This tested whether SIRST-UAV PConv/MSHNet sources should be added to the current strongest NUAA BGM+RPCANet fixed-fusion pool.

The candidate pool was:

```text
tpc_a
tpc_b
lca
pconv_uav_d05
pconv_uav_p43
official BGM
RPCANet++ SIRST-s3
background anomaly
```

This is a broader same-domain + external-method pool than the current fixed fusion, whose best full-test NUAA result is:

```text
81.0136 mIoU
```

## Scale

Output:

```text
reports/nuaa_fullpool_pconv_bgm_rpcanet_weighted_smoke.json
```

Settings:

```text
fit datasets: IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3
max_fit_samples: 80 per dataset
NUAA eval samples: 80
weight mode: coarse
weights screened: 198
gammas: 0.95, 1.0, 1.05
thresholds: 0.48, 0.50, 0.52, 0.55
top_fit full-metric rows: 24
```

No checkpoint or prediction archive was written.

## Result

Best eval row on the 80-image NUAA smoke:

| kind | threshold | gamma | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_external_weighted | 0.48 | 1.00 | 80.0979 | 81.8079 | 98.1651 | 9.4309e-6 |

Best weights:

| Source | Weight |
| --- | ---: |
| `tpc_a` | 0.1053 |
| `tpc_b` | 0.1053 |
| `lca` | 0.1053 |
| `pconv_uav_d05` | 0.0000 |
| `pconv_uav_p43` | 0.0000 |
| `bgm_nuaa` | 0.4211 |
| `rpcanet_sirst_s3` | 0.2105 |
| `bg_anomaly` | 0.0526 |

Additional diagnostic:

```text
top_fit full-metric rows: 24
rows with positive PConv weight: 0
```

## Interpretation

The weight search actively drops both PConv-UAV sources when BGM and RPCANet are present. This is stronger negative evidence than merely seeing a low score: the train-fit selection found no top candidate where either PConv map deserved nonzero weight.

Therefore, the SIRST-UAV PConv maps should not be added to the current NUAA fixed-fusion pool. This agrees with the earlier NUAA PConv component-fusion result: PConv gives some oracle curiosity but does not convert into a valid NUAA improvement.

## Current Bests

The reliable bests remain unchanged:

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.0136 | 82.64 | not passed |

## Cleanup

No bulky artifacts were created.

Retained small files only:

```text
reports/nuaa_fullpool_pconv_bgm_rpcanet_weighted_smoke.json  ~44K
reports/nuaa_fullpool_pconv_bgm_rpcanet_weighted_smoke.md    ~4K
```

