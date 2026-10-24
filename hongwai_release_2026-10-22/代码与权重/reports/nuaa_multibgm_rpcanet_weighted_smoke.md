# NUAA Multi-BGM + RPCANet Weighted Smoke

Generated: 2026-06-15 UTC

## Purpose

This tested whether cross-dataset official BGM checkpoints add useful complementary information to the current strongest NUAA fusion pool.

The previous best fixed fusion uses only the NUAA-trained BGM checkpoint:

```text
bgm_nuaa
```

This smoke added:

```text
bgm_irstd
bgm_nuaa
bgm_nudt
```

alongside:

```text
tpc_a
tpc_b
lca
RPCANet++ SIRST-s3
background anomaly
```

## Scale

Output:

```text
reports/nuaa_multibgm_rpcanet_weighted_smoke.json
```

Settings:

```text
fit datasets: IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3
max_fit_samples: 80 per dataset
NUAA eval samples: 80
weight mode: coarse
weights screened: 157
gammas: 0.95, 1.0, 1.05
thresholds: 0.48, 0.50, 0.52, 0.55
top_fit full-metric rows: 32
```

No training checkpoint or prediction archive was created.

## Result

Best eval row:

| kind | threshold | gamma | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| teacher_external_weighted | 0.48 | 0.95 | 79.7005 | 81.4469 | 96.3303 | 15.8438e-6 |

Best weights:

| Source | Weight |
| --- | ---: |
| `tpc_a` | 0.1176 |
| `tpc_b` | 0.1176 |
| `lca` | 0.1176 |
| `bgm_irstd` | 0.1569 |
| `bgm_nuaa` | 0.1569 |
| `bgm_nudt` | 0.1569 |
| `rpcanet_sirst_s3` | 0.1765 |
| `bg_anomaly` | 0.0000 |

## Interpretation

This is clearly below the current full-test best:

```text
current reliable NUAA best: 81.0136
multi-BGM smoke best:       79.7005
```

The coarse weight generator assigns equal weight to the three BGM checkpoints once BGM is selected. In this setting, adding the IRSTD-1K and NUDT-SIRST BGM maps hurts NUAA rather than improving it.

Decision:

```text
Do not run the full 214-image NUAA test for this multi-BGM configuration.
Do not add bgm_irstd or bgm_nudt to the current NUAA fixed-fusion pool.
```

## Cleanup

No bulky artifacts were produced.

Retained small files only:

```text
reports/nuaa_multibgm_rpcanet_weighted_smoke.json  ~48K
reports/nuaa_multibgm_rpcanet_weighted_smoke.md    ~4K
```

Current reliable bests remain:

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.0136 | 82.64 | not passed |

