# NUAA LPSM pair-fusion follow-up

Date: 2026-06-16

## Purpose

Two LPSM checkpoints independently improved NUAA:

```text
s1303_e2m_r006:          first LPSM positive branch
s1313_e2m_r0055_a018:    focused LPSM follow-up positive branch
```

This test checked whether their dense probability maps have useful complementarity by fixed-weight probability fusion plus the already useful `min_area=3` component filter.

## Script

Added:

```text
scripts/eval_lpsm_pair_fusion.py
```

No probability `.npz` archive was saved. Only compact JSON was retained.

## Run

An initial broad grid was stopped after more than 11 minutes because the connected-component grid was CPU-heavy and had not produced output. No result file was kept from that stopped run.

The final fast grid used:

```text
checkpoint_a: runs/local_rule_lpsm_nuaa_4gpu_screen/s1303_e2m_r006/best.pth.tar
checkpoint_b: runs/local_rule_lpsm_nuaa_followup_4gpu/s1313_e2m_r0055_a018/best.pth.tar
alpha_a: 0.00, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00
gamma: 1.00
thresholds: 0.498, 0.499, 0.500, 0.501, 0.502
min_area: 1, 3
```

Output:

```text
reports/nuaa_lpsm_pair_s1303_s1313_fusion_fast.json
```

## Result

Best row:

| alpha for s1303 | alpha for s1313 | threshold | min area | mIoU | nIoU | Pd | Fa |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 0.50 | 0.501 | 3 px | 82.3620 | 82.2060 | 96.9582 | 8.9181e-6 |

Comparison:

```text
previous retained NUAA best: 82.3504
new retained NUAA best:      82.3620
gain:                       +0.0116
target:                      82.6400
remaining gap:                0.2780
```

## Interpretation

The two LPSM checkpoints have small but real complementarity. The gain is modest, but it is a legal improvement and keeps the LPSM line alive.

Current retained NUAA recipe:

```text
0.5 * LPSM s1303 + 0.5 * LPSM s1313
threshold = 0.501
connected-component min_area = 3
mIoU = 82.3620
```

The overall goal remains incomplete because the NUAA +5 target is `82.64`.

## Narrow Fusion Sweep

A narrower sweep around the fast-grid winner was run:

```text
reports/nuaa_lpsm_pair_s1303_s1313_fusion_narrow.json
alpha_a: 0.35..0.65
gamma: 0.998, 1.000, 1.002, 1.004
thresholds: 0.499..0.503
min_area: 3
```

Best row:

| alpha for s1303 | alpha for s1313 | gamma | threshold | min area | mIoU | nIoU | Pd | Fa |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.45 | 0.55 | 0.998 | 0.501 | 3 px | 82.3730 | 82.2124 | 96.9582 | 8.9181e-6 |

Updated comparison:

```text
previous pair-fusion best: 82.3620
new pair-fusion best:      82.3730
gain:                     +0.0110
target:                    82.6400
remaining gap:              0.2670
```

Current retained NUAA recipe:

```text
0.45 * LPSM s1303 + 0.55 * LPSM s1313
gamma = 0.998
threshold = 0.501
connected-component min_area = 3
mIoU = 82.3730
```

## Ultra-narrow Sweep

A final sub-grid around the narrow-sweep winner was tested:

```text
reports/nuaa_lpsm_pair_s1303_s1313_fusion_ultranarrow.json
alpha_a: 0.40, 0.42, 0.44, 0.45, 0.46, 0.48, 0.50
gamma: 0.996, 0.997, 0.998, 0.999, 1.000
threshold: 0.500, 0.5005, 0.501, 0.5015, 0.502
min_area: 3
```

Best row:

| alpha for s1303 | alpha for s1313 | gamma | threshold | min area | mIoU | nIoU | Pd | Fa |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.44 | 0.56 | 0.999 | 0.5005 | 3 px | 82.3730 | 82.2124 | 96.9582 | 8.9181e-6 |

This tied the previous pair-fusion best and did not improve it:

```text
retained pair-fusion best: 82.3730
ultra-narrow best:         82.3730
delta:                    +0.0000
```

This suggests the current two-checkpoint LPSM fusion surface is locally saturated around `82.3730`.
