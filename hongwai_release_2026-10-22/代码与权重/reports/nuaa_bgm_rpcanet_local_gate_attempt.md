# NUAA BGM + RPCANet Local Gate Attempt

Generated: 2026-06-15 UTC

## Purpose

This tested a conservative local-gate extension of the current strongest NUAA fusion:

```text
tpc_a 0.14
tpc_b 0.14
lca 0.14
official BGM 0.22
RPCANet++ SIRST-s3 0.28
background anomaly 0.08
```

The gate was designed to avoid another unconstrained pixel MLP collapse. It keeps the fixed fusion as the base map and only tries small local changes:

- suppress low-saliency / low-agreement false positives;
- suppress background-anomaly-only responses;
- rescue RPCANet or BGM-supported weak target pixels;
- combine rescue and suppression in one rule.

## Code Added

```text
scripts/eval_bgm_rpcanet_local_gate.py
```

The script reuses the same TPC/LCA/BGM/RPCANet/background-normality sources as the current best fixed fusion. It does not save checkpoints or prediction archives.

## Smoke Result

Output:

```text
reports/nuaa_bgm_rpcanet_local_gate_smoke.json
```

Scale:

```text
fit samples: 48
NUAA eval samples: 24
candidates: 220
thresholds: 0.48, 0.50, 0.52
```

Best selected row:

| kind | threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0.52 | 81.6225 | 83.4882 | 100.0000 | 0.0000e-6 |

Decision from smoke:

```text
The selected rule was the unchanged base fusion.
```

## Medium Result

Output:

```text
reports/nuaa_bgm_rpcanet_local_gate_mid.json
```

Scale:

```text
fit samples: 240
NUAA eval samples: 80
candidates: 439
thresholds: 0.48, 0.50, 0.52, 0.54
```

Best selected row:

| kind | threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0.48 | 80.1564 | 81.4571 | 97.2477 | 14.7121e-6 |

Decision from medium run:

```text
Again, the selected rule was the unchanged base fusion.
```

## Interpretation

This route did not show evidence that local handcrafted gating can improve the current BGM+RPCANet fixed fusion. Since both the smoke and the larger diagnostic selected the base map itself, running the same gate over the full 214-image NUAA test split is unlikely to close the remaining NUAA gap:

```text
NUAA target: 82.64
current full-test best: 81.0136
remaining gap: 1.6264 mIoU
```

## Cleanup

No bulky artifacts were produced. Retained files are small:

```text
scripts/eval_bgm_rpcanet_local_gate.py         ~16K
reports/nuaa_bgm_rpcanet_local_gate_smoke.json ~32K
reports/nuaa_bgm_rpcanet_local_gate_mid.json   ~44K
```

The current reliable best remains unchanged:

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.0136 | 82.64 | not passed |

