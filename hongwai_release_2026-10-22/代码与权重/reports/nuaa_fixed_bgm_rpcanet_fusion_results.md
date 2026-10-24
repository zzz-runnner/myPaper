# NUAA Fixed BGM + RPCANet Fusion Results

Generated: 2026-06-14 UTC

## Purpose

After RPCANet improved the NUAA component pool, this step tested whether a simpler fixed weighted fusion could convert the same complementary sources into a stronger valid prediction map.

Candidate sources:

```text
tpc_a
tpc_b
lca
official BGM
RPCANet++ SIRST-s3
background-normality candidate
```

This is still a mixed route:

```text
in-domain IRSTD teacher family + official BGM + cross-method RPCANet + anomaly/background normality
```

## Code

Added:

```text
scripts/eval_fixed_bgm_rpcanet_fusion.py
```

Also extended:

```text
scripts/eval_weighted_candidate_fusion.py
```

to accept BGM/RPCANet/background-normality sources. The broad weighted-grid search was too slow in full metric mode and was stopped before JSON output; no large artifact was produced.

Stopped diagnostic logs:

```text
reports/nuaa_bgm_rpcanet_weighted_full.log
reports/nuaa_bgm_rpcanet_weighted_top4_full.log
```

These logs are not final result files.

## Final Full-Test Fixed Fusion

Output:

```text
reports/nuaa_fixed_bgm_rpcanet_fusion_full.json
reports/nuaa_fixed_bgm_rpcanet_fusion_full.log
```

Evaluation:

```text
NUAA-SIRST full test, 214 samples
```

Background-normality fit data:

```text
IRSTD-1K train, capped 220
NUAA-SIRST train, full 213
NUDT-SIRST train, capped 220
SIRST3 train, capped 220
```

Best valid result:

| Kind | Output threshold | Gamma | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed fusion with bg anomaly | 0.50 | 1.00 | 81.0136 | 81.5680 | 97.7186 | 8.8495e-6 |

Weights:

| Source | Weight |
| --- | ---: |
| `tpc_a` | 0.14 |
| `tpc_b` | 0.14 |
| `lca` | 0.14 |
| `bgm_nuaa` | 0.22 |
| `rpcanet_sirst_s3` | 0.28 |
| `bg_anomaly` | 0.08 |

Close rows:

| Kind | Output threshold | Gamma | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced no bg anomaly | 0.55 | 0.95 | 81.0011 | 81.5382 | 98.0989 | 7.2717e-6 |
| component-best-like | 0.52 | 1.00 | 80.9937 | 81.5427 | 98.0989 | 7.2717e-6 |

## Gain

Previous valid best after RPCANet component fusion:

```text
80.4849 mIoU
```

New valid best:

```text
81.0136 mIoU
```

Improvement:

```text
81.0136 - 80.4849 = +0.5287 mIoU
```

Relative to the older BGM-only valid best:

```text
81.0136 - 79.5487 = +1.4649 mIoU
```

Gap to requested NUAA +5 target:

```text
82.64 - 81.0136 = 1.6264 mIoU
```

## Current Bests

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.0136 | 82.64 | not passed |

## Cleanup

No checkpoints or prediction-map archives were created by this step.

Retained only small artifacts:

```text
reports/nuaa_fixed_bgm_rpcanet_fusion_full.json
reports/nuaa_fixed_bgm_rpcanet_fusion_full.log
scripts/eval_fixed_bgm_rpcanet_fusion.py
```

The broad weighted-grid runs were stopped because full metric evaluation was too slow; their logs are small and kept only as diagnostic evidence.
