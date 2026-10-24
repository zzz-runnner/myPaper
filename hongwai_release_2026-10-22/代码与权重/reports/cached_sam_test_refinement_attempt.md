# Cached SAM Test-Time Refinement Attempt

Generated: 2026-06-13 UTC

## Purpose

This tested SAM as a test-time refinement/candidate source, rather than only as training pseudo labels. It reuses frozen cached prediction maps as prompts and runs SAM-B on the original infrared test images.

This is a foundation-model transfer attempt, not SCTransNet parameter tuning.

## Code Added

```text
scripts/eval_cached_sam_refinement.py
```

The script:

- reads an existing prediction cache;
- uses a selected cached probability map as the teacher prompt map;
- prompts SAM-B with component points/boxes;
- evaluates `soft_rim` and `max` fusion against the original test masks;
- writes only small JSON, no masks/checkpoints.

## Smoke Results

NUAA-SIRST 20-image smoke:

```text
reports/cached_sam_refine_smoke_nuaa.json
```

| Row | mIoU | Pd | Fa |
|---|---:|---:|---:|
| cached baseline `tpc_b`, th 0.50 | 78.36 | 100.00 | 3.85 |
| SAM `soft_rim` | 78.36 | 100.00 | 3.85 |
| SAM `max` | about 23.74 | 72.73 | 1214.97 |

IRSTD-1K 20-image smoke:

```text
reports/cached_sam_refine_smoke_irstd.json
```

| Row | mIoU | Pd | Fa |
|---|---:|---:|---:|
| cached baseline avg, th 0.35 | 75.98 | 97.06 | 3.43 |
| SAM `soft_rim` | 75.92 or lower | 97.06 / 91.18 | 3.05-38.53 |
| SAM `max` | about 44.90 | 91.18 | 61.61 |

## Decision

Do not expand this exact SAM test-time refinement to full test evaluation. The conservative `soft_rim` path does not change the evaluated hard masks, while `max` fusion greatly over-expands regions and damages mIoU/false alarms.

This agrees with the earlier SAM pseudo-label evidence: SAM can draw larger object-like regions, but for infrared tiny targets it tends to add boundary/background area rather than reliable new target pixels.

Current reliable best remains unchanged:

| Dataset | Valid best | mIoU |
|---|---|---:|
| IRSTD-1K | dual-PConv average fusion region | 72.43 |
| NUAA-SIRST | background-normality component fusion | 79.34 |
