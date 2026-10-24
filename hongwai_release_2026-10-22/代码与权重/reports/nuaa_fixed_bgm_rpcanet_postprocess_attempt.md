# NUAA Fixed BGM + RPCANet Postprocess Attempt

Generated: 2026-06-14 UTC

## Purpose

This attempted to squeeze additional NUAA-SIRST mIoU from the current best fixed fusion result:

```text
81.0136 mIoU
```

by sweeping:

```text
threshold
connected-component min area
open / close / dilate / erode morphology
```

over the fixed BGM+RPCANet fusion map.

## Code

Added:

```text
scripts/eval_fixed_bgm_rpcanet_postprocess.py
```

The first implementation evaluated full PD/FA metrics for every postprocess row and was too slow. It was then changed to fast-rank candidates by pixel mIoU first and only run full metrics on a small top-k.

## Runs

The broad sweep was started and stopped because full metric evaluation was too slow for this sweep style:

```text
reports/nuaa_fixed_bgm_rpcanet_postprocess_sweep.log
```

A narrower full-metric sweep completed:

```text
reports/nuaa_fixed_bgm_rpcanet_postprocess_narrow.json
reports/nuaa_fixed_bgm_rpcanet_postprocess_narrow.log
```

Best row:

| Gamma | Threshold | Min area | Op | Kernel | mIoU | nIoU | Pd | Fa |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 0.50 | 1 | none | 1 | 81.0136 | 81.5680 | 97.7186 | 8.8495e-6 |

This is exactly the same as the no-postprocess fixed-fusion result. The best postprocess choice was therefore to do nothing.

Two later fast-top attempts were stopped after confirming the narrow full-metric sweep had already completed and showed no improvement:

```text
reports/nuaa_fixed_bgm_rpcanet_postprocess_fasttop.log
reports/nuaa_fixed_bgm_rpcanet_postprocess_top4.log
```

No checkpoint or prediction-map archive was written.

## Decision

Do not count this as an improvement. The current valid NUAA best remains:

```text
81.0136 mIoU
```

This path needs a faster metric implementation or cached postprocess maps before it is worth continuing.

## Current Bests

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.0136 | 82.64 | not passed |
