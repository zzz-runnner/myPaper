# NUAA Pixel Fusion BGM + RPCANet Results

Generated: 2026-06-14 UTC

## Purpose

This tested a train-selected pixel-level fusion route over the current best complementary sources:

```text
tpc_a
tpc_b
lca
official BGM
RPCANet++ SIRST-s3
background-normality candidate
local image cues
```

Unlike the previous fixed fusion result, this uses multi-dataset train labels to learn the fusion. The goal was to see whether a legal train-fitted calibrator can beat the current fixed NUAA result:

```text
81.0136 mIoU
```

## Code

Added:

```text
scripts/train_pixel_fusion_bgm_rpcanet.py
```

The script supports:

```text
MLP pixel fusion
residual_linear calibration around the fixed 81.0136 fusion
```

No checkpoint is saved unless explicitly requested.

## MLP Smoke

Output:

```text
reports/nuaa_pixel_fusion_bgm_rpcanet_smoke.json
```

Setup:

```text
max_fit_samples = 8 per dataset
max_eval_samples = 20
train_pixels = 18,130
positive_pixels = 1,746
```

Best result:

| Model | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: |
| MLP pixel fusion | 0.1325 | 0.3413 | 36.3636 | 552252.8551e-6 |

Decision:

```text
Plain MLP pixel fusion is unstable and unusable in this setup.
```

## Residual Linear Smoke

Output:

```text
reports/nuaa_pixel_fusion_bgm_rpcanet_residual_smoke.json
```

Setup:

```text
max_fit_samples = 8 per dataset
max_eval_samples = 20
train_pixels = 18,130
positive_pixels = 1,746
residual_scale = 0.10
```

Best result:

| Model | Threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: |
| residual linear calibrator | 0.50 | 83.0668 | 83.2314 | 100.0000 | 0.0000e-6 |

This only proves the residual calibrator can run and does not immediately collapse. It is not a final number because it uses only 20 eval samples.

## Full NUAA Test

Output:

```text
reports/nuaa_pixel_fusion_bgm_rpcanet_residual_full.json
reports/nuaa_pixel_fusion_bgm_rpcanet_residual_full.log
```

Training/fitting data:

```text
IRSTD-1K train, capped 220 samples
NUAA-SIRST train, full 213 samples
NUDT-SIRST train, capped 220 samples
SIRST3 train, capped 220 samples
```

Pixel sample count:

```text
train_pixels = 937,691
positive_pixels = 43,739
```

Evaluation:

```text
NUAA-SIRST full test, 214 samples
```

Best full result:

| Model | Threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: |
| residual linear calibrator | 0.50 | 80.8383 | 81.2403 | 97.7186 | 9.8785e-6 |

## Decision

The train-fitted residual calibrator is valid but below the current fixed fusion:

```text
fixed fusion: 81.0136
residual calibrator: 80.8383
```

So this does not update the current best.

## Cleanup

No checkpoint or prediction-map archive was created. Retained only small JSON/log files and the script.

## Current Bests

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.0136 | 82.64 | not passed |
