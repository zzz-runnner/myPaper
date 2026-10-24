# NUAA LPSM follow-up2 4-GPU attempt

Date: 2026-06-16

## Purpose

This run performed a narrower seed/parameter rescreen around the current winning LPSM configuration:

```text
state_stages = e2,mid
base_channels = 64
batch_size = 3
```

Reference before this run:

```text
current retained NUAA best: 82.3504
target:                     82.6400
remaining gap:               0.2896
```

## Runtime

All four valid branches ran with:

```text
about 9588 MiB / 11264 MiB per RTX 2080 Ti
GPU utilization: about 99-100%
```

## Results

| branch | residual | anchor | best epoch | threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s1324_r0055_a019 | 0.055 | 0.19 | 2 | 0.499 | 82.3174 | 82.0590 | 96.9582 | 9.1925e-6 |
| s1321_r0052_a019 | 0.052 | 0.19 | 2 | 0.504 | 82.2928 | 82.1412 | 96.9582 | 9.0553e-6 |
| s1323_r0058_a017 | 0.058 | 0.17 | 1 | 0.500 | 82.2715 | 82.1331 | 97.3384 | 8.3007e-6 |
| s1322_r0055_a018_salt | 0.055 | 0.18 | 3 | 0.499 | 82.2651 | 82.0274 | 96.9582 | 9.5355e-6 |

Comparison:

```text
current retained best: 82.3504
best follow-up2 row:   82.3174
delta:                -0.0330
```

## Cleanup

All four checkpoints were below the retained best, so they were deleted:

```text
runs/local_rule_lpsm_nuaa_followup2_4gpu/*/best.pth.tar
```

The run directory now keeps only logs and summaries and is about `72K`.

## Interpretation

This narrower rescreen did not improve the retained result. It suggests the current best is not simply improved by small residual/anchor/seed perturbations around `s1313`.

The retained NUAA best remains:

```text
LPSM follow-up s1313 + min_area=3
mIoU: 82.3504
```

The remaining gap to target is still:

```text
0.2896 mIoU
```
