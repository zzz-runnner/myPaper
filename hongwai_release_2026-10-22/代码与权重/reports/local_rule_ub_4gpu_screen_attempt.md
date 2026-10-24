# NUAA local-rule uncertainty-boundary 4-GPU screen

Date: 2026-06-16

## Purpose

This run retested the uncertainty/boundary residual refiner, but on the current strongest NUAA official-Mamba local-rule teacher rather than the older weak teacher cache.

The intent was to change the dense probability map itself, not to tune only thresholds or connected components.

Reference before this run:

```text
current NUAA best: 82.2928
target:            82.6400
remaining gap:      0.3472
```

## Launch

Added:

```text
scripts/run_local_rule_ub_one.sh
scripts/launch_local_rule_ub_nuaa_4gpu_screen.sh
```

Common setup:

```text
train caches:
  IRSTD-1K train
  NUAA-SIRST train
  NUDT-SIRST train
  SIRST3 train
test cache:
  NUAA-SIRST test
teacher_mode: local_rule
epochs: 3
batch_size: 4
patch_size: 512
base_channels: 64
max_train_samples_per_cache: 128
thresholds: 0.488..0.512
```

Runtime:

```text
4 x RTX 2080 Ti
about 10362 MiB / 11264 MiB per GPU
GPU utilization: 100% during training
```

## Results

| branch | seed | residual | anchor | boundary | best epoch | threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s1101_r006_a004_ub15 | 20260631 | 0.060 | 0.040 | 1.50 | 1 | 0.500 | 82.2671 | 82.0250 | 96.9582 | 9.6041e-6 |
| s1103_r010_a006_ub25 | 20260633 | 0.100 | 0.060 | 2.50 | 1 | 0.502 | 82.2519 | 82.0651 | 96.5779 | 11.5935e-6 |
| s1102_r008_a005_ub20 | 20260632 | 0.080 | 0.050 | 2.00 | 1 | 0.498 | 82.2480 | 82.0776 | 96.5779 | 11.5249e-6 |
| s1104_r012_a004_ub30 | 20260634 | 0.120 | 0.040 | 3.00 | 2 | 0.508 | 82.2480 | 82.0776 | 96.5779 | 11.5249e-6 |

Best row:

```text
s1101_r006_a004_ub15
mIoU: 82.2671
```

Comparison:

```text
current retained NUAA best: 82.2928
best local-rule UB row:     82.2671
delta:                     -0.0257
```

## Cleanup

All four branches were below the retained NUAA best, so their checkpoints were deleted:

```text
runs/local_rule_ub_nuaa_4gpu_screen/*/best.pth.tar
```

The run directory now keeps only logs and summaries and is about `72K`.

## Interpretation

This closes the obvious "rerun uncertainty/boundary on the new local-rule teacher" branch. The refiner is stable and uses the four GPUs heavily, but it stays in the same 82.24-82.27 region as the HFM-LK local-rule screen.

Do not extend this exact UB screen by only adding epochs. The current NUAA bottleneck still needs a genuinely new dense source or stronger cross-domain module, not another local loss weighting around the same local-rule map.
