# NUAA LPSM 256-sample 4-GPU attempt

Date: 2026-06-16

## Purpose

This run tested whether the current strongest NUAA LPSM line could improve by using more multi-dataset cache samples per source.

Reference before this run:

```text
current retained NUAA best: 82.3730
target:                     82.6400
remaining gap:               0.2670
```

The previous positive LPSM runs used `max_train_samples_per_cache=128`. This follow-up raised that to `256` while keeping the useful `e2,mid` local-preserving state-mixing path.

## Launch

Added:

```text
scripts/launch_local_rule_lpsm_nuaa_256sample_4gpu.sh
```

Common setup:

```text
teacher_mode: local_rule
train caches: IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3
test cache: NUAA-SIRST
state_stages: e2,mid
base_channels: 64
batch_size: 3
epochs: 3
max_train_samples_per_cache: 256
AMP: true
weighted_dice: true
thresholds: 0.498, 0.499, 0.500, 0.501, 0.502, 0.503, 0.504
```

Runtime:

```text
4 x RTX 2080 Ti
about 9588 MiB / 11264 MiB per GPU
GPU utilization: 99-100% during training
```

## Results

| branch | seed | residual | anchor | best epoch | threshold | mIoU | nIoU | Pd | Fa | delta vs retained best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s1331_r0055_a018_s71 | 20260681 | 0.055 | 0.18 | 2 | 0.499 | 82.2922 | 82.0504 | 96.9582 | 9.3983e-6 | -0.0808 |
| s1333_r0058_a017_s73 | 20260683 | 0.058 | 0.17 | 2 | 0.500 | 82.2742 | 82.0447 | 96.9582 | 9.6041e-6 | -0.0988 |
| s1334_r0052_a019_s74 | 20260684 | 0.052 | 0.19 | 1 | 0.499 | 82.2444 | 82.0452 | 96.9582 | 8.9181e-6 | -0.1286 |
| s1332_r0055_a018_s72 | 20260682 | 0.055 | 0.18 | 2 | 0.500 | 82.1876 | 82.0091 | 97.3384 | 7.8205e-6 | -0.1854 |

Best row:

```text
s1331_r0055_a018_s71 epoch 2
mIoU: 82.2922
```

Comparison:

```text
current retained NUAA best: 82.3730
best 256-sample row:        82.2922
delta:                     -0.0808
```

## Cleanup

All four branches were below the retained NUAA best, so their checkpoints were deleted:

```text
runs/local_rule_lpsm_nuaa_256sample_4gpu/*/best.pth.tar
```

The run directory now keeps only logs and summaries:

```text
runs/local_rule_lpsm_nuaa_256sample_4gpu
size after cleanup: 72K
```

## Interpretation

Increasing `max_train_samples_per_cache` from `128` to `256` did not improve the current LPSM path. The best row fell back near the older pre-LPSM retained level and stayed clearly below the current two-checkpoint fusion result.

The retained NUAA best remains:

```text
0.45 * LPSM s1303 + 0.55 * LPSM s1313
gamma = 0.998
threshold = 0.501
connected-component min_area = 3
mIoU = 82.3730
```

The overall goal remains incomplete because NUAA still has not crossed `82.64`.
