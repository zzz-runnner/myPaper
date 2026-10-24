# SIRST-UAVB400 Strong-Fusion HFM-LK NUAA Attempt

Generated: 2026-06-15 UTC

## Purpose

This tested a new-data-source extension of the current best NUAA line:

```text
corrected TPC/LCA + BGM + RPCANet++ + background-anomaly strong teacher
+ HFM-LK residual refiner
+ SIRST-UAVB train samples as an extra cached training domain
```

This was meant to test whether an additional UAV-based small-target distribution could improve the current NUAA-SIRST best of `81.1502` mIoU.

## Cache Build

First, a 4-sample smoke cache confirmed that `SIRST-UAVB` works with the corrected BGM/RPCANet cache pipeline.

Then a 400-sample SIRST-UAVB train cache was built with:

```text
rpcanet_sirst_s3 base_size = 256
fit datasets = IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3, SIRST-UAVB
samples = 400
```

The cache successfully wrote:

```text
reports/fusion_cache_bgm_rpcanet_sirst_uavb_train400/SIRST-UAVB__train.npz
samples = 400
size = 340 MB
```

The cache was deleted after the run because the experiment did not improve NUAA.

## 4GPU Training

Launcher:

```text
scripts/launch_strong_fusion_hfmlk_nuaa_uavb400_4gpu_tmux.sh
```

Training caches:

```text
IRSTD-1K train
NUAA-SIRST train
NUDT-SIRST train
SIRST3 train
SIRST-UAVB train400
```

Evaluation:

```text
NUAA-SIRST full test
```

Four branches were launched:

```text
nuaa_strong_fusion_uavb400_s781
nuaa_strong_fusion_uavb400_s782
nuaa_weighted_raw_uavb400_s783
nuaa_weighted_raw_uavb400_s784
```

GPU utilization during training:

```text
GPU0: 9048 MB, 99%
GPU1: 9048 MB, 99%
GPU2: 9048 MB, 99%
GPU3: 10450 MB, 99%
```

This met the requested 4GPU / 9GB+ utilization target.

## Results

The run was stopped after epoch 2 because none of the branches exceeded the retained NUAA best `81.1502` mIoU, and the best branch decreased from epoch 1 to epoch 2.

| run | teacher | best epoch | threshold | mIoU | nIoU | Pd | Fa |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nuaa_strong_fusion_uavb400_s781` | `strong_fusion` | 1 | 0.46 | 81.1396 | 81.2450 | 98.0989 | 10.4273e-6 |
| `nuaa_weighted_raw_uavb400_s784` | `weighted_raw` | 1 | 0.44 | 81.0793 | 81.6225 | 98.0989 | 7.4775e-6 |
| `nuaa_weighted_raw_uavb400_s783` | `weighted_raw` | 1 | 0.56 | 81.0546 | 81.6125 | 98.0989 | 7.4089e-6 |
| `nuaa_strong_fusion_uavb400_s782` | `strong_fusion` | 2 | 0.44 | 81.0457 | 81.6047 | 98.0989 | 7.4775e-6 |

Comparison to current best:

```text
current best NUAA: 81.1502
best with SIRST-UAVB400: 81.1396
delta: -0.0106 mIoU
```

The result is close, but it is still lower than the previous best and does not reduce the target gap.

## Cleanup

Deleted failed checkpoints:

```text
runs/strong_fusion_hfmlk_nuaa_uavb400_4gpu/*/best.pth.tar
```

Deleted failed temporary cache:

```text
reports/fusion_cache_bgm_rpcanet_sirst_uavb_train400/
reports/fusion_cache_bgm_rpcanet_sirst_uavb_train400.log
```

Retained only small logs:

```text
runs/strong_fusion_hfmlk_nuaa_uavb400_4gpu/
```

Retained size:

```text
52 KB
```

## Interpretation

SIRST-UAVB as a small 400-sample extra training domain does not improve the current NUAA refiner. The first epoch came very close to the retained best, but epoch 2 moved downward, so scaling this exact setup to full SIRST-UAVB is not justified without a better domain-balancing or source-selection mechanism.

The current NUAA best remains:

```text
runs/strong_fusion_hfmlk_nuaa_4gpu/nuaa_strong_fusion_s703/best.pth.tar
mIoU = 81.1502
```

The NUAA +5 target remains unmet:

```text
target = 82.64
gap = 1.4898 mIoU
```
