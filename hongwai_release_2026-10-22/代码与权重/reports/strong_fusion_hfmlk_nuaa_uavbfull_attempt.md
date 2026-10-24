# Full SIRST-UAVB Strong-Fusion HFM-LK NUAA Attempt

Generated: 2026-06-15 UTC

## Purpose

Tested whether using the full SIRST-UAVB train split as an additional source domain improves the current NUAA-SIRST blocker. This extends the previous SIRST-UAVB400 attempt from 400 samples to the full 2400 training samples.

## Cache Build

Built a full corrected BGM/RPCANet strong-fusion cache for:

```text
SIRST-UAVB train: 2400 samples
cache size: 2012.05 MB
```

Cache path before cleanup:

```text
reports/fusion_cache_bgm_rpcanet_sirst_uavb_full/SIRST-UAVB__train.npz
```

The cache was deleted after the failed training attempt to save space.

## 4GPU Training

Launcher:

```text
scripts/launch_strong_fusion_hfmlk_nuaa_uavbfull_4gpu_tmux.sh
```

Training caches:

```text
IRSTD-1K train
NUAA-SIRST train
NUDT-SIRST train
SIRST3 train
SIRST-UAVB train full 2400
```

Four branches:

```text
nuaa_strong_fusion_uavbfull_s881
nuaa_strong_fusion_uavbfull_s882
nuaa_weighted_raw_uavbfull_s883
nuaa_weighted_raw_uavbfull_s884
```

Observed GPU utilization during training:

```text
GPU0: 9048 MB, 99%
GPU1: 9048 MB, 99%
GPU2: 9048 MB, 99%
GPU3: 10450 MB, 99%
```

This met the requested 4x RTX 2080Ti / 9GB+ utilization target.

## Results

Stopped after epoch 2 because all branches were below the retained NUAA best of 81.2710 mIoU.

| run | best epoch seen | threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nuaa_weighted_raw_uavbfull_s883` | 1 | 0.44 | 81.1368 | 81.3556 | 98.10 | 10.2901e-6 |
| `nuaa_strong_fusion_uavbfull_s882` | 1 | 0.44 | 81.1196 | 81.2973 | 98.10 | 8.7809e-6 |
| `nuaa_weighted_raw_uavbfull_s884` | 2 | 0.54 | 81.1061 | 81.6456 | 98.10 | 7.2717e-6 |
| `nuaa_strong_fusion_uavbfull_s881` | 2 | 0.58 | 80.9660 | 81.5069 | 97.72 | 7.4089e-6 |

Best full-UAVB result:

```text
nuaa_weighted_raw_uavbfull_s883 epoch 1 mIoU = 81.1368
```

Comparison:

```text
current retained NUAA best: 81.2710
best full-UAVB result:   81.1368
delta:                   -0.1342
```

## Cleanup

Stopped tmux sessions:

```text
sf_hfmlk_uavbf_a
sf_hfmlk_uavbf_b
sf_hfmlk_uavbf_c
sf_hfmlk_uavbf_d
```

Deleted failed checkpoints:

```text
runs/strong_fusion_hfmlk_nuaa_uavbfull_4gpu/*/best.pth.tar
```

Deleted the full SIRST-UAVB cache:

```text
reports/fusion_cache_bgm_rpcanet_sirst_uavb_full/
```

Retained only lightweight logs; run directory is about 52 KB.

## Conclusion

Full SIRST-UAVB as an extra train domain does not improve the current NUAA solution under this HFM-LK strong-fusion setup. It is very close to the previous 400-sample result but still below the retained best, so continuing epochs would spend a lot of 4GPU time without evidence of closing the 1.3690 mIoU gap.

Current reliable bests remain:

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.2710 | 82.64 | not passed |

The next useful direction should add a genuinely new dense candidate source or a different NUAA-specific training objective; simply adding more SIRST-UAVB samples to this teacher-refiner line is now tested and not sufficient.
