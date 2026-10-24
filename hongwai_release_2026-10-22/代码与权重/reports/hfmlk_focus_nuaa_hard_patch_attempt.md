# NUAA-Focused HFM-LK Hard-Patch Sampler Attempt

Generated: 2026-06-15 UTC

## Purpose

After candidate-source, SIRST-UAVB, SAM, and small refusion attempts failed to close the NUAA gap, this tested a training-objective/data-sampling innovation instead of adding another map source.

The change targets NUAA-specific errors:

```text
teacher high-confidence false positives -> hard clutter patches
teacher low-confidence true target pixels -> missed-target patches
```

## Code Changes

Modified:

```text
scripts/train_teacher_anchor_refiner_from_cache.py
scripts/train_hfm_lk_anchor_refiner_from_cache.py
```

Added optional parameters:

```text
--focus_dataset
--focus_patch_prob
--focus_hard_threshold
--focus_miss_threshold
```

When enabled for `NUAA-SIRST`, the training dataset samples patches around high-confidence teacher clutter, missed target pixels, and true target pixels more often.

## 4GPU Training

Launcher:

```text
scripts/launch_hfmlk_focus_nuaa_4gpu_tmux.sh
```

Training data:

```text
IRSTD-1K train
NUAA-SIRST train
NUDT-SIRST train
SIRST3 train
```

Evaluation:

```text
NUAA-SIRST full test
```

Observed GPU usage:

```text
GPU0: about 9048-9050 MB, 99%
GPU1: about 9048-9050 MB, 99%
GPU2: about 9048-9050 MB, 99%
GPU3: about 10450 MB, 99%
```

This met the requested four-card / 9GB+ utilization target.

## Results

Stopped after epoch 2 because all branches stayed clearly below the retained NUAA best `81.2710` mIoU.

| run | best epoch seen | threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nuaa_focus_weighted_s903` | 2 | 0.44 | 81.0793 | 81.6225 | 98.10 | 7.4775e-6 |
| `nuaa_focus_weighted_s904` | 1 | 0.54 | 81.0668 | 81.6278 | 98.10 | 7.3403e-6 |
| `nuaa_focus_strong_s902` | 1 | 0.46 | 81.0609 | 81.5945 | 98.10 | 7.4775e-6 |
| `nuaa_focus_strong_s901` | 1 | 0.46 | 81.0485 | 81.4481 | 98.10 | 7.8205e-6 |

Best focused-sampler result:

```text
nuaa_focus_weighted_s903 epoch 2 mIoU = 81.0793
```

Comparison:

```text
current retained NUAA best: 81.2710
best focused sampler:      81.0793
delta:                     -0.1917
```

## Cleanup

Stopped tmux sessions:

```text
hfmlk_focus_nuaa_a
hfmlk_focus_nuaa_b
hfmlk_focus_nuaa_c
hfmlk_focus_nuaa_d
```

Deleted failed checkpoints:

```text
runs/hfmlk_focus_nuaa_4gpu/*/best.pth.tar
```

Retained only lightweight logs; run directory is about 52 KB.

## Conclusion

The hard-patch sampler and stronger NUAA false-positive weighting did not improve NUAA. It slightly over-optimized toward conservative calibration and stayed around 81.08 mIoU, below the current 81.2710 recipe.

This suggests the current HFM-LK teacher-refiner is not limited by seeing too few hard NUAA patches; the missing 1.369 mIoU likely requires a genuinely better candidate source, a stronger NUAA-specific pretrained model, or a different model family rather than more hard-negative weighting on the same teacher pool.
