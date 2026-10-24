# HFM-LK Copy-Paste NUAA 4GPU Attempt

Generated: 2026-06-15 UTC

## Purpose

Tested a RealScene-ISTD-inspired target/background augmentation without needing new uploaded assets. The idea was to copy real small-target pixels from existing training masks into other background locations inside the same training patch while leaving the teacher logit unchanged.

This simulates:

```text
image contains a real-looking target
teacher map does not explicitly mark it
```

The intended effect was to make the HFM-LK residual refiner less dependent on the existing teacher map and more able to recover missed target-like structures.

## Code Changes

Modified:

```text
scripts/train_teacher_anchor_refiner_from_cache.py
scripts/train_hfm_lk_anchor_refiner_from_cache.py
```

Added training dataset parameters:

```text
--copy_paste_prob
--copy_paste_max_area
```

Added launcher:

```text
scripts/launch_hfmlk_copypaste_nuaa_4gpu_tmux.sh
```

A single-GPU smoke test passed before the full run.

## 4GPU Full Multi-Dataset Setup

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

Branches:

```text
nuaa_cp_strong_s931
nuaa_cp_strong_s932
nuaa_cp_weighted_s933
nuaa_cp_weighted_s934
```

Observed GPU utilization:

```text
GPU0: 9048-9050 MB, 98%
GPU1: 10448 MB, 98-99%
GPU2: 9048-9050 MB, 98%
GPU3: 10448 MB, 98%
```

This met the requested four-card / 9GB+ utilization target.

## Results

Stopped after epoch 1 because all branches were far below the retained NUAA best `81.3192`.

| run | epoch | threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nuaa_cp_strong_s931` | 1 | 0.54 | 81.0125 | 81.5376 | 98.0989 | 7.2717e-6 |
| `nuaa_cp_strong_s932` | 1 | 0.56 | 81.0820 | 81.6296 | 98.0989 | 7.4089e-6 |
| `nuaa_cp_weighted_s933` | 1 | 0.52 | 81.0440 | 81.5523 | 98.0989 | 7.3403e-6 |
| `nuaa_cp_weighted_s934` | 1 | 0.52 | 81.0451 | 81.5446 | 98.0989 | 7.4089e-6 |

Best copy-paste branch:

```text
nuaa_cp_strong_s932 epoch 1 mIoU = 81.0820
```

Comparison:

```text
current retained NUAA best: 81.3192
best copy-paste result:    81.0820
delta:                     -0.2372
```

## Cleanup

Stopped tmux sessions:

```text
hfmlk_cp_nuaa_a
hfmlk_cp_nuaa_b
hfmlk_cp_nuaa_c
hfmlk_cp_nuaa_d
```

Deleted failed checkpoints:

```text
runs/hfmlk_copypaste_nuaa_4gpu/*/best.pth.tar
```

Retained only lightweight logs; run directory is about 52 KB.

## Conclusion

The in-place target copy-paste augmentation did not improve NUAA. It appears to push the model toward the same conservative 81.0 region as the rank-loss attempt, rather than changing the dense probability map in the needed direction.

Current reliable status remains:

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.3192 | 82.64 | not passed |
