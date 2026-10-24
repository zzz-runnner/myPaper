# NUAA S703 TTA and Conservative Resume Attempt

Generated: 2026-06-15 UTC

## Purpose

This follow-up tested two low-risk ways to improve the current best NUAA checkpoint:

```text
runs/strong_fusion_hfmlk_nuaa_4gpu/nuaa_strong_fusion_s703/best.pth.tar
```

Current best before this attempt:

```text
81.2277 mIoU
threshold = 0.468
connected component min_area = 2 px
```

## TTA Evaluation

Evaluated `s703` with:

```text
tta = none,hflip,vflip,hvflip
thresholds = 0.430:0.610:0.002
```

Top rows:

| mode | threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: |
| no_tta | 0.468 | 81.2034 | 81.3915 | 98.0989 | 9.1239e-6 |
| tta_4 | 0.468 | 81.1926 | 81.4267 | 98.0989 | 8.9867e-6 |
| tta_4 | 0.482 | 81.1877 | 81.5186 | 98.0989 | 8.6437e-6 |

TTA reduced false alarms slightly but lost enough overlap that it did not beat no-TTA.

Then `tta_4` maps were tested with connected-component min-area filtering:

| mode | threshold | min area | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tta_4 + component filter | 0.468 | 2 px | 81.2169 | 81.4465 | 97.7186 | 8.7123e-6 |

This is still below the current no-TTA + min-area result:

```text
81.2277 mIoU
```

Temporary TTA maps were deleted:

```text
reports/nuaa_s703_hfmlk_tta4_maps_tmp.npz
```

Retained small outputs:

```text
reports/nuaa_s703_hfmlk_tta4_fine_threshold_eval.json
reports/nuaa_s703_hfmlk_tta4_component_filter_search.json
```

## Conservative Resume Training

Added optional resume support to:

```text
scripts/train_hfm_lk_anchor_refiner_from_cache.py
```

New argument:

```text
--resume_checkpoint
```

Launched four conservative 4GPU fine-tuning branches from the retained `s703` checkpoint:

```text
scripts/launch_hfmlk_s703_resume_nuaa_4gpu_tmux.sh
```

Training data:

```text
IRSTD-1K train
NUAA-SIRST train
NUDT-SIRST train
SIRST3 train
```

GPU use:

```text
GPU0: about 8998 MB, 99%
GPU1: about 8998 MB, 99%
GPU2: about 8998 MB, 99%
GPU3: about 8998 MB, 99%
```

Epoch-1 results:

| run | lr | threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nuaa_s703_resume_lr03_s824` | 0.000003 | 0.46 | 81.1777 | 81.3832 | 98.0989 | 9.1239e-6 |
| `nuaa_s703_resume_lr05_s823` | 0.000005 | 0.468 | 81.1547 | 81.3643 | 98.0989 | 8.8495e-6 |
| `nuaa_s703_resume_lr10_s822` | 0.000010 | 0.468 | 81.1528 | 81.3891 | 98.0989 | 9.0553e-6 |
| `nuaa_s703_resume_lr20_s821` | 0.000020 | 0.468 | 81.1353 | 81.3844 | 98.0989 | 9.1925e-6 |

All were below the current best `81.2277`, so the run was stopped after epoch 1.

## Cleanup

Deleted failed resume checkpoints:

```text
runs/hfmlk_s703_resume_nuaa_4gpu/*/best.pth.tar
```

Retained only lightweight logs:

```text
runs/hfmlk_s703_resume_nuaa_4gpu
size = 52 KB
```

## Current Status

Current NUAA-SIRST best remains:

```text
81.2277 mIoU
```

from:

```text
s703 HFM-LK strong_fusion checkpoint
threshold = 0.468
connected component min_area = 2 px
```

Remaining gap:

```text
target = 82.64
gap = 1.4123 mIoU
```

Interpretation:

TTA and conservative fine-tuning do not solve the remaining NUAA gap. The `s703` checkpoint appears very close to its local optimum; continuing with small learning-rate fine-tuning is not a useful main path.
