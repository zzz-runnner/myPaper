# HFM-LK DFAware-Teacher NUAA 4GPU Attempt

Generated: 2026-06-15 UTC

## Purpose

The previous diagnostic showed that component-only filtering on the current dense map cannot reach the NUAA target. This attempt therefore changed the dense training anchor instead of doing more postprocessing.

The idea was to distill the validated external-source fusion into the HFM-LK refiner:

```text
s703_bgm = clip(0.96 * s703 - 0.04 * bgm_nuaa, 0, 1) ** 0.94
dfaware_fused = (0.88 * s703_bgm + 0.12 * DFAwareNet) ** 1.02
```

A new NUAA cache source named `dfaware_fused` was added and used as the NUAA teacher. Other datasets used fallback to `strong_fusion`, so the training remained full multi-dataset.

## Code / Cache Changes

Modified:

```text
scripts/train_teacher_anchor_refiner_from_cache.py
```

Added teacher fallback mode:

```text
--teacher_mode "dfaware_fused,strong_fusion"
```

This uses `dfaware_fused` when present, otherwise falls back to `strong_fusion`, while preserving the original 27-channel feature stack for checkpoint compatibility.

Added launcher:

```text
scripts/launch_hfmlk_dfaware_teacher_nuaa_4gpu_tmux.sh
```

Built temporary/experimental NUAA cache:

```text
reports/fusion_cache_dfaware_teacher_nuaa/NUAA-SIRST__train.npz
reports/fusion_cache_dfaware_teacher_nuaa/NUAA-SIRST__test.npz
```

Cache size:

```text
123 MB
```

## 4GPU Full Multi-Dataset Training

Training datasets:

```text
IRSTD-1K train
NUAA-SIRST train with dfaware_fused teacher
NUDT-SIRST train
SIRST3 train
```

Evaluation:

```text
NUAA-SIRST full test
```

Started from retained checkpoint:

```text
runs/strong_fusion_hfmlk_nuaa_4gpu/nuaa_strong_fusion_s703/best.pth.tar
```

Branches:

```text
nuaa_dfaware_teacher_s951
nuaa_dfaware_teacher_s952
nuaa_dfaware_teacher_fp_s953
nuaa_dfaware_teacher_focus_s954
```

Observed training GPU usage:

```text
GPU0: 8998 MiB, 98%
GPU1: 8998 MiB, 98%
GPU2: 9098 MiB, 97%
GPU3: 8998 MiB, 98%
```

This met the requested four-card and approximately 9GB+ utilization target.

## Results

Stopped after epoch 2 because all branches were below the retained NUAA best `81.7031`, and epoch 2 did not improve over epoch 1.

| run | best epoch seen | threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nuaa_dfaware_teacher_focus_s954` | 1 | 0.52 | 81.6665 | 81.8850 | 97.7186 | 7.9577e-6 |
| `nuaa_dfaware_teacher_s952` | 1 | 0.488 | 81.6375 | 81.8834 | 97.7186 | 8.0949e-6 |
| `nuaa_dfaware_teacher_fp_s953` | 1 | 0.52 | 81.6338 | 81.8744 | 97.7186 | 7.9577e-6 |
| `nuaa_dfaware_teacher_s951` | 1 | 0.52 | 81.6309 | 81.8351 | 97.7186 | 8.0263e-6 |

Best branch:

```text
nuaa_dfaware_teacher_focus_s954 epoch 1 mIoU = 81.6665
```

Comparison:

```text
current retained NUAA best:      81.7031
best DFAware-teacher training:   81.6665
delta:                          -0.0366
```

## Cleanup

Stopped tmux sessions:

```text
hfmlk_dfaware_nuaa_a
hfmlk_dfaware_nuaa_b
hfmlk_dfaware_nuaa_c
hfmlk_dfaware_nuaa_d
```

Deleted failed checkpoints:

```text
runs/hfmlk_dfaware_teacher_nuaa_4gpu/*/best.pth.tar
```

Deleted smoke run:

```text
runs/_smoke_dfaware_teacher
```

Retained only lightweight logs and this report. The experimental 123 MB DFAware teacher cache is retained for now because it is small and may be useful for a later, more conservative identity-distillation run.

## Conclusion

The DFAware-fused teacher was successfully integrated into full multi-dataset 4GPU HFM-LK training and used the requested GPU budget, but it did not beat the simpler test-time fusion + component drop-k result.

Current reliable status remains:

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.7031 | 82.64 | not passed |

The active goal remains open.
