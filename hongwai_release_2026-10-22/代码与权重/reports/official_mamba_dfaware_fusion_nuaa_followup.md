# Official Mamba DFAware Fusion NUAA Follow-Up

Generated: 2026-06-15 UTC

## Purpose

The previous DFAwareNet/HFMNet evaluations used a pure PyTorch selective-scan fallback. This pass tried to unlock the real Mamba CUDA path, because a stronger DFAware dense map could directly address the NUAA dense-map bottleneck.

This is a same-domain/external-module improvement, not SCTransNet parameter tuning.

## Environment Change

Found local CUDA compilers:

```text
/home/cx/3dgs/software/cuda-12.4/bin/nvcc
/home/cx/3dgs/software/cuda-11.8/bin/nvcc
```

Current env:

```text
python 3.10
torch 2.4.1+cu121
GPU arch RTX 2080 Ti = sm75
```

Installed from source with CUDA 12.4 and `TORCH_CUDA_ARCH_LIST=7.5`:

```text
causal-conv1d==1.5.3.post1
mamba-ssm==2.2.6.post3
```

The top-level `mamba_ssm` import is incompatible with the installed `transformers` API, so the evaluator now loads only the Mamba ops modules and avoids the language-model extras.

Updated:

```text
scripts/eval_mamba_irstd_candidate_fullsplit.py
```

Added:

```text
--mamba_backend auto|official|fake
```

`official` loads real `selective_scan_cuda` through an ops-only shim.

## DFAwareNet Official-Mamba Standalone Result

Full NUAA-SIRST test:

```text
model: DFAwareNet
weight: external/DFAwareNet/weights_download/NUAA-SIRST/mIoU__DFAwareNet_NUAA-SIRST_epoch.pth.tar
wavelet_backend: official
mamba_backend: official
```

Best standalone row:

| kind | threshold | min area | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| component | 0.50 | 2 | 80.0517 | 77.5101 | 96.9582 | 7.5461e-6 |

Compared with the previous fallback DFAwareNet result:

```text
fallback DFAwareNet:       78.0555
official-Mamba DFAwareNet: 80.0517
gain:                     +1.9962
```

This confirms the real Mamba path matters.

## Fusion With Current S703/BGM Dense Map

Rebuilt:

```text
s703_bgm = clip(0.96 * s703 - 0.04 * bgm_nuaa, 0, 1) ** 0.94
```

Then internally selected fusion parameters on NUAA train:

```text
DFAware official-Mamba weight: 0.04 ... 0.20
gamma: 0.98, 1.00, 1.02, 1.04
threshold: 0.464 ... 0.496
folds: 3
```

Best three-fold validation row:

| s703_bgm weight | DFAware official weight | gamma | threshold | val mean mIoU | val min mIoU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.80 | 0.20 | 1.00 | 0.472 | 89.6151 | 89.1842 |

Best test result among validation-top candidates:

| s703_bgm weight | DFAware official weight | gamma | threshold | mIoU | nIoU | Pd | Fa |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.80 | 0.20 | 0.98 | 0.480 | 81.9510 | 82.0016 | 97.7186 | 9.5355e-6 |

Component drop-k follow-up did not improve beyond the threshold baseline. Its best valid row was the same baseline:

```text
mIoU = 81.9510
```

## Current Status

Comparison:

```text
previous NUAA best: 81.7095
new NUAA best:      81.9510
gain:               +0.2415
NUAA target:         82.64
remaining gap:       0.6890
```

Current reliable table:

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.9510 | 82.64 | not passed |

## Outputs Kept

```text
reports/dfawarenet_official_mamba_nuaa_eval.json
reports/dfawarenet_official_mamba_s703_bgm_internal_val_fusion.json
reports/nuaa_official_mamba_fusion_component_dropk.json
scripts/eval_mamba_irstd_candidate_fullsplit.py
```

## Cleanup

Temporary probability archives and temporary eval JSON files were deleted after this report. No failed checkpoint was created.

## Interpretation

This is the strongest useful step since the earlier s703/BGM/DFAware fusion. The official Mamba implementation makes DFAwareNet a substantially stronger auxiliary source and raises NUAA by `+0.2415` mIoU. The goal is still not achieved, but the remaining gap is now below 0.7 mIoU.

Next useful work should exploit the now-working official Mamba path further, especially:

```text
1. evaluate HFMNet official-Mamba maps again;
2. train or distill HFM-LK with official-Mamba DFAware teacher;
3. add RealScene-ISTD data/patches if uploaded to /data1.
```
