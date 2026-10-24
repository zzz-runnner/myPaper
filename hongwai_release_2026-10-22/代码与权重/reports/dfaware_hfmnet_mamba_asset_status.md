# DFAwareNet / HFMNet Mamba Asset Status

Generated: 2026-06-15 UTC

## Purpose

NUAA-SIRST still needs a stronger dense candidate source. This pass searched for recently released official assets in the dynamic-frequency / state-space-model IRSTD family.

Two official repositories were found and downloaded:

```text
HFMNet     https://github.com/Fortuneteller6/HFMNet
DFAwareNet https://github.com/Fortuneteller6/DFAwareNet
```

These are relevant to the user's requested non-SCTransNet innovation direction:

```text
feature enhancement
dynamic frequency cue
VSSM / Mamba-style state-space modeling
```

## Downloaded Assets

### HFMNet

Downloaded official Google Drive weights:

```text
external/HFMNet/weights_download/IRSTD-1K/mIoU__HFMNet_IRSTD-1K_epoch.pth.tar
external/HFMNet/weights_download/NUAA-SIRST/mIoU__HFMNet_NUAA-SIRST_epoch.pth.tar
```

Reported README numbers:

| Dataset | IoU | Pd | Fa |
| --- | ---: | ---: | ---: |
| IRSTD-1K | 70.38 | 92.52 | 4.18 |
| NUAA-SIRST | 79.16 | 99.08 | 0.35 |

Retained size:

```text
external/HFMNet: about 53 MB
```

### DFAwareNet

Downloaded official Google Drive weights:

```text
external/DFAwareNet/weights_download/IRSTD-1K/mIoU__DFAwareNet_IRSTD-1K_epoch.pth.tar
external/DFAwareNet/weights_download/NUAA-SIRST/mIoU__DFAwareNet_NUAA-SIRST_epoch.pth.tar
```

Reported README numbers:

| Dataset | IoU | Pd | Fa |
| --- | ---: | ---: | ---: |
| IRSTD-1K | 70.96 | 90.82 | 4.40 |
| NUAA-SIRST | 79.65 | 99.08 | 1.42 |

Retained size:

```text
external/DFAwareNet: about 56 MB
```

For both repos, `.git` directories were deleted after download to avoid keeping unnecessary history.

## Environment Blocker

Both models import and use:

```text
mamba_ssm.ops.selective_scan_interface
```

HFMNet failed at import:

```text
ModuleNotFoundError: No module named 'mamba_ssm'
```

DFAwareNet has the same dependency in its model file and uses `VSSM -> selective_scan_fn` in the core network.

This is not a removable unused import. The downloaded official checkpoints depend on the VSSM/Mamba blocks, so replacing selective scan with a dummy implementation would invalidate the model.

Binary-only install check:

```text
pip install causal-conv1d==1.5.3.post1 mamba-ssm==2.2.6.post3 --only-binary=:all: --no-deps
```

failed with:

```text
No matching distribution found for causal-conv1d
```

This matches the earlier DCCS/MiM blocker:

```text
current env: torch 2.4.1+cu121
no nvcc available
no compatible prebuilt causal-conv1d / mamba_ssm wheel found by pip
```

## Decision

Do not attempt source builds in the current working environment. Previous DCCS dependency attempts already failed at the same boundary, and rebuilding the working IRSTD env risks breaking the current verified pipeline.

Keep both repos and weights because they are small and are useful if one of these becomes available:

```text
compatible prebuilt mamba_ssm + causal_conv1d wheels for torch 2.4.1/cu121
or a separate CUDA-build-capable environment with nvcc
or a dedicated conda env matching the authors' torch/CUDA stack
```

Current reliable bests remain:

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.0269 | 82.64 | not passed |

## Interpretation

These official dynamic-frequency/Mamba assets are not standalone enough to solve the target based on README numbers, but they could be useful as complementary candidate maps if the Mamba environment is solved. For now, they are a prepared but blocked candidate line, not an evaluated result.
