# HFMNet External Candidate Attempt

Generated: 2026-06-15 UTC

## Purpose

NUAA-SIRST still needs a stronger dense candidate source. This pass searched for updated executable assets from recent IRSTD methods and found:

```text
HFMNet: A Lightweight Feature Enhancement Model for Infrared Small Target Detection
Official repo: https://github.com/Fortuneteller6/HFMNet
```

The README reports:

| Dataset | Reported IoU | Pd | Fa |
| --- | ---: | ---: | ---: |
| IRSTD-1K | 70.38 | 92.52 | 4.18 |
| NUAA-SIRST | 79.16 | 99.08 | 0.35 |
| NUDT-SIRST | 91.34 | 99.30 | 6.05 |
| DenseSIRST | 69.35 | 91.34 | 8.52 |

These standalone numbers are below the current local bests, but HFMNet is a different VSSM/feature-enhancement family and could have been useful as a complementary fusion source.

## Assets Downloaded

Cloned:

```text
external/HFMNet
```

Downloaded official Google Drive checkpoints:

```text
external/HFMNet/weights_download/IRSTD-1K/mIoU__HFMNet_IRSTD-1K_epoch.pth.tar
external/HFMNet/weights_download/NUAA-SIRST/mIoU__HFMNet_NUAA-SIRST_epoch.pth.tar
```

Sizes:

```text
IRSTD-1K checkpoint: about 27.4 MB
NUAA-SIRST checkpoint: about 27.4 MB
external/HFMNet total after cleanup: about 53 MB
```

Cleanup:

```text
external/HFMNet/.git
```

was removed after cloning to avoid keeping unnecessary repository history.

## Environment Check

Tried importing and constructing the official model:

```text
from model.HFMNet import HFMNet
```

It failed immediately:

```text
ModuleNotFoundError: No module named 'mamba_ssm'
```

The HFMNet implementation is not merely importing Mamba accidentally. Its `HFMM` blocks use:

```text
VSSM -> selective_scan_fn
```

inside the encoder, so replacing it with a dummy function would invalidate the official weights.

Checked binary-only install:

```text
pip install causal-conv1d==1.5.3.post1 mamba-ssm==2.2.6.post3 --only-binary=:all: --no-deps
```

Result:

```text
No matching binary distribution found for causal-conv1d in this environment.
```

This is the same blocker as DCCS/MiM:

```text
current torch/CUDA env has no nvcc
Mamba selective-scan stack needs compatible prebuilt wheels or a build-capable CUDA toolchain
```

## Decision

HFMNet cannot be evaluated in the current environment without a working Mamba selective-scan stack.

Do not launch source builds or a long environment rebuild right now because:

1. DCCS already failed at the same dependency boundary;
2. HFMNet reported standalone NUAA/IRSTD numbers are below current local bests;
3. the likely value is complementarity, not standalone breakthrough, so it is not worth risking the working `irstd` environment.

Kept:

```text
external/HFMNet code + two official checkpoints
```

because they are only about 53 MB and can be used immediately if a compatible Mamba environment or wheel set becomes available.

Current reliable bests remain:

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.0269 | 82.64 | not passed |

## Next Useful Asset

The highest-value user-upload or external asset remains:

```text
RealScene-ISTD dataset and official weights
SAIST official code/weights if found
Text-IRSTD code/weights if released
prebuilt mamba_ssm + causal_conv1d wheels matching torch 2.4.1/cu121, or a separate CUDA-build-capable env
```
