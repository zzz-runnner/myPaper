# External Source Priority After NUAA Plateau

Generated: 2026-06-15 UTC

## Current Verified Bests

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.3192 | 82.64 | not passed |

The NUAA gap is still:

```text
82.64 - 81.3192 = 1.3208 mIoU
```

## What This Pass Checked

After calibration, component selection, per-image threshold transfer, and rank-loss training saturated or failed, this pass re-screened local and online external-source candidates for a genuinely new dense prediction source.

## Local Sources Already Evaluated And Not Worth Repeating

| Source / line | Status | Best / reason |
| --- | --- | --- |
| EFLNet official boxes | runnable, failed | NUAA best box-to-mask around 50.0311 mIoU; boxes too coarse for pixel mIoU |
| DFRL DNANet/PCSDL/RDIAN | runnable, failed | NUAA best 66.6145; far below current |
| AnomalyCLIP | runnable smoke, failed | NUAA smoke mIoU 4.33, huge false alarms |
| DHiF anchor refiner | runnable 4GPU, failed | NUAA best 79.2230 under older cache line |
| Wavelet-anchor refiner | runnable 4GPU, failed | NUAA best 79.2212 under older cache line |
| DWTFreqNet | runnable, not promising | limited probe NUAA best 42.43; full run too slow from scratch |
| SIRST-UAVB extra domain | tested 400 and full 2400 | did not beat current s703+BGM line |
| SAM / MobileSAM refinement | smoke failed | conservative unchanged, aggressive overexpanded masks |
| Rank-loss HFM-LK | runnable 4GPU, failed | epoch1 best 81.0909, stopped and cleaned |
| Per-image threshold transfer | runnable, failed | train-selected test 81.1612; no improvement |
| Component stump / drop-k | small valid gain | current NUAA best 81.3192, but saturated |

## Blocked But Potentially Useful Sources

### HFMNet / DFAwareNet

Assets are present:

```text
external/HFMNet/weights_download/NUAA-SIRST/mIoU__HFMNet_NUAA-SIRST_epoch.pth.tar
external/HFMNet/weights_download/IRSTD-1K/mIoU__HFMNet_IRSTD-1K_epoch.pth.tar
external/DFAwareNet/weights_download/NUAA-SIRST/mIoU__DFAwareNet_NUAA-SIRST_epoch.pth.tar
external/DFAwareNet/weights_download/IRSTD-1K/mIoU__DFAwareNet_IRSTD-1K_epoch.pth.tar
```

Blocked by environment:

```text
mamba_ssm missing
causal_conv1d missing
nvcc not available
pip binary probe: no causal-conv1d wheel available for this env
```

These cannot be evaluated safely in the current working env without either compatible wheels or a separate CUDA-build-capable environment.

## Online/Repo Recheck

### Text-IRSTD

Online paper and GitHub still point to:

```text
https://github.com/Zhengsy0407/Text-IRSTD
```

Current remote repository still contains only README, no code/weights/dataset. Local `.git` was removed after confirming this.

### RealScene-ISTD

Correct online repo:

```text
https://github.com/luy0222/RealScene-ISTD
```

Local code is present, but no dataset or weights are present. README provides Baidu links for:

```text
RealScene-ISTD dataset
pre-prepared small target patches
RealScene-ISTD official weights
IRSTD-1K official weights
```

No NUAA official weight is provided, but the dataset/patches are still valuable as new cross-domain training data and hard-background augmentation.

## Highest-Value User Uploads Now

If the goal is still to push NUAA over 82.64, the highest-value next assets to place under `/data1` are:

1. RealScene-ISTD dataset from the official Baidu link.
2. RealScene-ISTD pre-prepared small target patches.
3. RealScene-ISTD official weights and IRSTD-1K official weights.
4. A separate environment asset for Mamba models:
   - compatible `causal_conv1d` and `mamba_ssm` wheels for Python 3.10, torch 2.4.1+cu121;
   - or a CUDA/nvcc-capable conda env matching HFMNet/DFAwareNet.
5. Text-IRSTD code/data/weights if the authors release more than README later.
6. Any SAIST official code/weights if found.

## Practical Next Direction

Without new assets, the current local source pool appears saturated around `81.32` NUAA. The next meaningful experiment should not be another calibration pass on `s703+BGM`. It should be one of:

```text
A. add RealScene-ISTD data/patches into the multi-dataset cache/training line;
B. unlock HFMNet/DFAwareNet official weights in a compatible Mamba environment and evaluate them as new dense maps;
C. obtain a newly released Text-IRSTD / SAIST dense model or data source.
```

Until one of those assets exists locally, further work is likely to produce only tiny changes around the current `81.3192` NUAA plateau.
