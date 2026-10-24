# PAL / Text-IRSTD / SAIST Source Screen and PAL-MSDA Results

Generated: 2026-06-14 UTC

## Purpose

NUAA-SIRST remains below the requested +5 mIoU target. This step screened newer methods from the provided method list and tested a new non-SCTransNet backbone candidate.

Focus:

```text
Text-IRSTD
SAIST
PAL
PAL-MSDA frequency-attention backbone
```

## Source Screen

### Text-IRSTD

Local repository:

```text
external/Text-IRSTD
```

Updated with:

```text
git pull --ff-only
```

Finding:

```text
The repository is still only README.md.
No runnable code or checkpoint is available locally.
```

Decision:

```text
Cannot evaluate Text-IRSTD yet.
```

### SAIST

Web/source check did not find a reliable official runnable repository/checkpoint in this environment.

Decision:

```text
Cannot evaluate SAIST yet without official code/checkpoints or user-provided assets.
```

### PAL

Cloned:

```text
external/PAL
```

PAL contains runnable code and multiple models, but no pretrained checkpoint was included. The original PAL framework is point-supervised and expects its own generated dataset layout.

Useful local finding:

```text
PAL-MSDA, PAL-ACM, and PAL-DNA all instantiate and run a forward pass in the current env.
```

Patched:

```text
external/PAL/model/MSDA/MSDA_no_sigmoid.py
```

Reason:

```text
The original custom wavelet backward forced filters to float16, which broke float32 backward.
The patch makes filters follow dx dtype/device.
```

## PAL-MSDA Local Training Adapter

Added:

```text
scripts/train_pal_msda_multidata.py
scripts/launch_pal_msda_4gpu_tmux.sh
```

The adapter:

- reuses PAL's MSDA-Net frequency-attention model;
- trains on local full-mask IRSTD datasets instead of PAL's point-label folder layout;
- uses the same local full-test metric path as other experiments;
- supports AMP to fit 2080 Ti memory.

## GPU / Memory

Initial attempts with `patch_size=512` and batch sizes 16, 14, 12, and 8 all OOM'd near:

```text
10.3-10.5 GB allocated
```

Stable final configuration:

```text
patch_size = 320
batch_size = 16
AMP enabled
```

Runtime:

```text
4 x RTX 2080 Ti
about 10.1 GB per GPU
98-99% utilization
```

This met the requested 9GB+ GPU usage.

## Full Multi-Dataset Training

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

Runs:

| GPU | run | lr | target repeat | pos weight |
| ---: | --- | ---: | ---: | ---: |
| 0 | `nuaa_msda_s301` | 0.00030 | 8 | 25 |
| 1 | `nuaa_msda_s302` | 0.00022 | 10 | 30 |
| 2 | `nuaa_msda_s303` | 0.00016 | 12 | 35 |
| 3 | `nuaa_msda_s304` | 0.00012 | 14 | 40 |

## Results

Stopped after epoch 6 / epoch 5 region because the full-test results stayed far below the current NUAA best.

| run | epoch | mIoU | nIoU | Pd | Fa | threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nuaa_msda_s301` | 2 | 69.9069 | 73.3214 | 92.0152 | 30.4588e-6 | 0.50 |
| `nuaa_msda_s301` | 4 | 71.9624 | 75.4459 | 94.2966 | 15.6410e-6 | 0.70 |
| `nuaa_msda_s301` | 6 | 72.6830 | 76.2962 | 95.4373 | 42.8069e-6 | 0.10 |
| `nuaa_msda_s302` | 2 | 57.5834 | 66.1531 | 95.8175 | 169.0325e-6 | 0.90 |
| `nuaa_msda_s302` | 4 | 71.3961 | 74.0381 | 93.9163 | 51.4506e-6 | 0.20 |
| `nuaa_msda_s302` | 6 | 72.9422 | 75.8090 | 96.5779 | 45.1394e-6 | 0.10 |
| `nuaa_msda_s303` | 2 | 65.0419 | 69.6944 | 91.6350 | 31.9680e-6 | 0.20 |
| `nuaa_msda_s303` | 4 | 70.9825 | 73.7103 | 94.2966 | 46.5114e-6 | 0.25 |
| `nuaa_msda_s304` | 2 | 70.7754 | 72.3528 | 96.1977 | 62.7012e-6 | 0.75 |
| `nuaa_msda_s304` | 4 | 70.6322 | 74.4896 | 96.1977 | 68.8753e-6 | 0.85 |

Best PAL-MSDA result:

```text
72.9422 mIoU
```

Current NUAA best remains:

```text
79.5487 valid
80.6827 oracle
```

Requested target:

```text
82.64
```

## Cleanup

Stopped all PAL-MSDA tmux sessions:

```text
pal_msda_s301
pal_msda_s302
pal_msda_s303
pal_msda_s304
```

Deleted four weak checkpoints:

```text
runs/pal_msda_4gpu/*/best.pth.tar
```

Retained only lightweight logs:

```text
runs/pal_msda_4gpu/*/train.log
reports/pal_msda_logs/*.log
```

## Interpretation

PAL-MSDA is a valid external backbone with useful frequency-attention mechanisms, and it ran at the requested GPU scale. However, as a standalone NUAA candidate it is much weaker than the existing teacher/cache family.

It should not be added to the current component pool as a dense source. Its wavelet/FrequencyAttention module may still be useful as a small architectural idea inside a stronger teacher-anchored refiner, but the trained PAL-MSDA weights from this run are not worth keeping.
