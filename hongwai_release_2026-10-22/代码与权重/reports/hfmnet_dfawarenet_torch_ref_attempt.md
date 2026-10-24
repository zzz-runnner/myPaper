# HFMNet / DFAwareNet Torch Selective-Scan Fallback Attempt

Generated: 2026-06-15 UTC

## Purpose

After DCCS_hf was unlocked with a pure PyTorch selective-scan fallback, this attempt applied the same idea to the stronger local Mamba-family candidates:

```text
external/HFMNet
external/DFAwareNet
```

Both have local NUAA-SIRST and IRSTD-1K weights, so they were worth testing as possible new dense prediction sources for the current fusion/refiner line.

## Code Added

Updated:

```text
scripts/eval_mamba_irstd_candidate_fullsplit.py
```

The evaluator now:

1. injects a fake `mamba_ssm.ops.selective_scan_interface` module backed by `selective_scan_torch_ref`;
2. loads HFMNet / DFAwareNet without installing Mamba;
3. for DFAwareNet, injects a lightweight Haar `pytorch_wavelets` fallback for `DWTForward` and `DWTInverse`;
4. evaluates full local splits using the unified mIoU/nIoU/Pd/Fa metric and threshold/component sweeps.

No probability map archives were saved.

## Smoke Tests

HFMNet NUAA smoke passed.

DFAwareNet initially failed because `pytorch_wavelets` was missing. A Haar fallback was added, and DFAwareNet NUAA smoke then passed.

Temporary smoke JSON files were deleted.

## Full-Split Results

| Model | Dataset | weight | best postprocess | mIoU | nIoU | Pd | Fa |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| HFMNet | NUAA-SIRST | `mIoU__HFMNet_NUAA-SIRST_epoch.pth.tar` | threshold 0.25 | 70.9571 | 63.3420 | 80.2281 | 11.1133e-6 |
| HFMNet | IRSTD-1K | `mIoU__HFMNet_IRSTD-1K_epoch.pth.tar` | threshold 0.70 + min_area 6 | 64.4264 | 53.7465 | 84.1751 | 15.8851e-6 |
| DFAwareNet | NUAA-SIRST | `mIoU__DFAwareNet_NUAA-SIRST_epoch.pth.tar` | threshold 0.40 + min_area 2 | 78.0555 | 75.7665 | 94.6768 | 10.6331e-6 |
| DFAwareNet | IRSTD-1K | `mIoU__DFAwareNet_IRSTD-1K_epoch.pth.tar` | threshold 0.35 | 67.6433 | 59.9099 | 88.8889 | 9.3185e-6 |

Outputs:

```text
reports/hfmnet_torch_ref_nuaa_eval.json
reports/hfmnet_torch_ref_irstd_eval.json
reports/dfawarenet_torch_ref_nuaa_eval.json
reports/dfawarenet_torch_ref_irstd_eval.json
```

## Interpretation

The fallback successfully unlocked both model families in the current environment, but the results are too weak to help the active target:

```text
Current NUAA best: 81.3192
Best new Mamba-family NUAA candidate: DFAwareNet 78.0555

Current IRSTD best: 73.0367
Best new Mamba-family IRSTD candidate: DFAwareNet 67.6433
```

So these maps are not worth saving or fusing into `s703+BGM+drop4`.

The likely reasons are one or more of:

1. official preprocessing/base-size protocol mismatch;
2. numerical differences between fused selective-scan and the torch fallback;
3. Haar fallback for DFAwareNet not exactly matching `pytorch_wavelets` boundary behavior.

Even with those caveats, the gap is large enough that these candidates are not immediate breakthrough sources.

## Cleanup

Deleted:

```text
reports/hfmnet_torch_ref_smoke.json
reports/dfawarenet_torch_ref_smoke.json
```

No large map archives or new checkpoints were created.

## Current Reliable Status

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.3192 | 82.64 | not passed |

## Official NUAA Split Sanity Check

A follow-up checked whether the low NUAA scores were caused by evaluating on the local 214-image target split while the official HFMNet/DFAwareNet weights use their own `80_20/test.txt` split.

Split comparison:

```text
Official NUAA test split: 86 images
Local target NUAA test split: 214 images
Overlap: 42 images
```

Added evaluator option:

```text
--index_file
```

Results on the official 86-image NUAA split:

| Model | split | mIoU | nIoU | Pd | Fa |
| --- | --- | ---: | ---: | ---: | ---: |
| HFMNet | official 86-image NUAA split | 65.0357 | 62.4521 | 84.4037 | 7.8501e-6 |
| DFAwareNet | official 86-image NUAA split | 72.3688 | 74.1363 | 93.5780 | 3.5837e-6 |

Outputs:

```text
reports/hfmnet_torch_ref_nuaa_officialsplit_eval.json
reports/dfawarenet_torch_ref_nuaa_officialsplit_eval.json
```

This confirms the low scores are not simply due to using the local 214-image target split. The likely causes remain fallback numerical differences and/or subtle official protocol details not reproduced here. Since the official-split scores are even lower than the local-split scores, these candidates are closed for the current goal unless a true Mamba/selective-scan + pytorch_wavelets environment is provided.
