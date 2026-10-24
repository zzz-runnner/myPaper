# Current Status After ClearCLIP and Source Screening

Generated: 2026-06-14 UTC

## Machine State

At the latest check:

```text
active train/eval process: none
GPUs: 4 x RTX 2080 Ti idle
/ free: about 58 GB
/data1 free: about 3.4 TB
```

## Current Target Status

| Dataset | SCTransNet baseline | +5 target | Current best | Status |
| --- | ---: | ---: | ---: | --- |
| IRSTD-1K | 68.03 | 73.03 | 73.0367 | passed narrowly |
| NUAA-SIRST | 77.64 | 82.64 | 79.5487 valid / 80.6827 oracle | not passed |

NUAA-SIRST remains the blocker.

## New Work Completed In This Step

### ClearCLIP dense text prior

Added:

```text
scripts/eval_clearclip_dense_prior.py
reports/clearclip_dense_prior_attempt.md
reports/clearclip_nuaa_smoke20.json
```

Purpose:

```text
external-domain VLM dense prior:
tiny thermal target / infrared small target
minus
background / cloud clutter / sea clutter / sensor noise
```

Smoke result on 20 NUAA-SIRST test images:

| threshold | mIoU | nIoU | Pd | Fa |
| ---: | ---: | ---: | ---: | ---: |
| 0.95 | 1.0234 | 1.2923 | 15.1515 | 16557.7873e-6 |

Decision:

```text
Do not run full evaluation. Do not add ClearCLIP to the candidate pool.
```

Cleanup:

```text
Deleted external/ClearCLIP/weights_cache, about 335 MB.
Kept external/ClearCLIP itself, about 20 MB.
```

### Text-IRSTD repository check

Current local repository state:

```text
external/Text-IRSTD
```

Finding:

```text
Only README is currently available.
No runnable code or checkpoint is present locally.
```

Decision:

```text
Cannot evaluate Text-IRSTD until the authors release code/checkpoints or the assets are provided.
```

### MiM-ISTD repository check

Current local repository:

```text
external/MiM-ISTD
```

Finding:

```text
mim_network.py exists.
No local checkpoint exists.
Current env lacks causal_conv1d and mamba_ssm.
README checkpoint link is Baidu only.
```

Decision:

```text
Potentially useful, but not a quick candidate source without official weight + Mamba dependencies.
```

### IRSTD-Diff repository check

Current local repository:

```text
external/IRSTD-Diff
```

Finding:

```text
Training/sampling code exists.
No pretrained checkpoint exists locally.
From-scratch diffusion training would be costly and not a quick NUAA candidate source.
```

Decision:

```text
Do not start long diffusion training unless explicitly choosing diffusion as a main research line.
```

## Dataset State

Prepared datasets:

| Dataset | Images | Masks |
| --- | ---: | ---: |
| IRSTD-1K | 1001 | 1001 |
| NUAA-SIRST | 427 | 427 |
| NUDT-SIRST | 1327 | 1327 |
| SIRST-UAVB | 3000 | 3000 |
| SIRST3 | 2755 | 2755 |

No additional IRSTD/SIRST-like dataset was found under `/data1` at this check.

## Honest Interpretation

The current local candidate pool is still oracle-limited on NUAA. ClearCLIP did not add a useful external-domain signal. Text-IRSTD has no released code locally. MiM-ISTD and IRSTD-Diff are not immediately evaluable without extra assets or long setup/training.

The next meaningful move should be one of:

1. Add a new NUAA-like segmentation dataset under `/data1`, then build a converter and re-run multi-dataset training/fusion.
2. Provide official checkpoints for a strong method that is not already tested, especially SAIST, Text-IRSTD, MiM-ISTD, S3D/PAL, or a stronger IRSAM/HQ-SAM variant.
3. If no new assets are available, shift from quick candidate screening to a deliberate full 4-GPU training line based on one mechanism, such as background memory + target-clutter high-frequency decoupling, knowing this is a longer experiment rather than a quick checkpoint evaluation.
