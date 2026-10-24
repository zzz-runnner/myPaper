# RealScene-ISTD and Text-IRSTD Source Screen

Generated: 2026-06-14 UTC

## Purpose

After EFLNet and the spectral-normality branch failed to improve NUAA-SIRST, this step screened newer sources from the user's pasted method list:

```text
Text-IRSTD
RealScene-ISTD
SAIST / recent VLM-style IRSTD candidates
```

The goal was to find a genuinely new candidate source or new labeled data, not another SCTransNet training-parameter adjustment.

## Text-IRSTD

Local repository:

```text
external/Text-IRSTD
```

Current contents:

```text
README.md only
```

The README says:

```text
近期事务较多，相关代码将尽快整理并发布。
```

Decision:

```text
Text-IRSTD cannot be evaluated locally yet.
No code, weights, dataset, or runnable inference path is available in the current repository.
```

## SAIST

Search did not reveal a directly usable official code/checkpoint repository for SAIST in this step. No local asset was added.

Decision:

```text
Keep as a literature direction, but it is not currently an executable candidate source.
```

## RealScene-ISTD

Cloned:

```text
external/RealScene-ISTD
```

Repository size after cleanup:

```text
804 KB
```

The repository contains:

```text
Dataset_Alignment/
detection_model/
README.md
```

It does not contain local dataset files or pretrained checkpoints.

The README provides Baidu Drive links for:

```text
RealScene-ISTD dataset
pre-prepared small target patches
RealScene-ISTD official weights
IRSTD-1K official weights
```

Reported paper numbers from the README:

| Test set | SCTransNet | RealScene-ISTD Ours |
| --- | ---: | ---: |
| RealScene-ISTD | 75.01 | 79.32 |
| IRSTD-1K | 66.93 | 72.44 |

The IRSTD-1K reported `72.44` is close to but below the current local retained best:

```text
current local IRSTD-1K best = 73.0367
```

No NUAA-SIRST result or NUAA official weight is provided in the README.

## Mechanism Notes

The code is built from SCTransNet-like components but adds:

```text
cross-view representation / channel alignment style network code
dataset alignment scripts
gamma correction
top-k patch fusion / seamless clone target-background alignment
thermal/noise consistency training in train.py
```

These are useful ideas, especially for cross-dataset generalization and synthetic hard-background augmentation, but without the Baidu Drive dataset/weights they do not provide an immediate strong candidate map for NUAA.

## Cleanup

Deleted:

```text
external/RealScene-ISTD/.git
```

Freed about:

```text
808 KB
```

Kept the small code directory because it is only about 804 KB and may be useful if the user uploads the Baidu Drive assets.

## Decision

Current status:

```text
Text-IRSTD: unavailable locally, README only.
SAIST: no executable official asset found in this pass.
RealScene-ISTD: code available, but dataset and weights are Baidu-only and absent locally.
```

No new model evaluation was possible from these sources in the current server state.

Useful next user-upload assets:

```text
RealScene-ISTD dataset
RealScene-ISTD official weights
IRSTD-1K official weights from RealScene-ISTD
Text-IRSTD code/weights if released later
SAIST code/weights if found
```

If the RealScene-ISTD dataset is uploaded under `/data1`, it can be converted directly into the local dataset format because its README already matches the local structure:

```text
images/
masks/
img_idx/train_RealScene-ISTD.txt
img_idx/test_RealScene-ISTD.txt
```
