# New Candidate Source Screen

Generated: 2026-06-14 UTC

## Purpose

This screen focused on methods from the pasted lists that could provide a genuinely new dense candidate source, especially for NUAA-SIRST where current fusion is still far from the requested +5 mIoU target.

## Checked Sources

| Source | Status | Local path | Decision |
|---|---|---|---|
| Text-IRSTD | official repository cloned, but only README is present | `external/Text-IRSTD` | not runnable yet |
| SAIST | paper found, no reliable official runnable repo found in this check | none | cannot run now |
| UniVAD | official repo cloned; full pipeline requires CLIP-L, DINOv2-g, HQ-SAM, GroundingDINO | `external/UniVAD` | full stack too heavy for blind launch; lightweight normal-memory smoke was weak |
| DWTFreqNet | official repo cloned; wrapper implemented and smoke tested | `external/DWTFreqNet` | runnable but limited probe was far below current best |
| IRSAM | official repo cloned; MobileSAM checkpoint verified | `external/IRSAM` | runnable with MobileSAM baseline, but smoke far below current best |
| MiM-ISTD | official repo cloned, but only network file/checkpoint dependency state | `external/MiM-ISTD` | not immediately runnable |

## Key Findings

Text-IRSTD repository currently says:

```text
近期事务较多，相关代码将尽快整理并发布。
```

So it cannot be used as an experiment source yet.

SAIST remains promising conceptually, but without an official runnable repo/weights in this check, it is not a practical next experiment.

The most practical external-domain source was UniVAD, but the full VFM pipeline is heavy. A lightweight dense normal-memory transfer was implemented and tested; it achieved only 20.55 mIoU on a 20-image NUAA smoke subset, so it is not worth expanding in that lightweight form.

The most practical new same-domain source was DWTFreqNet. It was runnable, but a 400-sample multi-dataset probe only reached 21.65 IRSTD and 42.43 NUAA mIoU, far below current best.

## Current Best Still Stands

| Dataset | Current reliable best | mIoU |
|---|---|---:|
| IRSTD-1K | dual-PConv average fusion | 72.43 |
| NUAA-SIRST | background-normality component fusion | 79.34 |

## Next Practical Need

For a real jump on NUAA-SIRST, the workspace likely needs one of:

- official runnable SAIST/Text-IRSTD code and weights when released;
- official DWTFreqNet/IRSAM/MiM checkpoints obtained from their linked cloud storage;
- additional datasets under `/data1` that change the teacher/student distribution;
- a new trainable dense model that transplants only the useful module, such as wavelet-frequency preservation or VFM prompt distillation, into a proven local backbone rather than training a weak external model from scratch.
