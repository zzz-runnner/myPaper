# AnomalyCLIP External Candidate Attempt

Generated: 2026-06-14 UTC

## Purpose

NUAA-SIRST is still the blocker. Existing local candidates have an oracle ceiling around 80.49 mIoU, below the requested 82.64 target. This attempt tested an external-domain foundation/anomaly source from the pasted material:

```text
AnomalyCLIP, ICLR 2024, object-agnostic normal/abnormal prompt learning
```

The intended transfer idea was to treat infrared small targets as local abnormal thermal responses against a normal background.

## Setup

Cloned:

```text
external/AnomalyCLIP
```

Compatibility fixes:

- installed small missing dependency `ftfy`;
- patched AnomalyCLIP CLIP cache path from the author's `/remote-home/...` to local `~/.cache/clip`;
- added local evaluator:

```text
scripts/eval_anomalyclip_irstd.py
```

Downloaded by the model loader:

```text
~/.cache/clip/ViT-L-14-336px.pt
```

Size:

```text
891M
```

## Smoke Result

Dataset:

```text
NUAA-SIRST test, first 20 images
```

Output:

```text
reports/anomalyclip_nuaa_smoke20.json
```

Best threshold result:

| Threshold | mIoU | nIoU | Pd | Fa |
| ---: | ---: | ---: | ---: | ---: |
| 0.95 | 4.33 | 3.94 | 54.55 | 9302.67e-6 |

## Interpretation

AnomalyCLIP is runnable in this environment, but its zero-shot anomaly maps are not aligned with infrared small targets. It fires broadly on background texture and produces extremely high false alarms. The 20-image smoke is far below the current NUAA best region:

```text
current reliable NUAA best: about 79.34 mIoU
target: 82.64 mIoU
AnomalyCLIP smoke: 4.33 mIoU
```

Decision:

- Do not run full NUAA/IRSTD evaluation for this exact AnomalyCLIP zero-shot dense source.
- Do not add it into the candidate fusion pool unless a much stronger IR-specific prompt adaptation is implemented.
- Keep the evaluator script because it may be useful if a better anomaly/VLM checkpoint becomes available.

## Cleanup

The cloned repository initially included 30 prompt-learner epoch files, about 648M. All redundant epoch files were deleted, leaving only `epoch_15.pth` under the two provided checkpoint folders.

Current retained size:

```text
external/AnomalyCLIP: 94M
~/.cache/clip: 891M
```

The CLIP cache is retained because ClearCLIP/CAT-Seg-style dense text prior experiments can reuse it.
