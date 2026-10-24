# UniVAD-Inspired Dense Normal Memory Attempt

Generated: 2026-06-14 UTC

## Purpose

This continued the external-domain transfer line from the pasted method list. The selected source was UniVAD, a CVPR 2025 training-free few-shot visual anomaly detection method. The goal was to test whether a normal-memory / patch-matching anomaly mechanism can provide a new dense candidate source for NUAA-SIRST, where the current teacher pool is still far from the +5 target.

## Source Check

Official code was cloned:

```text
external/UniVAD
```

The full UniVAD pipeline requires large foundation-model components:

- CLIP ViT-L/14-336
- DINOv2 ViT-g/14
- HQ-SAM
- GroundingDINO
- local submodules and pretrained checkpoints

Because this would pull a large stack of weights and dependencies before knowing whether the mechanism helps IRSTD, the full VFM pipeline was not launched.

## Lightweight Transfer Implemented

Implemented:

```text
scripts/eval_dense_normal_memory.py
```

This is a lightweight UniVAD-style dense normal-memory map:

1. sample GT-background pixels from the full multi-dataset train pool;
2. extract local patch/intensity/frequency features;
3. build a small normal memory;
4. score test pixels by nearest normal-memory distance;
5. evaluate the resulting dense anomaly map.

No model checkpoint is written.

## NUAA Tiny Smoke

Command shape:

```text
python scripts/eval_dense_normal_memory.py \
  --eval_datasets NUAA-SIRST \
  --max_fit_images_per_dataset 50 \
  --samples_per_image 64 \
  --memory_size 512 \
  --eval_max_samples 20 \
  --chunk_pixels 262144 \
  --thresholds 0.40,0.50,0.60,0.70,0.80,0.90,0.95 \
  --output reports/dense_normal_memory_nuaa_tiny.json
```

Result:

| Dataset | Subset | Best threshold | mIoU | Pd | Fa |
|---|---:|---:|---:|---:|---:|
| NUAA-SIRST | first 20 test images | 0.95 | 20.55 | 81.82 | 1733.80 |

## Runtime Note

A larger run with 4096 memory vectors and 40+40 eval samples was stopped after more than two minutes because brute-force nearest-memory matching was too CPU-heavy. It produced no JSON and no large artifacts.

## Interpretation

The dense normal-memory map is much weaker than the existing NUAA teacher/fusion region. It may still be useful as a minor auxiliary feature, but it is not a competitive dense candidate source by itself.

Decision: do not expand this lightweight normal-memory implementation. If the UniVAD line is revisited, it should either use real VFM features from the official pipeline or be fused only as a weak auxiliary cue, not treated as a standalone detector.
