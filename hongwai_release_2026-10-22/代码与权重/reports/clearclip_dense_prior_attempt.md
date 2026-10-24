# ClearCLIP Dense Prior Attempt

Generated: 2026-06-14 UTC

## Purpose

NUAA-SIRST remains below the requested +5 mIoU target, so this attempt tested an external-domain vision-language dense prior rather than another SCTransNet parameter tweak.

The idea was to use ClearCLIP/GEM patch-level image-text similarity as a candidate source for:

```text
target prompts - background/clutter prompts
```

## Implementation

Added:

```text
scripts/eval_clearclip_dense_prior.py
```

The script avoids the full ClearCLIP `mmseg/mmcv` segmentation stack and directly uses:

```text
external/ClearCLIP/gem/gem.py
external/ClearCLIP/gem/gem_wrapper.py
```

A compatibility issue was found in the local ClearCLIP wrapper:

```text
GEMWrapper.forward passes an extra ignore_residual argument, but modified_vit_forward only accepts image.
```

The evaluator therefore computes dense similarity directly from GEM patch features and text embeddings.

## Smoke Test

Command:

```text
/home/cx/hongwai/irstd-lab/envs/irstd/bin/python scripts/eval_clearclip_dense_prior.py \
  --dataset_name NUAA-SIRST \
  --max_samples 20 \
  --device cuda:0 \
  --output reports/clearclip_nuaa_smoke20.json
```

Model:

```text
ViT-B/16 OpenAI
image size 224
GEM depth 7
```

Prompts:

```text
target:
- infrared small target
- tiny thermal target
- small bright infrared object
- dim point target

background:
- infrared background
- cloud clutter
- sea clutter
- sensor noise
- building edge
```

Best NUAA-SIRST 20-image smoke result:

| threshold | mIoU | nIoU | Pd | Fa |
| ---: | ---: | ---: | ---: | ---: |
| 0.95 | 1.0234 | 1.2923 | 15.1515 | 16557.7873e-6 |

Result JSON:

```text
reports/clearclip_nuaa_smoke20.json
```

## Interpretation

This route has essentially no useful segmentation signal for NUAA-SIRST small infrared targets under the current prompt setup. The false-alarm rate is extremely high and target recall is poor.

It is not worth running full NUAA evaluation or adding this source to the component-fusion pool.

## Cleanup

The first run downloaded a 335 MB ViT-B/16 cache under:

```text
external/ClearCLIP/weights_cache
```

Because the smoke result was weak, that cache was deleted. The retained ClearCLIP repository is about 20 MB, and the retained result/report files are small.

## Decision

Do not continue ClearCLIP dense text prior as a main line. The NUAA blocker still needs a stronger infrared-specific candidate source or additional NUAA-like labeled training data.
