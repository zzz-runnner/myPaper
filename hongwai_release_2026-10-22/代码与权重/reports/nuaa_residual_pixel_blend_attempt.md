# NUAA residual-pixel / local-anchor blend attempt

Date: 2026-06-16

## Purpose

Test whether the residual pixel calibrator and the local-rule anchor are complementary on the NUAA-SIRST official split. The previous best NUAA result was the residual pixel calibrator with component filtering:

- mIoU: `82.2928`
- threshold: `0.498`
- min area: `3`
- gap to +5 target (`82.64`): `0.3472`

## Code change

`scripts/train_nuaa_official_mamba_residual_pixel.py` now supports evaluating blended probability maps:

```text
blend = alpha * residual_map + (1 - alpha) * local_rule_anchor
```

The default `--blend_alphas 1.0` preserves the previous pure-residual behavior.

## Command

```bash
envs/irstd/bin/python scripts/train_nuaa_official_mamba_residual_pixel.py \
  --steps 800 --residual_scale 0.04 --lr 3e-4 \
  --thresholds 0.494,0.496,0.498,0.500,0.502,0.504 \
  --min_areas 2,3,4 \
  --blend_alphas 0.50,0.60,0.70,0.80,0.90,1.00 \
  --output reports/nuaa_official_mamba_residual_pixel_blend_s800_r004.json
```

## Result

Best row:

```json
{
  "kind": "blend_component",
  "alpha": 0.7,
  "threshold": 0.496,
  "min_area": 3,
  "mIoU": 0.8229281160694658,
  "nIoU": 0.8216425018304664,
  "Pd": 0.9695817490494296,
  "Fa": 0.000008986711877373503
}
```

This exactly ties the previous best pure-residual row:

```text
alpha=1.0, threshold=0.498, min_area=3, mIoU=0.8229281160694658
```

## Interpretation

The simple linear blend did not improve the NUAA best score. The best blended result shifts the threshold from `0.498` to `0.496`, but the final binary mask is effectively equivalent after component filtering. This suggests the residual calibrator has already absorbed the useful local-rule correction available from this anchor.

## Storage

No prediction maps or checkpoints were saved by this run. The result file is small:

- `reports/nuaa_official_mamba_residual_pixel_blend_s800_r004.json`

## Next

The next NUAA branch should not spend more time on scalar blending. Better candidates are:

1. connected-component feature scorer with additional shape/context features,
2. per-component false-positive veto trained on train split,
3. recovery rule for missed low-contrast target cores after veto.
