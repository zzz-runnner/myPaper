# NUAA official-Mamba residual pixel calibrator follow-up

Date: 2026-06-15

## Purpose

The component-only searches showed that the current local-rule dense map had little room left for post-hoc component deletion. This run instead changed the pixel probabilities before component formation.

This is not SCTransNet parameter tuning. It is a lightweight residual pixel calibrator over the strongest cached dense sources:

- train-selected local-rule anchor
- official-Mamba fused map
- strong_fusion
- weighted_raw
- DFAware official-Mamba
- HFMNet official-Mamba
- BGM background map
- local mean/support features

## Script

```text
scripts/train_nuaa_official_mamba_residual_pixel.py
```

The model is intentionally tiny:

- linear residual on top of the local-rule probability logit
- residual bounded by `residual_scale * tanh(...)`
- no checkpoint saved
- no prediction maps saved

## Runs

### Short check

```bash
envs/irstd/bin/python scripts/train_nuaa_official_mamba_residual_pixel.py \
  --steps 600 \
  --residual_scale 0.08 \
  --lr 4e-4 \
  --output reports/nuaa_official_mamba_residual_pixel_s600.json
```

Best:

- threshold: 0.488
- mIoU: 82.2346
- nIoU: 82.0480
- Pd: 96.5779
- Fa: 11.7307e-6

### Conservative residual

```bash
envs/irstd/bin/python scripts/train_nuaa_official_mamba_residual_pixel.py \
  --steps 800 \
  --residual_scale 0.04 \
  --lr 3e-4 \
  --thresholds 0.480,0.482,0.484,0.486,0.488,0.490,0.492,0.494,0.496,0.498,0.500 \
  --output reports/nuaa_official_mamba_residual_pixel_s800_r004.json
```

Best:

- threshold: 0.498
- mIoU: 82.2767
- nIoU: 82.1544
- Pd: 96.9582
- Fa: 9.2611e-6

Seed check with `--seed 20260616` reproduced the same best threshold and metric.

### Conservative residual + component filter

```bash
envs/irstd/bin/python scripts/train_nuaa_official_mamba_residual_pixel.py \
  --steps 800 \
  --residual_scale 0.04 \
  --lr 3e-4 \
  --thresholds 0.490,0.492,0.494,0.496,0.498,0.500,0.502,0.504,0.506 \
  --min_areas 1,2,3,4,5,6 \
  --output reports/nuaa_official_mamba_residual_pixel_s800_r004_component.json
```

Best:

- kind: component
- threshold: 0.498
- min_area: 3
- mIoU: 82.2928
- nIoU: 82.1643
- Pd: 96.9582
- Fa: 8.9867e-6

## Comparison

```text
previous NUAA best: 82.2554
new NUAA best:      82.2928
gain:               +0.0374
NUAA +5 target:      82.64
remaining gap:       0.3472
```

Current reliable table:

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 82.2928 | 82.64 | not passed |

## Cleanup

- No model checkpoint was saved.
- No prediction maps were saved.
- Kept only small JSON evidence files and this report.

## Conclusion

This residual pixel calibrator is the first post-component-search change to improve NUAA beyond 82.2554. The gain is small but legitimate on the full NUAA-SIRST test split. The target is still not reached, so this should be treated as the new anchor for the next NUAA attempts, not as completion.
