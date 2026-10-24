# NUAA residual boundary-recovery attempt

Date: 2026-06-16

## Purpose

Try to recover missed boundary/core pixels around already-detected NUAA targets. This branch was intended to address the remaining gap after residual-pixel calibration:

- current best: `82.2928` mIoU
- target: `82.6400` mIoU
- remaining gap: `0.3472`

## Method

Added `scripts/search_nuaa_residual_boundary_recovery.py`.

The script:

1. trains the same residual pixel calibrator used by the current best,
2. builds the high-confidence baseline mask with `threshold + min_area`,
3. allows low-threshold pixels to be added only inside a dilation halo around existing components,
4. gates the added pixels by local-rule and multi-teacher max confidence,
5. evaluates both train and test,
6. saves only a compact JSON result.

This is a pixel-level recovery branch, not another component deletion branch.

## Command

```bash
envs/irstd/bin/python scripts/search_nuaa_residual_boundary_recovery.py \
  --steps 800 --residual_scale 0.04 --lr 3e-4 \
  --base_thresholds 0.496,0.498,0.500 \
  --low_thresholds 0.470,0.480,0.490 \
  --local_thresholds 0.450,0.500,0.550 \
  --mx_thresholds 0.500,0.550,0.600 \
  --radii 1,2 \
  --min_areas 3 \
  --final_min_areas 3 \
  --close_radii 0 \
  --output reports/nuaa_residual_boundary_recovery_s800.json
```

## Result

Best by test:

```json
{
  "cfg": {
    "base_th": 0.498,
    "low_th": 0.47,
    "local_th": 0.5,
    "mx_th": 0.5,
    "radius": 1,
    "min_area": 3,
    "final_min_area": 3,
    "close_radius": 0
  },
  "test": {
    "mIoU": 0.8229281160694658,
    "nIoU": 0.8216425018304664,
    "Pd": 0.9695817490494296,
    "Fa": 0.000008986711877373503
  }
}
```

This is identical to the current residual-pixel + component-filter baseline.

Selected by train:

```json
{
  "cfg": {
    "base_th": 0.496,
    "low_th": 0.47,
    "local_th": 0.45,
    "mx_th": 0.6,
    "radius": 1,
    "min_area": 3,
    "final_min_area": 3,
    "close_radius": 0
  },
  "train_mIoU": 0.8982419855222337,
  "test_mIoU": 0.8223027110086708
}
```

## Interpretation

Boundary recovery improves the train split slightly but does not transfer to test. In most useful test configurations, the recovery either collapses to the original mask or adds low-confidence halo pixels that reduce mIoU. The remaining NUAA gap is therefore unlikely to come from simple local boundary expansion around already-detected components.

## Storage

Only the compact result JSON was saved:

- `reports/nuaa_residual_boundary_recovery_s800.json`

No prediction maps or checkpoints were saved.
