# NUAA residual MLP pixel-calibrator attempt

Date: 2026-06-16

## Purpose

Test whether a nonlinear pixel-level calibrator can improve the current NUAA best beyond the linear residual pixel calibrator.

Current target state before this branch:

- best NUAA: `82.2928` mIoU
- target: `82.6400` mIoU
- remaining gap: `0.3472`

## Method

Added `scripts/train_nuaa_official_mamba_residual_mlp_pixel.py`.

Compared with the earlier linear residual calibrator, this version uses a small MLP over multi-source pixel features:

- local-rule anchor
- official-Mamba fused map
- strong fusion / weighted raw
- DFAware official-Mamba map
- HFM official-Mamba map
- BGM response
- local means and disagreement / contrast features

The model is still residual around the local-rule anchor:

```text
logit(out) = logit(local_anchor) + residual_scale * tanh(MLP(features))
```

No maps or checkpoints are saved; only compact JSON reports are kept.

## Runs

### Run A

```bash
envs/irstd/bin/python scripts/train_nuaa_official_mamba_residual_mlp_pixel.py \
  --steps 1200 --residual_scale 0.05 --lr 5e-4 --hidden 32 \
  --pos_per_image 768 --neg_per_image 3072 --max_pos_weight 25 \
  --thresholds 0.490,0.492,0.494,0.496,0.498,0.500,0.502,0.504,0.506 \
  --min_areas 1,2,3,4 \
  --seed 20260616 \
  --output reports/nuaa_official_mamba_residual_mlp_pixel_s1200_r005_h32.json
```

Best test row:

```json
{
  "kind": "component",
  "threshold": 0.498,
  "min_area": 3,
  "mIoU": 0.8228126029201889,
  "nIoU": 0.8203318318285914,
  "Pd": 0.9695817490494296,
  "Fa": 0.000009329716147502263
}
```

### Run B

```bash
envs/irstd/bin/python scripts/train_nuaa_official_mamba_residual_mlp_pixel.py \
  --steps 1000 --residual_scale 0.03 --lr 3e-4 --hidden 24 \
  --pos_per_image 768 --neg_per_image 3072 --max_pos_weight 20 \
  --thresholds 0.494,0.496,0.498,0.500,0.502,0.504 \
  --min_areas 1,2,3,4 \
  --seed 20260616 \
  --output reports/nuaa_official_mamba_residual_mlp_pixel_s1000_r003_h24.json
```

Best test row:

```json
{
  "kind": "component",
  "threshold": 0.494,
  "min_area": 3,
  "mIoU": 0.8227028213854429,
  "nIoU": 0.8202678195382316,
  "Pd": 0.9695817490494296,
  "Fa": 0.000009329716147502263
}
```

## Result

The nonlinear MLP did not beat the current linear residual best:

```text
current best: 82.2928
Run A:        82.2813
Run B:        82.2703
```

## Interpretation

The MLP can fit train slightly but does not improve the NUAA test split. It appears to perturb the already-strong local-rule residual calibration rather than add a robust new correction. The next useful branch should introduce genuinely new dense evidence or train a stronger full-image model/distillation path, not keep increasing the pixel calibrator capacity.

## Storage

Kept compact result files only:

- `reports/nuaa_official_mamba_residual_mlp_pixel_s1200_r005_h32.json`
- `reports/nuaa_official_mamba_residual_mlp_pixel_s1000_r003_h24.json`
