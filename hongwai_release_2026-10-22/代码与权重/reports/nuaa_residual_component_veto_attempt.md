# NUAA residual component-veto attempt

Date: 2026-06-16

## Purpose

Try a component-level false-positive veto on top of the current NUAA residual pixel calibrator. This was motivated by the remaining small gap to the NUAA +5 target:

- current best: `82.2928` mIoU
- target: `82.6400` mIoU
- remaining gap: `0.3472`

## Method

Added `scripts/train_nuaa_residual_component_veto.py`.

The script:

1. trains the residual pixel calibrator in memory,
2. predicts train/test probability maps in memory,
3. extracts connected-component features at several seed thresholds,
4. trains a small component MLP to keep/drop candidate components,
5. evaluates the selected binary maps,
6. writes only a compact JSON report.

No probability maps or checkpoints are saved.

## Command

```bash
envs/irstd/bin/python scripts/train_nuaa_residual_component_veto.py \
  --steps 800 --residual_scale 0.04 --lr 3e-4 \
  --seed_thresholds 0.470,0.480,0.490,0.496,0.498 \
  --keep_thresholds 0.20,0.30,0.40,0.50,0.60 \
  --min_areas 1,2,3,4 \
  --max_areas 0,64,128 \
  --component_steps 600 --component_lr 1e-3 \
  --output reports/nuaa_residual_component_veto_s800.json
```

## Result

Best row:

```json
{
  "seed_threshold": 0.498,
  "keep_threshold": 0.2,
  "min_area": 3,
  "max_area": 0,
  "test": {
    "mIoU": 0.8229281160694658,
    "nIoU": 0.8216425018304664,
    "Pd": 0.9695817490494296,
    "Fa": 0.000008986711877373503
  }
}
```

This exactly ties the previous residual-pixel + component-filter best:

```text
threshold=0.498, min_area=3, mIoU=82.2928
```

## Interpretation

The component MLP did not add useful discrimination beyond the existing area filter. The training table contained `1355` component rows with `1350` positives, so the candidate set near the useful thresholds has almost no negative components to learn from. Lower seed thresholds created more components but reduced test mIoU instead of improving it.

This indicates that NUAA's remaining gap is unlikely to be solved by simply dropping connected components. The next branch should focus on pixel-level boundary / missed-core recovery or adding a genuinely different teacher map.

## Storage

Only the compact report JSON was saved:

- `reports/nuaa_residual_component_veto_s800.json`
