# NUAA component scorer attempt

Date: 2026-06-15

## Purpose

After component threshold/min-area search failed to improve beyond 82.2554 mIoU, this tried a component-level scorer. Instead of only deleting components by area, each connected component was described by local features and scored for whether it should be retained.

Features included:

- component area
- local-rule probability mean/max
- background-map mean
- local support mean
- strong-fusion mean
- weighted-raw mean
- official-Mamba fused mean

No sklearn dependency was available, so this used a pure numpy/grid scorer rather than installing new packages.

## Script

```text
scripts/search_nuaa_component_scorer.py
```

## Run

```bash
envs/irstd/bin/python scripts/search_nuaa_component_scorer.py \
  --output reports/nuaa_component_scorer_train_selected.json
```

Search:

- 7776 configs
- train-fold selected
- test evaluated with the exact existing metric

## Best Result

```text
threshold: 0.482
min_area: 2
w_mean: 0.6
w_max: 0.4
w_area: 0.02
w_support: 0.0
w_agree: 0.0
w_bg: 0.0
score_th: 0.2

mIoU: 82.2156
nIoU: 82.0106
Pd: 96.9582
Fa: 9.5355e-6
```

This is below:

```text
current NUAA best: 82.2554
component scorer:  82.2156
gap:              -0.0398
```

## Cleanup

- No model checkpoint was created.
- No prediction maps were saved.
- Kept only:
  - `scripts/search_nuaa_component_scorer.py`
  - `reports/nuaa_component_scorer_train_selected.json`
  - this report

## Conclusion

The component-level scorer did not improve the current local-rule map. The selected best effectively collapses back to a simple threshold/min-area rule, which confirms that the current dense map has limited room for post-hoc component selection. The remaining NUAA gain likely needs a genuinely new dense prediction source or a training/data pipeline change that alters pixel probabilities before components form.
