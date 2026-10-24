# NUAA component-table search attempt

Date: 2026-06-15

## Purpose

The current NUAA-SIRST best is 82.2554 mIoU from the train-selected local rule plus component min-area filtering. Previous component searches repeatedly recomputed connected components for every candidate, which made broader searches slow.

This attempt added a component-statistics table search:

- precompute connected components once per image and threshold
- store each component area, probability mean, probability max, and GT intersection
- use those records to quickly rank threshold/min-area/min-mean/min-max configs on train folds
- evaluate only top train-selected candidates with the exact existing metric

## Script

```text
scripts/search_nuaa_component_table.py
```

## Run

```bash
envs/irstd/bin/python scripts/search_nuaa_component_table.py \
  --output reports/nuaa_component_table_train_selected.json
```

Search space:

- 4080 configs
- thresholds: 0.472 to 0.504, step 0.002
- min_area: 1, 2, 3, 4, 5, 6, 8, 10
- min_mean: 0.0, 0.36, 0.40, 0.44, 0.48, 0.52
- min_max: 0.0, 0.48, 0.52, 0.56, 0.60

## Best Train-Selected Test Result

```text
threshold: 0.482
min_area: 3
min_mean: 0.0
min_max: 0.0
mIoU: 82.2337
nIoU: 82.0244
Pd: 96.9582
Fa: 9.3983e-6
```

This is below the current best:

```text
current NUAA best: 82.2554
component-table best: 82.2337
gap: -0.0217
```

## Cleanup

- No prediction maps or checkpoints were created.
- Kept only:
  - `scripts/search_nuaa_component_table.py`
  - `reports/nuaa_component_table_train_selected.json`
  - this report

## Conclusion

The component-table infrastructure works and is much better suited for broader component sweeps, but a wider threshold/min-area/min-mean/min-max search on the current local-rule map did not beat 82.2554. The remaining NUAA gap probably needs a new dense source or a rule that changes which pixels enter components, not just stricter component deletion.
