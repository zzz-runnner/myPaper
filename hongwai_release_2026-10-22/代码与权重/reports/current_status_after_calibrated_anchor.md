# Current Status After Calibrated Anchor

Generated: 2026-06-13 03:43 UTC

## New Code Change

Added a teacher-logit anchor term to `teacher_preservation_loss` in:

```text
scripts/train_hybrid_methods.py
```

New CLI argument:

```text
--teacher_logit_weight
```

Purpose:

- The earlier TCD/LCA/BNG composition branches loaded correctly and started near the best checkpoints.
- But normal training pulled the strong checkpoints down.
- The new logit anchor constrains the student/refiner to stay close to the strong teacher logits while learning only small target/clutter corrections.

## Branches Stopped This Round

| Branch | Reason | Evidence |
|---|---|---|
| `tcd_lca_bng_irstd_gpu0` | IRSTD started at `68.23`, dropped to `64.96` by epoch 2 | `reports/tcd_lca_bng_irstd_gpu0_drift_preserved/` |
| `tcd_lca_bng_nuaa_gpu1` | NUAA started at `79.11`, drifted down to `78.17` by epoch 4 | `reports/tcd_lca_bng_nuaa_gpu1_drift_preserved/` |
| `tcd_lca_bng_irstd_gpu2` | IRSTD started at `68.23`, dropped to `65.88` by epoch 2 | `reports/tcd_lca_bng_irstd_gpu2_drift_preserved/` |

The bulky run directories for those stopped branches were removed after preserving logs/checkpoints.

## Current Active Sessions

| GPU | Session | Dataset | Method | Current signal |
|---:|---|---|---|---|
| 0 | `cal_tcd_lca_bng_irstd_gpu0` | IRSTD-1K | calibrated TCD/LCA/BNG | epoch 0 `68.24`, epoch 1 `68.22` |
| 1 | `cal_tcd_lca_bng_nuaa_gpu1` | NUAA-SIRST | calibrated TCD/LCA/BNG + pseudo | epoch 0 `79.12`, epoch 1 `79.06` |
| 2 | `cal_tcd_lca_bng_irstd_gpu2` | IRSTD-1K | stricter calibrated TCD/LCA/BNG | epoch 0 `68.24`, first epoch pending |
| 3 | `pseudo_nuaa_gpu3` | NUAA-SIRST | PConvTCF + LCA/TCD pseudo labels | epoch 10 `73.83`, improving but below best |

## Interpretation

- The calibrated anchor stopped the severe drift seen in the first TCD/LCA/BNG runs.
- It has not yet beaten the current best:
  - IRSTD best remains `68.28`.
  - NUAA best remains `79.13`.
- The current strategy is now safer: keep the cross-domain/background-normality/local-anomaly modules, but restrict them to small corrections near a verified strong checkpoint.

## Next Decisions

- Keep GPU0/GPU1/GPU2 through a few short eval intervals.
- If calibrated branches stay flat, try an even more selective residual-gate or validation-threshold/ensemble calibration rather than stronger training.
- Keep GPU3 until epoch 15 because it improved from `68.74` to `73.83`; stop if it plateaus far below NUAA best.
