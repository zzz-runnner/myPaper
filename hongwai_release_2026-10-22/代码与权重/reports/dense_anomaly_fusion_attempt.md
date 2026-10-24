# Dense Anomaly Fusion Attempt

Generated: 2026-06-14 UTC

## Purpose

This was a cross-domain transfer attempt, treating infrared small target segmentation as dense background-anomaly detection rather than as another SCTransNet training tweak.

The candidate source combined:

- multi-scale local-z bright anomaly,
- white top-hat small target enhancement,
- DoG / Laplacian high-frequency cues,
- coarse background reconstruction residual,
- optional sparse top-K connected components.

The anomaly map was then fused with cached teacher maps using train-selected parameters.

## Code

```text
scripts/search_dense_anomaly_fusion.py
```

The script:

- reads existing prediction caches,
- computes dense handcrafted anomaly maps from original images,
- selects fusion parameters on train cache,
- evaluates the selected rows on test cache using the same `metric_from_maps` evaluator.

This line does not create model checkpoints.

## NUAA-SIRST smoke/full-cache result

Command used the NUAA train/test caches with a small parameter grid.

Output:

```text
reports/dense_anomaly_fusion_smoke_nuaa.json
```

Result:

| row | mIoU | Pd | Fa |
| --- | ---: | ---: | ---: |
| selected dense anomaly fusion | 78.93 | 96.20 | 30.60e-6 |
| best test row in train-top-k | 78.97 | 96.20 | 30.18e-6 |
| cached teacher baseline in same run | 79.11 | 96.58 | 17.70e-6 |
| current reliable NUAA best | 79.34 | 96.58 | about 15.9e-6 |

Interpretation:

- The dense anomaly map increased false alarms and lowered mIoU.
- It did not improve NUAA and is far from the +5 target.

## IRSTD-1K diagnostic

The first full-cache IRSTD diagnostic was stopped because the dense anomaly score path was too slow for the train/test cache size.

Then the script was optimized:

- anomaly score is computed once per sample,
- image long side is downsampled to 320 before scoring,
- repeated grid evaluation reuses score maps,
- `--max_train_samples` and `--max_test_samples` were added.

Even after this, a 120 train / 80 test subset diagnostic remained too slow for useful iteration and was stopped.

No IRSTD result JSON was produced from the stopped subset run.

## Cleanup

- No checkpoint files were created.
- The stopped IRSTD Python processes were killed.
- Only the script and small NUAA JSON/report evidence remain.

## Decision

Do not continue this CPU connected-component dense-anomaly fusion path in its current form.

The NUAA evidence is clearly negative, and the IRSTD implementation is too slow relative to its expected value. If the anomaly idea is revisited, it should be implemented as a GPU-batched learned candidate generator rather than CPU-heavy handcrafted component search.

Current bests remain unchanged:

| dataset | best mIoU |
| --- | ---: |
| IRSTD-1K | 72.8351 |
| NUAA-SIRST | 79.34 |
