# HFM-LK FP-TopK Suppression NUAA 4GPU Attempt

Generated: 2026-06-15 UTC

## Purpose

Tested a new training-objective variant starting from the retained `s703` HFM-LK checkpoint rather than training from scratch. The idea was to explicitly suppress the highest-scoring background pixels while preserving target peaks, aiming to fix the false-positive component errors behind the current NUAA plateau.

This is not SCTransNet parameter tuning. It adds a cross-domain/anomaly-style hard-negative objective to the HFM/LK + dynamic high-frequency residual refiner.

## Code Changes

Modified:

```text
scripts/train_hfm_lk_anchor_refiner_from_cache.py
```

Added parameters:

```text
--fp_topk_weight
--fp_topk_ratio
--fp_topk_min
--pos_peak_weight
--pos_peak_margin
```

Added launcher:

```text
scripts/launch_hfmlk_fp_suppress_nuaa_4gpu_tmux.sh
```

`fp_topk_loss` penalizes the top-scoring background logits per patch. `pos_peak_loss` protects at least one strong positive target logit per target-containing patch so FP suppression does not collapse small targets.

## 4GPU Full Multi-Dataset Setup

Training data:

```text
IRSTD-1K train
NUAA-SIRST train
NUDT-SIRST train
SIRST3 train
```

Evaluation:

```text
NUAA-SIRST full test
```

Initialization:

```text
runs/strong_fusion_hfmlk_nuaa_4gpu/nuaa_strong_fusion_s703/best.pth.tar
```

Branches:

```text
nuaa_fp_strong_s941
nuaa_fp_strong_s942
nuaa_fp_rank_s943
nuaa_fp_focus_s944
```

Observed GPU utilization after cache load:

```text
GPU0: 9098 MiB, 97%
GPU1: 9098 MiB, 97%
GPU2: 9102-9104 MiB, 96%
GPU3: 9098 MiB, 96-97%
```

This met the requested four-card / 9GB+ utilization target.

## Epoch 1 Results

Stopped after epoch 1 because all branches were below the retained NUAA best `81.3192`.

| run | epoch | threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nuaa_fp_strong_s941` | 1 | 0.468 | 81.2189 | 81.4406 | 98.0989 | 8.6437e-6 |
| `nuaa_fp_strong_s942` | 1 | 0.468 | 81.1804 | 81.4507 | 98.0989 | 8.5065e-6 |
| `nuaa_fp_rank_s943` | 1 | 0.468 | 81.1184 | 81.3719 | 98.0989 | 9.3297e-6 |
| `nuaa_fp_focus_s944` | 1 | 0.468 | 81.1549 | 81.3864 | 98.0989 | 9.1239e-6 |

Best branch:

```text
nuaa_fp_strong_s941 epoch 1 mIoU = 81.2189
```

Comparison:

```text
current retained NUAA best: 81.3192
best FP-suppress result:   81.2189
delta:                    -0.1003
```

## Cleanup

Stopped tmux sessions:

```text
hfmlk_fp_nuaa_a
hfmlk_fp_nuaa_b
hfmlk_fp_nuaa_c
hfmlk_fp_nuaa_d
```

Deleted failed checkpoints:

```text
runs/hfmlk_fp_suppress_nuaa_4gpu/*/best.pth.tar
```

Retained only lightweight logs and this report.

## Conclusion

The explicit FP-topK suppression objective reduced false alarms slightly in some branches but hurt mIoU relative to the current `s703+BGM+drop4` recipe. It did not solve the NUAA +5 target.

Current reliable status remains:

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.3192 | 82.64 | not passed |
