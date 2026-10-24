# MSHNet Official Weights NUAA Evaluation

Generated: 2026-06-14 UTC

## Purpose

NUAA-SIRST remains the blocker. This turn tested official MSHNet/SLS weights from the CVPR 2024 MSHNet repository as another non-SCTransNet dense source.

## Assets Downloaded

Downloaded with `gdown` from the links in `external/MSHNet/README.md`:

```text
external/MSHNet/weights_download/irstd_new_weights.tar
external/MSHNet/weights_download/irstd_old_weights.tar
external/MSHNet/weights_download/nudt_weights.tar
```

Total retained size:

```text
47 MB
```

## Evaluation Setup

Used existing evaluator:

```text
scripts/eval_official_mshnet_pconv.py
```

Settings:

```text
dataset = NUAA-SIRST test
preprocess = official_mshnet
base_size = 256
warm_flags = false,true
threshold sweep = 0.05 ... 0.90
```

## Best NUAA Results

| Official weight | Train source | Best warm flag | Best threshold | mIoU | nIoU | Pd | Fa |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `irstd_new_weights.tar` | IRSTD-1K | false | 0.50 | 68.2428 | 71.5182 | 92.0152 | 40.9547e-6 |
| `irstd_old_weights.tar` | IRSTD-1K | false | 0.50 | 69.4252 | 72.4857 | 92.7757 | 13.6516e-6 |
| `nudt_weights.tar` | NUDT-SIRST | false | 0.25 | 70.0429 | 71.9624 | 95.4373 | 24.9021e-6 |

Result JSON:

```text
reports/mshnet_official_irstd_new_nuaa_eval.json
reports/mshnet_official_irstd_old_nuaa_eval.json
reports/mshnet_official_nudt_nuaa_eval.json
```

## Interpretation

The official MSHNet/SLS weights are valid and runnable, but they do not transfer strongly to NUAA-SIRST. The best result is:

```text
70.0429 mIoU
```

This is far below:

```text
current NUAA best valid region: about 79.55
requested NUAA target: 82.64
```

Therefore these official MSHNet weights are not useful enough to add to the NUAA candidate pool. They would mostly add weak false-positive/false-negative patterns and increase component-search cost.

## Cleanup

No training output, prediction archive, or large temporary file was created. The 47 MB official weights are retained as checked external assets; all result files are small JSON/log files.
