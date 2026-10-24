# BGM Official Checkpoint Evaluation

Generated: 2026-06-14 UTC

## Purpose

NUAA-SIRST is still below the requested +5 mIoU target. Since the BGM repository provides official trained checkpoints, this attempt tested whether those checkpoints give a stronger non-SCTransNet dense source than our previous from-scratch BGM runs.

## Assets

Downloaded official BGM checkpoints from the repository Google Drive link:

```text
external/BGM/checkpoints_download/weights/BasicUNet_plus-2024-09-20-19-40-03-IRSTD-1K/best_mIoU_on_IRSTD-1K.pth.tar
external/BGM/checkpoints_download/weights/BasicUNet_plus-2024-09-20-21-17-03-NUAA-SIRST/best_mIoU_on_NUAA-SIRST.pth.tar
external/BGM/checkpoints_download/weights/BasicUNet_plus-2024-09-20-21-47-11-NUDT-SIRST/best_mIoU_on_NUDT-SIRST.pth.tar
```

Each checkpoint is about 16.5 MB.

Added local evaluator:

```text
scripts/eval_bgm_official_local.py
```

The evaluator uses the local prepared test splits and the same `metric_from_maps` path used by the previous external-candidate evaluations.

## Official Log Reference

The downloaded BGM logs report:

| Official checkpoint | Official mIoU |
| --- | ---: |
| IRSTD-1K | 70.7856 |
| NUAA-SIRST | 78.9491 |
| NUDT-SIRST | 94.0884 |

## Full Local Test Results

| Checkpoint | Eval dataset | Best threshold | mIoU | nIoU | Pd | Fa |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| BGM-IRSTD | IRSTD-1K | 0.50 | 70.8405 | 66.1087 | 93.6027 | 16.7201e-6 |
| BGM-IRSTD | NUAA-SIRST | 0.95 | 73.9907 | 74.2190 | 92.3954 | 26.9601e-6 |
| BGM-NUAA | IRSTD-1K | 0.10 | 48.2648 | 59.3728 | 87.2054 | 35.4900e-6 |
| BGM-NUAA | NUAA-SIRST | 0.25 | 79.0733 | 80.0089 | 96.5779 | 20.1687e-6 |
| BGM-NUDT | IRSTD-1K | 0.05 | 32.6928 | 48.0059 | 76.0943 | 39.8550e-6 |
| BGM-NUDT | NUAA-SIRST | 0.05 | 58.6634 | 65.2319 | 90.8745 | 49.1868e-6 |

Result JSON:

```text
reports/bgm_official_full_test.json
```

## Interpretation

The official BGM checkpoints reproduce the repository-reported region under the local metric path. The NUAA checkpoint is stronger than our previous from-scratch BGM run, but still below the current best:

| Dataset | Current reliable best | Best official BGM |
| --- | ---: | ---: |
| IRSTD-1K | 73.0367 | 70.8405 |
| NUAA-SIRST | 79.34 | 79.0733 |

So official BGM is not a standalone breakthrough source. It may still be useful as an additional independent candidate map in a later component-pool screen, but it should not become the main long-run direction.

## Cleanup

No training output or prediction archive was created. Persistent additions are small:

```text
external/BGM/checkpoints_download      about 48 MB checkpoints + logs
reports/bgm_official_full_test.json
reports/bgm_official_smoke5.json
reports/bgm_gdrive_download.log
scripts/eval_bgm_official_local.py
```

The official checkpoints are retained because they are small and directly reusable.
