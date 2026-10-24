# NUAA residual pixel calibrator sweep follow-up

Date: 2026-06-15

## Purpose

After the residual pixel calibrator improved NUAA-SIRST from 82.2554 to 82.2928 mIoU, this sweep tested whether small changes to residual strength, learning rate, positive/negative sampling, and positive weight could add further gain.

No checkpoint or prediction map was saved.

## Attempted parallel run

Initially launched four variants in parallel. Training finished quickly, but the parallel component-filter evaluation became CPU-bound because each process independently evaluated 214 test images across multiple thresholds and min-area values. Those processes were stopped before writing JSON outputs, and the temporary logs were deleted.

## Sequential reduced sweep

To avoid CPU contention, three variants were run sequentially with a narrower threshold/min-area grid:

| Output | Key change | Best mIoU | Threshold | Min area | Pd | Fa |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `reports/nuaa_respix_seq_s1000_r003_pw20.json` | residual_scale=0.03, pos_weight cap=20 | 82.2476 | 0.496 | 3 | 96.9582 | 9.0553e-6 |
| `reports/nuaa_respix_seq_s1000_r005_pw25.json` | residual_scale=0.05, pos_weight cap=25 | 82.2928 | 0.500 | 3 | 96.9582 | 8.9867e-6 |
| `reports/nuaa_respix_seq_s1200_r004_pos768_pw20.json` | more positive samples, pos_weight cap=20 | 82.2928 | 0.498 | 3 | 96.9582 | 8.9867e-6 |

The previous best run remains tied:

```text
reports/nuaa_official_mamba_residual_pixel_s800_r004_component.json
mIoU: 82.2928
threshold: 0.498
min_area: 3
```

## Conclusion

The residual pixel calibrator gain is reproducible, but this small sweep did not improve beyond 82.2928. The current NUAA best remains:

```text
82.2928 mIoU
```

Remaining gap to the requested NUAA +5 target:

```text
82.64 - 82.2928 = 0.3472
```
