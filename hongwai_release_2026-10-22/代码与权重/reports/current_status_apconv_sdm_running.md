# Current Status: APConv-SDM Running

Generated: 2026-06-12 22:35 UTC

## Stopped weak line

Stopped tmux session:

```text
paired_bgm_4gpu
```

Removed weak run directory:

```text
/home/cx/hongwai/irstd-lab/runs/paired_bgm_4gpu
```

Preserved reports:

```text
/home/cx/hongwai/irstd-lab/reports/paired_bgm_4gpu_results.md
/home/cx/hongwai/irstd-lab/reports/paired_bgm_4gpu_stopped_note.md
```

Reason: the official-style paired-BGM run occupied all four GPUs, but the only early evaluation was `NUAA-SIRST / BGM-PairedBasicUNet` at epoch 10 with mIoU `71.86`, which is `-5.64` below the SCTransNet README target `77.50`. Earlier direct BGM/TCF evidence also did not beat SCTransNet, so this line was stopped early to avoid wasting four-GPU time.

## Active line

Active tmux session:

```text
apconv_sdm_4gpu
```

Run root:

```text
/home/cx/hongwai/irstd-lab/runs/apconv_sdm_4gpu
```

This is the next innovation line:

- official PConv asymmetric-padding convolution from the PConv + SD Loss repo.
- mask-format SDM IoU loss.
- TCF/BGM-style background suppression.
- multi-prototype background normality memory, inspired by anomaly-detection normal-memory methods.
- full multi-dataset training with `IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3`.

## Running jobs

| GPU | Dataset | Method | Model | Batch | Train datasets |
|---:|---|---|---|---:|---|
| 0 | IRSTD-1K | APConv-TCF-SDM | `apconv_tcf` | 7 | IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3 |
| 1 | IRSTD-1K | APConv-TCF-PairedSDM | `apconv_tcf` | 5 | IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3 |
| 2 | NUAA-SIRST | APConv-TCF-SDM | `apconv_tcf` | 7 | IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3 |
| 3 | NUAA-SIRST | APConv-TCF-PairedSDM | `apconv_tcf` | 5 | IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3 |

Observed at launch:

| GPU | Memory | Utilization |
|---:|---:|---:|
| 0 | ~5.6G / 11G | 97% |
| 1 | ~7.8G / 11G | 98% |
| 2 | ~5.6G / 11G | 97% |
| 3 | ~7.8G / 11G | 98% |

No OOM, traceback, or runtime error was observed during the launch checks.
