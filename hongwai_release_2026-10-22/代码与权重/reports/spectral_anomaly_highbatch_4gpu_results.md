# Spectral-Anomaly High-Batch 4GPU Results

Generated: 2026-06-14 UTC

## Purpose

This run tested a non-plain-SCTransNet candidate line matching the user's requested direction:

```text
SET-style spectral enhancement
+ UniVAD/anomaly-style background normality
+ DHiF-style dynamic high-frequency gating
+ full multi-dataset training
```

The model is implemented as:

```text
sct_spectral_anomaly_refiner
```

in:

```text
scripts/train_hybrid_methods.py
```

The high-batch launcher was:

```text
scripts/launch_spectral_anomaly_highbatch_4gpu_tmux.sh
```

This launcher had not produced a prior completed result directory, so it was run as the next external-mechanism candidate after EFLNet was ruled out.

## Training Setup

All four branches used full multi-dataset training:

```text
IRSTD-1K
NUAA-SIRST
NUDT-SIRST
SIRST3
```

Common settings:

```text
batch_size = 28
epochs planned = 80
eval_before_train = true
eval every epoch = true
```

The goal of this run was twofold:

1. test whether the spectral-normality branch creates a new candidate signal;
2. check whether the high-batch setup moves closer to the requested 9GB+ per GPU usage.

## Runs

| GPU | run | eval dataset | teacher |
| ---: | --- | --- | --- |
| 0 | `SpectralAnomalyHB-IRSTD-A` | IRSTD-1K | TPC-DHiF-B |
| 1 | `SpectralAnomalyHB-IRSTD-B` | IRSTD-1K | TPC-DHiF-A |
| 2 | `SpectralAnomalyHB-NUAA-LCA` | NUAA-SIRST | LCA |
| 3 | `SpectralAnomalyHB-NUAA-TCD` | NUAA-SIRST | TCD-Decouple |

## GPU Use

Observed during training:

```text
GPU0: about 7.1 GB, 98-99% utilization
GPU1: about 7.1 GB, 98-99% utilization
GPU2: about 7.1 GB, 98-99% utilization
GPU3: about 7.2 GB, 98-99% utilization
```

So the four GPUs were fully utilized computationally, but this configuration did not reach the preferred 9GB+ memory target.

## Early Results

The run was stopped early because the metrics stayed near the old 68/79 region and far below the retained bests.

| run | epoch | mIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: |
| `SpectralAnomalyHB-IRSTD-A` | 0 | 68.2654 | 93.6027 | 12.6018e-6 |
| `SpectralAnomalyHB-IRSTD-A` | 1 | 68.2512 | 93.6027 | 12.6587e-6 |
| `SpectralAnomalyHB-IRSTD-B` | 0 | 68.1634 | 93.6027 | 12.2981e-6 |
| `SpectralAnomalyHB-IRSTD-B` | 1 | 68.1594 | 93.6027 | 12.2981e-6 |
| `SpectralAnomalyHB-NUAA-LCA` | 0 | 79.1123 | 96.5779 | 17.6990e-6 |
| `SpectralAnomalyHB-NUAA-LCA` | 1 | 79.1230 | 96.5779 | 17.6990e-6 |
| `SpectralAnomalyHB-NUAA-LCA` | 2 | 79.1083 | 96.5779 | 17.7676e-6 |
| `SpectralAnomalyHB-NUAA-TCD` | 0 | 79.1315 | 96.5779 | 17.6990e-6 |
| `SpectralAnomalyHB-NUAA-TCD` | 1 | 79.1145 | 96.5779 | 17.6990e-6 |

Reference retained bests:

| Dataset | retained best | target |
| --- | ---: | ---: |
| IRSTD-1K | 73.0367 | 73.03 |
| NUAA-SIRST | 79.5487 valid / 80.6827 oracle | 82.64 |

## Decision

Stop this run.

Reason:

```text
The branch remained in the old 68/79 mIoU region, did not approach the current retained bests, and did not satisfy the desired 9GB+ memory usage despite batch_size=28.
```

This is useful negative evidence for the SET/UniVAD-style spectral-normality route as currently implemented. It does not create the missing NUAA candidate source.

## Cleanup

Stopped sessions:

```text
spectral_hb_irstd_gpu0
spectral_hb_irstd_gpu1
spectral_hb_nuaa_gpu2
spectral_hb_nuaa_gpu3
```

Deleted failed checkpoints:

```text
runs/spectral_anomaly_highbatch_4gpu/*/checkpoints/*/best_mIoU.pth.tar
```

Each checkpoint was about 45.7 MB. Four were removed, freeing about 183 MB.

Retained only small logs:

```text
runs/spectral_anomaly_highbatch_4gpu/**/*.log
```

After cleanup:

```text
tmux sessions: none
GPUs: idle, 1 MiB each
```
