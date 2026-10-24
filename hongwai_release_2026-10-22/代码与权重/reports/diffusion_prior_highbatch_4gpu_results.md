# Diffusion-Prior High-Batch 4GPU Results

Generated: 2026-06-14 UTC

## Purpose

This run tested the remaining local external-mechanism branch:

```text
thermal diffusion / point-supervision-inspired pseudo-mask prior
+ teacher/base anchored residual refinement
+ full multi-dataset training
```

Model:

```text
sct_diffusion_prior_refiner
```

Launcher:

```text
scripts/launch_diffusion_prior_highbatch_4gpu_tmux.sh
```

This corresponds to the pasted Diffuse-to-Detect / point-supervised diffusion idea, adapted here as a train-time prior rather than a new dataset annotation protocol.

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
batch_size = 22
epochs planned = 90
eval_before_train = true
eval every epoch = true
```

## Runs

| GPU | run | eval dataset | teacher |
| ---: | --- | --- | --- |
| 0 | `DiffusionPrior-IRSTD-A` | IRSTD-1K | TPC-DHiF-B |
| 1 | `DiffusionPrior-IRSTD-B` | IRSTD-1K | TPC-DHiF-A |
| 2 | `DiffusionPrior-NUAA-LCA` | NUAA-SIRST | LCA |
| 3 | `DiffusionPrior-NUAA-TCD` | NUAA-SIRST | TCD-Decouple |

## GPU Use

Observed during training:

```text
GPU0: about 6.8 GB, 98-99% utilization
GPU1: about 6.8 GB, 98-99% utilization
GPU2: about 6.8 GB, 98-99% utilization
GPU3: about 6.8 GB, 98-99% utilization
```

The branch kept all four GPUs busy, but it was below the requested 9GB+ memory target.

## Early Results

Stopped early because the branch stayed in the old 68/79 region and did not approach retained bests.

| run | epoch | mIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: |
| `DiffusionPrior-IRSTD-A` | 0 | 68.2327 | 93.6027 | 12.4689e-6 |
| `DiffusionPrior-IRSTD-B` | 0 | 68.2456 | 93.6027 | 12.3361e-6 |
| `DiffusionPrior-NUAA-LCA` | 0 | 79.1145 | 96.5779 | 17.6990e-6 |
| `DiffusionPrior-NUAA-LCA` | 1 | 79.0936 | 96.5779 | 17.7676e-6 |
| `DiffusionPrior-NUAA-TCD` | 0 | 79.1208 | 96.5779 | 17.6990e-6 |
| `DiffusionPrior-NUAA-TCD` | 1 | 79.1195 | 96.5779 | 17.7676e-6 |

Reference retained bests:

| Dataset | retained best | target |
| --- | ---: | ---: |
| IRSTD-1K | 73.0367 | 73.03 |
| NUAA-SIRST | 79.5487 valid / 80.6827 oracle | 82.64 |

## Decision

Stop this run.

Reason:

```text
The diffusion-prior branch did not produce a stronger dense candidate source, remained below current retained bests, and used less memory than the prompt/spectral high-batch branches.
```

## Cleanup

Stopped sessions:

```text
diffusion_prior_irstd_gpu0
diffusion_prior_irstd_gpu1
diffusion_prior_nuaa_gpu2
diffusion_prior_nuaa_gpu3
```

Deleted failed checkpoints:

```text
runs/diffusion_prior_highbatch_4gpu/*/checkpoints/*/best_mIoU.pth.tar
```

Each checkpoint was about 45.7 MB. Four were removed, freeing about 183 MB.

Retained only small logs:

```text
runs/diffusion_prior_highbatch_4gpu/**/*.log
```

After cleanup:

```text
tmux sessions: none
GPUs: idle, 1 MiB each
```
