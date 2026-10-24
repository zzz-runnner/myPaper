# Prompt-Morph High-Batch 4GPU Results

Generated: 2026-06-14 UTC

## Purpose

This run tested another non-plain-SCTransNet external-mechanism route:

```text
SAMRefiner-inspired coarse-mask prompt excavation
+ thermal spot shape cues
+ morphology-like split/merge context
+ full multi-dataset training
```

Model:

```text
sct_prompt_morph_refiner
```

Launcher:

```text
scripts/launch_prompt_morph_highbatch_4gpu_tmux.sh
```

This route was selected after EFLNet and spectral-normality failed, because it matches the user's pasted external-domain direction: using foundation-model/refinement-style mask priors rather than just tuning SCTransNet training parameters.

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
batch_size = 24
epochs planned = 80
eval_before_train = true
eval every epoch = true
```

## Runs

| GPU | run | eval dataset | teacher |
| ---: | --- | --- | --- |
| 0 | `PromptMorph-IRSTD-A` | IRSTD-1K | TPC-DHiF-B |
| 1 | `PromptMorph-IRSTD-B` | IRSTD-1K | TPC-DHiF-A |
| 2 | `PromptMorph-NUAA-LCA` | NUAA-SIRST | LCA |
| 3 | `PromptMorph-NUAA-TCD` | NUAA-SIRST | TCD-Decouple |

## GPU Use

Observed during training:

```text
GPU0: about 7.7 GB, 98-99% utilization
GPU1: about 7.7 GB, 98-99% utilization
GPU2: about 7.6 GB, 98-99% utilization
GPU3: about 7.7 GB, 98-99% utilization
```

The branch kept all four GPUs busy, but it did not meet the preferred 9GB+ memory target.

## Early Results

Stopped early because the branch stayed in the old 68/79 region and did not approach retained bests.

| run | epoch | mIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: |
| `PromptMorph-IRSTD-A` | 0 | 68.2644 | 93.6027 | 12.6018e-6 |
| `PromptMorph-IRSTD-B` | 0 | 68.2007 | 93.6027 | 12.2981e-6 |
| `PromptMorph-NUAA-LCA` | 0 | 79.1208 | 96.5779 | 17.6990e-6 |
| `PromptMorph-NUAA-LCA` | 1 | 79.1048 | 96.5779 | 17.7676e-6 |
| `PromptMorph-NUAA-TCD` | 0 | 79.1208 | 96.5779 | 17.6990e-6 |
| `PromptMorph-NUAA-TCD` | 1 | 79.1172 | 96.5779 | 17.7676e-6 |

Reference retained bests:

| Dataset | retained best | target |
| --- | ---: | ---: |
| IRSTD-1K | 73.0367 | 73.03 |
| NUAA-SIRST | 79.5487 valid / 80.6827 oracle | 82.64 |

## Decision

Stop this run.

Reason:

```text
The SAMRefiner-style prompt-morph branch did not create a stronger dense candidate source. It stayed below current retained bests and below the requested memory target.
```

This is useful negative evidence for prompt-morph refinement as currently implemented.

## Cleanup

Stopped sessions:

```text
prompt_morph_irstd_gpu0
prompt_morph_irstd_gpu1
prompt_morph_nuaa_gpu2
prompt_morph_nuaa_gpu3
```

Deleted failed checkpoints:

```text
runs/prompt_morph_highbatch_4gpu/*/checkpoints/*/best_mIoU.pth.tar
```

Each checkpoint was about 45.8 MB. Four were removed, freeing about 183 MB.

Retained only small logs:

```text
runs/prompt_morph_highbatch_4gpu/**/*.log
```

After cleanup:

```text
tmux sessions: none
GPUs: idle, 1 MiB each
```
