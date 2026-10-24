# Current Status After SAM-Rim Pseudo Experiments

Generated: 2026-06-13 07:00 UTC

## Runtime State

- No active tmux training sessions after cleanup.
- All four RTX 2080 Ti GPUs are free.
- The bulky `runs/sam_rim_pseudo_4gpu` directory was deleted after preserving logs and best checkpoints.

## New Code / Assets Added

- `scripts/generate_sam_refined_pseudo_labels.py`
  - Generates conservative SAM-assisted pseudo labels from strong teacher predictions.
  - Uses teacher connected components as point/box prompts.
  - Adds negative points around the target box.
  - Uses `soft_rim` fusion so SAM regions are soft boundary supervision, not hard foreground.
  - Supports sharding for multi-GPU generation.
- `scripts/summarize_sam_pseudo_quality.py`
  - Merges shard summaries into one quality report.
- `scripts/launch_sam_refined_pseudo_4gpu_tmux.sh`
  - Generated two full pseudo-label sets on 4 GPUs.
- `scripts/launch_sam_rim_pseudo_train_4gpu_tmux.sh`
  - Ran four full multi-dataset training branches with SAM soft-rim pseudo supervision.
- `scripts/train_hybrid_methods.py`
  - Added pseudo-label rim loss args:
    - `--pseudo_rim_weight`
    - `--pseudo_rim_low`
    - `--pseudo_rim_high`
    - `--pseudo_rim_target`

## Pseudo Labels Generated

Two complete pseudo-label sets were generated, each covering the full train pool:

| Directory | IRSTD-1K | NUAA-SIRST | NUDT-SIRST | SIRST3 | Size |
|---|---:|---:|---:|---:|---:|
| `generated/sam_refined_irstd_tpc_rim_train` | 800 | 213 | 663 | 1676 | 176M |
| `generated/sam_refined_nuaa_lca_tcd_rim_train` | 800 | 213 | 663 | 1676 | 305M |

A mixed NUAA directory was also created:

```text
generated/sam_refined_mix_for_nuaa_rim_train
```

It uses NUAA-teacher pseudo labels for NUAA-SIRST and IRSTD-teacher pseudo labels for IRSTD-1K, NUDT-SIRST, and SIRST3.

## Pseudo Quality Summary

Hard SAM masks were first tested and rejected because they over-expanded small targets badly. Example IRSTD smoke: teacher mIoU 73.89 at threshold 0.5 fell to 36.92 after hard SAM fusion.

The final `soft_rim` version preserves the teacher hard mask at thresholds 0.35/0.5 and only adds low-confidence SAM boundary/rim pixels.

Merged train-set quality at threshold 0.5:

| Pseudo source | Dataset | Teacher/refined mIoU at 0.5 |
|---|---|---:|
| IRSTD-TPC source | IRSTD-1K | 76.35 |
| IRSTD-TPC source | NUAA-SIRST | 87.31 |
| IRSTD-TPC source | NUDT-SIRST | 96.81 |
| IRSTD-TPC source | SIRST3 | 82.85 |
| NUAA-LCA/TCD source | IRSTD-1K | 48.93 |
| NUAA-LCA/TCD source | NUAA-SIRST | 90.12 |
| NUAA-LCA/TCD source | NUDT-SIRST | 50.53 |
| NUAA-LCA/TCD source | SIRST3 | 51.08 |

Conclusion: the NUAA teacher is useful for NUAA but weak cross-domain; the IRSTD-TPC source is stronger across the multi-dataset pool.

## SAM-Rim Training Results

All four branches used full train datasets:

```text
IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3
```

| Branch | Best observed | Later signal | Decision |
|---|---:|---:|---|
| `SAMRim-TPCDHiF-IRSTD-A` | epoch0 mIoU 0.682638 | epoch5 mIoU 0.675108 | stopped; steady drift down |
| `SAMRim-TPCDHiF-IRSTD-B` | epoch1 mIoU 0.683039 | epoch5 mIoU 0.679268 | stopped; tiny early bump then drift down |
| `SAMRim-LCA-NUAA` | epoch5 mIoU 0.791515 | epoch10 mIoU 0.791018 | stopped; noise-level gain only |
| `SAMRim-TCD-NUAA` | epoch4 mIoU 0.791471 | epoch7 mIoU 0.790889 | stopped; noise-level gain only |

Preserved evidence:

```text
reports/sam_rim_pseudo_stopped_20260613_0700/
```

## Honest Interpretation

This was a legitimate external-domain/foundation-model pseudo-supervision attempt, not a SCTransNet parameter-only tweak. It produced working full-pool pseudo labels and trained correctly on all four GPUs, but it did not open the requested +5 mIoU region.

Best current reliable region remains approximately:

- IRSTD-1K: 68.30 mIoU (`SAMRim-TPCDHiF-IRSTD-B` got 68.3039, tiny over previous best but not meaningful)
- NUAA-SIRST: 79.15 mIoU (`SAMRim-LCA-NUAA` got 79.1515, tiny over previous best but not meaningful)

The +5 mIoU objective is still not achieved.
