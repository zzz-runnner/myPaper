# Current Full Status Report

Generated: 2026-06-13 02:08 UTC

This report is based on the files and processes currently visible on the server, not only on memory.

## 1. Important Correction

The server is not fully idle right now. As of 2026-06-13 02:08 UTC, four MTMLNet training jobs are still running on four RTX 2080 Ti GPUs.

Current active tmux sessions:

| Session | Purpose |
|---|---|
| `mtmlnet_segheavy_gpu0` | original full-data MTMLNet seg-heavy 512 branch |
| `mtmlnet_balanced_seg08_gpu1` | balanced-target MTMLNet, segmentation gain 0.80, from scratch |
| `mtmlnet_balanced_fromseg_gpu2` | balanced-target MTMLNet, initialized from seg-heavy 512 best checkpoint |
| `mtmlnet_balanced_seg09_gpu3` | balanced-target MTMLNet, segmentation gain 0.90, from scratch |

Latest GPU snapshot:

| GPU | Used / Total | Utilization | Temperature |
|---:|---:|---:|---:|
| 0 | 7138 MiB / 11264 MiB | 76% | 60 C |
| 1 | 8238 MiB / 11264 MiB | 81% | 64 C |
| 2 | 6618 MiB / 11264 MiB | 72% | 60 C |
| 3 | 7378 MiB / 11264 MiB | 82% | 60 C |

## 2. Disk and Data Placement

Latest disk snapshot:

| Mount | Size | Used | Available | Use |
|---|---:|---:|---:|---:|
| `/` | 437G | 357G | 62G | 86% |
| `/data1` | 3.6T | 195G | 3.4T | 6% |

Project size:

```text
/home/cx/hongwai/irstd-lab  8.5G
```

Conclusion:

- The project itself is not large after cleanup.
- Root `/` has only about 62G available, so future large datasets should not be placed under `/home/cx/hongwai/irstd-lab` directly.
- `/data1` has about 3.4T available and is the right place for uploaded datasets.
- Earlier check showed `/data1` root is owned by `user:user` with mode `755`, so the current user `cx` may not be able to create directories directly at `/data1` root. If the upload tool can create `/data1/<your_dataset_folder>`, that is fine; otherwise an admin-created writable folder under `/data1` is needed.

## 3. Project and Environment Prepared

Main project:

```text
/home/cx/hongwai/irstd-lab
```

Python environment:

```text
/home/cx/hongwai/irstd-lab/envs/irstd
```

External repositories present:

```text
/home/cx/hongwai/irstd-lab/external/SCTransNet
/home/cx/hongwai/irstd-lab/external/BGM
/home/cx/hongwai/irstd-lab/external/MSHNet
/home/cx/hongwai/irstd-lab/external/PConv-SDloss-Data
/home/cx/hongwai/irstd-lab/external/MTMLNet
```

Main datasets prepared under:

```text
/home/cx/hongwai/irstd-lab/datasets
```

Dataset counts:

| Dataset | Total images | Train | Test |
|---|---:|---:|---:|
| IRSTD-1K | 1001 | 800 | 201 |
| NUAA-SIRST / SIRST-v1 | 427 | 213 | 214 |
| NUDT-SIRST | 1327 | 663 | 664 |
| SIRST3 | 2755 | 1676 | 1079 |

One raw-data issue was fixed:

- NUAA/SIRST sample `Misc_111` had an image/mask size mismatch.
- The mask was resized with nearest-neighbor interpolation to match the image.

## 4. Baseline Verification

Official SCTransNet SIRST3 checkpoint:

```text
/home/cx/hongwai/irstd-lab/pretrained/SCTransNet_SIRST3_official.pth.tar
```

It reproduced the official README numbers closely:

| Dataset | README mIoU | Reproduced mIoU | Reproduced Pd | Reproduced Fa |
|---|---:|---:|---:|---:|
| NUAA-SIRST / SIRST-v1 | 77.50 | 77.64 | 96.96 | 13.93 |
| NUDT-SIRST | 94.09 | 94.11 | 98.62 | 4.30 |
| IRSTD-1K | 68.03 | 68.03 | 93.27 | 10.74 |

Meaning:

- Dataset conversion and evaluation are aligned with the official SCTransNet code.
- The later comparisons are on a reliable evaluation path.

## 5. From-Scratch SCTransNet Runs

Four SCTransNet from-scratch baselines were run to 1000 epochs.

| Dataset | Best epoch | Best mIoU | README / target mIoU | Delta | Pd | Fa |
|---|---:|---:|---:|---:|---:|---:|
| SIRST3 | 678 | 82.96 | n/a | n/a | 96.94 | 7.83 |
| NUDT-SIRST | 547 | 93.76 | 94.09 | -0.33 | 98.94 | 4.37 |
| IRSTD-1K | 500 | 67.32 | 68.03 | -0.71 | 93.60 | 13.61 |
| NUAA-SIRST | 823 | 77.52 | 77.50 | +0.02 | 96.20 | 18.04 |

Conclusion:

- Training from scratch works.
- It does not give the desired improvement over the official SCTransNet baseline.

## 6. Current Best Results Before MTMLNet

Best target results achieved before the MTMLNet line:

| Target dataset | Best method so far | Best mIoU | Baseline target | Gain |
|---|---|---:|---:|---:|
| IRSTD-1K | `SCT-TPC-DHiF-B` | 68.28 | 68.03 | +0.25 |
| NUAA-SIRST | `SCT-LCA-B` / `SCT-TCD-Decouple` family | 79.13 | 77.50 | +1.63 |

This is positive, but still far from the desired `+5 mIoU` target.

## 7. SCTransNet-Family / Refiner Attempts

These lines used SCTransNet-family checkpoints, extra heads, refiners, feature fusion, gating, or teacher-like branches.

The broader `our_method_4gpu` sweep contained:

| Dataset | Variant count in report | Best mIoU | Baseline target | Gain |
|---|---:|---:|---:|---:|
| IRSTD-1K | 21 variants | 68.05 | 68.03 | +0.02 |
| NUAA-SIRST | 12 variants | 79.10 | 77.50 | +1.60 |

Other focused SCTransNet-family/refiner lines:

| Line | IRSTD-1K best | NUAA-SIRST best | Conclusion |
|---|---:|---:|---|
| `SCT-Refiner-Fixed` | 68.09 | 79.12 | small gain only |
| `SCT-DHiF-Balanced` | 68.05 | 79.11 | no meaningful improvement |
| `SCT-DHiF-FromFixed` | 68.09 | 79.12 | same plateau |
| `SCT-DHiF-Multi` | 68.06 | 79.11 | no meaningful improvement |
| `SCT-TPC-DHiF` | 68.28 | 79.12 | best IRSTD so far |
| `SCT-LCA` | 68.09 | 79.13 | best NUAA so far |
| `BNG/FeatUp` variants | 68.09 | 79.12 | did not move beyond initialization |
| `ExpertGate` | 68.10 | 79.12 | gating did not unlock large gain |
| `ExpertOracleGate` | 68.10 | 79.12 | even oracle-style gating plateaued |
| `TCD-Decouple` | 68.09 | 79.13 | ties best NUAA, IRSTD unchanged |

Complementarity checks showed limited ensemble headroom:

- IRSTD sample oracle among similar variants reached only about 68.9 mIoU.
- NUAA sample oracle was effectively the same as the best single model, about 79.13 mIoU.

Interpretation:

- These models are highly correlated.
- More small SCTransNet refiner/gating variants are unlikely to produce +5.

## 8. External / Non-SCTransNet Lines Tested

### 8.1 Official MSHNet / PConv-MSHNet

Official/pretrained MSHNet and PConv-MSHNet were evaluated, then used as a full-data finetune line.

Official-style IRSTD evaluation:

| Model | Best observed IRSTD mIoU |
|---|---:|
| MSHNet D-0.5 official-style | about 67.67 |
| P43/PConv-MSHNet official-style | about 66.82 |

Full-data finetune results:

| Dataset | Method | Init mIoU | Best mIoU | Existing best target | Delta |
|---|---|---:|---:|---:|---:|
| IRSTD-1K | MSHNet-D05-FullData | 67.29 | 67.29 | 68.28 | -0.99 |
| IRSTD-1K | MSHNet-P43-FullData | 66.70 | 66.70 | 68.28 | -1.58 |
| NUAA-SIRST | MSHNet-D05-FullData | 70.02 | 71.40 | 79.13 | -7.73 |
| NUAA-SIRST | MSHNet-P43-FullData | 69.46 | 72.77 | 79.13 | -6.36 |

Decision:

- Stopped.
- Run directory deleted after preserving reports.
- Not competitive in this form.

### 8.2 Paired BGM

| Dataset | Method | Progress | Best mIoU | Target | Delta |
|---|---|---:|---:|---:|---:|
| NUAA-SIRST | BGM-PairedBasicUNet | epoch 10 | 71.86 | 77.50 | -5.64 |
| IRSTD-1K | BGM/TCF paired variants | epoch 5 only | no useful positive eval | 68.03 | n/a |

Decision:

- Stopped.
- Run directory deleted after preserving reports.

### 8.3 APConv + SDM + TCF

| Dataset | Method | Progress | Best mIoU | Target | Delta |
|---|---|---:|---:|---:|---:|
| NUAA-SIRST | APConv-TCF-SDM | epoch 10 | 73.21 | 77.50 | -4.29 |
| IRSTD-1K | APConv variants | early only | no useful positive result | 68.03 | n/a |

Decision:

- Stopped.
- Run directory deleted after preserving reports.

### 8.4 APConv/TPC Refiner

| Dataset | Method | Progress | Best mIoU | Later behavior |
|---|---|---:|---:|---|
| IRSTD-1K | SCT-APConvTPC-A/B | epoch 2 | 68.02 / 68.03 | dropped afterward |
| NUAA-SIRST | SCT-APConvTPC-A/B | epoch 4 | 79.09 / 79.10 | no new gain |

Decision:

- Stopped.
- Run directory deleted after preserving reports.

### 8.5 Unfrozen APConv/TPC

| Dataset | Method | Progress | Best mIoU | Later mIoU |
|---|---|---:|---:|---:|
| IRSTD-1K | SCT-UnfrozenAPConvTPC-A/B | epoch 1 | about 68.01 / 68.02 | 66.83 / 66.45 |
| NUAA-SIRST | SCT-UnfrozenAPConvTPC-A/B | epoch 1 | about 79.10 / 79.11 | 75.35 / 76.04 |

Decision:

- Stopped.
- Run directory deleted after preserving reports.

## 9. MTMLNet Main Line

This is the current external-method main line. It is not just another SCTransNet parameter tweak.

Official MTMLNet repo:

```text
/home/cx/hongwai/irstd-lab/external/MTMLNet
```

Important prepared files:

```text
/home/cx/hongwai/irstd-lab/scripts/prepare_mtmlnet_dataset.py
/home/cx/hongwai/irstd-lab/scripts/eval_mtmlnet_segmentation.py
/home/cx/hongwai/irstd-lab/scripts/launch_mtmlnet_full_4gpu_tmux.sh
/home/cx/hongwai/irstd-lab/scripts/launch_mtmlnet_segheavy_gpu0_tmux.sh
/home/cx/hongwai/irstd-lab/scripts/launch_mtmlnet_segheavy640_gpu1_tmux.sh
/home/cx/hongwai/irstd-lab/scripts/launch_mtmlnet_balanced_seg08_gpu1_tmux.sh
/home/cx/hongwai/irstd-lab/scripts/launch_mtmlnet_balanced_seg09_gpu3_tmux.sh
/home/cx/hongwai/irstd-lab/scripts/launch_mtmlnet_balanced_fromseg_gpu2_tmux.sh
/home/cx/hongwai/irstd-lab/configs/mtmlnet_hyp_sgd.yaml
/home/cx/hongwai/irstd-lab/configs/mtmlnet_hyp_adamw.yaml
```

Patched MTMLNet files:

```text
/home/cx/hongwai/irstd-lab/external/MTMLNet/val.py
/home/cx/hongwai/irstd-lab/external/MTMLNet/train.py
```

Patch purposes:

- `val.py`: create `metric_results/` before validation writes segmentation CSV metrics.
- `train.py`: add configurable `--det-gain`, `--seg-gain`, `--det-val-batch-mult`, and `--seg-val-batch`, so the multi-task detection/segmentation balance can be controlled on 2080 Ti.

### 9.1 Full MTMLNet Dataset Conversion

Generated dataset:

```text
/home/cx/hongwai/irstd-lab/generated/mtmlnet_full
```

Composition:

| Split | IRSTD-1K | NUAA-SIRST | NUDT-SIRST | SIRST3 | Total |
|---|---:|---:|---:|---:|---:|
| train | 800 | 213 | 663 | 1676 | 3352 |
| val | 201 | 214 | 0 | 0 | 415 |

Boxes:

| Split | Boxes |
|---|---:|
| train | 4772 |
| val | 560 |

Empty masks: none.

### 9.2 First Launch Failure and Fix

The first MTMLNet 4-GPU launch failed because the official code interpreted `--device 0/1/2/3` in a way that put jobs on physical GPU0 despite `CUDA_VISIBLE_DEVICES`, causing OOM.

Actions taken:

- Preserved failure note at `/home/cx/hongwai/irstd-lab/reports/mtmlnet_full_first_launch_failure_note.md`.
- Deleted the failed bulky run directory.
- Relaunched with `CUDA_VISIBLE_DEVICES=<gpu>` and `--device ""`.

### 9.3 MTMLNet Branches Already Stopped

| Branch | Progress / official result | Independent target eval | Decision |
|---|---|---|---|
| `mtm_full_s512_b3_sgd_a` | epoch 0 mIoU 0.00082194, mAP50 0.019105 | not useful | stopped/deleted |
| `mtm_full_s640_b2_sgd` | epoch 0 mIoU 0.0059368, mAP50 0.027621 | not useful | stopped/deleted |
| `mtm_full_s512_b3_adamw_b` | epoch 0/1/2 mIoU 0.12247 / 0.27620 / 0.35976 | epoch0 IRSTD 22.14, NUAA 32.18 | stopped/deleted |
| `mtm_segheavy_s640_b2_adamw` | epoch 0 mIoU 0.40480, mAP50 0.61033 | IRSTD 39.27, NUAA 48.60 | stopped/deleted |
| `mtm_full_s512_b3_adamw_a` | epoch 0/1/2/3 mIoU 0.33252 / 0.45366 / 0.49843 / 0.39254 | epoch1 IRSTD 39.76, NUAA 53.27 | stopped/deleted |

For these stopped MTMLNet branches, small evidence files were preserved under:

```text
/home/cx/hongwai/irstd-lab/reports/mtmlnet_*_stopped_note.md
/home/cx/hongwai/irstd-lab/reports/mtmlnet_*_preserved/
```

### 9.4 Current Original Seg-Heavy 512 Branch

Active run:

```text
/home/cx/hongwai/irstd-lab/runs/mtmlnet_full_4gpu/mtm_segheavy_s512_b3_adamw
```

Settings:

- data: full converted MTMLNet dataset
- image size: 512
- batch size: 3
- optimizer: AdamW + cosine LR
- `det_gain=0.35`
- `seg_gain=0.65`
- epochs target: 160
- current state: epoch 3 has started after completing epoch 2 validation

Official validation:

| Epoch | MTMLNet val mIoU | Detection mAP@0.5 |
|---:|---:|---:|
| 0 | 0.52381 | 0.50052 |
| 1 | 0.53051 | 0.60829 |
| 2 | 0.49701 | 0.62850 |

Independent original-split eval:

| Checkpoint timing | Dataset | Best mIoU | Best threshold |
|---|---|---:|---:|
| epoch0 best | IRSTD-1K | 42.40 | 0.600 |
| epoch0 best | NUAA-SIRST | 60.91 | 0.300 |
| latest/best at later check | IRSTD-1K | 46.42 | 0.700 |
| latest/best at later check | NUAA-SIRST | 57.58 | 0.800 |

Interpretation:

- This is the best early MTMLNet signal so far.
- It is still far below SCTransNet on both IRSTD-1K and NUAA-SIRST.
- Epoch 2 official mIoU dropped from 0.53051 to 0.49701, so this branch is not yet trending clearly upward.

### 9.5 Balanced-Target MTMLNet Dataset

To reduce domination by SIRST3/NUDT and emphasize the target datasets, I patched the dataset preparation script to support train repeat factors.

Generated dataset:

```text
/home/cx/hongwai/irstd-lab/generated/mtmlnet_balanced_target
```

Repeat factors:

```text
IRSTD-1K: 2
NUAA-SIRST: 6
```

Composition:

| Split | IRSTD-1K | NUAA-SIRST | NUDT-SIRST | SIRST3 | Total |
|---|---:|---:|---:|---:|---:|
| train | 1600 | 1278 | 663 | 1676 | 5217 |
| val | 201 | 214 | 0 | 0 | 415 |

Boxes:

| Split | Boxes |
|---|---:|
| train | 7320 |
| val | 560 |

Empty masks: none.

### 9.6 Balanced Branches Currently Running

| GPU | Branch | Init | Key settings | Current status |
|---:|---|---|---|---|
| 1 | `mtm_balanced_s512_b3_seg08_adamw` | from scratch | `det_gain=0.20`, `seg_gain=0.80`, epochs 160 | epoch 0, no validation yet |
| 2 | `mtm_balanced_fromseg_s512_b3_seg08_adamw` | from `mtm_segheavy_s512_b3_adamw/weights/best.pt` | `det_gain=0.20`, `seg_gain=0.80`, epochs 120 | epoch 0, no validation yet |
| 3 | `mtm_balanced_s512_b3_seg09_adamw` | from scratch | `det_gain=0.10`, `seg_gain=0.90`, epochs 160 | epoch 0, no validation yet |

These have not produced `results.csv` rows yet, so there is no target metric to judge them by at this timestamp.

## 10. What Was Deleted

Large failed/weak experiment directories were deleted after preserving concise reports/configs. This was done to keep the project small and avoid filling `/`.

Deleted or cleaned after preservation:

- MTMLNet first failed OOM launch
- MTMLNet SGD branches
- MTMLNet weaker AdamW branches
- MTMLNet seg-heavy 640 branch
- official MSHNet finetune runs
- paired BGM runs
- APConv/SDM runs
- APConv/TPC refiner runs
- unfrozen APConv/TPC runs

Current project size after cleanup is about 8.5G.

## 11. Current Honest Assessment

What is solid:

- Environment and datasets are usable.
- Official SCTransNet reproduction is correct.
- Four-dataset training/evaluation infrastructure is working.
- 2080 Ti memory limits are understood; current MTMLNet settings fit.
- Failed MTMLNet branches were not left consuming disk/GPU after becoming clearly unpromising.

What has worked best so far:

- IRSTD-1K: `SCT-TPC-DHiF-B`, 68.28 mIoU, only +0.25 over 68.03.
- NUAA-SIRST: `SCT-LCA-B` / `TCD-Decouple`, 79.13 mIoU, +1.63 over 77.50.

What has not worked:

- Simply training SCTransNet from scratch does not exceed the official target by much.
- Many SCTransNet-family refiners/gates plateaued because the predictions are highly correlated.
- MSHNet/PConv, BGM, APConv-SDM were valid external attempts but were not competitive in the tested form.
- MTMLNet is training, but current independent IRSTD/NUAA scores are still far below SCTransNet.

Current risk:

- The running balanced MTMLNet branches may still fail to close the gap.
- If their first one or two validations are not much stronger than original seg-heavy 512, continuing them for 160 epochs would waste GPU time.

## 12. Recommended Next Decision

Immediate next step:

1. Let the three balanced MTMLNet branches reach their first validation.
2. Evaluate each `best.pt` independently on original IRSTD-1K and NUAA-SIRST splits.
3. Keep only a branch that clearly beats the original seg-heavy 512 trajectory.
4. Stop and preserve/delete weak balanced branches quickly.

Decision rule I recommend:

- If a balanced branch is still far below about 55 mIoU IRSTD and 65 mIoU NUAA after early validation/eval, it is not on track to beat SCTransNet and should be stopped.
- If none of the MTMLNet branches gets close, the next serious direction should not be more SCTransNet refiner tuning. It should be a teacher/pseudo-label/distillation line that uses the strong SCTransNet checkpoints as training-time teachers for a different/student architecture or for MTMLNet segmentation supervision.

Rationale for the next serious direction:

- The provided materials include training-time external-prior / teacher-style ideas.
- The current best SCTransNet-family models are already strong but correlated.
- A teacher-distillation or pseudo-label line may transfer their precise masks while allowing a different architecture/training objective to improve target-domain behavior.

