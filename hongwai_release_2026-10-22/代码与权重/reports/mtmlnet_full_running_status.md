# MTMLNet Full-Data Running Status

Generated: 2026-06-13 00:39 UTC

## Purpose

This is the current external-method line after SCTransNet-family refiners saturated. It uses the official MTMLNet repository as a multi-task detection + segmentation backbone, not another SCTransNet parameter-tuning run.

## Data Conversion

Converted the four prepared IRSTD-style datasets into MTMLNet's required detection + segmentation format:

- source: `/home/cx/hongwai/irstd-lab/datasets`
- converted root: `/home/cx/hongwai/irstd-lab/generated/mtmlnet_full`
- train datasets: `IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3`
- validation/test split for early monitoring: `IRSTD-1K` test + `NUAA-SIRST` test
- train images: 3352
- val images: 415
- train boxes generated from masks: 4772
- val boxes generated from masks: 560
- empty masks: 0

Images are symlinked; segmentation masks and YOLO labels are generated files. The generated directory is about 60 MB.

## Code Added

- `/home/cx/hongwai/irstd-lab/scripts/prepare_mtmlnet_dataset.py`
- `/home/cx/hongwai/irstd-lab/scripts/launch_mtmlnet_full_4gpu_tmux.sh`
- `/home/cx/hongwai/irstd-lab/scripts/eval_mtmlnet_segmentation.py`
- `/home/cx/hongwai/irstd-lab/configs/mtmlnet_hyp_sgd.yaml`
- `/home/cx/hongwai/irstd-lab/configs/mtmlnet_hyp_adamw.yaml`

Small external-repo fix:

- `/home/cx/hongwai/irstd-lab/external/MTMLNet/val.py` now creates `metric_results/` before writing CSV metrics.

## Launch Correction

The first launch failed before any valid epoch because the launcher passed `--device 0` in every tmux window. MTMLNet resets `CUDA_VISIBLE_DEVICES` when `--device` is non-empty, so all jobs landed on physical GPU0 and caused OOM. This failed run was deleted after preserving `/home/cx/hongwai/irstd-lab/reports/mtmlnet_full_first_launch_failure_note.md`.

The launcher now passes `--device ""`, allowing each tmux window's `CUDA_VISIBLE_DEVICES=$gpu` binding to work.

## Active Jobs

tmux session: `mtmlnet_full_4gpu`

| GPU | Run | Image size | Batch | Optimizer |
|---:|---|---:|---:|---|
| 0 | `mtm_full_s512_b3_sgd_a` | 512 | 3 | SGD |
| 1 | `mtm_full_s640_b2_sgd` | 640 | 2 | SGD |
| 2 | `mtm_full_s512_b3_adamw_a` | 512 | 3 | AdamW |
| 3 | `mtm_full_s512_b3_adamw_b` | 512 | 3 | AdamW |

At the last check, all four GPUs were training with about 10 GB used and 80-90% GPU utilization. The jobs were still in epoch 0, with the 512 runs around 20% through the epoch and the 640 run around 13% through the epoch.

## Current Risk

Epoch-end validation uses no-grad inference, but it still uses larger validation batches than training in the official script. If validation OOMs, the next fix should reduce the validation dataloader batch sizes in `external/MTMLNet/train.py` or add explicit CLI controls, then restart after deleting the failed run directory.
