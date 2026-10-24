# MTMLNet Full-Data First Launch Failure Note

Generated: 2026-06-13 00:34 UTC

The first four-window launch of MTMLNet failed before completing epoch 0. The launcher passed `--device 0` to MTMLNet in every tmux window. MTMLNet's `select_device()` resets `CUDA_VISIBLE_DEVICES` when `--device` is non-empty, so all four jobs were placed on physical GPU0 instead of GPU0-GPU3. This caused immediate CUDA OOM across the four processes.

Fix applied: `/home/cx/hongwai/irstd-lab/scripts/launch_mtmlnet_full_4gpu_tmux.sh` now passes `--device ""`, leaving each window's external `CUDA_VISIBLE_DEVICES=$gpu` binding intact.

The failed run directory was removed after preserving this note and the launcher logs were not kept because no checkpoint or valid epoch result existed.
