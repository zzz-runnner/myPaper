# MTMLNet Current Status

Generated: 2026-06-13 01:30 UTC

## Active Branches

Four MTMLNet branches are currently running:

| GPU | Branch | Status |
|---:|---|---|
| 0 | `mtm_segheavy_s512_b3_adamw` | running, epoch 1 after strong epoch 0 |
| 1 | `mtm_segheavy_s640_b2_adamw` | running, epoch 0 |
| 2 | `mtm_full_s512_b3_adamw_a` | running, epoch 2 |
| 3 | `mtm_full_s512_b3_adamw_b` | running, epoch 2 |

Stopped and removed weak branches after preserving reports/configs:

| Branch | Reason |
|---|---|
| `mtm_full_s512_b3_sgd_a` | epoch 0 segmentation mIoU 0.00082194; val cls loss abnormal |
| `mtm_full_s640_b2_sgd` | epoch 0 segmentation mIoU 0.0059368; val cls loss abnormal |

## Official MTMLNet Validation Results

| Branch | Epoch | mIoU | Detection mAP@0.5 | Notes |
|---|---:|---:|---:|---|
| `mtm_full_s512_b3_adamw_a` | 0 | 0.33252 | 0.48604 | usable but below SCTransNet |
| `mtm_full_s512_b3_adamw_a` | 1 | 0.45366 | 0.62811 | improving |
| `mtm_full_s512_b3_adamw_b` | 0 | 0.12247 | 0.50763 | weak |
| `mtm_full_s512_b3_adamw_b` | 1 | 0.27620 | 0.59064 | improving but behind A |
| `mtm_segheavy_s512_b3_adamw` | 0 | 0.52381 | 0.50052 | best early MTMLNet signal so far |
| `mtm_full_s512_b3_sgd_a` | 0 | 0.00082194 | 0.019105 | stopped/deleted |
| `mtm_full_s640_b2_sgd` | 0 | 0.0059368 | 0.027621 | stopped/deleted |

## Independent IRSTD / NUAA Evaluation

Independent evaluation does threshold sweeps on the original dataset test splits using `/home/cx/hongwai/irstd-lab/scripts/eval_mtmlnet_segmentation.py`.

| Branch | Epoch | Dataset | Best mIoU | Best threshold |
|---|---:|---|---:|---:|
| `mtm_full_s512_b3_adamw_a` | 0 | IRSTD-1K | 35.69 | 0.08 |
| `mtm_full_s512_b3_adamw_a` | 0 | NUAA-SIRST | 53.27 | 0.08 |
| `mtm_full_s512_b3_adamw_a` | 1 | IRSTD-1K | 39.76 | 0.30 |
| `mtm_full_s512_b3_adamw_a` | 1 | NUAA-SIRST | 53.27 | 0.15 |
| `mtm_full_s512_b3_adamw_b` | 0 | IRSTD-1K | 22.14 | 0.02 |
| `mtm_full_s512_b3_adamw_b` | 0 | NUAA-SIRST | 32.18 | 0.02 |
| `mtm_segheavy_s512_b3_adamw` | 0 | IRSTD-1K | 42.40 | 0.60 |
| `mtm_segheavy_s512_b3_adamw` | 0 | NUAA-SIRST | 60.91 | 0.30 |

## Current Interpretation

- The original AdamW-A branch is learning, but its independent IRSTD/NUAA performance is still far below SCTransNet.
- The seg-heavy 512 branch is the best MTMLNet candidate so far: its first official validation mIoU is already higher than AdamW-A after one epoch, and its independent IRSTD/NUAA scores are also better than AdamW-A at comparable early checkpoints.
- The SGD branches failed badly and were removed to free disk/GPU.
- The most useful immediate action is to keep seg-heavy 512 and seg-heavy 640 running to their first few validations, then preserve only branches that show a plausible path toward the SCTransNet baselines.
