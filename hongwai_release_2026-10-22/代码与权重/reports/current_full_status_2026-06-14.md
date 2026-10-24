# Current Full Status - 2026-06-14

## Machine and process state

- Project root: `/home/cx/hongwai/irstd-lab`
- Active training processes: none after the latest 4-GPU run finished.
- GPU state after completion: all 4 GPUs idle, about 1 MiB used on each.
- Disk:
  - `/`: 437G total, 359G used, 59G free.
  - `/data1`: 3.6T total, 195G used, 3.4T free.
  - Project directory: about 11G.

## Datasets currently prepared

- IRSTD-1K: 800 train / 201 test.
- NUAA-SIRST: 213 train / 214 test. `Misc_111` mask mismatch was fixed.
- NUDT-SIRST: 663 train / 664 test.
- SIRST3: 1676 train / 1079 test.

## Target

The working target was to exceed SCTransNet by +5 mIoU on both main datasets.

- IRSTD-1K SCTransNet reproduced/official baseline: 68.03 mIoU.
  - +5 target: 73.03 mIoU.
- NUAA-SIRST SCTransNet reproduced baseline: 77.64 mIoU.
  - +5 target: 82.64 mIoU.

This target has not been reached yet.

## Current best strict results

- IRSTD-1K best: 72.8351 mIoU.
  - Method: full multi-dataset teacher-anchored residual refiner, run `irstd_multi_r05_a002_s81`.
  - Gap to +5 target: 73.03 - 72.8351 = about 0.19 mIoU.
  - This improves SCTransNet by about +4.81 mIoU.
- NUAA-SIRST best: 79.34 mIoU.
  - Method: background-normality/component fusion from previous cached-component experiments.
  - Gap to +5 target: 82.64 - 79.34 = about 3.30 mIoU.
  - This improves SCTransNet by about +1.70 mIoU.

## Latest completed run: full multi-dataset teacher-anchored residual refiner

Script:

- `/home/cx/hongwai/irstd-lab/scripts/train_teacher_anchor_refiner_from_cache.py`

Idea:

- Use strong teacher prediction caches as the anchor.
- Add image/local residual cues: robust image normalization, local contrast/local-z, top-hat, DoG/Laplacian, low-rank/background residual proxy.
- Add teacher statistics: average, max, min, std, and individual teacher maps.
- Train a bounded residual refiner:
  - `final_logit = teacher_logit + residual_scale * tanh(refiner(features))`

Resources:

- Ran 4 jobs in parallel on 4 GPUs.
- Batch size 5, patch size 512, base channels 64.
- Used `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Runtime memory was about 10.4-10.7GB per GPU, so it did occupy the 2080Ti GPUs as requested.

Training data:

- IRSTD run used full multi-dataset train caches from IRSTD-oriented dual family teachers:
  - IRSTD-1K train
  - NUAA-SIRST train
  - NUDT-SIRST train
  - SIRST3 train
- NUAA run used full multi-dataset train caches from NUAA-oriented TPC/LCA teacher family:
  - IRSTD-1K train
  - NUAA-SIRST train
  - NUDT-SIRST train
  - SIRST3 train

Final metrics:

| run | eval dataset | best mIoU | nIoU | Pd | Fa | threshold | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `irstd_multi_r05_a002_s81` | IRSTD-1K | 72.8351 | 67.7155 | 94.6128 | 1.2412e-5 | 0.35 | current IRSTD best, kept checkpoint |
| `irstd_multi_r035_a001_s82` | IRSTD-1K | 72.7797 | 67.0795 | 94.6128 | 1.0837e-5 | 0.40 | weaker than kept IRSTD run, checkpoint deleted |
| `nuaa_multi_r05_a002_s83` | NUAA-SIRST | 79.1429 | 79.7999 | 96.1977 | 3.0253e-5 | 0.60 | weaker than previous NUAA best, checkpoint deleted |
| `nuaa_multi_r035_a001_s84` | NUAA-SIRST | 79.1217 | 79.2796 | 96.5779 | 1.7425e-5 | 0.55 | weaker than previous NUAA best, checkpoint deleted |

Kept checkpoint:

- `/home/cx/hongwai/irstd-lab/runs/anchor_refiner_multidata_probe/irstd_multi_r05_a002_s81/best.pth.tar`

Deleted from this family:

- Single-dataset old IRSTD anchor-refiner checkpoint, because it was superseded.
- Multi-dataset `irstd_multi_r035_a001_s82` checkpoint, because it was slightly weaker.
- Both NUAA multi-dataset anchor-refiner checkpoints, because neither beat the previous 79.34 best.

Logs:

- `/home/cx/hongwai/irstd-lab/reports/anchor_refiner_multidata_logs/irstd_multi_r05_a002_s81.log`
- `/home/cx/hongwai/irstd-lab/reports/anchor_refiner_multidata_logs/irstd_multi_r035_a001_s82.log`
- `/home/cx/hongwai/irstd-lab/reports/anchor_refiner_multidata_logs/nuaa_multi_r05_a002_s83.log`
- `/home/cx/hongwai/irstd-lab/reports/anchor_refiner_multidata_logs/nuaa_multi_r035_a001_s84.log`

## Main things already tried

### Baselines

- Official/reproduced SCTransNet:
  - IRSTD-1K: 68.03 mIoU.
  - NUAA-SIRST: 77.64 mIoU.
  - NUDT-SIRST: 94.11 mIoU.
- From-scratch SCTransNet:
  - IRSTD-1K: 67.32 mIoU.
  - NUAA-SIRST: 77.52 mIoU.
  - NUDT-SIRST: 93.76 mIoU.
  - SIRST3: 82.96 mIoU.

### Stronger local variants and combinations

- TCD / DHiF / LCA / PConv / APConv / SCT refiner style variants.
- Cached prediction infrastructure was built and used for fusion/refiner experiments.
- Dual-PConv validation-selected fusion previously reached about 72.48 mIoU on IRSTD-1K.
- Background-normality/component fusion previously reached about 79.34 mIoU on NUAA-SIRST.
- The latest teacher-anchored residual refiner improved IRSTD further to 72.8351 mIoU.

### External or paper-inspired candidates

The following were tried and did not beat the current best; bulky weak checkpoints were deleted where appropriate.

- LRRNet standalone full multi-dataset 4-GPU probe:
  - IRSTD best about 64.59.
  - NUAA best about 70.84.
  - Report: `/home/cx/hongwai/irstd-lab/reports/lrrnet_multidata_4gpu_probe_results.md`
- LRRNet official checkpoint direct evaluation:
  - IRSTD full test about 65.42.
  - NUAA about 67.25.
  - Report: `/home/cx/hongwai/irstd-lab/reports/lrrnet_external_candidate_attempt.md`
- BasicIRSTD family, including ACM/DNANet/ISTDU/UIUNet:
  - Weaker than current line.
  - Report: `/home/cx/hongwai/irstd-lab/reports/basicirstd_mim_external_candidate_results.md`
- IRSAM / MobileSAM:
  - Weak for this strict evaluation.
  - Report: `/home/cx/hongwai/irstd-lab/reports/irsam_mobilesam_candidate_attempt.md`
- DWTFreqNet:
  - Weak under this setup.
  - Report: `/home/cx/hongwai/irstd-lab/reports/dwtfreqnet_external_candidate_attempt.md`
- UniVAD-style dense normal-memory attempt:
  - Did not provide the needed gain.
  - Report: `/home/cx/hongwai/irstd-lab/reports/univad_dense_normal_memory_attempt.md`
- High-frequency student from cache:
  - Weak.
  - Report: `/home/cx/hongwai/irstd-lab/reports/hf_student_distillation_attempt.md`
- SAM/RIM pseudo-label refinements:
  - Not enough to pass current best.

## Important reusable assets

Do not delete these unless intentionally rebuilding all caches:

- `/home/cx/hongwai/irstd-lab/reports/prediction_cache_irstd_dual_train_a`
- `/home/cx/hongwai/irstd-lab/reports/prediction_cache_irstd_dual_train_b_test`
- `/home/cx/hongwai/irstd-lab/reports/prediction_cache_nuaa_tpc_lca_train_a`
- `/home/cx/hongwai/irstd-lab/reports/prediction_cache_nuaa_tpc_lca_train_b_test`
- `/home/cx/hongwai/irstd-lab/runs/selected_checkpoints`

## Honest conclusion

- IRSTD-1K is now close to the requested +5 improvement. The current best is 72.8351 mIoU, only about 0.19 below the 73.03 target.
- NUAA-SIRST remains the harder blocker. The best is still 79.34 mIoU, far below the 82.64 target.
- The latest full multi-dataset residual-refiner line is useful for IRSTD but did not solve NUAA.
- The server is idle now, with no active training processes.
