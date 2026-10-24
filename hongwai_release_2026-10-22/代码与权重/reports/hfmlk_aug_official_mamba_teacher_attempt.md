# HFM-LK official-Mamba teacher augmentation attempt

Date: 2026-06-15

## Motivation

The active NUAA-SIRST best is 82.2554 mIoU, still below the target line. This attempt added cross-domain robustness augmentation from `train_our_method.py` into the cache-based HFM-LK teacher-anchor refiner:

- gamma intensity augmentation
- additive Gaussian noise
- existing full multi-cache teacher-anchor training
- existing official-Mamba fused teacher for NUAA
- existing HFM-LK large-kernel/frequency residual refiner

The goal was to keep the stable high-memory 2080Ti setting while testing a real method/module combination rather than only tuning SCTransNet parameters.

## Code changes

- `scripts/train_teacher_anchor_refiner_from_cache.py`
  - Added `gamma_prob` and `noise_std` to `AnchorTrainSet`.
  - Augmentation is applied to the raw infrared image before `image_feature_stack`.
  - Teacher maps, labels, and teacher logits are not altered.

- `scripts/train_hfm_lk_anchor_refiner_from_cache.py`
  - Exposed `--gamma_prob`, `--noise_std`, and `--num_workers`.

- Added launchers:
  - `scripts/run_hfmlk_aug_official_mamba_teacher_one.sh`
  - `scripts/launch_hfmlk_aug_official_mamba_teacher_nuaa_4gpu_tmux.sh`
  - `scripts/launch_hfmlk_aug_official_mamba_teacher_nuaa_4gpu_nohup.sh`

## Sanity checks

Small-sample smoke with augmentation completed successfully:

- train cache: NUAA official-Mamba teacher cache
- max train samples: 4
- eval max samples: 8
- result was only a sanity check, not a formal metric
- debug output was deleted after verification

No code error was observed in the augmentation path.

## Full four-GPU attempt

Started four branches:

- `aug_s971`
- `aug_s972`
- `aug_fp_s973`
- `aug_focus_s974`

Training configuration:

- train caches:
  - IRSTD-1K
  - NUAA-SIRST
  - NUDT-SIRST
  - SIRST3
- test cache:
  - NUAA-SIRST
- teacher mode:
  - `official_mamba_fused,strong_fusion`
- batch size: 2
- patch size: 512
- base channels: 48
- AMP: enabled
- resume checkpoint:
  - `runs/strong_fusion_hfmlk_nuaa_4gpu/nuaa_strong_fusion_s703/best.pth.tar`

Observed GPU usage:

- GPU0: about 8810 MiB
- GPU1: about 8810 MiB
- GPU2: about 8832 MiB
- GPU3: about 8810 MiB

This is close to the stable limit on RTX 2080Ti. The earlier batch-size-3 version OOMed, so this is the practical stable upper bound for this architecture.

## Failure mode

The four-branch run repeatedly exited before the first epoch JSON metric was written. Logs contained only startup messages and no Python traceback.

I then tested a single full-data branch in the foreground with `/usr/bin/time -v`:

- GPU memory reached about 8810 MiB.
- CPU RSS reached about 20 GB for one process.
- It ran for more than 15 minutes without finishing one epoch.
- The process was stopped manually because the branch was too slow to be a useful current experiment.

The most likely cause is data-pipeline cost:

- `AnchorTrainSet.__getitem__` opens the source image and recomputes `image_feature_stack` for every sampled patch.
- With gamma/noise augmentation, those features cannot be reused from the unchanged image.
- Full multi-cache training makes this CPU-side feature recomputation the bottleneck.
- Four concurrent full-cache runs increase startup/load pressure and exited without useful metrics.

## Cleanup

- Debug output directories were deleted.
- No useful augmented checkpoints were produced.
- No large prediction maps were saved.
- Only small logs and this report are kept as evidence.

## Conclusion

The augmentation idea is technically implemented and passes small smoke tests, but the current on-the-fly feature pipeline is too slow for full multi-dataset high-throughput training.

Do not rerun this full four-cache augmented HFM-LK setup unchanged. If this route is revisited, first optimize the data path by precomputing reusable image features or moving augmentation to a cheaper feature-space perturbation.

## Feature-space follow-up

Added a cheaper mode after the first failed run:

- `--image_aug_mode feature`
- `--gamma_prob`
- `--noise_std`

This applies gamma/noise perturbation to the already computed image feature stack instead of perturbing the raw image before `image_feature_stack`.

Smoke test passed, and the four-GPU run reached the expected high-memory state:

- about 8810-8832 MiB per 2080Ti
- about 97-98% GPU utilization during the training section

However, four concurrent full-cache runs still exited before the first epoch metric, probably because each process plus dataloader workers expands large cache objects. A two-GPU `num_workers=0` run was stable but too slow for full screening.

I then ran a limited four-cache screening run:

```bash
NUM_WORKERS=2 IMAGE_AUG_MODE=feature bash scripts/run_hfmlk_aug_official_mamba_teacher_one.sh \
  0 screen_feature_aug_s971_128 20260619 4e-5 0.06 0.14 0.35 0.006 \
  --epochs 2 \
  --max_train_samples_per_cache 128 \
  --uncertainty_weight 0.25
```

Best NUAA result:

- epoch: 1
- threshold: 0.54
- mIoU: 82.1034
- nIoU: 82.1865
- Pd: 97.7186
- Fa: 7.3403e-6

Epoch 2 fell slightly to 82.0950 mIoU.

This did not beat the old HFM-LK official-Mamba teacher run at 82.1330 mIoU, and it is also below the current rule-based NUAA best of 82.2554 mIoU.

Cleanup:

- Deleted `runs/hfmlk_aug_official_mamba_teacher_nuaa_4gpu/screen_feature_aug_s971_128/best.pth.tar`.
- Kept only small `summary.json` and `train.log`.

Updated conclusion: feature-space augmentation is technically viable but does not show a positive signal in the limited screen. Do not prioritize this branch unless the data pipeline is rewritten and a stronger augmentation hypothesis is added.
