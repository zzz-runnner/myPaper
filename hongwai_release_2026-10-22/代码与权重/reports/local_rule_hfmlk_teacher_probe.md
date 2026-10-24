# Local-rule HFM-LK teacher probe

Date: 2026-06-16

## Purpose

Try a new dense-source branch after the NUAA postprocessing / pixel-calibration plateau. This branch trains the HFM-LK large-kernel/frequency residual refiner from cached multi-dataset teacher maps, but uses the current strong NUAA local-rule teacher instead of the older raw cache modes.

This is not SCTransNet parameter tuning. It combines:

- HFM/DFAware-style large-kernel context,
- dynamic high-frequency gating,
- multi-dataset cache training,
- the current NUAA official-Mamba local-rule teacher.

## Code change

Updated `scripts/train_teacher_anchor_refiner_from_cache.py`:

- added `teacher_mode=local_rule`;
- for NUAA official-Mamba cache, it builds the same local-rule teacher used by the current best residual-pixel branch;
- for other datasets without `official_mamba_fused`, it falls back to a stable average over available strong/weighted/TPC/LCA maps;
- feature channels are fixed across datasets by filling missing prediction sources with the teacher map.

## Smoke

Small smoke passed:

```bash
envs/irstd/bin/python scripts/train_hfm_lk_anchor_refiner_from_cache.py \
  --train_cache_spec 'IRSTD-1K|reports/fusion_cache_bgm_rpcanet_train_a/IRSTD-1K__train.npz' \
  --train_cache_spec 'NUAA-SIRST|reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__train.npz' \
  --test_cache_spec 'NUAA-SIRST|reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__test.npz' \
  --teacher_mode local_rule \
  --epochs 1 --batch_size 1 --patch_size 256 --base_channels 16 \
  --thresholds 0.49,0.498 \
  --eval_interval 1 --max_train_samples_per_cache 4 --eval_max_samples 8 \
  --num_workers 0 --device cuda:0 \
  --output_dir runs/local_rule_hfmlk_smoke
```

The first smoke attempt exposed a channel-count mismatch across datasets. The fixed-channel `local_rule` implementation resolved it.

## Probe command

```bash
envs/irstd/bin/python scripts/train_hfm_lk_anchor_refiner_from_cache.py \
  --train_cache_spec 'IRSTD-1K|reports/fusion_cache_bgm_rpcanet_train_a/IRSTD-1K__train.npz' \
  --train_cache_spec 'NUAA-SIRST|reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__train.npz' \
  --train_cache_spec 'NUDT-SIRST|reports/fusion_cache_bgm_rpcanet_train_b_test/NUDT-SIRST__train.npz' \
  --train_cache_spec 'SIRST3|reports/fusion_cache_bgm_rpcanet_train_b_test/SIRST3__train.npz' \
  --test_cache_spec 'NUAA-SIRST|reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__test.npz' \
  --teacher_mode local_rule \
  --epochs 3 --batch_size 2 --patch_size 512 --base_channels 48 \
  --thresholds 0.468,0.476,0.484,0.492,0.498,0.504,0.512,0.520,0.540 \
  --eval_interval 1 --max_train_samples_per_cache 128 --num_workers 1 \
  --focus_dataset NUAA-SIRST --focus_patch_prob 0.45 \
  --focus_hard_threshold 0.45 --focus_miss_threshold 0.18 \
  --weighted_dice --residual_scale 0.08 --anchor_weight 0.12 \
  --lr 4e-5 --seed 20260616 --amp --device cuda:0 \
  --output_dir runs/local_rule_hfmlk_nuaa_probe_s1001
```

## Result

Best row:

```json
{
  "epoch": 1,
  "threshold": 0.492,
  "mIoU": 0.8224586029169865,
  "nIoU": 0.8210093442392881,
  "Pd": 0.9695817490494296,
  "Fa": 0.000009055312731399256
}
```

Comparison:

```text
current NUAA best: 82.2928
local-rule HFM-LK probe: 82.2459
gap: -0.0469
```

This is below the retained best, but stronger than the earlier HFM-LK official-Mamba teacher runs around `82.13`.

## Resource use

During the training section:

- GPU0 memory: about `9052 MiB` / `11264 MiB`
- GPU0 utilization: about `98%`

This is a stable 2080Ti-sized configuration and can be parallelized across four GPUs if a stronger hyperparameter direction appears.

## Cleanup

The probe checkpoint did not beat the current best, so it was deleted:

```text
runs/local_rule_hfmlk_nuaa_probe_s1001/best.pth.tar
```

Kept only:

- `runs/local_rule_hfmlk_nuaa_probe_s1001/summary.json`
- `runs/local_rule_hfmlk_nuaa_probe_s1001/train.log`

The run directory is now about `12K`.

## Interpretation

The local-rule teacher is a viable new teacher mode and the HFM-LK refiner can learn from it, but the limited screen still does not exceed the current residual-pixel + component-filter best. Since epoch 1 was best and later epochs drifted, a larger run should only be attempted with a more conservative residual scale / anchor or with multi-seed parallel screening, not by simply extending this exact run.
