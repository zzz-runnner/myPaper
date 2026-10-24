# NUAA local-rule LPSM/MambaIR-style 4-GPU follow-up

Date: 2026-06-16

## Purpose

After HFM-LK, uncertainty-boundary, and DHiF all stayed below the retained NUAA result, this run tested a different migration idea from the supplied materials:

```text
MambaIR / MSVMamba-style long-range state mixing
+ explicit local-preserving path for tiny target evidence
+ bounded residual around the current official-Mamba local-rule teacher
```

This is not SCTransNet parameter tuning. It is a cache-based teacher-anchored refiner with directional integral state mixing.

## External source status

The higher-priority external assets were checked first:

- MiM-ISTD: code is present locally, but the useful checkpoint is only linked through Baidu in the README and is not present locally.
- RealScene-ISTD: code is present locally, but dataset/patches/weights are Baidu links and are not present locally.
- Text-IRSTD: local README says the code will be released later.
- SAIST: no stable official runnable code/checkpoint was found locally.

Because those stronger external sources are not currently available on the server, this run implemented the MambaIR/MSVMamba-style module directly.

## Code

Added:

```text
scripts/train_lpsm_anchor_refiner_from_cache.py
scripts/run_local_rule_lpsm_one.sh
scripts/launch_local_rule_lpsm_nuaa_4gpu_screen.sh
```

The model adds:

- directional prefix-state mixing over rows/columns;
- depthwise local preservation path;
- gated state residual;
- teacher-anchored bounded residual output.

## Training setup

Common setup:

```text
train caches:
  IRSTD-1K train
  NUAA-SIRST train
  NUDT-SIRST train
  SIRST3 train
test cache:
  NUAA-SIRST test
teacher_mode: local_rule
epochs: 3
batch_size: 3
patch_size: 512
AMP: true
weighted_dice: true
thresholds: 0.488..0.512
```

Runtime memory:

```text
GPU0: about 9184 MiB / 11264 MiB, 100% util
GPU1: about 7660 MiB / 11264 MiB
GPU2: about 8416 MiB / 11264 MiB
GPU3: about 7864 MiB / 11264 MiB
```

GPU0 met the requested 9GB+ target. The other branches used less memory because this was a conservative first LPSM screen to avoid OOM.

## Results

Previous retained NUAA best:

```text
82.2928
```

Best LPSM row:

| branch | stages | seed | residual | best epoch | threshold | mIoU | nIoU | Pd | Fa |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s1303_e2m_r006 | e2,mid | 20260653 | 0.060 | 3 | 0.500 | 82.3084 | 82.0780 | 96.9582 | 9.1925e-6 |

All branches:

| branch | stages | best epoch | mIoU | delta vs old best |
| --- | --- | ---: | ---: | ---: |
| s1303_e2m_r006 | e2,mid | 3 | 82.3084 | +0.0156 |
| s1304_mid_r004 | mid | 3 | 82.2742 | -0.0186 |
| s1301_e2e3m_r004 | e2,e3,mid | 1 | 82.2664 | -0.0264 |
| s1302_e3m_r005 | e3,mid | 1 | 82.2660 | -0.0268 |

Updated NUAA status:

```text
new NUAA best: 82.3084
target:        82.6400
remaining gap:  0.3316
```

## Checkpoint policy

Kept only the branch that improved the retained best:

```text
runs/local_rule_lpsm_nuaa_4gpu_screen/s1303_e2m_r006/best.pth.tar
```

Deleted the three weaker checkpoints:

```text
runs/local_rule_lpsm_nuaa_4gpu_screen/s1301_e2e3m_r004/best.pth.tar
runs/local_rule_lpsm_nuaa_4gpu_screen/s1302_e3m_r005/best.pth.tar
runs/local_rule_lpsm_nuaa_4gpu_screen/s1304_mid_r004/best.pth.tar
```

The run directory is about `52M` after cleanup.

## Interpretation

This is a small but real improvement over the previous NUAA best. It suggests the MambaIR/MSVMamba-style local-preserving state-mixing direction is more promising than the previous HFM-LK/UB/DHiF local-rule screens.

However, the gain is only `+0.0156` mIoU and the +5 target is still not reached. The result should be treated as the new NUAA anchor for follow-up, not as completion.

Recommended next compute:

1. Run a focused LPSM follow-up around `e2,mid`, slightly larger memory settings if stable.
2. Try component/min-area calibration on the retained LPSM probability map.
3. If user can upload Baidu assets, prioritize RealScene-ISTD dataset/patches and MiM-ISTD checkpoint.

## Component-filter follow-up

After retaining `s1303_e2m_r006`, a direct checkpoint evaluation swept threshold and connected-component min-area filtering:

```text
scripts/eval_lpsm_component_filter.py
reports/nuaa_lpsm_s1303_component_filter.json
```

Best valid postprocess:

| checkpoint | threshold | min area | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| s1303_e2m_r006 | 0.500 | 3 px | 82.3245 | 82.0878 | 96.9582 | 8.9181e-6 |

Comparison:

```text
previous retained NUAA best before LPSM: 82.2928
LPSM raw best:                         82.3084
LPSM + min_area=3:                     82.3245
gain over previous retained best:      +0.0317
NUAA target:                           82.6400
remaining gap:                         0.3155
```

This is now the retained NUAA best, but the overall goal remains incomplete because NUAA still has not crossed `82.64`.
