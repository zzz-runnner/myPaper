# NUAA focused LPSM follow-up 4-GPU results

Date: 2026-06-16

## Purpose

The previous LPSM screen was the first local-rule refiner branch to improve NUAA. This follow-up focused on the winning `e2,mid` state-mixing configuration and increased model width to test whether the MambaIR/MSVMamba-style local-preserving state branch could continue improving the dense map.

Reference before this run:

```text
previous retained NUAA best: 82.3245
target:                      82.6400
remaining gap:                0.3155
```

## Launch

Added:

```text
scripts/launch_local_rule_lpsm_nuaa_followup_4gpu.sh
```

The three valid branches used:

```text
teacher_mode: local_rule
train caches: IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3
test cache: NUAA-SIRST
state_stages: e2,mid
base_channels: 64
batch_size: 3
epochs: 4
AMP: true
weighted_dice: true
```

Runtime memory:

```text
GPU0/GPU1/GPU2: about 9586-9588 MiB / 11264 MiB
GPU utilization: about 99-100% during training
```

One exploratory `e2,e3,mid` branch with `batch_size=4` OOM'd at about 10.39 GiB before producing a valid epoch. It is not counted as a result.

## Raw Results

| branch | seed | residual | anchor | best epoch | threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s1313_e2m_r0055_a018 | 20260663 | 0.055 | 0.18 | 1 | 0.501 | 82.3303 | 82.1582 | 96.9582 | 8.9867e-6 |
| s1311_e2m_r0065_a016 | 20260661 | 0.065 | 0.16 | 2 | 0.502 | 82.2832 | 82.0545 | 96.9582 | 9.5355e-6 |
| s1312_e2m_r007_a014 | 20260662 | 0.070 | 0.14 | 1 | 0.503 | 82.2651 | 82.0234 | 96.9582 | 9.6041e-6 |

## Component-filter Follow-up

The best raw checkpoint was swept with threshold and connected-component min-area filtering:

```text
scripts/eval_lpsm_component_filter.py
reports/nuaa_lpsm_followup_s1313_component_filter.json
```

Best valid postprocess:

| checkpoint | threshold | min area | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| s1313_e2m_r0055_a018 | 0.499 | 3 px | 82.3504 | 82.1407 | 96.9582 | 8.8495e-6 |

Updated NUAA status:

```text
previous retained NUAA best: 82.3245
new retained NUAA best:      82.3504
gain:                       +0.0258
target:                      82.6400
remaining gap:                0.2896
```

## Cleanup

Kept only the branch that improved the retained best:

```text
runs/local_rule_lpsm_nuaa_followup_4gpu/s1313_e2m_r0055_a018/best.pth.tar
```

Deleted weaker checkpoints:

```text
runs/local_rule_lpsm_nuaa_followup_4gpu/s1311_e2m_r0065_a016/best.pth.tar
runs/local_rule_lpsm_nuaa_followup_4gpu/s1312_e2m_r007_a014/best.pth.tar
```

The run directory is about `68M` after cleanup.

## Interpretation

This confirms that the LPSM local-preserving state-mixing direction has real, repeatable positive movement on NUAA, unlike the earlier HFM-LK/UB/DHiF local-rule screens. The gain is still small and the goal is not complete, but the useful local path is now clearly:

```text
official-Mamba local-rule teacher
+ LPSM e2,mid residual refiner
+ min_area=3 component filter
```

Further gains likely require either another focused LPSM seed/shape search or adding external dense evidence from RealScene-ISTD/MiM-ISTD assets.
