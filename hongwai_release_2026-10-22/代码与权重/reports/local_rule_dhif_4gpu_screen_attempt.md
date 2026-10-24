# NUAA local-rule DHiF 4-GPU screen

Date: 2026-06-16

## Purpose

This run tested the DHiF dynamic high-frequency module from the supplied method directions on top of the current strongest NUAA official-Mamba local-rule teacher.

The goal was to change the dense map itself by separating target-like high-frequency evidence from clutter high-frequency evidence, not to continue threshold or component tuning.

Reference before this run:

```text
current NUAA best: 82.2928
target:            82.6400
remaining gap:      0.3472
```

## Launch

Added:

```text
scripts/run_local_rule_dhif_one.sh
scripts/launch_local_rule_dhif_nuaa_4gpu_screen.sh
```

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
patch_size: 512
thresholds: 0.488..0.512
weighted_dice: true
AMP: true
```

Main branch runtime:

```text
GPU0 e1 branch:       about 9666 MiB / 11264 MiB, 100% util
GPU1 e1,e2 branch:   about 10788-10804 MiB / 11264 MiB, 100% util
GPU3 e2 branch:      about 9666 MiB / 11264 MiB, high util
```

The initial `e1,e2,e3` branch on GPU2 OOM'd at about 10.35 GiB allocated. It was replaced by a smaller safety branch, which ran at about 6970 MiB and is therefore treated as supplementary evidence rather than a 9GB+ branch.

## Results

| branch | stages | seed | residual | hard clutter | miss | best epoch | threshold | mIoU | nIoU | Pd | Fa |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| s1204_e2_r006_hc10_m12 | e2 | 20260644 | 0.060 | 1.00 | 1.20 | 1 | 0.500 | 82.2671 | 82.0250 | 96.9582 | 9.6041e-6 |
| s1201_e1_r004_hc04_m08 | e1 | 20260641 | 0.040 | 0.40 | 0.80 | 1 | 0.496 | 82.2651 | 82.0234 | 96.9582 | 9.6041e-6 |
| s1202_e12_r005_hc06_m10 | e1,e2 | 20260642 | 0.050 | 0.60 | 1.00 | 1 | 0.498 | 82.2651 | 82.0234 | 96.9582 | 9.6041e-6 |
| s1205_e13_r004_safe | e1,e3 | 20260645 | 0.040 | 0.70 | 0.80 | 1 | 0.496 | 82.2651 | 82.0234 | 96.9582 | 9.6041e-6 |

OOM branch:

```text
s1203_e123_r004_hc08_m06
stages: e1,e2,e3
status: CUDA OOM before first valid epoch
```

Comparison:

```text
current retained NUAA best: 82.2928
best local-rule DHiF row:   82.2671
delta:                     -0.0257
```

## Cleanup

All valid DHiF rows were below the retained NUAA best, so their checkpoints were deleted:

```text
runs/local_rule_dhif_nuaa_4gpu_screen/*/best.pth.tar
```

The run directory now keeps only logs and summaries and is about `84K`.

## Interpretation

This tests the supplied "dynamic high-frequency target/clutter decoupling" idea on the current strongest local-rule dense teacher. It is stable for e1/e2-style configurations and uses the 2080 Ti memory target on the main branches, but it again lands in the same 82.26 region as local-rule HFM-LK and local-rule UB.

Do not continue this exact DHiF local-rule setup by only adding epochs or nearby weights. The NUAA bottleneck is increasingly unlikely to be solved by another bounded residual refiner around the same local-rule map. The next useful step should bring a stronger external dense source or new data distribution into the candidate pool.
