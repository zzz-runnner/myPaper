# DCCS_hf Torch Selective-Scan Fallback Full Attempt

Generated: 2026-06-15 UTC

## Purpose

After the first DCCS_hf screen showed that official mode was blocked by missing `selective_scan_fn`, this attempt implemented a pure PyTorch selective-scan fallback so the full DCCS weights could run without Mamba/custom CUDA.

This is a source-unlocking attempt, not SCTransNet parameter tuning.

## Code Added

Updated evaluator:

```text
scripts/eval_dccs_hf_fullsplit.py
```

Added mode:

```text
--mode torch_ref
```

The fallback implements the SSM recurrence used by DCCS:

```text
x_t = A_t * x_{t-1} + q_t
y_t = C_t * x_t + D * u_t
```

using a vectorized `cumprod/cumsum` scan. This is slower and less numerically polished than the official fused selective-scan kernel, but it lets the full model forward pass run in the current environment.

## Smoke Test

A one-sample NUAA smoke passed with the full DCCS model and official NUAA weight:

```text
mode=torch_ref forward: ok
```

The temporary smoke JSON was deleted after full evaluation.

## Full-Split Results

No probability maps were saved.

| Dataset | weight | mode | best postprocess | mIoU | nIoU | Pd | Fa |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| NUAA-SIRST | `nuaa_dccs.pkl` | torch_ref | threshold 0.525 + min_area 6 | 65.1010 | 66.9232 | 87.0722 | 60.0943e-6 |
| IRSTD-1K | `irstd1k_dccs.pkl` | torch_ref | threshold 0.575 + min_area 3 | 67.6581 | 59.6258 | 92.5926 | 7.8192e-6 |

Outputs:

```text
reports/dccs_hf_torch_ref_nuaa_eval.json
reports/dccs_hf_torch_ref_irstd_eval.json
```

## Interpretation

The fallback successfully unlocks the full DCCS forward path, but the resulting maps are not strong enough under the local evaluation protocol:

```text
NUAA current best: 81.3192
DCCS torch_ref NUAA: 65.1010

IRSTD current best: 73.0367
DCCS torch_ref IRSTD: 67.6581
```

This is too weak to justify saving maps or fusing into the current recipe. The gap is large enough that small preprocessing tweaks are unlikely to make DCCS competitive here without the official environment/protocol.

## Cleanup

Deleted:

```text
reports/dccs_hf_torch_ref_smoke.json
```

No large map archives or checkpoints were created.

## Current Reliable Status

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.3192 | 82.64 | not passed |
