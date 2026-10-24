# DCCS_hf External Candidate Attempt

Generated: 2026-06-15 UTC

## Purpose

Re-screened the local `external/DCCS_hf` source because it contains official-looking NUAA-SIRST and IRSTD-1K weights:

```text
external/DCCS_hf/weight/nuaa_dccs.pkl
external/DCCS_hf/weight/irstd1k_dccs.pkl
```

The goal was to see whether DCCS could provide a new dense prediction source for the current fusion/refiner pipeline, instead of continuing to tune SCTransNet or the existing `s703` output.

## What Worked

Model import and checkpoint loading work in the current environment:

```text
DCCS import: ok
nuaa_dccs.pkl load: ok, strict load possible
irstd1k_dccs.pkl load: ok, strict load possible
```

Added evaluator:

```text
scripts/eval_dccs_hf_fullsplit.py
```

It supports:

```text
official: run the full DCCS model
zero_vssm: replace the missing VSSM/selective-scan branch with zero feature maps
```

## Official Mode Blocker

Official forward fails because the environment still lacks the selective-scan/Mamba operator used by the VSSM branch:

```text
NameError: name 'selective_scan_fn' is not defined
```

This confirms DCCS_hf belongs to the same blocked Mamba/selective-scan family as HFMNet/DFAwareNet under the current env.

## Zero-VSSM Fallback Test

A pragmatic fallback was tested: keep the DCCS CNN/GLFA decoder and official weights, but replace VSSM outputs with zero tensors. This is not the official DCCS result; it is only a source-ablation test to see whether the non-Mamba branch is useful as a dense map.

Full-split results:

| Dataset | mode | best postprocess | mIoU | nIoU | Pd | Fa |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| NUAA-SIRST | zero_vssm | threshold 0.95 + min_area 2 | 3.3313 | 15.0968 | 82.8897 | 4726.5988e-6 |
| IRSTD-1K | zero_vssm | threshold 0.05 | 4.3777 | 5.8618 | 10.1010 | 2.8468e-6 |

Outputs:

```text
reports/dccs_hf_zero_vssm_nuaa_eval.json
reports/dccs_hf_zero_vssm_irstd_eval.json
```

No probability maps were saved.

## Cleanup

Deleted the failed official smoke JSON. No large checkpoint or map archive was produced.

## Conclusion

DCCS_hf cannot currently contribute a useful dense source:

1. Full official mode is blocked by missing `selective_scan_fn`.
2. The zero-VSSM fallback is numerically unusable and far below existing baselines.

Therefore this candidate should not be fused into the current `s703+BGM+drop4` NUAA recipe. The useful path remains either obtaining compatible selective-scan/Mamba wheels/env, or adding a genuinely new dataset/source such as RealScene-ISTD.

Current reliable status remains:

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.3192 | 82.64 | not passed |
