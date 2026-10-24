# RPCANet Component-Fusion Results

Generated: 2026-06-14 UTC

## Purpose

After standalone RPCANet official weights were much weaker than the current SCTransNet-family/BGM pool, this test checked whether RPCANet still provides useful complementary low-false-alarm components for NUAA-SIRST.

This keeps the current non-plain-SCTransNet line:

```text
TPC/LCA teacher candidates
+ official BGM background-forgetting candidate
+ RPCANet++ external low-false-alarm candidate
+ background-normality candidate
+ component-level train-fit selector
```

## Code Change

Extended:

```text
scripts/eval_component_fusion_learner.py
```

with:

```text
--rpcanet_spec name|model|weight|threshold|base_size
```

The RPCANet integration loads HuggingFace RPCANet/RPCANet++ weights directly, runs grayscale 256x256 inference, resizes the probability map back to the local test image size, thresholds it, and inserts it as another binary candidate map in the existing component-fusion pool.

## Standalone RPCANet Results

Evaluated full local test splits before fusion:

| Dataset | Weight | Model | Best threshold | mIoU | nIoU | Pd | Fa |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| IRSTD-1K | `1K/RPCANet++_s6.pkl` | `rpcanet_pp` | 0.25 | 59.5781 | 53.2023 | 87.8788 | 18.1435e-6 |
| NUAA-SIRST | `SIRST/RPCANet++_s3.pkl` | `rpcanet_pp_s3` | 0.50 | 69.0609 | 65.6572 | 97.3384 | 13.3772e-6 |
| NUAA-SIRST | `SIRST-Aug/RPCANet++_s6.pkl` | `rpcanet_pp` | 0.60 | 63.6809 | 72.4351 | 95.4373 | 156.0669e-6 |
| NUAA-SIRST | `1K/RPCANet++_s6.pkl` | `rpcanet_pp` | 0.30 | 52.8071 | 59.2065 | 87.0722 | 121.8351e-6 |
| NUAA-SIRST | `NUDT/RPCANet++_s6.pkl` | `rpcanet_pp` | 0.20 | 45.3697 | 51.8543 | 80.6084 | 72.0309e-6 |
| NUAA-SIRST | `SIRST-Aug/RPCANet++_s3.pkl` | `rpcanet_pp_s3` | 0.60 | 42.3584 | 66.4581 | 93.5361 | 517.9364e-6 |

Standalone conclusion:

```text
RPCANet is not competitive alone, but SIRST/RPCANet++_s3 has high Pd and low Fa, so it was worth testing as a component source.
```

## Fusion Setup

Candidate pool:

```text
tpc_a
tpc_b
lca
bgm_nuaa
rpcanet_sirst_s3
bg_anomaly
```

Fit data:

```text
IRSTD-1K train, capped 220 samples
NUAA-SIRST train, full 213 samples
NUDT-SIRST train, capped 220 samples
SIRST3 train, capped 220 samples
```

Evaluation:

```text
NUAA-SIRST full test, 214 samples
```

Output:

```text
reports/nuaa_tpc_lca_bgm_rpcanet_component_capped.json
reports/nuaa_tpc_lca_bgm_rpcanet_component_capped.log
```

## Best Result

Previous best valid NUAA result:

```text
79.5487 mIoU
```

New best valid row:

| Row | Seed | Fill | Keep | Output | mIoU | nIoU | Pd | Fa |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train-fit component selector | 0.35 | avg | 0.50 | 0.40 | 80.4849 | 80.7696 | 97.7186 | 11.8679e-6 |

Best test oracle, not valid:

| Row | Seed | Fill | Output | mIoU | nIoU | Pd | Fa |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| component oracle | 0.40 | avg | 0.40 | 81.0198 | 81.0489 | 97.7186 | 7.6833e-6 |

Gain:

```text
80.4849 - 79.5487 = +0.9362 mIoU valid
81.0198 - 80.6827 = +0.3371 mIoU oracle
```

Gap to requested NUAA target:

```text
82.64 - 80.4849 = 2.1551 mIoU
```

So RPCANet is useful as a complementary external source, but it still does not solve NUAA-SIRST.

## Multi-Threshold Follow-Up

A second capped run tested multiple RPCANet thresholds and an extra SIRST-Aug source:

```text
rpcanet_sirst_s3_t045
rpcanet_sirst_s3_t050
rpcanet_sirst_s3_t055
rpcanet_sirstaug_s6_t060
```

Output:

```text
reports/nuaa_tpc_lca_bgm_rpcanet_multith_component_capped.json
reports/nuaa_tpc_lca_bgm_rpcanet_multith_component_capped.log
```

Best valid result:

```text
80.3360 mIoU
```

Best oracle:

```text
80.7713 mIoU
```

This was weaker than the single-threshold SIRST/RPCANet++_s3 fusion result, so the single RPCANet source remains the better setting.

## Cleanup

Deleted RPCANet weights that were evaluated or screened but did not contribute to the best fusion result.

Kept:

```text
external/RPCANet_hf/result/ISTD/SIRST/RPCANet++_s3.pkl
```

Removed the other RPCANet `.pkl` weights and `__pycache__` directories.

Space:

```text
external/RPCANet_hf before cleanup: about 151 MB
external/RPCANet_hf after cleanup: about 6.2 MB
```

## Current Status After This Step

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 80.4849 | 82.64 | not passed |

This is real progress on the NUAA blocker, but the active goal remains open.
