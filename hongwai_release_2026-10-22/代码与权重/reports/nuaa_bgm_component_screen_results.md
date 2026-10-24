# NUAA Official BGM Component Screen Results

Generated: 2026-06-14 UTC

## Purpose

After evaluating official BGM checkpoints as standalone dense sources, this run tested whether the best BGM checkpoint adds complementary components to the existing NUAA candidate pool.

This is still aligned with the non-plain-SCTransNet route:

```text
SCTransNet-family teachers + official BGM background-forgetting checkpoint + background-normality candidate
```

The goal was to check whether official BGM raises the NUAA candidate-pool oracle beyond the previous best region.

## Code Change

Extended:

```text
scripts/eval_component_fusion_learner.py
```

with:

```text
--official_bgm_spec name|norm_dataset|weight|threshold
```

The BGM integration loads the official `BasicUNet_plus` checkpoint directly and adds its binarized prediction map as a candidate source.

Smoke output:

```text
reports/bgm_component_smoke.json
```

## Candidate Pool

Models:

```text
tpc_a
tpc_b
lca
bgm_nuaa
bg_anomaly
```

Official BGM checkpoint:

```text
external/BGM/checkpoints_download/weights/BasicUNet_plus-2024-09-20-21-17-03-NUAA-SIRST/best_mIoU_on_NUAA-SIRST.pth.tar
```

Fitting data:

```text
IRSTD-1K train
NUAA-SIRST train
NUDT-SIRST train
SIRST3 train
```

Evaluation:

```text
NUAA-SIRST test
```

## Full Attempt Note

A full uncapped run was launched first:

```text
reports/nuaa_tpc_lca_bgm_bg_component.log
```

It completed full candidate collection but became too slow in the final component search stage because the component count was high. It was stopped before JSON output to avoid wasting time. No large artifact was produced.

## Capped Screen Result

Output:

```text
reports/nuaa_tpc_lca_bgm_bg_component_capped.json
reports/nuaa_tpc_lca_bgm_bg_component_capped.log
```

Caps:

```text
max_fit_samples = 220 per dataset
max_eval_samples = full NUAA-SIRST test
max_components_per_image = 48
```

Best valid result:

| Row | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: |
| baseline average of candidate maps, output threshold 0.50 | 79.5487 | 80.0684 | 96.5779 | 29.0182e-6 |

Best test oracle, not valid:

| Row | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: |
| component oracle, seed 0.45, avg fill | 80.6827 | 80.7211 | 96.9582 | 7.4089e-6 |

## Interpretation

Official BGM adds a small amount of complementarity:

```text
previous strongest NUAA oracle region: about 80.49
BGM capped oracle: 80.6827
```

The best valid result also edges above the previous 79.34 region:

```text
79.34 -> 79.5487
```

However, this is still far below the requested target:

```text
NUAA target = 82.64
gap from BGM oracle = 82.64 - 80.6827 = about 1.96 mIoU
gap from BGM valid = 82.64 - 79.5487 = about 3.09 mIoU
```

So official BGM is useful as a minor candidate-source addition, but it is not enough to solve the NUAA blocker. The next useful source still needs to raise the candidate-pool oracle into the 82.64+ region.

## Cleanup

No checkpoint, prediction map archive, or large file was produced. The retained artifacts are small JSON/log/report files plus the 48 MB official BGM checkpoints already recorded in:

```text
reports/bgm_official_checkpoint_eval.md
```
