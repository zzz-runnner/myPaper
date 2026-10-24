# Contrast/Spectral Candidate Attempt

Generated: 2026-06-13 UTC

## Purpose

This branch tested a non-SCTransNet candidate source inspired by traditional infrared small-target priors:

- multi-scale center-surround local contrast;
- top-hat small bright object enhancement;
- DoG/high-frequency cues;
- spectral-residual saliency;
- top-K small connected-component filtering.

The goal was to create new candidates beyond the current teacher/PConv pool, then feed them into the existing component learner.

## Code Added

```text
scripts/eval_component_fusion_learner.py
scripts/eval_contrast_candidate_diagnostic.py
```

New component-fusion flags:

```text
--contrast_candidates
--contrast_topk
--contrast_threshold
```

## Smoke Result

Output:

```text
reports/contrast_candidate_smoke_irstd.json
```

On a 5-sample IRSTD smoke split, the new candidate path ran correctly and wrote JSON, but the best valid row was still the existing baseline teacher region rather than the new contrast candidate.

Best smoke valid:

| Dataset | Best row | mIoU | Pd | Fa |
|---|---|---:|---:|---:|
| IRSTD-1K smoke | baseline `baseq`, out 0.40 | 75.80 | 88.89 | 13.73 |

## NUAA Diagnostic

Output:

```text
reports/contrast_candidate_diag_nuaa.json
```

Best standalone contrast candidate:

| Dataset | Top-K | Threshold | mIoU | Pd | Fa |
|---|---:|---:|---:|---:|---:|
| NUAA-SIRST test | 4 | 0.82 | 10.97 | 66.16 | 2232.89 |

This is far too weak to be a useful new candidate source for the NUAA +5 target.

## IRSTD Diagnostic

The full IRSTD diagnostic was stopped because the contrast/spectral candidate extractor was too slow on this data even after precomputing score maps. A 50-sample reduced run was also stopped after about a minute without output.

This is a negative engineering signal: the route is both computationally awkward and not promising enough from the NUAA/IRSTD smoke evidence to justify a full four-GPU component-fusion run.

## Decision

Do not continue this exact contrast/spectral candidate branch. It can remain as code for reference, but it should not consume more GPU/CPU time unless rewritten much more cheaply and paired with a stronger learned selector.

Current reliable best remains unchanged:

| Dataset | Valid best | mIoU |
|---|---|---:|
| IRSTD-1K | dual-PConv average fusion region | 72.43 |
| NUAA-SIRST | background-normality component fusion | 79.34 |
