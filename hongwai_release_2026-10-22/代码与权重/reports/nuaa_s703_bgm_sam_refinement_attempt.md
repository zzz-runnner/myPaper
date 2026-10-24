# NUAA S703+BGM SAM Refinement Smoke

Generated: 2026-06-15 UTC

## Purpose

Retested SAM refinement using the current strongest NUAA probability map, instead of older cached teachers.

Prompt source:

```text
s703 HFM-LK no-TTA probability
fused = clip(0.96 * s703 - 0.04 * bgm_nuaa, 0, 1) ** 0.94
```

This is the current retained NUAA best recipe before component filtering.

## Code

Added:

```text
scripts/eval_npz_sam_refinement.py
```

It reads an exported probability-map npz, prompts SAM-B with connected components from the teacher map, and evaluates conservative `soft_rim` and aggressive `max` fusion.

## Smoke Setup

```text
dataset: NUAA-SIRST test
samples: first 30
SAM checkpoint: pretrained/sam/sam_vit_b_01ec64.pth
teacher thresholds: 0.468, 0.472, 0.480
eval thresholds: 0.468, 0.472, 0.476, 0.480
max prompts: 6
```

Output:

```text
reports/nuaa_s703_bgm_sam_refine_smoke30.json
```

## Result

Best row was still the baseline teacher map, not SAM refinement:

```text
{'kind': 'baseline', 'threshold': 0.472, 'pixAcc': 0.9427950978279114, 'mIoU': 0.8454545454545455, 'nIoU': 0.8379562087880915, 'Pd': 1.0, 'Fa': 0.0}
```

Conservative `soft_rim` accepted SAM masks but produced identical hard-mask metrics to baseline at the best thresholds.

Best aggressive `max` row:

```text
{'kind': 'sam_refined', 'fusion': 'max', 'teacher_threshold': 0.48, 'threshold': 0.468, 'accepted': 36, 'components': 44, 'pixAcc': 0.9956552982330322, 'mIoU': 0.3273809523809524, 'nIoU': 0.386930706714766, 'Pd': 0.9545454545454546, 'Fa': 0.0002379793278739496}
```

## Cleanup

Deleted temporary probability archives:

```text
reports/nuaa_s703_sam_refine_base_tmp.npz
reports/nuaa_s703_bgm_best_sam_refine_tmp.npz
```

## Conclusion

Even when prompted by the current strongest s703+BGM map, SAM-B does not improve NUAA. The conservative path does not change the final binary prediction, while aggressive fusion expands masks and is not promising for the tiny-target mIoU objective.

Do not run full-test SAM refinement for this recipe. Current NUAA best remains `81.2710` mIoU.
