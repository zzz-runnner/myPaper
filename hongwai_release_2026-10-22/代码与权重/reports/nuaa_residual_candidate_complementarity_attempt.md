# NUAA residual candidate complementarity diagnostic

Date: 2026-06-16

## Purpose

The external-Mamba residual pixel calibrator almost tied the retained NUAA best but did not exceed it. This diagnostic checked whether the two near-identical masks still have usable complementarity:

- retained current best residual pixel mask;
- external-Mamba residual pixel mask.

The goal was to see whether a train-selected image-level selector could choose between them and improve NUAA-SIRST test mIoU.

## Script

Added/updated:

```text
scripts/diagnose_nuaa_residual_candidate_complementarity.py
```

The script trains both tiny residual calibrators in memory, predicts train/test maps, binarizes with their best known operating points, and evaluates:

- base mask;
- external mask;
- union;
- intersection;
- per-image test oracle;
- simple train-selected one-dimensional rule;
- small train-fold logistic selector.

No checkpoint or probability map archive is saved.

## Results

| row | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: |
| retained base residual | 82.2928 | 82.1643 | 96.9582 | 8.9867e-6 |
| external-Mamba residual | 82.2813 | 82.0330 | 96.9582 | 9.3297e-6 |
| union | 82.2813 | 82.0330 | 96.9582 | 9.3297e-6 |
| intersection | 82.2928 | 82.1643 | 96.9582 | 8.9867e-6 |
| per-image test oracle | 82.3717 | 82.2061 | 96.9582 | 9.2611e-6 |
| train-selected 1D rule | 82.2896 | 82.1347 | 96.9582 | 8.9867e-6 |
| train-fold logistic selector | 82.2928 | 82.1643 | 96.9582 | 8.9867e-6 |

Best non-oracle valid result remains the retained base residual:

```text
82.2928
```

The per-image test oracle shows that the two masks have a small amount of complementarity:

```text
oracle gain over base: +0.0789 mIoU
oracle score:          82.3717
```

But even this test-set oracle is still below the NUAA +5 target:

```text
target:         82.6400
oracle gap:      0.2683
```

## Interpretation

This line should not be pushed further by only selecting between the current residual mask and the external-Mamba residual mask. There is some test-only complementarity, but:

1. the train-selected selectors do not generalize into a gain;
2. the oracle upper bound for this two-mask choice is still below the required NUAA target.

The next useful NUAA attempt needs a new dense source or a model that changes recall/shape quality more substantially, not another selector over these two residual masks.

## Cleanup

Only compact JSON diagnostics were saved:

```text
reports/nuaa_residual_candidate_complementarity.json
reports/nuaa_residual_candidate_complementarity_logistic.json
```

No large checkpoint or map archive was produced.
