# Dual PConv Component Gate Attempt

Generated: 2026-06-13 UTC

## Purpose

After fixed weighted averaging failed, this attempt tested a more local/non-linear IRSTD-1K route:

```text
baseq + fixed refiner + PConv-D05 + PConv-P43
```

The idea was to use full multi-dataset train data to gate only the PConv components that might rescue targets missed by SCTransNet-family teachers.

## Code

Added:

```text
scripts/eval_dual_pconv_component_gate.py
```

The script collects teacher + dual-PConv maps, applies component-level PConv gates using local shape/contrast/teacher-support rules, and can select parameters by train fast nIoU before evaluating TopK rows.

## What Completed

A small smoke run completed:

```text
reports/dual_pconv_gate_smoke_irstd.json
```

On only 4 train / 4 test samples, the gate improved over the tiny-sample teacher baseline, but this is not a valid dataset result.

## Full Attempts

Two full multi-dataset attempts were launched with:

```text
IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3
```

Both successfully completed forward collection over the full training pool and IRSTD-1K test split, but then became too slow in the parameter rendering / ranking stage:

1. full TopK metric attempt: stopped after collection and prolonged CPU post-processing.
2. smaller `--fast_only` grid: also stopped after collection because per-parameter full-map rendering over all train samples was still too slow.

No valid full JSON metric was produced.

## Current Valid Best Remains Unchanged

| Dataset | Best valid method | mIoU |
|---|---|---:|
| IRSTD-1K | dual-PConv average fusion from component MLP selector fast run | 72.41 |
| NUAA-SIRST | background-normality component fusion | 79.34 |

## Decision

The local PConv gating concept is still aligned with the remaining IRSTD gap, but this implementation is not efficient enough. Continuing this exact script would waste CPU time.

If this direction is resumed, the next implementation should precompute component-level features once and score parameter rows over component tables, instead of re-rendering full prediction maps for every parameter combination.

No large artifacts or checkpoints were created. The failed full attempts were stopped, and the server is idle after stopping them.
