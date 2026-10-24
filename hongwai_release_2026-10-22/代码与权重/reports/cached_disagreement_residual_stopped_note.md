# Cached Disagreement Residual Stopped Note

Generated: 2026-06-14 UTC

## Purpose

This was intended as a targeted follow-up to the dual-PConv IRSTD-1K best result. Instead of re-rendering hard components, it preserves the dense average probability map and tries small local residual changes in PConv/teacher disagreement regions.

Script added:

```text
scripts/eval_cached_disagreement_residual.py
```

## What Happened

Two CPU-only attempts were launched over the full cached train pool:

```text
IRSTD-1K train
NUAA-SIRST train
NUDT-SIRST train
SIRST3 train
```

Both were stopped because the current implementation recomputes dense maps and metrics for too many candidates over too many pixels. The runs consumed CPU for minutes without writing a result JSON.

A smaller one-off direct diagnostic over IRSTD-1K test was also stopped for the same reason before producing a JSON file.

No GPU time and no large artifacts were consumed. No partial JSON remains.

## Interpretation

This is an implementation-efficiency failure, not evidence that local residual calibration is mathematically impossible. But combined with previous evidence:

- full pixel MLP calibration failed badly,
- hard component insertion/suppression failed,
- componentized avg-gating oracle falls below the dense avg baseline,

the current practical conclusion is that small cached fusion rules are unlikely to be the fastest path to the remaining target.

The next useful work should prioritize a genuinely new dense candidate source, especially for NUAA-SIRST, where the current best is still far from the +5 target.
