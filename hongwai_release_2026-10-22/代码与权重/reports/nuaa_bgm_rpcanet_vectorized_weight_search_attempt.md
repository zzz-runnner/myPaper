# NUAA BGM + RPCANet Vectorized Weight Search Attempt

Generated: 2026-06-14 UTC

## Purpose

This attempted to improve the current NUAA-SIRST best:

```text
81.0136 mIoU
```

by searching global weights over the six current complementary sources:

```text
tpc_a
tpc_b
lca
bgm_nuaa
rpcanet_sirst_s3
bg_anomaly
```

The idea was to avoid slow row-by-row full metric search by vectorizing the pixel mIoU ranking.

## Code

Added:

```text
scripts/search_bgm_rpcanet_weight_vectorized.py
```

The script:

1. collects the same full NUAA test candidate maps in memory;
2. stacks the six candidate sources into a dense matrix;
3. evaluates many weight/gamma/threshold rows by vectorized pixel mIoU;
4. runs the expensive full metric only on top-ranked rows.

Follow-up change:

```text
The full metric inside this script was replaced with an OpenCV connected-components implementation.
```

This removed the slow `skimage.measure.regionprops` path, but the search was still too slow at the current matrix/search scale because the full test prediction matrix and repeated candidate rendering remain CPU-heavy.

## Runs

Started:

```text
reports/nuaa_bgm_rpcanet_vectorized_weight_search.log
reports/nuaa_bgm_rpcanet_vectorized_weight_search_top6.log
reports/nuaa_bgm_rpcanet_vectorized_weight_search_fastmetric.log
reports/nuaa_bgm_rpcanet_vectorized_weight_search_small.log
reports/nuaa_bgm_rpcanet_vectorized_weight_search_top1.log
```

The larger runs completed candidate collection and started the vectorized search:

```text
1988 weights x 6 gammas x 11 thresholds
```

The fastmetric run used:

```text
2788 weights x 6 gammas x 11 thresholds
```

The small/top1 runs used:

```text
688 weights x 4 gammas x 6 thresholds
```

All were stopped before JSON output because they were still too slow to be useful interactively.

The runs were stopped before JSON output. No checkpoint, prediction-map archive, or large file was created.

## Decision

No new valid result from this attempt. Current NUAA-SIRST best remains:

```text
81.0136 mIoU
```

The next optimization, if continuing this branch, should cache the candidate prediction stack to a small temporary array or compute the weighted predictions in smaller vectorized chunks, then run the OpenCV full metric only once on the final best row.

## Current Bests

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.0136 | 82.64 | not passed |
