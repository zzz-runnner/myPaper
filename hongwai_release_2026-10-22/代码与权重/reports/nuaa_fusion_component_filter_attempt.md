# NUAA Fusion Component Filter Attempt

Generated: 2026-06-15 UTC

## Purpose

NUAA-SIRST remains below the requested +5 mIoU target. The current best is the six-source BGM/RPCANet fusion:

```text
tpc_a 0.1900
tpc_b 0.0960
lca 0.1044
bgm_nuaa 0.4005
rpcanet_sirst_s3 0.1750
bg_anomaly 0.0342
gamma 0.95
threshold 0.52
```

This attempt checked whether train-selected connected-component filtering can remove residual false alarms from that fused probability map.

## Code

Added:

```text
scripts/search_nuaa_fusion_component_filter.py
```

The script:

- rebuilds the current six-source NUAA candidate maps;
- renders the current best weighted fusion;
- sweeps connected-component filters over area, max side length, and fill ratio;
- selects rows on the multi-dataset train/fit side with a small false-alarm penalty;
- evaluates only the fit-selected top rows on NUAA-SIRST test;
- writes only a small JSON file and no checkpoint or prediction archive.

## Tiny Probe

Output:

```text
reports/nuaa_fusion_component_filter_tiny.json
```

Scale:

```text
fit samples: 20 per dataset from IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3
eval samples: first 20 NUAA-SIRST test images
threshold: 0.52
gamma: 0.95
min_area: 1,2,3,4,5,6
max_area: 0
max_side: 0,16,24,32
min_fill: 0.0
```

Selected-by-fit row:

| kind | threshold | min area | max side | fit mIoU | test mIoU | test Pd | test Fa |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 0.52 | 1 | 0 | 71.0960 | 83.4286 | 100.0000 | 0.0000e-6 |

Top evaluated rows showed no useful positive signal from filtering:

| kind | min area | max side | test mIoU |
| --- | ---: | ---: | ---: |
| raw | 1 | 0 | 83.4286 |
| component filter | 2 | 0 | 83.4286 |
| component filter | 5 | 0 | 83.4286 |
| component filter | 3 | 0 | 83.4286 |
| component filter | 1 | 24 | 65.2510 |
| component filter | 1 | 32 | 65.2510 |

## Stopped Larger Smoke

An initial 80-sample smoke with a broader parameter grid was stopped manually because it was CPU-bound in repeated connected-component sweeps and produced no JSON output. It did not create any large artifact.

## Decision

Do not expand this component-filter route to a full NUAA run in its current form.

Reason:

```text
The fit-selected row is the unfiltered raw current-best fusion. Simple area/side/fill filtering does not add a new valid signal, and shape limits can destroy true targets on the tiny probe.
```

Current reliable bests therefore remain:

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.0269 | 82.64 | not passed |

Next useful progress still requires a stronger new dense candidate source or additional NUAA-like labeled data, rather than more simple postprocessing on the existing six-source fusion.
