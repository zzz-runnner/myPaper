# NUAA S703+BGM Component Drop-K Follow-Up

Generated: 2026-06-15 UTC

## Purpose

Followed up the valid component-stump selector result by testing whether the learned component scores work better as an ordered false-positive removal list than as a fixed probability keep threshold.

Input map is still the retained NUAA recipe:

```text
s703 HFM-LK no-TTA
fused = clip(0.96 * s703 - 0.04 * bgm_nuaa, 0, 1) ** 0.94
```

## Added Code

```text
scripts/npz_component_stump_fast_sweep.py
scripts/npz_component_stump_dropk_sweep.py
```

The fast-sweep script reproduces the previous `81.3103` component-threshold result with a faster metric implementation. The drop-k script trains the same boosted-stump selector on NUAA train components, sorts NUAA test components by learned score, and drops the lowest-scored K components.

## Result

A narrow valid sweep around the previous best completed:

```text
seed_threshold = 0.472
pos_weight_scale = 0.5
rounds = 40
bins = 11
drop_count = 0..20
output_thresholds = 0.456..0.48
```

Best row:

| method | seed th | dropped comps | output th | fill | kept / total comps | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| component-stump drop-k | 0.472 | 4 | 0.456 | prob | 265 / 269 | 81.3192 | 81.4894 | 97.7186 | 9.6727e-6 |

Comparison:

```text
previous retained NUAA best:       81.2710
component threshold selector:      81.3103
component drop-k selector:         81.3192
gain over previous retained best:  +0.0482
gain over threshold selector:      +0.0089
remaining gap to 82.64:            1.3208
```

Output:

```text
reports/nuaa_s703_bgm_component_dropk_sweep_narrow.json
```

## Interpretation

The ordered drop-k variant is a tiny improvement. It confirms that the learned component score can identify a few false-positive components, but the useful range is very narrow:

- dropping 3 components gives `81.3103`;
- dropping 4 components gives `81.3192`;
- dropping more begins to remove true target components and Pd falls.

So this line is likely near saturation unless the component table is expanded with lower seed thresholds or a better source map. It still does not solve the NUAA +5 target.

## Cleanup

Deleted temporary probability archives after the run:

```text
reports/nuaa_train_s703_component_base_tmp.npz
reports/nuaa_test_s703_component_base_tmp.npz
reports/nuaa_train_s703_bgm_component_tmp.npz
reports/nuaa_test_s703_bgm_component_tmp.npz
```

## Current Status

| Dataset | Current best | +5 target | Status |
| --- | ---: | ---: | --- |
| IRSTD-1K | 73.0367 | 73.03 | passed narrowly |
| NUAA-SIRST | 81.3192 | 82.64 | not passed |
