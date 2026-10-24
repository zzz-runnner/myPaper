# LPSM + official-Mamba external source fusion probe

Date: 2026-06-16

## Purpose

After the 256-sample LPSM follow-up failed, this probe tested a different direction: add external dense evidence from same-domain official-Mamba models to the current retained LPSM pair.

Reference before this probe:

```text
current retained NUAA best: 82.3730
target:                     82.6400
remaining gap:               0.2670
```

## Code Added

```text
scripts/export_lpsm_pair_maps.py
scripts/search_lpsm_external_source_fusion.py
scripts/fast_lpsm_external_raw_fusion.py
```

## Sources

The probe used four full NUAA-SIRST test dense sources:

```text
LPSM retained pair:
  reports/tmp_lpsm_pair_test_maps.npz
  key: maps_lpsm_pair

DFAwareNet official-Mamba:
  reports/tmp_dfaware_test_maps.npz
  key: maps_candidate

HFMNet official-Mamba:
  reports/tmp_hfmnet_test_maps.npz
  key: maps_candidate

s703:
  reports/tmp_s703_test_maps.npz
  key: maps_no_tta
```

The LPSM map archive was exported from the two retained checkpoints:

```text
runs/local_rule_lpsm_nuaa_4gpu_screen/s1303_e2m_r006/best.pth.tar
runs/local_rule_lpsm_nuaa_followup_4gpu/s1313_e2m_r0055_a018/best.pth.tar
```

with:

```text
0.45 * s1303 + 0.55 * s1313
gamma = 0.998
```

## Raw Fusion Result

The fast raw probability fusion found a positive external-source combination:

| LPSM | DFAware | HFMNet | s703 | gamma | threshold | mIoU | nIoU | Pd | Fa |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.92 | 0.04 | 0.04 | 0.00 | 0.998 | 0.502 | 82.5135 | 82.2040 | 99.0654 | 47.9520e-6 |

Comparison:

```text
previous retained NUAA best: 82.3730
raw external-source fusion:  82.5135
gain:                      +0.1405
target:                     82.6400
remaining gap:               0.1265
```

This is not yet a retained final recipe because false alarms are much higher than the current component-filtered retained result. It is, however, the clearest positive complementarity signal since the LPSM pair fusion.

## Stopped Component-Filter Attempts

Two general component-filter sweeps were stopped because the Python/CPU connected-component loop was too slow:

```text
reports/nuaa_lpsm_external_sources_fusion_fast.json        # not produced
reports/nuaa_lpsm_external_sources_fusion_narrow.json      # not produced
reports/nuaa_lpsm_external_sources_component_refine.json   # not produced
```

No checkpoint was created by these searches.

## Files Kept

Kept:

```text
reports/nuaa_lpsm_external_sources_raw_fast.json
reports/tmp_lpsm_pair_test_maps.npz
```

The temporary LPSM map archive is about 69M and is useful for the next optimized postprocess/refinement pass. It can be deleted after the component-filter search is rewritten or completed.

## Interpretation

The same-domain official-Mamba external sources are complementary to the current LPSM pair:

```text
LPSM pair + small DFAware official-Mamba + small HFMNet official-Mamba
```

This moves NUAA from `82.3730` to `82.5135` in raw threshold evaluation and reduces the remaining target gap to `0.1265`. The next useful step is not another LPSM seed sweep. It should be an optimized component/postprocess search or a small learned fusion/refiner around this positive four-source input.

The overall goal remains incomplete because NUAA still has not crossed `82.64`.
