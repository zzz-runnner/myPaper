# NUAA S703 Multisource Refusion Narrow Search

Generated: 2026-06-15 UTC

## Purpose

Tested whether the current best s703+BGM-negative recipe could be improved by adding small RPCANet, weighted_raw, and bg_anomaly terms.

Current retained NUAA best before this test:

```text
81.2710 mIoU
s703 no-TTA + clip(0.96*s703 - 0.04*bgm_nuaa,0,1)**0.94
threshold = 0.472
connected-component min_area = 2 px
```

## Search

Added vectorized search script:

```text
scripts/search_s703_multisource_refusion_vectorized.py
```

Narrow grid searched around the known useful BGM-negative point:

```text
w_bgm in [-0.045, -0.04, -0.035]
w_rpc in [0, 0.005, 0.01]
w_weighted in [0, 0.005, 0.01]
w_bg_anom in [0, -0.005]
gamma in [0.93, 0.94, 0.95]
threshold in [0.468, 0.472, 0.476, 0.480]
```

Best threshold-only row:

```text
{'kind': 'multisource', 'w_bgm': -0.04, 'w_rpc': 0.01, 'w_weighted': 0.005, 'w_bg_anom': -0.005, 'gamma': 0.94, 'threshold': 0.48, 'mIoU_fast': 0.8125, 'tp': 7514, 'fp': 860, 'fn': 874, 'mIoU': 0.8125, 'nIoU': 0.8123516394813138, 'Pd': 0.9953271028037384, 'Fa': 5.899673446214667e-05}
```

That is `81.2500` mIoU, below the retained `81.2710`.

## Component Filter Follow-Up

Generated a temporary fused-map archive for the best row and ran min-area filtering.

Best component-filtered row:

```text
{'kind': 'min_area', 'threshold': 0.48, 'min_area': 2, 'pixAcc': 0.8955650925636292, 'mIoU': 0.812547322877231, 'nIoU': 0.8123695304585277, 'Pd': 0.9771863117870723, 'Fa': 8.300703337115984e-06}
```

That is `81.2547` mIoU, still below the retained `81.2710`.

## Cleanup

Deleted temporary probability archives:

```text
reports/nuaa_s703_vectorized_refusion_maps_tmp.npz
reports/nuaa_s703_multisource_top_tmp.npz
```

## Conclusion

Small positive RPCANet/weighted_raw additions and bg_anomaly suppression do not improve the current NUAA best. They mostly increase high-threshold false positives, and component filtering recovers some false alarms but not enough to beat the existing BGM-negative refusion recipe.

Current NUAA best remains:

```text
81.2710 mIoU
```

Gap to target remains:

```text
82.64 - 81.2710 = 1.3690 mIoU
```
