# Multi-Dataset Component Fusion Learner Results

Generated: 2026-06-13 UTC

This is the full multi-dataset version of the component-level teacher fusion learner. The component keeper is fitted on the union of all four training datasets: `IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3`, then evaluated separately on IRSTD-1K and NUAA-SIRST test splits.

## Valid Results

| Dataset | Teacher set | Best setting | mIoU | Pd | Fa |
|---|---|---|---:|---:|---:|
| IRSTD-1K | TPC teachers | seed 0.45, keep 0.15, fill max, output 0.5 | 69.16 | 90.91 | 7.61 |
| IRSTD-1K | TPC + fixed refiner | seed 0.45, keep 0.15, fill avg, output 0.45 | 69.08 | 90.57 | 7.65 |
| NUAA-SIRST | LCA + TCD teachers | seed 0.45, keep 0.15, fill avg, output 0.5 | 79.27 | 96.58 | 16.46 |
| NUAA-SIRST | TPC + LCA teachers | seed 0.45, keep 0.15, fill avg, output 0.5 | 79.28 | 96.58 | 16.46 |

## Comparison

| Dataset | Previous component single-domain best | Multi-dataset best | Interpretation |
|---|---:|---:|---|
| IRSTD-1K | 69.17 | 69.16 | tied/slightly lower; extra domains did not improve the keeper |
| NUAA-SIRST | 79.27 | 79.28 | tiny +0.01; not meaningful |

## Test Oracle Reference Only

The oracle rows use test-set GT to decide component keeps, so they are not valid final method results.

| Dataset | Teacher set | Oracle mIoU | Pd | Fa |
|---|---|---:|---:|---:|
| IRSTD-1K | TPC teachers | 70.33 | 93.60 | 3.66 |
| IRSTD-1K | TPC + fixed refiner | 70.41 | 93.60 | 3.66 |
| NUAA-SIRST | LCA + TCD teachers | 80.27 | 96.58 | 8.09 |
| NUAA-SIRST | TPC + LCA teachers | 80.27 | 96.58 | 8.09 |

## Conclusion

Full multi-dataset component learning is aligned with the requested full-data setting, but it does not unlock the requested +5 mIoU. It confirms that the available teacher candidates contain roughly 1-2 mIoU of component-selection headroom, not a five-point jump, unless a much more diverse candidate generator is introduced. No large run directory was kept; only small JSON/log/report evidence remains.
