| Dataset | Latest epoch | Latest loss | Best epoch | Best mIoU | README mIoU | Delta | Best Pd | README Pd | Best Fa | README Fa |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SIRST3 | 1000 | 0.002862 | 678 | 82.96 | - | - | 96.94 | - | 7.83 | - |
| NUDT-SIRST | 1000 | 0.002630 | 547 | 93.76 | 94.09 | -0.33 | 98.94 | 98.62 | 4.37 | 4.29 |
| IRSTD-1K | 1000 | 0.003003 | 500 | 67.32 | 68.03 | -0.71 | 93.60 | 93.27 | 13.61 | 10.74 |
| SIRST-v1 / NUAA-SIRST | 1000 | 0.002671 | 823 | 77.52 | 77.50 | +0.02 | 96.20 | 96.95 | 18.04 | 13.92 |

SIRST3 has no single README target in the authors' table; after that run finishes, its best checkpoint is evaluated on SIRST-v1, NUDT-SIRST, and IRSTD-1K for the paper-style comparison.
