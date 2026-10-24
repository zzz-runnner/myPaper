# SIRST-UAVB NUAA Refiner Early Stop

Generated: 2026-06-14 UTC

## Purpose

NUAA-SIRST remains below the requested +5 mIoU target. This run tested whether adding the newly converted SIRST-UAVB dataset to the full multi-dataset train pool improves NUAA-focused hybrid refiner models.

Train pool:

```text
IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3, SIRST-UAVB
```

Target/eval dataset:

```text
NUAA-SIRST
```

SIRST-UAVB was converted to the project layout:

```text
datasets/SIRST-UAVB
train: 2400
test: 600
```

## Runs

Launched 4 GPU branches:

| GPU | Method | Model | Batch |
| ---: | --- | --- | ---: |
| 0 | NUAA-UAVB-LCA | sct_lca_refiner | 56 |
| 1 | NUAA-UAVB-TCD | sct_tcd_decouple_refiner | 56 |
| 2 | NUAA-UAVB-BNG | sct_bng_featup_refiner | 28 |
| 3 | NUAA-UAVB-APConv | sct_apconv_tpc_refiner | 32 |

Peak observed memory after batch increase:

```text
GPU0 about 7.7G
GPU1 about 8.7G
GPU2 about 9.0G
GPU3 about 8.3G
```

## Early Results

| Method | Best early mIoU | Epochs observed | Status |
| --- | ---: | ---: | --- |
| NUAA-UAVB-LCA | 79.1208 | 3 | stopped |
| NUAA-UAVB-TCD | 79.1145 | 3 | stopped |
| NUAA-UAVB-BNG | 79.1315 | 2 | stopped |
| NUAA-UAVB-APConv | 79.0976 | 2 | stopped |

Current reliable NUAA best remains:

```text
79.34 mIoU
```

## Interpretation

Adding SIRST-UAVB to the SCTransNet-family refiner training pool did not produce a positive early signal. All four variants stayed around 79.10-79.13 mIoU and below the existing 79.34 background-normality component result.

This suggests that simply adding SIRST-UAVB to frozen-SCTransNet/refiner fine-tuning is not the right route. The next SIRST-UAVB use should be as a genuinely new candidate/source model, not as more data for the same refiner family.

## Cleanup

Stopped tmux session:

```text
sirst_uavb_nuaa_4gpu
```

Deleted failed checkpoint and tensorboard files. Kept only small logs and this report:

```text
runs/sirst_uavb_nuaa_4gpu 136K
```
