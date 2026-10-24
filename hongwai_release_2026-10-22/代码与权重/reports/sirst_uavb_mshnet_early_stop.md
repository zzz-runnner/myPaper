# SIRST-UAVB MSHNet/P43 Early Stop

Generated: 2026-06-14 UTC

## Purpose

After SIRST-UAVB did not help frozen-SCTransNet-family NUAA refiners, this run tested a more independent same-domain source: official MSHNet-style RGB training and the P43/PConv first-block variant.

Train pool:

```text
IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3, SIRST-UAVB
```

Target/eval dataset:

```text
NUAA-SIRST
```

Input/eval mode:

```text
official_mshnet, base_size=512, patch_size=512
```

## Runs

| GPU | Method | Model | Batch | Peak memory signal |
| ---: | --- | --- | ---: | --- |
| 0 | NUAA-UAVB-MSH3-A | mshnet3_sls | 10 | about 8.1G |
| 1 | NUAA-UAVB-P43-A | mshnet_p43_sls | 10 | about 10.8G |
| 2 | NUAA-UAVB-MSH3-B | mshnet3_sls | 12 | about 9.7G |
| 3 | NUAA-UAVB-P43-B | mshnet_p43_sls | 12 -> 8 | 12 OOM, retry 8 about 8.7G |

The first P43-B attempt with batch 12 OOMed and was immediately removed/retried at batch 8.

## Early Results

| Method | Epoch 1 mIoU | Pd | Fa | Status |
| --- | ---: | ---: | ---: | --- |
| NUAA-UAVB-MSH3-A | 6.3066 | 79.0875 | 3012.6065e-6 | stopped |
| NUAA-UAVB-MSH3-B | 0.0081 | 0.3802 | 995050.7228e-6 | stopped |
| NUAA-UAVB-P43-A | 2.1853 | 87.0722 | 21589.4434e-6 | stopped |
| NUAA-UAVB-P43-B | 0.0134 | 1.1407 | 995209.6710e-6 | stopped |

These are far below both:

```text
current reliable NUAA best: 79.34 mIoU
target: 82.64 mIoU
```

## Interpretation

From-scratch official MSHNet/P43 training on the mixed full-data pool is not a useful NUAA breakthrough route in this setup. The outputs are unstable at epoch 1 and dominated by false alarms or near-empty/near-full masks.

The useful SIRST-UAVB direction should therefore shift away from training MSHNet/P43 from scratch and toward:

1. using official SIRST-UAVB PConv weights as candidate maps, and/or
2. building SIRST-UAVB-informed prediction caches/pseudo-labels for a lighter NUAA selector/refiner.

## Cleanup

Stopped tmux session:

```text
sirst_uavb_mshnet_4gpu
```

Deleted failed checkpoints and tensorboard files. Kept only small logs and this report:

```text
runs/sirst_uavb_mshnet_4gpu 136K
```
