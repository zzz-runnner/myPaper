# IRSAM / MobileSAM Candidate Attempt

Generated: 2026-06-14 UTC

## Purpose

This round tested a non-SCTransNet foundation-model line from the infrared small-target / segmentation family. The public IRSAM repository was cloned and paired with the official MobileSAM checkpoint so the run was not random-initialized.

Important source caveat: the public IRSAM README says the currently released `IRSAM_encoder` is the MobileSAM baseline, while the paper's WPMD IRSAM encoder is not released yet. Therefore this experiment should be treated as "IRSAM structure with MobileSAM initialization", not the full paper IRSAM.

## Assets

External code:

```text
external/IRSAM
```

Downloaded official MobileSAM checkpoint:

```text
pretrained/mobile_sam/mobile_sam.pt
```

The checkpoint is 39M and was verified to load into `build_sam_IRSAM(checkpoint=...)`.

Training wrapper:

```text
scripts/train_irsam_multidata.py
```

It trains on the full prepared pool when not sample-limited:

```text
IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3
```

and evaluates:

```text
IRSTD-1K, NUAA-SIRST
```

## Runs

### Random-Initialized Smoke

Before downloading MobileSAM weights, two tiny smoke runs were made:

| Setting | Train samples | Epochs | IRSTD-1K mIoU | NUAA-SIRST mIoU |
|---|---:|---:|---:|---:|
| decoder only | 25 per dataset | 2 | 0.00 | 0.00 |
| decoder + prompt | 25 per dataset | 2 | 0.00 | 0.00 |

This showed that random-initialized decoder/prompt training is unusable.

### MobileSAM-Initialized Smoke

Command shape:

```text
python scripts/train_irsam_multidata.py \
  --checkpoint pretrained/mobile_sam/mobile_sam.pt \
  --epochs 5 \
  --batch_size 1 \
  --lr 0.0002 \
  --trainable decoder_prompt \
  --eval_interval 1 \
  --eval_max_samples 50 \
  --max_train_samples_per_dataset 100 \
  --output_dir runs/irsam_mobilesam_smoke \
  --device cuda:0
```

Best default-threshold epoch:

| Epoch | IRSTD-1K mIoU | IRSTD Pd | IRSTD Fa | NUAA-SIRST mIoU | NUAA Pd | NUAA Fa |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 28.81 | 80.25 | 0.000225 | 38.07 | 69.86 | 0.000092 |

Threshold sweep on the same 50-sample eval subset:

| Dataset | Best threshold | Best mIoU |
|---|---:|---:|
| IRSTD-1K | 0.80 | 29.21 |
| NUAA-SIRST | 0.05 | 39.62 |

## Cleanup

Failed smoke checkpoints were deleted:

```text
runs/irsam_multidata_smoke/*/best.pth.tar
runs/irsam_mobilesam_smoke/*/best.pth.tar
```

Only small logs and summaries remain. The MobileSAM pretrained checkpoint is kept because it is a reusable 39M external asset.

## Interpretation

MobileSAM initialization fixes the all-empty/random failure mode, but the dense masks after short decoder+prompt adaptation are far below the existing teachers:

| Dataset | Current reliable best | MobileSAM-initialized IRSAM smoke |
|---|---:|---:|
| IRSTD-1K | 72.43 | 29.21 on 50-sample smoke |
| NUAA-SIRST | 79.34 | 39.62 on 50-sample smoke |

This is not competitive enough to justify an immediate full 4-GPU long run. The line may only become useful if the unreleased IRSAM/WPMD weights become available, or if we redesign the training target into a prompt/candidate generator rather than direct dense segmentation.

Decision: do not expand this exact IRSAM setup. Continue with the stronger IRSTD PConv candidate/gating line and search for a different NUAA candidate source.
