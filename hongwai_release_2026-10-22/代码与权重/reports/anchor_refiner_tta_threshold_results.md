# Anchor Refiner TTA and Fine Threshold Results

Generated: 2026-06-14 UTC

## Purpose

The current best IRSTD-1K result is very close to the requested +5 target:

```text
SCTransNet baseline: 68.03
+5 target: 73.03
current best: 72.8351
gap: about 0.19
```

This round checked whether inference-time fine thresholding or flip TTA can close the remaining gap without new training.

## Code

New evaluator:

```text
scripts/eval_anchor_refiner_tta.py
```

It loads the retained anchor-refiner checkpoint:

```text
runs/anchor_refiner_multidata_probe/irstd_multi_r05_a002_s81/best.pth.tar
```

and evaluates:

- no TTA,
- horizontal flip,
- vertical flip,
- horizontal+vertical flip averaged TTA,
- fine thresholds around the previous best threshold.

## Result

Best row:

| mode | threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: |
| no TTA | 0.368 | 72.8619 | 67.6527 | 94.6128 | 12.0894e-6 |

Previous best:

| mode | threshold | mIoU | nIoU | Pd | Fa |
| --- | ---: | ---: | ---: | ---: | ---: |
| no TTA | 0.350 | 72.8351 | 67.7155 | 94.6128 | 12.4120e-6 |

Change:

```text
72.8351 -> 72.8619  (+0.0268 mIoU)
```

This is a small real improvement, but it still does not reach the +5 target:

```text
73.03 - 72.8619 = about 0.1681 mIoU remaining
```

## TTA

4-way flip TTA did not improve the result. The best TTA row was below the best no-TTA row.

## Component postprocess attempt

A connected-component filter sweep was started over saved probability maps, but repeated connected-component evaluation over the 201-image test split was too slow for this turn and was stopped. No useful postprocess result was produced.

## Outputs

```text
reports/anchor_refiner_tta_irstd_fine.json
reports/anchor_refiner_tta_irstd_maps_eval.json
scripts/eval_anchor_refiner_tta.py
```

The saved probability map archive was temporary and can be deleted if space is preferred:

```text
reports/anchor_refiner_tta_irstd_maps.npz
```

## Conclusion

Fine thresholding slightly improves IRSTD-1K but does not meet the requested +5 target. TTA is not useful here. The next IRSTD attempt should focus on a targeted high-frequency target-vs-clutter selector or validation-stable component correction, not generic TTA.
