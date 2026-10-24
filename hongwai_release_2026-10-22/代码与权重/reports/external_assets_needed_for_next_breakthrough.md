# External assets needed for the next NUAA breakthrough

Date: 2026-06-16

Current NUAA best is now `82.3084`, still below the `82.64` target. Local refiner variants around the same official-Mamba local-rule teacher are nearly saturated. The next likely useful source is external data/weights.

## Highest priority

### RealScene-ISTD dataset

Source listed in local README:

```text
external/RealScene-ISTD/README.md
dataset: https://pan.baidu.com/s/1BAY9N4xEtw57_v3TjHrSpg?pwd=trnh
```

Expected placement after upload:

```text
/data1/RealScene-ISTD/
  images/
  masks/
  img_idx/
    train_RealScene-ISTD.txt
    test_RealScene-ISTD.txt
```

Why needed:

```text
new real-scene distribution for background/clutter generalization
```

### RealScene-ISTD small target patches

Source listed in local README:

```text
patches: https://pan.baidu.com/s/1zSDV6cw2xcciaKpxpurXdQ?pwd=vjkb
```

Why needed:

```text
can be used for copy-paste / alignment / hard clutter augmentation
```

### MiM-ISTD checkpoint

Source listed in local README:

```text
checkpoint: https://pan.baidu.com/s/13g2v_M9tPxq_ze02fpaYGw
extraction code: DY4h
```

Why needed:

```text
would provide a genuinely new Mamba-in-Mamba dense source instead of only local modules
```

## Lower priority

Text-IRSTD currently has only a README locally:

```text
external/Text-IRSTD/README.md
```

The README says code will be released later, so it is not immediately actionable here.

SAIST should be revisited if an official runnable repository/checkpoint becomes available.
