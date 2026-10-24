# 数据与划分

本包保留 `datasets/*/img_idx/` 与其他原有文本划分，不重新分配 train/test。原图与原始标注、数据 ZIP、生成伪标签图像不随代码交付；预测缓存为复算保留，部分包含测试掩码，应遵守相应数据集条款。

| 数据集 | train | test |
|---|---:|---:|
| IRSTD-1K | 800 | 201 |
| NUAA-SIRST | 213 | 214 |
| NUDT-SIRST | 663 | 664 |
| SIRST3 | 1676 | 1079 |

运行权重推理/训练时，把合法取得的数据放入：

```text
datasets/<dataset>/images/<name>.png
datasets/<dataset>/masks/<name>.png
datasets/<dataset>/img_idx/train_<dataset>.txt
datasets/<dataset>/img_idx/test_<dataset>.txt
```

数据名、划分以本包文本为准；NUAA-SIRST 对应 SIRST-v1。`Misc_111` 曾因图像/掩码尺寸不一致，将掩码最近邻调整；需保留该预处理记录。原始数据可参考各第三方 README 的官方获取方式。本包不授权任何数据再分发。
