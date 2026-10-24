# Infrared Small Target Lab

这个目录是给红外小目标检测准备的实验环境。主线选择 **SCTransNet**：

- 它是红外小目标检测专用 baseline，不需要 SAM/CLIP 这种大模型，适合 2080 Ti 11GB。
- 官方代码轻，已经在 `external/SCTransNet` 中拉好。
- 后续创新可以围绕“高频目标-背景杂波解耦”“背景遗忘/false alarm 抑制”“轻量并行 baseline”展开。

## 已配置内容

```text
/home/cx/hongwai/irstd-lab
├── envs/irstd                 # Python 3.10 + PyTorch 2.4.1/cu121
├── external/SCTransNet         # 主线官方代码
├── scripts/check_env.py        # 环境/GPU/模型前向检查
├── scripts/check_dataset.py    # 数据集结构检查
├── scripts/train_sctransnet.sh # SCTransNet 训练入口
└── runs/                       # 训练输出，首次训练时生成
```

## 数据放置

推荐把数据上传到 `/data1`，默认训练脚本会找：

```text
/data1/irstd/datasets
```

如果你的上传目录不是这个名字，可以运行时设置 `IRSTD_DATASET_DIR`。

SCTransNet 需要的数据结构是：

```text
/data1/irstd/datasets/SIRST3
├── images
│   ├── xxx.png
│   └── ...
├── masks
│   ├── xxx.png
│   └── ...
└── img_idx
    ├── train_SIRST3.txt
    └── test_SIRST3.txt
```

`train_*.txt` 和 `test_*.txt` 里写不带扩展名的样本名，例如 `XDU0`。

## 我建议你先传的数据集

优先级如下：

1. **SIRST3**：由 `SIRST-v1 + NUDT-SIRST + IRSTD-1K` 合并，最适合直接复现 SCTransNet 主线。
2. **NUDT-SIRST**：样本多，训练比较稳，适合作为第一轮单数据集实验。
3. **IRSTD-1K**：常用公开基准，适合泛化/对比。
4. **SIRST-v1 / NUAA-SIRST**：小一些，可以做补充对比。

如果暂时只传一个，我建议先传 **NUDT-SIRST** 或 **SIRST3**。

## 常用命令

环境检查：

```bash
/home/cx/hongwai/irstd-lab/envs/irstd/bin/python /home/cx/hongwai/irstd-lab/scripts/check_env.py
```

数据集检查：

```bash
IRSTD_DATASET_DIR=/data1/你的上传目录/datasets \
  /home/cx/hongwai/irstd-lab/envs/irstd/bin/python \
  /home/cx/hongwai/irstd-lab/scripts/check_dataset.py SIRST3
```

单卡训练：

```bash
IRSTD_DATASET_DIR=/data1/你的上传目录/datasets \
  GPU=0 EPOCHS=1000 BATCH_SIZE=16 \
  bash /home/cx/hongwai/irstd-lab/scripts/train_sctransnet.sh SIRST3
```

四张 2080 Ti 可以并行跑不同数据集或不同参数：

```bash
IRSTD_DATASET_DIR=/data1/你的上传目录/datasets \
  bash /home/cx/hongwai/irstd-lab/scripts/parallel_examples.sh
```

## 2080 Ti 设置建议

- `PATCH_SIZE=256`
- `BATCH_SIZE=16` 作为起点
- 如果显存紧张，先降到 `BATCH_SIZE=8`
- 多实验并行时，每个实验绑定一张卡：`GPU=0/1/2/3`

## 后续实验主线

第一阶段先复现 SCTransNet。跑通后最值得做的改进方向是：

```text
SCTransNet baseline
  -> 高频目标-背景杂波解耦模块
  -> hard clutter / false alarm 抑制
  -> 跨数据集泛化：NUDT-SIRST -> IRSTD-1K -> SIRST-v1
```

这条线比直接上 SAM/CLIP 更稳，也更贴合 2080 Ti 的显存条件。
