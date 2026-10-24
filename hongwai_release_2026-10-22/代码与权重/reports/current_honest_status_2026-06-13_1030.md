# 当前真实情况汇总

生成时间：2026-06-13 10:30 UTC

## 结论先说

目标是做红外小目标检测/分割，在 IRSTD-1K 和 NUAA-SIRST 上尽量超过 SCTransNet，理想目标是大约 +5 mIoU。

到目前为止，这个目标没有达到。

当前可靠最好结果：

| 数据集 | 当前可靠最好方法 | mIoU | SCTransNet 官方复现 | 提升 |
|---|---|---:|---:|---:|
| IRSTD-1K | component fusion learner，TPC + fixed refiner | 69.17 | 68.03 | +1.14 |
| NUAA-SIRST | background-normality component fusion，TPC + LCA / LCA + TCD | 79.34 | 77.64 | +1.70 |

注意：oracle 结果不是正式结果，因为它用了测试集 GT 来选组件，只能看理论上限，不能当论文/最终方法指标。

## 当前服务器状态

现在服务器不是完全空闲。还有 4 个 tmux session 在跑：

| tmux session | 数据集 | 输出文件 | 当前状态 |
|---|---|---|---|
| `proto_irstd_mixed` | IRSTD-1K | `reports/target_proto_gated_multidata_irstd_mixed.json` | 未写出 json |
| `proto_irstd_tpc` | IRSTD-1K | `reports/target_proto_gated_multidata_irstd_tpc.json` | 未写出 json |
| `proto_nuaa_tpc_lca` | NUAA-SIRST | `reports/target_proto_gated_multidata_nuaa_tpc_lca.json` | 未写出 json |
| `proto_nuaa_lca_tcd` | NUAA-SIRST | `reports/target_proto_gated_multidata_nuaa_lca_tcd.json` | 未写出 json |

这 4 个任务已经完成了 train/test 样本收集，日志最后停在：

- IRSTD：`collected test 200/201`
- NUAA：`collected test 200/214`

GPU 利用率当前是 0%，每张卡只占约 748-790 MiB 显存；但 4 个 Python 进程 CPU 占用很高，说明它们卡在 component 后处理、阈值网格或 logistic/metric 计算阶段，还没有形成正式结果文件。

因此：target prototype gated full run 目前不能算完成，也不能算有有效指标。

磁盘：

| 路径 | 状态 |
|---|---:|
| `/` | 437G 总量，359G 已用，60G 可用 |
| `/data1` | 3.6T 总量，195G 已用，3.4T 可用 |
| `/home/cx/hongwai/irstd-lab` | 约 11G |

后续大数据集继续放 `/data1`。

## 环境和数据

主项目目录：

```text
/home/cx/hongwai/irstd-lab
```

Python 环境：

```text
/home/cx/hongwai/irstd-lab/envs/irstd
```

已准备数据集：

| 数据集 | train | test | 备注 |
|---|---:|---:|---|
| IRSTD-1K | 800 | 201 | 主评测集之一 |
| NUAA-SIRST | 213 | 214 | 主评测集之一，之前修过 `Misc_111` mask 尺寸问题 |
| NUDT-SIRST | 663 | 664 | 用于全数据训练/辅助 |
| SIRST3 | 1676 | 1079 | 用于全数据训练/辅助 |

## 基线复现

官方 SCTransNet checkpoint 复现结果：

| 数据集 | 官方 README mIoU | 本地复现 mIoU | 说明 |
|---|---:|---:|---|
| IRSTD-1K | 68.03 | 68.03 | 对齐 |
| NUAA-SIRST | 77.50 | 77.64 | 对齐，略高 |
| NUDT-SIRST | 94.09 | 94.11 | 对齐 |

自己从头训练 SCTransNet 1000 epoch：

| 数据集 | best epoch | best mIoU |
|---|---:|---:|
| IRSTD-1K | 500 | 67.32 |
| NUAA-SIRST | 823 | 77.52 |
| NUDT-SIRST | 547 | 93.76 |
| SIRST3 | 678 | 82.96 |

结论：训练和评测链路是通的，但单纯从头训练 SCTransNet 没有超过官方 checkpoint。

## 做过的主线和结果

### 1. SCTransNet 系列模块改造

试过的方向包括：

- fixed refiner
- DHiF / Balanced DHiF
- TPC-DHiF
- LCA local contrast anomaly
- TCD decouple
- BNG / FeatUp / background-normality 类辅助
- expert gate / oracle gate

大致结果：

| 方法线 | IRSTD-1K 最好 | NUAA-SIRST 最好 | 判断 |
|---|---:|---:|---|
| SCT-Refiner-Fixed | 68.09 | 79.12 | 小涨 |
| SCT-DHiF-Balanced | 68.05 | 79.11 | 无明显提升 |
| SCT-DHiF-FromFixed | 68.09 | 79.12 | 平台期 |
| SCT-DHiF-Multi | 68.06 | 79.11 | 平台期 |
| SCT-TPC-DHiF | 68.28 | 79.12 | 早期 IRSTD 最好 |
| SCT-LCA | 68.09 | 79.13 | 早期 NUAA 最好 |
| TCD-Decouple | 68.09 | 79.13 | NUAA 接近最好 |

结论：这些模型之间相关性很高，互补性有限，只靠继续调这些模块很难冲 +5。

### 2. 外部方法线

不是只调 SCTransNet，也试了外部/其他论文路线：

| 方法 | 数据集 | 结果 | 判断 |
|---|---|---:|---|
| MSHNet-D05-FullData | IRSTD-1K | 67.29 | 低于当前最好 |
| MSHNet-P43-FullData | IRSTD-1K | 66.70 | 低于当前最好 |
| MSHNet-D05-FullData | NUAA-SIRST | 71.40 | 明显低 |
| MSHNet-P43-FullData | NUAA-SIRST | 72.77 | 明显低 |
| BGM-PairedBasicUNet | NUAA-SIRST | epoch 10 时 71.86 | 低，停止 |
| APConv-TCF-SDM | NUAA-SIRST | epoch 10 时 73.21 | 低，停止 |
| MTMLNet 多个变体 | IRSTD/NUAA | 分割指标明显低 | 停止 |

这些大目录已经尽量清理，只保留报告、小日志和必要证据。

### 3. SAM pseudo-label / RIM 路线

准备过 SAM-B checkpoint：

```text
pretrained/sam/sam_vit_b_01ec64.pth
```

生成过 SAM/refined pseudo label：

```text
generated/sam_refined_irstd_tpc_rim_train
generated/sam_refined_nuaa_lca_tcd_rim_train
generated/sam_refined_mix_for_nuaa_rim_train
```

结果：

| 数据集 | 方法 | mIoU |
|---|---|---:|
| IRSTD-1K | SAMRim | 68.30 |
| NUAA-SIRST | SAMRim | 79.15 |

结论：没有超过 component fusion 当前最好，后续 bulky 训练已停。

### 4. Component Fusion Learner

这是目前最有效的路线之一。

思路：不是重新训练一个大网络，而是把多个强 teacher 的预测连通域拿出来，做 component-level false alarm filtering。每个候选组件提取面积、形状、teacher 概率统计、局部热对比、Laplacian、模型一致性等特征，然后训练一个轻量 logistic keeper 来决定保留哪些组件。

脚本：

```text
scripts/eval_component_fusion_learner.py
```

单数据训练/测试正式 valid 结果：

| 数据集 | teacher 组合 | mIoU | Pd | Fa |
|---|---|---:|---:|---:|
| IRSTD-1K | TPC teachers | 69.13 | 90.24 | 7.06 |
| IRSTD-1K | TPC + fixed refiner | 69.17 | 89.90 | 6.09 |
| NUAA-SIRST | LCA + TCD | 79.27 | 96.58 | 16.40 |
| NUAA-SIRST | TPC + LCA | 79.26 | 96.58 | 16.40 |

多数据训练，即用 `IRSTD-1K, NUAA-SIRST, NUDT-SIRST, SIRST3` 全部 train split 拟合 keeper：

| 数据集 | teacher 组合 | mIoU | Pd | Fa |
|---|---|---:|---:|---:|
| IRSTD-1K | TPC teachers | 69.16 | 90.91 | 7.61 |
| IRSTD-1K | TPC + fixed refiner | 69.08 | 90.57 | 7.65 |
| NUAA-SIRST | LCA + TCD | 79.27 | 96.58 | 16.46 |
| NUAA-SIRST | TPC + LCA | 79.28 | 96.58 | 16.46 |

oracle 上限：

| 数据集 | teacher 组合 | oracle mIoU |
|---|---|---:|
| IRSTD-1K | TPC / mixed | 约 70.3-70.4 |
| NUAA-SIRST | LCA/TCD/TPC 组合 | 约 80.27 |

结论：这是有效提升，尤其 IRSTD 从 68.03 到 69.17，但 oracle 显示当前 teacher 候选集自身上限也只有约 70-80，不像有 +5 空间。

### 5. Background Normality Component

这是从 anomaly detection / background normality 思路引入的路线，不是 SCTransNet 调参。

实现位置：

```text
scripts/eval_component_fusion_learner.py
scripts/launch_bg_normality_remaining_tmux.sh
```

思路：在四个数据集 train split 的 GT-background 像素上拟合背景正常性模型，抽取局部强度、top-hat、DoG、Laplacian、gradient、local-z 等特征，形成 background anomaly score，再作为 component keeper 的附加特征。

正式 full valid 结果：

| 数据集 | teacher 组合 | mIoU | Pd | Fa |
|---|---|---:|---:|---:|
| IRSTD-1K | TPC + fixed refiner | 69.03 | 90.24 | 6.68 |
| IRSTD-1K | TPC teachers | 68.93 | 88.55 | 7.23 |
| NUAA-SIRST | TPC + LCA | 79.34 | 96.58 | 15.92 |
| NUAA-SIRST | LCA + TCD | 79.34 | 96.58 | 15.92 |

对比：

| 数据集 | 之前最好 | background-normality 后 |
|---|---:|---:|
| IRSTD-1K | 69.17 | 69.03 |
| NUAA-SIRST | 79.28 | 79.34 |

结论：NUAA 有一个很小的新高，主要是降低 false alarm；IRSTD 反而略降。它可以作为小模块保留，但不是突破 +5 的主线。

### 6. Handcrafted / top-hat / sparse candidate

在 `eval_component_fusion_learner.py` 加过 handcrafted candidates：

- top-hat
- black-hat
- local-z
- LoG
- hand_sparse / hand_topk

这些分支因为全量后处理太慢或没有看到明显有效趋势而停止，保留了 stopped note：

```text
reports/component_fusion_handsparse_stopped_note.md
reports/component_fusion_handtopk_stopped_note.md
```

结论：没有正式可报提升。

### 7. Target Prototype Gated Candidate

这是当前还在跑、但没有结果的最新尝试。

实现位置：

```text
scripts/eval_component_fusion_learner.py
scripts/launch_target_proto_gated_4gpu_tmux.sh
```

思路：从 GT target / hard background negative 拟合一个 dense target prototype model，生成新的 target-prototype score map，并用 teacher suppression、local-z、top-k connected components 做 gate，尝试引入 teacher 之外的新候选。

保留的 smoke test：

```text
reports/target_proto_gated_smoke_irstd.json
```

smoke test 只用了很小样本，不是正式全量结果：

| 设置 | mIoU | Pd | Fa |
|---|---:|---:|---:|
| best valid smoke | 74.92 | 92.31 | 21.46 |
| smoke oracle | 76.51 | 93.85 | 16.69 |

这个 smoke 分数看起来比同 smoke 的 baseline 略好，所以才启动了全量 4GPU。

全量 4GPU 当前命令共同设置：

```text
--fit_datasets IRSTD-1K,NUAA-SIRST,NUDT-SIRST,SIRST3
--target_prototype_candidates
--prototype_topk 8
--prototype_threshold 0.82
--prototype_teacher_suppress 0.25
--seed_thresholds 0.40,0.45,0.50
--keep_thresholds 0.10,0.15,0.20,0.30,0.40
--output_thresholds 0.40,0.45,0.50
--steps 1600
```

当前判断：这条线还不能汇报成成功，也不能汇报成失败。它的问题是全量后处理非常慢，目前还没有 json 指标。

## 代码层面实际做了什么

主要改动集中在：

```text
scripts/eval_component_fusion_learner.py
```

已经加入：

- component-level teacher fusion learner
- train-fit / test-eval 的 valid 评测
- test oracle 评测，用于上限诊断
- full multi-dataset fitting
- handcrafted candidate maps
- background normality feature / optional candidate
- target prototype gated candidate
- 多组 seed/keep/output threshold sweep

辅助启动脚本：

```text
scripts/launch_bg_normality_remaining_tmux.sh
scripts/launch_target_proto_gated_4gpu_tmux.sh
```

报告和证据主要在：

```text
reports/component_fusion_learner_results.md
reports/component_fusion_multidata_results.md
reports/bg_normality_component_results.md
reports/target_proto_gated_smoke_irstd.json
reports/target_proto_gated_logs/
```

## 删除/保留情况

已经按“不要占太多盘”的原则处理：

- 多数失败的大训练目录已经停止并清理。
- 保留小型报告、json、日志、stopped note。
- 当前项目整体约 11G，没有近期因为 prototype 产生的大目录。
- `/data1` 还有约 3.4T，后续数据集应该放那里。

## 当前最重要的判断

1. 目前已经有真实提升，但远不到 +5。
2. 最可靠的路线不是继续训练 SCTransNet 本体，而是 component-level fusion / false-alarm suppression。
3. 当前 teacher 候选集的 oracle 上限有限：IRSTD 大约 70.4，NUAA 大约 80.3。这说明只筛掉假阳性不够，想 +5 需要真正带来更多正确目标候选或更强的新模型。
4. background-normality 只能带来很小改善。
5. target prototype gated 是为了解决“候选不够新”的问题，但当前全量运行还没出结果，而且后处理很慢。

## 建议下一步

短期：

- 再观察 target prototype gated 全量任务是否能写出 json。
- 如果继续长时间无 json，应停止这 4 个 tmux 任务，记录为“全量后处理过慢，未产出正式结果”。
- 然后把这条线改成更小的阈值网格或先只跑一个数据集/一个 teacher 组合，避免 CPU 后处理卡死。

中期：

- 如果 target prototype 全量结果有效，再优化候选生成和 keeper。
- 如果无效，应转向更强的新候选来源，而不是继续在同一批 SCTransNet teacher 上筛选。

当前不要把 target prototype full run 当作已经完成的有效实验。
