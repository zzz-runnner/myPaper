# 红外小目标论文工作

本仓库保存论文项目源码、配置和实验记录。权重已单独备份到网盘；Git 仓库不包含权重、预测缓存、数据图像、安装环境或交付包中的第三方代码副本。

## 从这里开始

- [项目交付说明](hongwai_release_2026-10-22/README.md)
- [实验结果主表](hongwai_release_2026-10-22/实验结果主表.md)
- [SCTransNet 对比表](hongwai_release_2026-10-22/SCTransNet对比表.md)
- [研究思路与历史实验审计](hongwai_release_2026-10-22/代码与权重/reports/research_idea_and_effective_experiments_2026-09-17.md)
- [代码入口与复现说明](hongwai_release_2026-10-22/代码与权重/docs/REPRODUCIBILITY.md)

历史最终探索 mIoU：IRSTD-1K 为 73.2142%，NUAA-SIRST 为 82.7020%。历史测试集曾参与模型与参数选择，两数据集采用的主修正网络不同，且保留 LPSM 权重的方向门控有效值为 0。上述数值不能直接作为统一新方法在独立留出测试上的增益证明，解释以审计为准。

## Git 中保存什么

| 路径 | 内容 |
|---|---|
| `hongwai_release_2026-10-22/代码与权重/scripts/` | 项目训练、评估、融合和历史探索脚本 |
| `hongwai_release_2026-10-22/代码与权重/tools/` | 冻结指标复算、权重检查、训练配置迁移和归档校验工具 |
| `hongwai_release_2026-10-22/代码与权重/configs/` | 历史配置及三个最终主源权重的可移植训练参数 |
| `hongwai_release_2026-10-22/代码与权重/docs/` | 安装、数据、实验协议和复现文档 |
| `hongwai_release_2026-10-22/代码与权重/reports/` | 文本报告、评估 JSON 和相关小文件；不含预测数组 |
| `hongwai_release_2026-10-22/代码与权重/runs/` | 训练参数、summary 和文本日志；不含权重 |
| `hongwai_release_2026-10-22/代码与权重/datasets/` | 数据集划分列表；不含原图和掩码 |
| `baseline&study/` | 仓库原有的 baseline 与学习代码，沿用已有提交 |

`requirements.txt` 和 `environment/requirements-mamba.txt` 是安装依赖的清单，应当随代码保存；实际安装的库、虚拟环境和第三方仓库副本不随本次提交上传。

## 克隆后如何恢复运行资产

```bash
git clone git@github.com:zzz-runnner/myPaper.git
cd myPaper
```

这是源码和实验记录仓库，克隆后不能立即执行完整训练或推理。原交付文档描述完整本地归档；其中的权重、缓存和第三方目录链接在 GitHub 上可能没有对应文件。

在 `hongwai_release_2026-10-22/代码与权重/` 下按原路径恢复需要的资产：

1. 从网盘备份恢复 `runs/` 中的权重，以及 `pretrained/`、`reports/` 中的预测缓存。三个主要 checkpoint 的位置见 [模型清单](hongwai_release_2026-10-22/代码与权重/docs/MODEL_ZOO.md)。
2. 准备 `external/` 下所需第三方源码。项目含历史本地适配，优先恢复原归档中的副本；来源说明见 [THIRD_PARTY_NOTICES](hongwai_release_2026-10-22/代码与权重/THIRD_PARTY_NOTICES.md)。仅重新下载上游最新版本不保证兼容。部分评估脚本在导入时就需要这些源码，即使只进行缓存复算也需要恢复对应依赖。
3. 如需从原图推理或训练，按 [数据说明](hongwai_release_2026-10-22/代码与权重/docs/DATASETS.md) 准备原图和掩码；仅固定缓存复算无需原图。
4. 按 [安装说明](hongwai_release_2026-10-22/代码与权重/docs/INSTALL.md) 安装依赖，再按照复现文档运行所需命令。

`MANIFEST.json` 和 `SHA256SUMS` 标识原始完整交付包，包含刻意不上传 Git 的资产。只有恢复全部归档文件后，`tools/verify_release.py` 才能进行完整校验；它不是对精简 Git 克隆执行的通过检查。网盘下载链接尚未写入仓库。

## 日常同步

在仓库根目录检查变更后通过 SSH 同步：

```bash
git status --short
git add <需要提交的源码或文档路径>
git diff --cached --stat
git commit -m "Describe the research change"
git push origin main
```

`.gitignore` 排除大文件和外部资产。更新权重时继续备份到网盘，保留原始相对路径以便恢复。
