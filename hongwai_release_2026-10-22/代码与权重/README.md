# Hongwai: Infrared Small Target Research Snapshot

本目录保存当前红外小目标项目可复用的代码、现存权重、配置、历史日志与预测缓存。交付日期：2026-10-22。最终探索结果：IRSTD-1K mIoU **73.2142%**，NUAA-SIRST **82.7020%**。

这是可审计的科研归档。历史测试集参与 checkpoint/参数选择，且 NUAA 保留权重的方向分支有效门控为 0；目前不构成严格留出验证或统一新方法的顶会开源发布。完整说明见 [实验协议](docs/EXPERIMENT_PROTOCOL.md) 与 [历史审计](reports/research_idea_and_effective_experiments_2026-09-17.md)。

## 快速开始

建议 Linux、Python 3.10。Windows 可下载、解压与查看文档；CUDA/Mamba 推理建议在 Linux 或 WSL2 运行。

```bash
python3 tools/verify_release.py
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/reproduce_final_metrics.py
```

最后一条命令只执行**冻结配方**，核对六份现存概率图的样本名、尺寸、掩码与参考五指标，不搜索阈值。无需 GPU、原始数据集或 Mamba。它证明缓存数字可复算，不代表所有上游模型从原图到结果已端到端复现。

当前主机也可用原实验解释器执行：`/home/cx/hongwai/irstd-lab/envs/irstd/bin/python tools/reproduce_final_metrics.py`。

## 目录

```text
scripts/       历史训练、评估、缓存导出、探索入口（保留原始实现）
external/      25 份现存第三方源码/配套资产（含各自权重、原许可证）
pretrained/    官方 SCTransNet、SAM、MobileSAM 权重
runs/          当前仍存在的训练权重、summary、日志
reports/       历史证据与预测缓存，含最终六份概率档案
configs/       原始配置与 release/ 可移植复现参数
datasets/      数据划分与说明，不含原始图像/掩码
results/       最终参考 JSON、CSV 与本次复算结果
tools/         固定配方复算、训练参数迁移和 SHA-256 验证工具
```

完整文件内容以 `MANIFEST.json` 为准。所有原始代码路径关系保持一致，避免移动权重破坏导入/默认路径。`runs/` 不等于全部训练都有效，现存权重也不自动等于有增益；分类见 [MODEL_ZOO](docs/MODEL_ZOO.md)。

## 复现层级

1. **最终指标复算**：上面的 `tools/reproduce_final_metrics.py`，已提供参考输出。
2. **现存权重生成主源预测**：[REPRODUCIBILITY](docs/REPRODUCIBILITY.md) 提供固定参数命令，需要自行准备原始数据；已有上游预测缓存随包提供。
3. **训练重跑**：`configs/release/*.json` 保存保留权重的训练参数；`tools/run_retained_training.py` 将历史绝对路径改成当前包路径并运行原训练入口。历史训练程序仍按测试集选 checkpoint，仅用于重现历史，不能当新论文协议。
4. **完整上游重建**：第三方模型需额外依赖及原始数据。未宣称本次已从原图重跑全部教师；Mamba 必须显式使用 `--mamba_backend official`。

## 环境、权重与证据

- 核心固定版本：[requirements.txt](requirements.txt)；原环境完整记录：[pip freeze](environment/pip-freeze.original.txt)。
- [安装说明](docs/INSTALL.md)、[数据说明](docs/DATASETS.md)、[模型清单](docs/MODEL_ZOO.md)、[第三方来源](THIRD_PARTY_NOTICES.md)。
- [最终结果](reports/final_common_fusion_metrics_table.md)；所有历史报告仅为时间快照，出现旧的“current best”时以交付目录主表与 2026-09-17 审计为准。
- 第三方资产保留原授权；项目未发现自有代码的统一开源许可证，作者、论文名称、引用信息尚未提供，因此不擅自添加授权或虚构 BibTeX。发布前事项见 [RELEASE_CHECKLIST](docs/RELEASE_CHECKLIST.md)。
