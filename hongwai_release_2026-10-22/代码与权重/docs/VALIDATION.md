# 本次交付验证记录

执行日期：2026-10-22。

| 检查 | 结果 |
|---|---|
| `tools/reproduce_final_metrics.py` 固定配方复算 | PASS；IRSTD-1K mIoU=73.2142%、F1=84.5355%；NUAA-SIRST mIoU=82.7020%、F1=90.5316%，均与参考 JSON 精确一致 |
| `tools/check_retained_checkpoints.py` | PASS；IRSTD Anchor、NUAA s1303、NUAA s1313 均 strict-load，CPU 32×32 前向输出有限且形状正确 |
| `tools/verify_release.py` | PASS；14,131 个文件 SHA-256/大小一致 |
| 密钥/凭据扫描 | PASS；未发现代码/文本中的常见私钥或 API key 模式 |
| 原始数据与环境 | 未复制原图/掩码和 7.4GB 环境；使用者需按文档准备 |
| 全新环境端到端训练 | 未执行；本次交付目标是整理历史资产和冻结结果复算 |

固定复算不搜索权重、阈值或融合系数，只读取随包的历史概率图并核对样本名、尺寸、掩码和指标。它验证的是结果归档的一致性，不是独立留出泛化或从原图重建全部教师网络。
