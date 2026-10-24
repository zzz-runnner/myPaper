# 固定配方复现

以下命令在本包根目录执行。`tools/reproduce_final_metrics.py` 是本次新增并实际运行的最终指标复算入口，不进行参数搜索。原图→权重推理与完整教师重建需要数据及可选依赖，下面命令已按历史参数整理，本次未重新运行全量推理或训练。

## 1. 不使用 GPU 的缓存复算

```bash
python tools/reproduce_final_metrics.py
python tools/check_retained_checkpoints.py
```

前者检查三路预测的样本名/掩码/尺寸，按固定系数、幂次、阈值和面积过滤复算五指标，若偏离参考值则报错。后者严格加载三个最终主源权重，做 32×32 CPU 前向并记录方向门控；小尺寸前向只证明结构/权重兼容，不等同于完整数据推理。

NUAA `tmp_lpsm_pair_test_maps.npz` 已包含 `(0.45*A+0.55*B)^0.998`，不得再次重复加 pair gamma。

## 2. 由现存权重重新生成主源图

准备 `datasets/` 中的原始图像与标注后运行。新文件写入 results，不覆盖历史证据。

IRSTD Anchor：

```bash
python scripts/eval_anchor_refiner_tta.py \
  --dataset_name IRSTD-1K --dataset_dir datasets \
  --cache reports/prediction_cache_irstd_dual_train_b_test/IRSTD-1K__test.npz \
  --checkpoint runs/anchor_refiner_multidata_probe/irstd_multi_r05_a002_s81/best.pth.tar \
  --teacher_mode avg --base_channels 64 --residual_scale 0.5 \
  --thresholds 0.4 --tta none --device cuda:0 \
  --output results/anchor.export.json --save_maps results/anchor.maps.npz
```

NUAA LPSM pair（历史采用 AMP，复现时保持）：

```bash
python scripts/export_lpsm_pair_maps.py \
  --dataset_dir datasets --dataset_name NUAA-SIRST \
  --cache reports/fusion_cache_official_mamba_teacher_nuaa/NUAA-SIRST__test.npz \
  --teacher_mode local_rule \
  --checkpoint_a runs/local_rule_lpsm_nuaa_4gpu_screen/s1303_e2m_r006/best.pth.tar \
  --base_a 56 --stages_a e2,mid --residual_a 0.06 \
  --checkpoint_b runs/local_rule_lpsm_nuaa_followup_4gpu/s1313_e2m_r0055_a018/best.pth.tar \
  --base_b 64 --stages_b e2,mid --residual_b 0.055 \
  --alpha_a 0.45 --gamma 0.998 --amp --device cuda:0 \
  --output results/lpsm_pair.maps.npz
```

这两条命令使用随包上游预测缓存，并非只靠单一学生从原图独立推理。先验证重导出图与历史概率图差异，再替换缓存运行最终固定配方；不同硬件/算子版本的浮点差异可能影响临界像素。

## 3. 外部模型预测

历史参数见 `reports/tmp_dfaware_test_eval.json`、`tmp_hfmnet_test_eval.json`、`dfawarenet_official_mamba_irstd_eval_cross.json`、`hfmnet_official_mamba_irstd_eval_cross.json` 的 args。原入口 `scripts/eval_mamba_irstd_candidate_fullsplit.py` **默认 mamba_backend=fake**，历史原样保留；重建最终来源必须显式传入 `--mamba_backend official` 和 `--wavelet_backend official`，不能使用 fake 结果代替。

示例（NUAA DFAwareNet）：

```bash
python scripts/eval_mamba_irstd_candidate_fullsplit.py \
  --model dfawarenet --dataset_dir datasets --dataset_name NUAA-SIRST --split test \
  --weight external/DFAwareNet/weights_download/NUAA-SIRST/mIoU__DFAwareNet_NUAA-SIRST_epoch.pth.tar \
  --base_size 256 --thresholds 0.5 --min_areas 1 \
  --mamba_backend official --wavelet_backend official --device cuda:0 \
  --output results/dfaware.export.json --save_maps results/dfaware.maps.npz
```

## 4. 重跑保留训练配置

```bash
python tools/run_retained_training.py --config configs/release/irstd_anchor_s81.json
python tools/run_retained_training.py --config configs/release/nuaa_lpsm_s1303.json
python tools/run_retained_training.py --config configs/release/nuaa_lpsm_s1313.json
```

默认只打印可移植命令；加 `--execute` 才启动训练，输出到 results/，不会覆盖保留权重。`--dataset-dir`、`--output-dir`、`--device` 可指定。历史训练入口仍按测试集评估选择 checkpoint，保留是为重现历史，不建议用作新论文的验证协议。

历史脚本和日志中旧的 `/home/cx/hongwai/irstd-lab` 绝对路径用于溯源；新工具/配置使用包根目录。其他探索脚本需按 `--help` 覆盖旧路径。已删除的权重只保留记录，无法通过归档恢复。
