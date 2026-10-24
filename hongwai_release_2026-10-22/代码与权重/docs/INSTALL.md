# 环境安装

原环境：Linux x86_64，Python 3.10，PyTorch 2.4.1+cu121，torchvision 0.19.1+cu121；历史计算硬件为 4×RTX 2080 Ti（11GB）。`environment/pip-freeze.original.txt` 是本次读取的完整环境记录，含历史无关包与一个不可移植的 pip 构建路径，供溯源，不能直接作为可移植安装锁文件。

建议新建环境并按根目录 `requirements.txt` 安装。核心依赖版本已固定；未在一台全新机器上重新安装验证。CPU 缓存复算可安装匹配的 CPU 版 PyTorch/torchvision，但本次验证使用现有 CUDA 版环境在 CPU 上执行。

从原图生成 DFAwareNet/HFMNet 预测还需 `environment/requirements-mamba.txt` 中的官方算子。安装可能需要匹配的 CUDA toolkit/nvcc 与 PyTorch ABI：

```bash
python -m pip install packaging ninja
python -m pip install --no-build-isolation -r environment/requirements-mamba.txt
```

此命令仅为安装入口，不保证所有平台都有预编译 wheel。原机器已安装 mamba-ssm 2.2.6.post3 与 causal-conv1d 1.5.3.post1；本包不复制整个 7.4GB 环境或与机器绑定的 .so。历史报告曾记录安装失败，后续官方后端已启用，应结合日期阅读。

探索脚本可能另需 ultralytics、transformers、segment-anything、mmcv 等，详见原第三方 README 与完整环境快照。核心缓存复算不需要安装这些可选依赖。
