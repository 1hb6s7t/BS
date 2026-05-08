# SRGAN 真实 CPU/GPU 性能补充说明

本目录补充的是 **SRGAN 超分阶段** 的真实 GPU 性能数据。
注意：这不是声称当前 Windows 主机已跑通完整 MindSpore deep GPU 版 CRA+SRGAN；
当前主机的 MindSpore wheel 仍只能 CPU，因此这里采用“同一原始 SRGAN 检查点 + PyTorch CUDA 适配器”的方式补充真实 GPU 超分性能。

## 运行环境
- GPU：`NVIDIA GeForce RTX 4060 Laptop GPU`
- PyTorch：`2.11.0+cu128`，CUDA：`12.8`
- MindSpore：`2.9.0`（CPU 基线）
- SRGAN 检查点：`ckpt/pretrained_generator_epoch100000.ckpt`
- 环境 JSON：`paper_results/gpu_benchmark/metrics/srgan_gpu_environment.json`

## 结论摘要
- 平均 GPU / CPU 加速比（`torch_cpu` → `torch_cuda`）：`11.24×`
- CPU / GPU 输出平均 PSNR：`53.96 dB`
- CPU / GPU 输出平均 MAE：`0.1015`
- GPU 峰值显存（本批次最大）：`776.4 MB`

## 解释边界
- 论文中建议重点引用 `torch_cpu ↔ torch_cuda` 的真实加速比，因为二者使用完全相同的转换后 SRGAN 权重，输出几乎一致。
- `mindspore_cpu` 结果仍保留在 CSV 中，作为当前仓库原始实现的 CPU 参考性能。
- 不建议把 `mindspore_cpu ↔ torch_cuda` 的输出差异当作质量结论；该差异只说明当前 Windows 主机无法给出完整、同实现的 MindSpore GPU 对照。

## 生成文件
- `paper_results/gpu_benchmark/figures/srgan_cpu_gpu_latency.png`
- `paper_results/gpu_benchmark/figures/srgan_cpu_gpu_speedup.png`
- `paper_results/gpu_benchmark/metrics/srgan_cpu_gpu_benchmark.csv`
- `paper_results/gpu_benchmark/metrics/srgan_cpu_gpu_comparison.csv`

## 论文写法建议
可在实验或工程实现章节中补充说明：在当前 Windows 主机上，完整 MindSpore deep GPU 后端受官方 wheel/算子支持限制；
因此保留 MindSpore CPU 结果作为原始工程基线，并基于同一 SRGAN 原始权重实现 PyTorch CPU/CUDA 对照推理，获得真实的 NVIDIA GPU 超分推理性能，用于补充加速效果分析。