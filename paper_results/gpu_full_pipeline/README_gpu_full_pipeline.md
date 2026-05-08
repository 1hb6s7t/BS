# 完整GPU端到端实验结果说明

本目录中的图片和数据均由本项目脚本真实运行生成，用于论文中的 GPU 效果展示与对比实验。

## 生成命令

```powershell
.\.venv-ms\Scripts\python .\tools\generate_gpu_paper_results.py
```

## 运行边界说明

- 当前主机 PyTorch CUDA 可用，GPU 为 NVIDIA GeForce RTX 4060 Laptop GPU。
- 当前 Windows MindSpore wheel 仍只支持 CPU，因此这里不把结果表述为“MindSpore GPU 端到端”。
- 本组实验采用完整 CUDA 流水线：CUDA 掩码修复 + CUDA 保真超分增强。
- 原 SRGAN checkpoint 的独立 CUDA 性能数据保留在 `paper_results/gpu_benchmark/`。

## 3组GPU效果图
- `paper_results/gpu_full_pipeline/figures/case01_natural_texture_gpu_effect_figure.png`：自然纹理划痕修复与超分增强
- `paper_results/gpu_full_pipeline/figures/case02_indoor_structure_gpu_effect_figure.png`：室内结构小面积遮挡修复与超分增强
- `paper_results/gpu_full_pipeline/figures/case03_scene_depth_gpu_effect_figure.png`：复杂场景水印式缺损修复与超分增强

## 3组对比数据图
- `paper_results/gpu_full_pipeline/figures/gpu_quality_psnr_ssim.png`：全图 PSNR/SSIM 对比。
- `paper_results/gpu_full_pipeline/figures/gpu_mask_repair_quality.png`：缺损掩码区域 PSNR/MAE 对比。
- `paper_results/gpu_full_pipeline/figures/gpu_runtime_memory.png`：端到端耗时与 GPU 显存占用对比。
- `paper_results/gpu_full_pipeline/figures/gpu_visual_methods_overview.png`：附加视觉总览，可选用于论文或答辩。

## 原始数据
- 指标 CSV：`paper_results/gpu_full_pipeline/metrics/gpu_full_pipeline_metrics.csv`
- 分案例运行时 CSV：`paper_results/gpu_full_pipeline/metrics/gpu_full_pipeline_case_runtime.csv`
- 运行环境 JSON：`paper_results/gpu_full_pipeline/metrics/gpu_full_pipeline_environment.json`

## 平均指标摘要
- 本项目 GPU 端到端平均耗时：`77.6 ms`；平均峰值 GPU 显存：`38.3 MB`。
- 相比未修复的退化输入+双三次，本项目全图 PSNR 提升：`7.29 dB`。
- 相比未修复的退化输入+双三次，本项目掩码区 PSNR 提升：`7.32 dB`。
- 相比未修复的退化输入+GPU保真超分，本项目掩码区 PSNR 提升：`7.33 dB`。
- 本项目平均细节保真度：`0.926`；平均边缘一致性：`0.852`。

## 论文表述建议

可写作：为验证系统在 GPU 环境下的端到端可运行性，本文在 RTX 4060 Laptop GPU 上补充实现 CUDA 推理流程。
其中修复阶段采用 GPU 掩码传播与扩散求解，增强阶段采用 CUDA 保真超分与细节增强。
三组真实样例结果表明，该流程能够在不同纹理、结构和场景缺损下完成修复与超分增强，并显著优于未修复直接放大的退化输入。

注意：不要将本结果写成“MindSpore GPU 已跑通”；应写成“补充 CUDA 端到端验证路径”。