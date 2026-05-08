# 论文效果图与对比测试数据说明

本目录中的图片均由本项目代码真实运行生成，生成命令：

```powershell
python tools\generate_paper_results.py
```

## 运行后端与环境
- 论文结果生成后端：`classic`；设备参数：`CPU`；放大倍数：`2x`。
- 实际 Python：`3.12.2`；MindSpore：`2.9.0`；MindSpore 当前 device_target：`CPU`。
- 环境记录：`paper_results/metrics/runtime_environment.json`。
- 说明：若使用 Windows CPU 版 MindSpore，CRA 注意力算子 `ExtractImagePatches` 不支持 CPU，完整 CRA 推理需要 GPU/Ascend 后端；本脚本可通过参数切换 deep/auto 并复用同一评测流程。

## 3组效果图片
- `paper_results/figures/case01_natural_texture_effect_figure.png`：自然纹理划痕修复与细节增强
- `paper_results/figures/case02_indoor_structure_effect_figure.png`：室内结构小面积遮挡修复
- `paper_results/figures/case03_scene_depth_effect_figure.png`：复杂场景水印式缺损修复增强

## 2–3组对比测试数据图
- `paper_results/figures/comparison_quality_psnr_ssim.png`：PSNR 与 SSIM 对比。
- `paper_results/figures/comparison_edge_detail_fidelity.png`：边缘一致性与细节保真度对比。
- `paper_results/figures/comparison_case_psnr_heatmap.png`：不同测试组 PSNR 热力图。
- `paper_results/figures/comparison_visual_methods_overview.png`：多方法视觉对比总览。
- `paper_results/figures/comparison_average_metrics_table.png`：平均量化指标表格。

## 原始量化数据
- CSV：`paper_results/metrics/comparison_metrics.csv`

## 平均指标摘要
- 本项目平均 PSNR：31.84 dB；相较退化输入+双三次提升 9.78 dB。
- 本项目平均 SSIM：0.982；相较传统修复+双三次放大变化 -0.001。
- 本项目平均边缘一致性：0.843；平均细节保真度：0.896。
- 结论建议：论文中可表述为“项目端到端联合流程显著优于未修复退化输入与简单放大；在 classic 后端下，与传统修复+双三次插值质量接近，并额外提供可配置、可视化和批处理的一体化流程”。

注：当前脚本默认使用项目内置 classic 后端（OpenCV 修复 + 双三次超分 + 轻量锐化）生成论文可复现实验；若在可用的 MindSpore GPU/Ascend 环境下切换 `--backend deep/auto` 并加载权重，可复用同一评测脚本扩展深度模型结果。

## 论文图注建议
1. 图A：三组典型场景下的修复与超分辨率增强效果。每组从左到右依次为原始参考图、退化输入、掩码区域、项目修复结果和联合增强输出。
2. 图B：多方法视觉对比总览。与直接放大退化输入相比，本项目端到端流程能够先消除局部缺损，再进行分辨率增强，输出更适合后续观察和展示。
3. 图C：PSNR、SSIM、边缘一致性和细节保真度等量化指标对比。实验结果表明，完整流程相较未修复退化输入和简单插值具有明显质量提升。

## 正文引用建议
可在论文实验章节写作：为验证系统在不同场景中的泛化表现，选取自然纹理、室内结构和复杂户外场景构建三组受损样例。所有结果均由系统命令行流程自动生成，并与最近邻放大、双三次放大、单独修复后放大等基线方法进行对比。实验数据保存在 comparison_metrics.csv 中，保证结果可复现。