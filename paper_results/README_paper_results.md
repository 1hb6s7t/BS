# 论文效果图与对比测试数据说明

本目录中的图片均由本项目代码真实运行生成，生成命令：

```powershell
python tools\generate_paper_results.py
```

## 3组效果图片
- `paper_results/figures/case01_natural_texture_effect_figure.png`：自然纹理划痕修复与细节增强
- `paper_results/figures/case02_indoor_structure_effect_figure.png`：室内结构小面积遮挡修复
- `paper_results/figures/case03_scene_depth_effect_figure.png`：复杂场景水印式缺损修复增强

## 2–3组对比测试数据图
- `paper_results/figures/comparison_quality_psnr_ssim.png`：PSNR 与 SSIM 对比。
- `paper_results/figures/comparison_edge_detail_fidelity.png`：边缘一致性与细节保真度对比。
- `paper_results/figures/comparison_case_psnr_heatmap.png`：不同测试组 PSNR 热力图。

## 原始量化数据
- CSV：`paper_results/metrics/comparison_metrics.csv`

## 平均指标摘要
- 本项目平均 PSNR：31.82 dB；相较退化输入+双三次提升 9.76 dB。
- 本项目平均 SSIM：0.982；相较传统修复+双三次放大变化 -0.001。
- 本项目平均边缘一致性：0.842；平均细节保真度：0.896。
- 结论建议：论文中可表述为“项目端到端联合流程显著优于未修复退化输入与简单放大；在 classic 后端下，与传统修复+双三次插值质量接近，并额外提供可配置、可视化和批处理的一体化流程”。

注：当前机器未安装 MindSpore，脚本使用项目内置 classic 后端（OpenCV 修复 + 双三次超分 + 轻量锐化）完成真实运行验证；若在 MindSpore 环境下切换 deep/auto 并加载权重，可复用同一评测脚本扩展深度模型结果。