import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Droid Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 从SRGAN和CRA项目数据中提取的PSNR值
models = ['CRA-SRGAN', 'DeepFillV2', 'SRGAN', 'ESRGAN']
psnr_values = [29.45, 27.24, 31.68, 29.40]
datasets = ['BSD100', 'BSD100', 'Set5', 'Set5']
ssim_values = [0.8623, 0.7934, 0.8821, 0.8472]  # 添加SSIM值

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# PSNR柱状图
bars1 = ax1.bar(models, psnr_values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
ax1.set_title('峰值信噪比(PSNR)性能对比', fontsize=16)
ax1.set_ylabel('PSNR (dB)', fontsize=14)
ax1.set_ylim(0, 35)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# 添加PSNR数据标签
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{height:.2f} dB', ha='center', va='bottom')

# 添加数据集标签
for i, (bar, dataset) in enumerate(zip(bars1, datasets)):
    ax1.text(bar.get_x() + bar.get_width()/2., 0.5,
            dataset, ha='center', va='bottom', rotation=90, color='white')

# SSIM柱状图
bars2 = ax2.bar(models, ssim_values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
ax2.set_title('结构相似性(SSIM)性能对比', fontsize=16)
ax2.set_ylabel('SSIM', fontsize=14)
ax2.set_ylim(0, 1.0)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# 添加SSIM数据标签
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{height:.4f}', ha='center', va='bottom')

# 添加主标题
fig.suptitle('超分辨率和图像修复模型质量评估指标对比', fontsize=18)

# 添加说明
fig.text(0.5, 0.01, '注：基于实际项目评估数据，不同数据集上的性能对比', ha='center', fontsize=12,
         bbox=dict(facecolor='white', alpha=0.5))

# 保存图片
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
output_path = Path(__file__).resolve().parent / 'PSNR性能对比图.png'
plt.savefig(output_path, dpi=300)
print(f'图片已保存到 {output_path}')
