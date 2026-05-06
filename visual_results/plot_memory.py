import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Droid Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 实际测量的显存占用数据
models = ['传统方法', 'CRA-SRGAN']
vram_usage = [14.2, 5.3]  # 单位：GB
reduction = round((1 - 5.3/14.2) * 100)  # 计算减少百分比

# 创建柱状图
fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.bar(models, vram_usage, color=['#e74c3c', '#3498db'], width=0.5)

# 添加数据标签
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f'{height:.1f} GB', ha='center', va='bottom', fontsize=14)

# 图表装饰
ax.set_title('8K图像处理显存占用对比', fontsize=18)
ax.set_ylabel('显存使用量 (GB)', fontsize=15)
ax.set_ylim(0, 16)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 添加说明文本
textstr = f'CRA-SRGAN优化后显存占用减少了{reduction}%\n处理8K图像仅需{vram_usage[1]:.1f}GB显存'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.5, 0.15, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='center', horizontalalignment='center', bbox=props)

# 添加箭头指示优化效果
ax.annotate('显存优化效果', xy=(1, vram_usage[1]), xytext=(1.1, 10),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
            fontsize=13)

# 保存图片
plt.tight_layout()
output_path = Path(__file__).resolve().parent / '显存优化对比图.png'
plt.savefig(output_path, dpi=300)
print(f'图片已保存到 {output_path}')
