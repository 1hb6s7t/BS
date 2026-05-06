import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Droid Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 实际测量的处理速度数据（秒/张图像）
models = ['CRA-SRGAN', 'ESRGAN', 'SRGAN', 'DeepFillV2']
# 4K分辨率处理时间 (秒/张)
time_4k = [0.32, 0.86, 0.53, 0.67]
# 8K分辨率处理时间 (秒/张)
time_8k = [0.75, 2.13, 1.48, 1.92]

# 计算相对CRA-SRGAN的速度提升百分比
def calc_speedup(baseline_times, cra_time):
    return [(t/cra_time - 1) * 100 for t in baseline_times]

speedup_4k = calc_speedup(time_4k, time_4k[0])
speedup_8k = calc_speedup(time_8k, time_8k[0])

# 图表设置
width = 0.35
x = np.arange(len(models))

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# 4K处理时间
bars1 = ax1.bar(x, time_4k, width, label='处理时间', color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
ax1.set_title('4K图像处理时间对比', fontsize=15)
ax1.set_ylabel('处理时间 (秒/图像)', fontsize=13)
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.set_ylim(0, max(time_4k) * 1.2)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# 在4K图表上添加数据标签
for i, bar in enumerate(bars1):
    height = bar.get_height()
    if i > 0:
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                f'{height:.2f}s\n(+{speedup_4k[i]:.0f}%)', 
                ha='center', va='bottom', fontsize=10)
    else:
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.03,
                f'{height:.2f}s\n(基准)', 
                ha='center', va='bottom', fontsize=10, weight='bold')

# 8K处理时间
bars2 = ax2.bar(x, time_8k, width, label='处理时间', color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
ax2.set_title('8K图像处理时间对比', fontsize=15)
ax2.set_ylabel('处理时间 (秒/图像)', fontsize=13)
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.set_ylim(0, max(time_8k) * 1.2)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# 在8K图表上添加数据标签
for i, bar in enumerate(bars2):
    height = bar.get_height()
    if i > 0:
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}s\n(+{speedup_8k[i]:.0f}%)', 
                ha='center', va='bottom', fontsize=10)
    else:
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}s\n(基准)', 
                ha='center', va='bottom', fontsize=10, weight='bold')

# 添加主标题
fig.suptitle('CRA-SRGAN实时处理性能对比\n8K图像处理速度比ESRGAN快185%，比SRGAN快97%', 
             fontsize=16, y=0.98)

# 添加说明文本
fig.text(0.5, 0.01, 
         '测试环境：RTX 3090 GPU，批处理大小=1，数值为单张图像平均处理时间\n'
         'CRA-SRGAN在高分辨率图像处理时优势更明显', 
         ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

# 保存图片
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
output_path = Path(__file__).resolve().parent / '实时处理性能图.png'
plt.savefig(output_path, dpi=300)
print(f'图片已保存到 {output_path}')
