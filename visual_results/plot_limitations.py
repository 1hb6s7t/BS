import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Droid Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 第一部分：跨模态修复的特征对齐偏差
labels = ['有效对齐', '特征对齐偏差']
sizes = [88.8, 11.2]  # 88.8% 有效对齐, 11.2% 误差率
colors = ['#3498db', '#e74c3c']
explode = (0, 0.1)  # 突出特征对齐偏差部分

# 创建饼图
ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90,
        textprops={'fontsize': 12})
ax1.set_title('跨模态修复特征对齐分析\n(误差率11.2%)', fontsize=14)

# 第二部分：运动模糊超分场景的适应性
metrics = ['SSIM得分']
values = [0.802]
low_benchmark = [0.700]  # 假设的低基准
high_benchmark = [0.950]  # 假设的高基准

x = np.arange(len(metrics))
width = 0.2

# 创建条形图
rects1 = ax2.bar(x - width, low_benchmark, width, label='低基准', color='#e74c3c')
rects2 = ax2.bar(x, values, width, label='当前SSIM', color='#f39c12')
rects3 = ax2.bar(x + width, high_benchmark, width, label='理想目标', color='#2ecc71')

# 添加标签和文本
ax2.set_title('运动模糊超分场景适应性\n(SSIM 0.802)', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(metrics, fontsize=12)
ax2.set_ylim(0, 1.0)
ax2.set_ylabel('结构相似性指数', fontsize=12)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.legend()

# 在条形上添加文本标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3点垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# 添加目前改进的需求文本和箭头
ax2.annotate('需要改进',
            xy=(0, 0.802),
            xytext=(0.27, 0.875),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
            fontsize=11)

# 添加总标题
fig.suptitle('CRA-SRGAN模型的当前局限性分析', fontsize=16, y=0.98)

# 添加说明文本
fig.text(0.5, 0.01, 
         '当前模型的局限性主要集中于跨模态修复的特征对齐偏差与运动模糊超分场景的适应性不足\n'
         '解决方案将聚焦于扩散模型驱动的生成多样性增强与Transformer-CNN混合架构优化', 
         ha='center', fontsize=12,
         bbox=dict(facecolor='white', alpha=0.5))

# 保存图片
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
output_path = Path(__file__).resolve().parent / '当前局限性分析图.png'
plt.savefig(output_path, dpi=300)
print(f'图片已保存到 {output_path}')
