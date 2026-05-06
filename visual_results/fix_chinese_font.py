import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import subprocess
import glob

# 设置支持中文的字体
def set_chinese_font():
    # 尝试使用系统中可用的中文字体
    font_list = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Droid Sans Fallback']
    
    font_found = False
    for font_name in font_list:
        try:
            # 检查字体是否可用
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if font_path:
                plt.rcParams['font.family'] = font_name
                print(f"使用字体: {font_name}")
                font_found = True
                break
        except:
            continue
    
    if not font_found:
        # 如果未找到指定字体，使用系统默认的sans-serif字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print("使用默认字体配置")

# 获取所有Python绘图脚本
def fix_all_plots():
    # 设置中文字体
    set_chinese_font()
    
    # 获取当前目录所有的Python脚本
    plot_scripts = glob.glob('*.py')
    
    # 排除当前脚本
    if 'fix_chinese_font.py' in plot_scripts:
        plot_scripts.remove('fix_chinese_font.py')
    
    # 执行每个绘图脚本
    for script in plot_scripts:
        print(f"执行脚本: {script}")
        try:
            subprocess.run(['python3', script], check=True)
            print(f"成功重新生成图表: {script}")
        except subprocess.CalledProcessError:
            print(f"执行失败: {script}")

if __name__ == "__main__":
    # 确保我们在visual_results目录
    if not os.path.basename(os.getcwd()) == 'visual_results':
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
    fix_all_plots()
    print("所有图表中文字体修复完成!") 