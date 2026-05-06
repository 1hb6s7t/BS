#!/usr/bin/env python3
"""
图像修复与超分辨率工具快速启动脚本
双击此文件即可启动图形界面
"""

import sys
import os
import traceback

def main():
    """主启动函数"""
    try:
        print("正在启动图像修复与超分辨率处理工具...")
        print("=" * 50)
        
        # 检查Python版本0
        if sys.version_info < (3, 7):
            print("错误: 需要Python 3.7或更高版本")
            input("按回车键退出...")
            return
        
        # 添加当前目录到路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        # 检查必要依赖。MindSpore 是深度模型后端的可选依赖；缺失时 GUI 会使用 classic 后端。
        required_modules = [
            'tkinter', 'PIL', 'numpy', 'cv2'
        ]
        optional_modules = ['mindspore']
        
        missing_modules = []
        for module in required_modules:
            try:
                if module == 'PIL':
                    import PIL
                elif module == 'cv2':
                    import cv2
                else:
                    __import__(module)
                print(f"✓ {module} 已安装")
            except ImportError:
                missing_modules.append(module)
                print(f"✗ {module} 未安装")
        
        if missing_modules:
            print("\n缺少以下依赖包:")
            for module in missing_modules:
                if module == 'PIL':
                    print(f"  pip install Pillow")
                elif module == 'cv2':
                    print(f"  pip install opencv-python")
                else:
                    print(f"  pip install {module}")
            print("\n请安装缺少的依赖包后重试")
            input("按回车键退出...")
            return

        for module in optional_modules:
            try:
                __import__(module)
                print(f"✓ {module} 已安装（可使用深度模型后端）")
            except ImportError:
                print(f"! {module} 未安装，将使用 classic 后端运行")
        
        print("\n所有依赖检查完成！")
        print("正在启动GUI界面...")
        
        # 导入并启动GUI
        from combined_repair_sr_gui import launch_gui
        launch_gui()
        
    except Exception as e:
        print(f"\n启动失败: {e}")
        print("\n详细错误信息:")
        traceback.print_exc()
        input("按回车键退出...")

if __name__ == "__main__":
    main() 
