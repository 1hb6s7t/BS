#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import time
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def check_graphviz_installed():
    """检查是否安装了系统级别的Graphviz库"""
    try:
        # 尝试运行dot -V命令检查Graphviz是否已安装
        subprocess.run(['dot', '-V'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("系统级Graphviz已安装。")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("警告: 系统级Graphviz未安装。")
        print("请执行 'sudo apt-get install -y graphviz' (Ubuntu/Debian) 或对应的系统包管理器命令安装Graphviz。")
        return False

def run_script(script_name):
    """运行指定的Python脚本"""
    try:
        script_path = os.path.join(SCRIPT_DIR, script_name)
        result = subprocess.run([sys.executable, script_path],
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True,
                               cwd=SCRIPT_DIR,
                               check=True)
        print(f"成功运行 {script_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"运行 {script_name} 时出错:")
        print(f"标准输出: {e.stdout}")
        print(f"标准错误: {e.stderr}")
        return False

def generate_diagrams():
    """生成所有架构图"""
    # 检查并确保首先安装了Graphviz
    if not check_graphviz_installed():
        print("错误: 缺少系统级Graphviz，无法生成图表。")
        return False

    # 待运行的图表生成脚本列表
    diagram_scripts = [
        'generate_architecture_diagram.py',
        'generate_data_flow_diagram.py',
        'generate_model_architecture.py',
        'generate_data_preprocessing_training.py',
        'generate_inference_flow.py',
        'generate_model_call_relationships.py',
        'generate_separate_data_processing.py',
        'generate_separate_model_inference.py',
        'generate_separate_model_training.py'
    ]
    
    # 运行每个脚本
    total_scripts = len(diagram_scripts)
    successful_scripts = 0
    
    for script in diagram_scripts:
        if os.path.exists(os.path.join(SCRIPT_DIR, script)):
            print(f"正在运行 {script}...")
            success = run_script(script)
            if success:
                successful_scripts += 1
            time.sleep(1)  # 短暂延迟，避免资源争用
        else:
            print(f"警告: 找不到脚本 {script}")
    
    print(f"成功生成 {successful_scripts}/{total_scripts} 个图表")
    
    # 列出生成的文件
    diagram_extensions = ['.png']
    generated_files = []
    
    for root, _, files in os.walk('.'):
        for file in files:
            if any(file.endswith(ext) for ext in diagram_extensions):
                file_path = os.path.join(root, file)
                generated_files.append((file_path, os.path.getsize(file_path) / 1024))
    
    if generated_files:
        print("\n生成的图表文件:")
        for file_path, size_kb in sorted(generated_files):
            print(f"- {file_path} ({size_kb:.1f} KB)")
    
    return successful_scripts == total_scripts

if __name__ == "__main__":
    print("开始生成CRA和SRGAN模型的所有架构图...")
    success = generate_diagrams()
    if success:
        print("所有架构图生成完成！")
    else:
        print("生成架构图过程中遇到问题，请检查上述输出。") 
