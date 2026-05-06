#!/usr/bin/env python3
"""
批量图像处理示例脚本
演示如何使用优化后的联合框架进行批量图像修复和超分辨率处理
"""

import os
import sys
import time
from pathlib import Path
import argparse

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from combined_repair_sr_optimized import CombinedProcessor, ModelConfig, ImageProcessor

def batch_process(input_dir, mask_dir, output_dir, cra_ckpt, srgan_ckpt, config=None):
    """
    批量处理图像
    
    Args:
        input_dir: 输入图像目录
        mask_dir: 掩码图像目录
        output_dir: 输出目录
        cra_ckpt: CRA模型路径
        srgan_ckpt: SRGAN模型路径
        config: 模型配置（可选）
    """
    
    # 创建配置
    if config is None:
        config = ModelConfig()
    
    # 创建处理器
    processor = CombinedProcessor(config)
    
    # 加载模型
    print("正在加载模型...")
    cra_success, srgan_success = processor.load_models(cra_ckpt, srgan_ckpt)
    
    if not (cra_success and srgan_success):
        print("错误: 模型加载失败")
        return False
    
    print("模型加载成功！")
    
    # 获取所有图像文件
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    input_files = []
    
    for file_path in Path(input_dir).rglob('*'):
        if file_path.suffix.lower() in image_extensions:
            input_files.append(file_path)
    
    if not input_files:
        print(f"错误: 在 {input_dir} 中未找到图像文件")
        return False
    
    print(f"找到 {len(input_files)} 个图像文件")
    
    # 批量处理
    processed_count = 0
    failed_count = 0
    total_time = 0
    
    for i, input_file in enumerate(input_files):
        try:
            print(f"\n处理进度: {i+1}/{len(input_files)} - {input_file.name}")
            
            # 构建掩码文件路径
            mask_file = Path(mask_dir) / input_file.name
            if not mask_file.exists():
                # 尝试不同的扩展名
                mask_file = Path(mask_dir) / f"{input_file.stem}.png"
                if not mask_file.exists():
                    mask_file = Path(mask_dir) / f"{input_file.stem}.jpg"
            
            if not mask_file.exists():
                print(f"  警告: 未找到对应的掩码文件，跳过 {input_file.name}")
                failed_count += 1
                continue
            
            # 创建输出子目录
            output_subdir = Path(output_dir) / input_file.stem
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            # 记录开始时间
            start_time = time.time()
            
            # 处理图像
            def progress_callback(message):
                print(f"  {message}")
            
            success, result = processor.process_image(
                str(input_file), 
                str(mask_file), 
                str(output_subdir),
                progress_callback
            )
            
            # 计算处理时间
            process_time = time.time() - start_time
            total_time += process_time
            
            if success:
                processed_count += 1
                print(f"  ✓ 处理完成 ({process_time:.2f}秒): {result}")
            else:
                failed_count += 1
                print(f"  ✗ 处理失败: {result}")
                
        except Exception as e:
            failed_count += 1
            print(f"  ✗ 处理 {input_file.name} 时出错: {e}")
    
    # 输出统计信息
    print(f"\n" + "="*50)
    print("批量处理完成！")
    print(f"总文件数: {len(input_files)}")
    print(f"成功处理: {processed_count}")
    print(f"处理失败: {failed_count}")
    print(f"总处理时间: {total_time:.2f}秒")
    if processed_count > 0:
        print(f"平均处理时间: {total_time/processed_count:.2f}秒/图像")
    print(f"结果保存在: {output_dir}")
    
    return processed_count > 0

def create_example_config():
    """创建示例配置"""
    config = ModelConfig()
    
    # 可以根据需要调整参数
    config.input_size = 512
    config.scale = 4
    config.device_target = 'GPU'  # 或 'CPU'
    
    return config

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="批量图像修复和超分辨率处理")
    parser.add_argument("--input_dir", type=str, required=True, 
                       help="输入图像目录")
    parser.add_argument("--mask_dir", type=str, required=True, 
                       help="掩码图像目录")
    parser.add_argument("--output_dir", type=str, required=True, 
                       help="输出目录")
    parser.add_argument("--cra_ckpt", type=str, default="",
                       help="CRA模型检查点路径")
    parser.add_argument("--srgan_ckpt", type=str, default="",
                       help="SRGAN模型检查点路径")
    parser.add_argument("--backend", type=str, default="auto",
                       choices=["auto", "deep", "classic"],
                       help="处理后端")
    parser.add_argument("--device", type=str, default="GPU",
                       choices=['GPU', 'CPU', 'Ascend'],
                       help="运行设备")
    parser.add_argument("--scale", type=int, default=4,
                       choices=[2, 4, 8],
                       help="超分辨率倍数")
    parser.add_argument("--input_size", type=int, default=512,
                       choices=[256, 512, 1024],
                       help="输入图像尺寸")
    
    args = parser.parse_args()
    
    # 验证输入
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        return
    
    if not os.path.exists(args.mask_dir):
        print(f"错误: 掩码目录不存在: {args.mask_dir}")
        return
    
    if args.backend == "deep" and not os.path.exists(args.cra_ckpt):
        print(f"错误: CRA模型文件不存在: {args.cra_ckpt}")
        return
    
    if args.backend == "deep" and not os.path.exists(args.srgan_ckpt):
        print(f"错误: SRGAN模型文件不存在: {args.srgan_ckpt}")
        return
    
    # 创建配置
    config = ModelConfig()
    config.backend = args.backend
    config.device_target = args.device
    config.scale = args.scale
    config.input_size = args.input_size
    config.validate()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("批量图像处理开始...")
    print(f"输入目录: {args.input_dir}")
    print(f"掩码目录: {args.mask_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"运行设备: {args.device}")
    print(f"超分辨率倍数: {args.scale}x")
    print(f"输入尺寸: {args.input_size}")
    
    # 执行批量处理
    success = batch_process(
        args.input_dir,
        args.mask_dir, 
        args.output_dir,
        args.cra_ckpt,
        args.srgan_ckpt,
        config
    )
    
    if success:
        print("\n批量处理成功完成！")
    else:
        print("\n批量处理失败！")

if __name__ == "__main__":
    # 使用示例
    print("批量图像处理示例脚本")
    print("=" * 50)
    
    # 如果没有提供命令行参数，显示使用示例
    if len(sys.argv) == 1:
        print("使用方法:")
        print("python batch_process_example.py \\")
        print("    --input_dir /path/to/input/images \\")
        print("    --mask_dir /path/to/mask/images \\")
        print("    --output_dir /path/to/output \\")
        print("    --cra_ckpt /path/to/cra.ckpt \\")
        print("    --srgan_ckpt /path/to/srgan.ckpt \\")
        print("    --device GPU \\")
        print("    --scale 4 \\")
        print("    --input_size 512")
        print("\n目录结构示例:")
        print("input_dir/")
        print("├── image1.jpg")
        print("├── image2.png")
        print("└── ...")
        print("\nmask_dir/")
        print("├── image1.jpg  (或 image1.png)")
        print("├── image2.png  (或 image2.jpg)")
        print("└── ...")
        print("\n处理后的结果将保存在output_dir中，每个图像会创建单独的子目录")
    else:
        main() 
