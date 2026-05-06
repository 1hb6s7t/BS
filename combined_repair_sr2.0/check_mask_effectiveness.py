#!/usr/bin/env python3
"""
掩码有效性检查工具
帮助用户验证掩码是否能产生明显的修复效果
"""

import cv2
import numpy as np
import os
import argparse
from PIL import Image

def analyze_mask(mask_path, image_path=None):
    """分析掩码的有效性"""
    if not os.path.exists(mask_path):
        print(f"❌ 掩码文件不存在: {mask_path}")
        return False
    
    # 加载掩码
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"❌ 无法读取掩码: {mask_path}")
        return False
    
    print(f"📊 掩码分析报告: {mask_path}")
    print("=" * 50)
    
    # 基本信息
    print(f"📏 掩码尺寸: {mask.shape[1]} x {mask.shape[0]} 像素")
    
    # 二值化掩码
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 统计像素
    total_pixels = mask.shape[0] * mask.shape[1]
    white_pixels = np.sum(binary_mask == 255)
    black_pixels = np.sum(binary_mask == 0)
    white_ratio = white_pixels / total_pixels * 100
    
    print(f"⚪ 白色像素 (需修复区域): {white_pixels:,} ({white_ratio:.1f}%)")
    print(f"⚫ 黑色像素 (保留区域): {black_pixels:,} ({100-white_ratio:.1f}%)")
    
    # 效果预测
    print("\n🔍 修复效果预测:")
    if white_ratio < 0.1:
        print("❗ 警告: 需修复区域过小 (<0.1%)，修复效果可能不明显")
        effectiveness = "很低"
    elif white_ratio < 1.0:
        print("⚠️  注意: 需修复区域较小 (<1%)，修复效果可能较轻微")
        effectiveness = "较低"
    elif white_ratio > 50:
        print("❗ 警告: 需修复区域过大 (>50%)，可能影响修复质量")
        effectiveness = "风险较高"
    else:
        print("✅ 修复区域大小合适，预期会有明显的修复效果")
        effectiveness = "良好"
    
    # 检查连通区域
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_regions = len(contours)
    print(f"🔗 独立修复区域数量: {num_regions}")
    
    if num_regions > 10:
        print("⚠️  修复区域过于分散，建议合并小的区域")
    
    # 最大区域分析
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        largest_area = cv2.contourArea(largest_contour)
        largest_ratio = largest_area / white_pixels * 100 if white_pixels > 0 else 0
        print(f"📍 最大修复区域占比: {largest_ratio:.1f}% (占所有修复区域)")
    
    # 如果提供了原图，进行对比分析
    if image_path and os.path.exists(image_path):
        print(f"\n🖼️  与原图对比分析:")
        img = cv2.imread(image_path)
        if img is not None:
            img_h, img_w = img.shape[:2]
            mask_h, mask_w = mask.shape
            
            if (img_h, img_w) != (mask_h, mask_w):
                print(f"⚠️  尺寸不匹配! 原图: {img_w}x{img_h}, 掩码: {mask_w}x{mask_h}")
                print("   建议调整掩码尺寸以匹配原图")
            else:
                print("✅ 尺寸匹配")
                
                # 创建效果预览
                preview = create_preview(img, binary_mask)
                preview_path = mask_path.replace('.png', '_preview.png').replace('.jpg', '_preview.png')
                cv2.imwrite(preview_path, preview)
                print(f"💡 已生成效果预览: {preview_path}")
    
    print("\n" + "=" * 50)
    print(f"📈 总体评估: {effectiveness}")
    
    # 建议
    print("\n💡 建议:")
    if white_ratio < 1.0:
        print("- 增加需修复区域的大小以获得更明显的效果")
    if num_regions > 10:
        print("- 合并临近的小区域")
    if white_ratio > 50:
        print("- 减少修复区域，保留更多原始内容")
    
    print("- 确保掩码中白色区域对应需要修复的部分")
    print("- 使用图像编辑软件细化掩码边缘")
    
    return True

def create_preview(img, mask):
    """创建修复效果预览"""
    # 将掩码应用到图像上，显示哪些区域会被修复
    preview = img.copy()
    
    # 在需修复区域添加红色overlay
    overlay = preview.copy()
    overlay[mask == 255] = [0, 0, 255]  # 红色标记修复区域
    
    # 混合原图和overlay
    alpha = 0.3
    preview = cv2.addWeighted(preview, 1-alpha, overlay, alpha, 0)
    
    return preview

def create_test_mask(output_path, width=512, height=512, pattern='center_circle'):
    """创建测试掩码"""
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if pattern == 'center_circle':
        # 中心圆形
        center = (width//2, height//2)
        radius = min(width, height) // 8
        cv2.circle(mask, center, radius, 255, -1)
    elif pattern == 'random_spots':
        # 随机斑点
        for _ in range(20):
            x = np.random.randint(50, width-50)
            y = np.random.randint(50, height-50)
            radius = np.random.randint(10, 30)
            cv2.circle(mask, (x, y), radius, 255, -1)
    elif pattern == 'scratch':
        # 划痕
        cv2.line(mask, (100, 100), (width-100, height-100), 255, 15)
        cv2.line(mask, (200, 50), (width-50, 200), 255, 10)
    
    cv2.imwrite(output_path, mask)
    print(f"✅ 测试掩码已创建: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="掩码有效性检查工具")
    parser.add_argument("--mask", type=str, help="掩码图像路径")
    parser.add_argument("--image", type=str, help="原始图像路径（可选）")
    parser.add_argument("--create_test", type=str, help="创建测试掩码的路径")
    parser.add_argument("--pattern", type=str, default="center_circle",
                       choices=['center_circle', 'random_spots', 'scratch'],
                       help="测试掩码模式")
    
    args = parser.parse_args()
    
    if args.create_test:
        create_test_mask(args.create_test, pattern=args.pattern)
        return
    
    if not args.mask:
        print("请提供掩码路径，或使用 --create_test 创建测试掩码")
        print("使用示例:")
        print("  python check_mask_effectiveness.py --mask mask.png --image input.jpg")
        print("  python check_mask_effectiveness.py --create_test test_mask.png --pattern center_circle")
        return
    
    analyze_mask(args.mask, args.image)

if __name__ == "__main__":
    main() 