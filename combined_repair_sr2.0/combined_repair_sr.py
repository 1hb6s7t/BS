#!/usr/bin/env python3
import os
import time
import argparse
import cv2
import numpy as np
import logging
import traceback
import glob
import math
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# MindSpore导入
import mindspore
from mindspore import Tensor, context, load_checkpoint, load_param_into_net
import mindspore.nn as nn
import mindspore.ops as ops

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 全局图像引用字典，防止垃圾回收
image_refs = {}

# 预定义全局变量
ApplyAttention2 = None

# 定义完整的ApplyAttention2实现，取代导入版本
class ApplyAttention2Impl(nn.Cell):
    """完整重写的CRA中的注意力应用层"""
    def __init__(self, shp=None, shp_att=None, sz=None):
        super(ApplyAttention2Impl, self).__init__()
        self.kernel = 3
        self.stride = 1
        self.sz = sz
        
        # 重新映射所需操作
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.bmm = ops.BatchMatMul()
        self.resize_bilinear = ops.ResizeBilinearV2()
        self.tile = ops.Tile()
        
        # 记录初始化参数
        self.shp = shp
        self.shp_att = shp_att
        
        logger.debug(f"初始化ApplyAttention2: sz={sz}, kernel={self.kernel}, stride={self.stride}")
    
    def construct(self, residual1, corres):
        """应用注意力机制的实现 - 移除try-except，适应MindSpore GRAPH模式"""
        # 获取输入形状
        orig_shape = residual1.shape
        b, c, h, w = orig_shape
        
        # 确保correspondence与residual1的batch_size匹配
        if corres.shape[0] != b:
            repeat_times = (b,) + (1,) * (len(corres.shape) - 1)
            corres = self.tile(corres, repeat_times)
        
        # 使用ops.unfold
        h_x = ops.unfold(residual1, kernel_size=self.kernel, 
                      stride=self.stride, padding=0)
        
        # 计算滑窗数量
        unfold_h_w = ((h - self.kernel) // self.stride + 1) ** 2
        kernel_size = self.kernel ** 2
        
        # 重塑展开的特征图
        h_x_reshaped = self.reshape(h_x, (b, c * kernel_size, unfold_h_w))
        h_x_transposed = self.transpose(h_x_reshaped, (0, 2, 1))
        
        # 执行批次矩阵乘法 - 添加安全检查
        corres_shape = corres.shape
        if len(corres_shape) == 3:
            output = self.bmm(h_x_transposed, corres)
            
            # 重塑回特征图格式
            output = self.transpose(output, (0, 2, 1))
            output = self.reshape(output, (b, c * kernel_size, unfold_h_w))
            
            # 处理输出尺寸
            output_h_w = int((unfold_h_w ** 0.5) * self.stride + self.kernel - self.stride)
            
            # 如果输出尺寸与输入不匹配，进行上采样
            if output_h_w != h:
                output_h = output_w = int(unfold_h_w ** 0.5)
                output = self.reshape(output, (b, c, output_h, output_w))
                output = self.resize_bilinear(output, (h, w))
            else:
                output = self.reshape(output, (b, c, h, w))
            
            return output
        else:
            # 如果corres形状不正确，直接返回原始输入
            return residual1

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="CRA图像修复和SRGAN超分辨率组合工具")
    parser.add_argument("--input_image", type=str, default=os.path.join(PROJECT_ROOT, 'CRA', 'test', 'images', '2.png'),
                      help="输入图像的路径")
    parser.add_argument("--mask_image", type=str, default=os.path.join(PROJECT_ROOT, 'CRA', 'test', 'masks', '2.png'),
                      help="掩码图像的路径")
    parser.add_argument('--output_dir', type=str, default='./output', help='输出目录路径')
    parser.add_argument('--cra_ckpt', type=str, default=os.path.join(PROJECT_ROOT, 'CRA', 'cra.ckpt'), help='CRA模型检查点路径')
    parser.add_argument('--srgan_ckpt', type=str, default=os.path.join(PROJECT_ROOT, 'ckpt', 'pretrained_generator_epoch100000.ckpt'), help='SRGAN模型检查点路径')
    parser.add_argument('--input_size', type=int, default=512, help='CRA输入图像大小')
    parser.add_argument('--times', type=int, default=8, help='CRA处理的放大倍数')
    parser.add_argument('--scale', type=int, default=4, help='SRGAN超分辨倍数')
    parser.add_argument('--device_target', type=str, default='GPU', choices=['Ascend', 'GPU', 'CPU'], help='运行平台')
    parser.add_argument('--train_batchsize', type=int, default=1, help='CRA训练批次大小')
    parser.add_argument('--attention_type', type=str, default='SOFT', help='CRA注意力类型')
    return parser.parse_args()

def setup_context(args):
    """设置MindSpore上下文"""
    # 使用PYNATIVE_MODE而非GRAPH_MODE以支持更多Python语法
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)
    logger.info(f"运行平台设置为: {args.device_target}, 模式: PYNATIVE_MODE")
    
    # 设置CPU线程数
    if args.device_target == 'CPU':
        context.set_context(max_call_depth=2000)
        context.set_context(max_device_memory="1GB")

# ====================== CRA 模型定义 ======================
class GatedGenerator(nn.Cell):
    """
    CRA生成器网络 - 简化定义，需要从CRA实际代码中导入
    """
    def __init__(self, args):
        super(GatedGenerator, self).__init__()
        # 这里是简化的网络定义
        # 实际使用时，应从CRA/src/models/inpainting_network.py中导入GatedGenerator类
        pass

    def construct(self, x, mask):
        # 实际使用时，由CRA模型实现
        pass

# ====================== SRGAN 模型定义 ======================
class ResidualBlock(nn.Cell):
    """SRGAN残差块结构"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, has_bias=True, pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, has_bias=True, pad_mode='pad')
        self.bn2 = nn.BatchNorm2d(channels)

    def construct(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        return out

class SubpixelConvolutionLayer(nn.Cell):
    """SRGAN子像素卷积层"""
    def __init__(self, channels):
        super(SubpixelConvolutionLayer, self).__init__()
        self.conv = nn.Conv2d(channels, channels*4, kernel_size=3, stride=1, padding=1, has_bias=True, pad_mode='pad')
        self.pixel_shuffle = ops.DepthToSpace(2)
        self.prelu = nn.PReLU(channels)

    def construct(self, x):
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        out = self.prelu(out)
        return out

class Generator(nn.Cell):
    """SRGAN生成器网络"""
    def __init__(self, upscale_factor):
        super(Generator, self).__init__()
        # 计算子像素卷积层数量
        num_subpixel_convolution_layers = int(math.log(upscale_factor, 2))
        # 第一层
        self.conv1 = nn.SequentialCell(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4, has_bias=True, pad_mode='pad'),
            nn.PReLU(channel=64))

        # 16个残差块
        trunk = []
        for _ in range(16):
            trunk.append(ResidualBlock(64))
        self.trunk = nn.SequentialCell(*trunk)

        # 残差块后的第二层卷积
        self.conv2 = nn.SequentialCell(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, has_bias=True, pad_mode='pad'),
            nn.PReLU(channel=64)
        )

        # 子像素卷积层
        subpixel_conv_layers = []
        for _ in range(num_subpixel_convolution_layers):
            subpixel_conv_layers.append(SubpixelConvolutionLayer(64))
        self.subpixel_conv = nn.SequentialCell(*subpixel_conv_layers)

        # 最终输出层
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4, has_bias=True, pad_mode='pad')
        self.tanh = nn.Tanh()

    def construct(self, x):
        conv1 = self.conv1(x)
        trunk = self.trunk(conv1)
        conv2 = self.conv2(trunk)
        out = conv1+conv2
        out = self.subpixel_conv(out)
        out = self.conv3(out)
        out = self.tanh(out)
        return out

# ====================== 功能函数 ======================
def load_image(image_path):
    """加载图像文件"""
    try:
        # 使用OpenCV读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法加载图像: {image_path}")
        # BGR转RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        logger.info(f"已加载图像: {image_path}，尺寸: {img.shape}")
        return img
    except Exception as e:
        logger.error(f"加载图像时出错: {e}")
        logger.debug(traceback.format_exc())
        return None

def load_mask(mask_path, img_shape=None):
    """加载掩码图像并调整尺寸匹配输入图像"""
    try:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"无法加载掩码: {mask_path}")
            
        # 记录原始尺寸
        original_shape = mask.shape
        logger.info(f"原始掩码尺寸: {original_shape}")
        
        # 如果提供了图像尺寸且掩码尺寸不匹配，则调整掩码尺寸
        if img_shape is not None and (original_shape[0] != img_shape[0] or original_shape[1] != img_shape[1]):
            logger.info(f"调整掩码尺寸以匹配图像: {img_shape[:2]}")
            mask = cv2.resize(mask, (img_shape[1], img_shape[0]))
            
        # 确保掩码是二值的（0或255）
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        logger.info(f"处理后掩码尺寸: {mask.shape}")
        return mask
    except Exception as e:
        logger.error(f"加载掩码时出错: {e}")
        logger.debug(traceback.format_exc())
        return None

def preprocess_for_cra(img, mask, args):
    """预处理图像和掩码，为CRA推理做准备"""
    try:
        # 图像预处理
        img_tensor = Tensor(img, dtype=mindspore.float32)
        
        # 掩码预处理 - 简化处理以避免维度错误
        mask_tensor = Tensor(mask, dtype=mindspore.float32)
        
        return img_tensor, mask_tensor
    except Exception as e:
        logger.error(f"预处理图像和掩码时出错: {e}")
        logger.debug(traceback.format_exc())
        return None, None

def build_inference_net(raw_img_ph, raw_mask_ph, model_gen, args):
    """
    完整的CRA网络测试模型，包括图像预处理、生成器生成和输出以及图像后处理操作。
    
    参数:
        raw_img_ph: 从文件读取的图像，处理为[1,3,512,512]格式，数据类型为float32，并归一化。
        raw_mask_ph: 从文件读取的掩码，处理为[1,3,512,512]格式，数据类型为float32，并归一化。
        model_gen: 生成网络。
        args: 选项类。
        
    返回:
        raw_size_output: 大型测试输出结果。
        raw_img_ph: 从文件读取的图像。
        raw_mask_ph: 从文件读取的掩码。
    """
    # 保持图像引用，防止垃圾回收
    global image_refs
    ref_id = id(raw_img_ph)
    image_refs[ref_id] = raw_img_ph
    
    # 处理输入图像
    raw_img = ops.ExpandDims()(raw_img_ph, 0)
    raw_img = raw_img.astype(mindspore.float32)
    raw_img = ops.Transpose()(raw_img, (0, 3, 1, 2))
    resize = ops.ResizeNearestNeighbor((args.times * args.input_size, args.times * args.input_size))
    large_img = resize(raw_img)
    large_img = ops.Reshape()(large_img, (1, 3, args.times * args.input_size, args.times * args.input_size))
    large_img = large_img / 127.5 - 1
    net = nn.Unfold([1, args.times, args.times, 1], [1, args.times, args.times, 1], [1, 1, 1, 1], 'same')
    small_img = net(large_img)
    small_img = ops.Transpose()(small_img, (0, 2, 3, 1))
    small_img = ops.Reshape()(small_img, (1, args.input_size, args.input_size, args.times, args.times, 3))
    small_img = ops.ReduceMean(False)(small_img, axis=(3, 4))
    small_img = ops.Transpose()(small_img, (0, 3, 1, 2))
    
    # 处理输入掩码 - 修复掩码的维度问题
    raw_mask = raw_mask_ph.astype(mindspore.float32)
    # 确保掩码是2D的 [H,W]
    if len(raw_mask.shape) == 3 and raw_mask.shape[2] == 1:
        raw_mask = raw_mask[:, :, 0]  # 移除单通道维度
        
    # 添加批次维度
    raw_mask = ops.ExpandDims()(raw_mask, 0)
    # 添加通道维度
    raw_mask = ops.ExpandDims()(raw_mask, 0)
    
    # 调整到正确的尺寸
    resize = ops.ResizeNearestNeighbor((args.input_size, args.input_size))
    small_mask = resize(raw_mask)
    
    # 确保掩码是归一化的
    small_mask = small_mask / 255.0
    
    # CRA期望掩码值为[0,1]，1表示需要修复的区域，0表示保留区域
    small_mask = 1.0 - small_mask  # 反转掩码，确保1表示要修复的区域
    
    # 输入图像和掩码到生成器
    x2, _, corres = build_inference_graph(real=small_img, mask=small_mask, model_gen=model_gen)
    
    # 后处理
    large_output, _, _, _ = post_processing(large_img, small_img, x2, small_mask, corres, args)
    
    # 调整回原尺寸
    raw_size_output = resize_back(raw_img, large_output, small_mask)
    return raw_size_output, raw_img_ph, raw_mask_ph

def build_inference_graph(real, mask, model_gen):
    """将real和mask输入到生成器并输出结果"""
    # 不使用try-except以适应MindSpore
    mask = mask[0:1, 0:1, :, :]
    x = real * (1. - mask)
    _, x2, corres = model_gen(x, mask)
    fake = x2
    fake_patched = fake * mask + x * (1 - mask)
    return x2, fake_patched, corres

def gaussian_kernel(size, std):
    """返回高斯核"""
    x = np.arange(-(size // 2), (size // 2) + 1)
    gauss = np.exp(-0.5 * np.square(x) / np.square(std))
    gauss = gauss / np.sum(gauss)
    kk = np.outer(gauss, gauss)
    return kk / np.sum(kk)

def resize_back(raw_img, large_output, small_mask):
    """将测试输出结果处理为与原始输入图像相同的尺寸"""
    raw_shp = raw_img.shape
    resize_op = ops.ResizeBilinearV2()
    raw_size_output = resize_op(large_output, (raw_shp[2], raw_shp[3]))
    raw_size_output = raw_size_output.astype(mindspore.float32)
    
    gauss_kernel = gaussian_kernel(7, 1.)
    gauss_kernel = Tensor(gauss_kernel)
    gauss_kernel = gauss_kernel.astype(mindspore.float32)
    gauss_kernel = ops.ExpandDims()(gauss_kernel, 2)
    gauss_kernel = ops.ExpandDims()(gauss_kernel, 3)
    a, b, c, d = ops.Shape()(gauss_kernel)
    gauss_kernel = ops.Transpose()(gauss_kernel, (3, 2, 0, 1))
    conv = nn.Conv2d(c, d, (a, b), 1, pad_mode='same', padding=0, weight_init=gauss_kernel, data_format='NCHW')
    
    mask = conv(small_mask[:, 0:1, :, :])
    mask = resize_op(mask, (raw_shp[2], raw_shp[3]))
    mask = mask.astype(mindspore.float32)
    raw_size_output = raw_size_output * mask + raw_img * (1 - mask)
    raw_size_output = ops.Transpose()(raw_size_output, (0, 2, 3, 1))
    raw_size_output = raw_size_output.astype(mindspore.uint8)
    return raw_size_output

def post_processing(large_img, small_img, low_base, small_mask, corres, args):
    """
    从原始输入中减去大的模糊图像以计算上下文残差，并通过注意力传输模块计算聚合残差。
    将聚合残差添加到上采样的生成器修复结果中。
    """
    try:
        high_raw = large_img
        resize_op = ops.ResizeBilinearV2()
        scale_factor = args.times
        input_shape = ops.Shape()(small_img)
        output_h = int(input_shape[2] * scale_factor)
        output_w = int(input_shape[3] * scale_factor)
        low_raw = resize_op(small_img, size=(output_h, output_w))
        
        mask = 1 - small_mask
        to_shape = list(ops.Shape()(mask))[2:]
        to_shape[0], to_shape[1] = int(to_shape[0] * args.times), int(to_shape[1] * args.times)
        resize = ops.ResizeNearestNeighbor((to_shape[0], to_shape[1]))
        mask = resize(mask)
        
        residual1 = (high_raw - low_raw) * mask
        
        # 使用我们的新实现
        sz = args.input_size * args.times
        
        # 直接创建ApplyAttention2Impl实例
        attention_module = ApplyAttention2Impl([1, 3, sz, sz], [1, 1024, 32, 32], sz)
        
        # 正常情况下应用注意力机制
        residual = attention_module(residual1, corres)
        logger.info("注意力处理成功完成")
        
        low_base_shape = ops.Shape()(low_base)
        low_base_output_h = int(low_base_shape[2] * scale_factor)
        low_base_output_w = int(low_base_shape[3] * scale_factor)
        low_base = resize_op(low_base, size=(low_base_output_h, low_base_output_w))
        
        # 确保residual和low_base尺寸相同
        if residual.shape != low_base.shape:
            logger.warning(f"尺寸不匹配: residual {residual.shape}, low_base {low_base.shape}")
            # 调整residual尺寸以匹配low_base
            residual = resize_op(residual, size=(low_base_output_h, low_base_output_w))
        
        x = low_base + residual
        x = ops.clip_by_value(x, -1, 1)
        x = (x + 1.) * 127.5
        return x, low_raw, low_base, residual
    except Exception as e:
        logger.error(f"后处理过程出错: {e}")
        logger.debug(traceback.format_exc())
        # 发生错误时，尝试返回一个有效的结果
        try:
            # 直接返回放大的low_base作为结果
            low_base_shape = ops.Shape()(low_base)
            low_base_output_h = int(low_base_shape[2] * scale_factor)
            low_base_output_w = int(low_base_shape[3] * scale_factor)
            low_base_resized = resize_op(low_base, size=(low_base_output_h, low_base_output_w))
            result = (low_base_resized + 1.) * 127.5
            return result, low_raw, low_base, None
        except:
            # 如果上面的尝试也失败，直接抛出原始异常
            raise

def repair_image(img, mask, args):
    """使用CRA修复图像"""
    try:
        logger.info("使用CRA模型修复图像")
        
        # 添加CRA模型期望的所有必要参数
        args.train_batchsize = 1
        args.attention_type = getattr(args, 'attention_type', 'SOFT')
        args.sc = getattr(args, 'sc', False)
        args.dilation = getattr(args, 'dilation', 2)
        args.seed = getattr(args, 'seed', 2022)
        args.nThreads = getattr(args, 'nThreads', 1)
        args.max_dataset_size = getattr(args, 'max_dataset_size', float("inf"))
        args.gpu_ids = getattr(args, 'gpu_ids', [0])
        args.dis_iter = getattr(args, 'dis_iter', 1)
        
        # 直接使用已设置的ApplyAttention2Impl
        global ApplyAttention2
        if ApplyAttention2 is None:
            ApplyAttention2 = ApplyAttention2Impl
            logger.info("设置ApplyAttention2为自定义实现")
        
        # 导入必要的CRA模型
        from CRA.src.models.inpainting_network import GatedGenerator
        
        # 加载CRA模型
        cra_model = GatedGenerator(args)
        param_dict = load_checkpoint(args.cra_ckpt)
        load_param_into_net(cra_model, param_dict)
        logger.info("CRA模型加载成功")
        
        # 预处理图像和掩码
        input_img_tensor = Tensor(img)
        input_mask_tensor = Tensor(255 - mask)
        
        # 执行推理
        outputs, _, _ = build_inference_net(input_img_tensor, input_mask_tensor, cra_model, args)
        logger.info("CRA推理完成")
        
        # 提取修复后的图像
        repaired_img = outputs[0].asnumpy()
        
        # 释放资源
        del cra_model, param_dict
        import gc
        gc.collect()
        
        logger.info("CRA修复完成，资源已释放")
        return repaired_img
        
    except Exception as e:
        logger.error(f"修复图像时出错: {e}")
        logger.debug(traceback.format_exc())
        # 返回原始图像作为备选
        logger.warning("返回原始图像作为备选结果")
        return img

def enhance_image(repaired_img, args):
    """使用SRGAN增强图像"""
    try:
        logger.info("使用SRGAN模型进行超分辨率处理")
        
        # 创建SRGAN生成器
        generator = Generator(args.scale)
        
        # 加载预训练参数
        logger.debug(f"加载SRGAN模型检查点: {args.srgan_ckpt}")
        params = load_checkpoint(args.srgan_ckpt)
        load_param_into_net(generator, params)
        logger.info("SRGAN模型加载成功")
        
        # 执行推理
        logger.info("正在执行SRGAN推理...")
        
        # 记录输入图像形状
        logger.debug(f"SRGAN输入图像形状: {repaired_img.shape}")
        
        # 预处理图像为NCHW格式，范围[-1, 1]
        repaired_img = repaired_img.astype(np.float32) / 127.5 - 1.0
        repaired_img = np.transpose(repaired_img, (2, 0, 1))  # HWC -> CHW
        repaired_img = np.expand_dims(repaired_img, 0)  # 添加批次维度
        repaired_tensor = Tensor(repaired_img, dtype=mindspore.float32)
        logger.debug(f"预处理后张量形状: {repaired_tensor.shape}")
        
        # 推理
        enhanced_output = generator(repaired_tensor)
        logger.debug(f"SRGAN推理输出形状: {enhanced_output.shape}")
        
        # 后处理
        # 使用ReduceSum去除批次维度
        op = ops.ReduceSum(keep_dims=False)
        enhanced_output = op(enhanced_output, 0).asnumpy()
        
        # 限制到[-1, 1]范围
        enhanced_output = np.clip(enhanced_output, -1.0, 1.0)
        
        # 转换到[0, 1]范围并调整维度顺序为(H,W,C)
        enhanced_output = ((enhanced_output + 1.0) / 2.0).transpose(1, 2, 0)
        logger.debug(f"增强后的图像形状: {enhanced_output.shape}, 范围: [{enhanced_output.min():.4f}, {enhanced_output.max():.4f}]")
        
        # 清除模型和内存
        del generator, repaired_tensor
        import gc
        gc.collect()
        
        logger.info("SRGAN超分完成")
        return enhanced_output
    except Exception as e:
        logger.error(f"SRGAN增强过程中出错: {e}")
        logger.debug(traceback.format_exc())
        raise

def save_result(img, filename, output_dir):
    """保存结果图像"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        # 确保数值在[0, 255]范围内
        if img.max() <= 1.0:
            img = img * 255.0
        
        img = img.astype(np.uint8)
        result = Image.fromarray(img)
        result.save(output_path)
        logger.info(f"结果已保存到: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"保存结果图像时出错: {e}")
        logger.debug(traceback.format_exc())
        raise

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置MindSpore上下文
    setup_context(args)
    
    # 尝试替换CRA模块的ApplyAttention2
    try:
        import sys
        import importlib.util
        # 检查CRA模块是否存在
        if importlib.util.find_spec("CRA.src.models.compute_attention") is not None:
            # 先导入我们的ApplyAttention2Impl类
            sys.modules['CRA.src.models.compute_attention.ApplyAttention2'] = ApplyAttention2Impl
            logger.info("已替换CRA模块中的ApplyAttention2实现")
    except Exception as e:
        logger.warning(f"替换CRA模块ApplyAttention2失败: {e}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载输入图像和掩码
    input_img = load_image(args.input_image)
    if input_img is None:
        logger.error("加载图像失败，程序终止")
        return

    # 加载掩码并调整大小以匹配输入图像
    mask_img = load_mask(args.mask_image, input_img.shape)
    
    if mask_img is None:
        logger.error("加载掩码失败，程序终止")
        return
    
    # 保存中间结果的文件名
    base_filename = os.path.splitext(os.path.basename(args.input_image))[0]
    repaired_filename = f"{base_filename}_repaired.png"
    enhanced_filename = f"{base_filename}_enhanced.png"
    
    # 全局计算进度标志
    continue_to_srgan = True
    repaired_img = None
    
    try:
        # 步骤1：使用CRA修复图像
        logger.info("开始图像修复流程")
        repaired_img = repair_image(input_img, mask_img, args)
        
        if repaired_img is None:
            logger.error("图像修复失败，使用原始图像作为备选")
            repaired_img = input_img  # 使用原始图像作为备选
            continue_to_srgan = True  # 仍然继续SRGAN处理
        
        # 保存修复后的图像
        repaired_path = save_result(repaired_img, repaired_filename, args.output_dir)
        logger.info(f"修复后的图像已保存到: {repaired_path}")
        
    except Exception as e:
        logger.error(f"修复过程中发生错误: {e}")
        logger.debug(traceback.format_exc())
        # 使用原始图像作为备选
        repaired_img = input_img
        continue_to_srgan = True  # 仍然继续SRGAN处理
        
        try:
            # 尝试保存原始图像作为"修复"结果
            repaired_path = save_result(repaired_img, repaired_filename, args.output_dir)
            logger.info(f"使用原始图像作为修复结果，已保存到: {repaired_path}")
        except:
            logger.error("无法保存原始图像作为修复结果")
    
    # 步骤2：使用SRGAN进行超分辨率处理
    if continue_to_srgan and repaired_img is not None:
        try:
            logger.info("开始图像超分辨率处理")
            enhanced_img = enhance_image(repaired_img, args)
            
            if enhanced_img is None:
                logger.error("图像超分辨率处理失败")
                return
            
            # 保存增强后的图像
            enhanced_path = save_result(enhanced_img, enhanced_filename, args.output_dir)
            logger.info(f"增强后的图像已保存到: {enhanced_path}")
            
            logger.info("处理完成！")
        except Exception as e:
            logger.error(f"超分辨率处理过程中发生错误: {e}")
            logger.debug(traceback.format_exc())
            # 尝试保存修复后的图像作为最终结果
            try:
                final_path = save_result(repaired_img, enhanced_filename, args.output_dir)
                logger.info(f"使用修复图像作为最终结果，已保存到: {final_path}")
            except:
                logger.error("无法保存最终结果")
    else:
        logger.error("无法继续进行超分辨率处理，程序终止")

if __name__ == "__main__":
    main() 
