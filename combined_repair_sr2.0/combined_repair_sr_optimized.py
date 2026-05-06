#!/usr/bin/env python3
"""
图像修复与超分辨率联合处理框架
整合CRA图像修复和SRGAN超分辨率技术
优化版本 - 更好的代码结构和错误处理
"""

import os
import time
import argparse
import sys
import cv2
import json
import numpy as np
import logging
import traceback
import math
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple, Union

# MindSpore 是深度模型后端依赖。Windows + Python 3.13 当前通常没有可安装
# wheel，因此这里做成可选依赖，让项目至少能通过经典 OpenCV 后端直接运行。
try:
    import mindspore
    from mindspore import Tensor, context, load_checkpoint, load_param_into_net, set_seed
    import mindspore.nn as nn
    import mindspore.ops as ops
    HAS_MINDSPORE = True
    MINDSPORE_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - exercised on machines without MindSpore
    mindspore = None
    Tensor = Any  # type: ignore[assignment]
    context = None
    load_checkpoint = None
    load_param_into_net = None
    set_seed = None
    ops = None

    class _MissingNN:
        class Cell:
            pass

    nn = _MissingNN()  # type: ignore[assignment]
    HAS_MINDSPORE = False
    MINDSPORE_IMPORT_ERROR = exc

# 确保可以导入同仓库中的 CRA / srgan 包
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

APP_VERSION = "2.1.0"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _str_to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_environment(env_file: Optional[Union[str, Path]] = None) -> None:
    """加载简单 .env 文件，不引入额外依赖。已存在的环境变量不会被覆盖。"""
    candidates: List[Path] = []
    if env_file:
        candidates.append(Path(env_file))
    else:
        candidates.extend([Path.cwd() / ".env", SCRIPT_DIR / ".env", PROJECT_ROOT / ".env"])

    for candidate in candidates:
        if not candidate.exists() or not candidate.is_file():
            continue
        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _env(name: str, default: Any = None) -> Any:
    return os.environ.get(f"CRSR_{name}", default)


def _env_int(name: str, default: int) -> int:
    value = _env(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("忽略非法整数环境变量 CRSR_%s=%r，使用默认值 %s", name, value, default)
        return default


def _env_float(name: str, default: float) -> float:
    value = _env(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("忽略非法浮点环境变量 CRSR_%s=%r，使用默认值 %s", name, value, default)
        return default

# 全局图像引用字典，防止垃圾回收
image_refs = {}

# 预定义全局变量
ApplyAttention2 = None

# 定义完整的ApplyAttention2实现
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

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(SCRIPT_DIR / 'combined_repair_sr.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
load_environment()

# ====================== 检查点路径发现 ======================

def _unique_existing_files(paths: List[Path]) -> List[str]:
    """去重并保留存在的文件路径。"""
    result: List[str] = []
    seen = set()
    for p in paths:
        try:
            rp = str(p.resolve())
        except Exception:
            rp = str(p)
        key = rp.lower()
        if key in seen:
            continue
        if Path(rp).exists() and Path(rp).is_file():
            seen.add(key)
            result.append(rp)
    return result


def discover_checkpoint_paths(base_dir: Optional[Union[str, Path]] = None) -> Dict[str, Union[str, List[str]]]:
    """
    自动发现 CRA/SRGAN 可用检查点路径。
    返回:
      - cra_default / srgan_default: 推荐默认路径
      - cra_candidates / srgan_candidates: 可尝试路径列表
    """
    script_dir = Path(base_dir).resolve() if base_dir else Path(__file__).resolve().parent
    project_root = script_dir.parent
    ckpt_root = project_root / "ckpt"

    cra_candidates: List[Path] = [
        script_dir / "checkpoints" / "cra.ckpt",
        project_root / "CRA" / "cra.ckpt",
        ckpt_root / "cra.ckpt",
    ]
    srgan_candidates: List[Path] = [
        script_dir / "checkpoints" / "srgan_generator.ckpt",
        ckpt_root / "generator_epoch11_batch56358.ckpt",
        ckpt_root / "pretrained_generator_epoch100000.ckpt",
        project_root / "srgan" / "ckpt" / "pretrained_generator_epoch10000.ckpt",
    ]

    if ckpt_root.exists():
        for f in ckpt_root.glob("*.ckpt"):
            name = f.name.lower()
            if "generator" in name and "discriminator" not in name:
                srgan_candidates.append(f)
            if "cra" in name:
                cra_candidates.append(f)

    cra_files = _unique_existing_files(cra_candidates)
    srgan_files = _unique_existing_files(srgan_candidates)

    return {
        "cra_default": cra_files[0] if cra_files else "",
        "srgan_default": srgan_files[0] if srgan_files else "",
        "cra_candidates": cra_files,
        "srgan_candidates": srgan_files,
    }

# ====================== CRA 推理辅助函数 ======================

def build_inference_net(raw_img_ph, raw_mask_ph, model_gen, args):
    """
    完整的CRA网络测试模型，包括图像预处理、生成器生成和输出以及图像后处理操作。
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
    
    # 处理输入掩码
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
        
        # 使用注意力机制
        sz = args.input_size * args.times
        attention_module = ApplyAttention2Impl([1, 3, sz, sz], [1, 1024, 32, 32], sz)
        residual = attention_module(residual1, corres)
        logger.info("注意力处理成功完成")
        
        low_base_shape = ops.Shape()(low_base)
        low_base_output_h = int(low_base_shape[2] * scale_factor)
        low_base_output_w = int(low_base_shape[3] * scale_factor)
        low_base = resize_op(low_base, size=(low_base_output_h, low_base_output_w))
        
        # 确保residual和low_base尺寸相同
        if residual.shape != low_base.shape:
            logger.warning(f"尺寸不匹配: residual {residual.shape}, low_base {low_base.shape}")
            residual = resize_op(residual, size=(low_base_output_h, low_base_output_w))
        
        x = low_base + residual
        x = ops.clip_by_value(x, -1, 1)
        x = (x + 1.) * 127.5
        return x, low_raw, low_base, residual
    except Exception as e:
        logger.error(f"后处理过程出错: {e}")
        # 发生错误时，尝试返回一个有效的结果
        try:
            low_base_shape = ops.Shape()(low_base)
            low_base_output_h = int(low_base_shape[2] * scale_factor)
            low_base_output_w = int(low_base_shape[3] * scale_factor)
            low_base_resized = resize_op(low_base, size=(low_base_output_h, low_base_output_w))
            result = (low_base_resized + 1.) * 127.5
            return result, low_raw, low_base, None
        except:
            raise

class ModelConfig:
    """模型配置类。支持环境变量、JSON 配置文件和命令行覆盖。"""

    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        ckpt_info = discover_checkpoint_paths()

        # 默认路径配置
        self.default_cra_ckpt = _env("CRA_CKPT", ckpt_info.get("cra_default") or "")
        self.default_srgan_ckpt = _env("SRGAN_CKPT", ckpt_info.get("srgan_default") or "")
        self.default_output_dir = _env("OUTPUT_DIR", str(SCRIPT_DIR / "output"))
        self.cra_ckpt_candidates = list(ckpt_info.get("cra_candidates", []))
        self.srgan_ckpt_candidates = list(ckpt_info.get("srgan_candidates", []))
        
        # 模型参数
        self.backend = str(_env("BACKEND", "auto")).lower()
        self.input_size = _env_int("INPUT_SIZE", 512)
        self.times = _env_int("TIMES", 8)
        self.scale = _env_int("SCALE", 2)
        self.device_target = str(_env("DEVICE_TARGET", "CPU"))
        self.train_batchsize = 1
        self.attention_type = 'SOFT'
        self.inpaint_radius = _env_int("INPAINT_RADIUS", 3)
        self.inpaint_method = str(_env("INPAINT_METHOD", "telea")).lower()
        self.sharpen_strength = _env_float("SHARPEN_STRENGTH", 0.15)
        self.allow_classic_fallback = _str_to_bool(str(_env("ALLOW_CLASSIC_FALLBACK", "true")))

        if config_file:
            self.apply_json_config(config_file)

        self.validate()

    @property
    def deep_backend_available(self) -> bool:
        return HAS_MINDSPORE

    def apply_json_config(self, config_file: Union[str, Path]) -> None:
        """从 JSON 文件覆盖配置项。未知字段会忽略，避免配置写错破坏启动。"""
        path = Path(config_file)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("配置文件必须是 JSON object")

        allowed = {
            "backend", "default_cra_ckpt", "default_srgan_ckpt", "default_output_dir",
            "input_size", "times", "scale", "device_target", "train_batchsize",
            "attention_type", "inpaint_radius", "inpaint_method", "sharpen_strength",
            "allow_classic_fallback",
        }
        for key, value in data.items():
            if key in allowed:
                setattr(self, key, value)

    def validate(self) -> None:
        """校验配置并规范化取值。"""
        self.backend = str(self.backend).lower()
        if self.backend not in {"auto", "deep", "classic"}:
            raise ValueError("backend 必须是 auto、deep 或 classic")

        self.device_target = str(self.device_target)
        if self.device_target not in {"CPU", "GPU", "Ascend"}:
            raise ValueError("device_target 必须是 CPU、GPU 或 Ascend")

        self.scale = int(self.scale)
        if self.scale not in {1, 2, 4, 8}:
            raise ValueError("scale 必须是 1、2、4 或 8")

        self.input_size = int(self.input_size)
        if self.input_size <= 0:
            raise ValueError("input_size 必须大于 0")

        self.times = int(self.times)
        if self.times <= 0:
            raise ValueError("times 必须大于 0")

        self.inpaint_radius = max(1, int(self.inpaint_radius))
        self.inpaint_method = str(self.inpaint_method).lower()
        if self.inpaint_method not in {"telea", "ns"}:
            raise ValueError("inpaint_method 必须是 telea 或 ns")

        self.sharpen_strength = float(self.sharpen_strength)
        self.sharpen_strength = min(max(self.sharpen_strength, 0.0), 1.0)

    def should_use_classic_backend(self) -> bool:
        """是否直接使用 OpenCV 经典后端。"""
        return self.backend == "classic" or (self.backend == "auto" and not HAS_MINDSPORE)

class ImageProcessor:
    """图像处理工具类"""
    
    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        """加载图像文件"""
        try:
            path = Path(image_path)
            if not path.exists() or not path.is_file():
                logger.error(f"图像文件不存在: {image_path}")
                return None
            if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                logger.error("不支持的图像格式: %s", path.suffix)
                return None

            with Image.open(path) as image:
                img = np.array(image.convert("RGB"))
            logger.info(f"成功加载图像: {image_path}, 尺寸: {img.shape}")
            return img
        except Exception as e:
            logger.error(f"加载图像时出错: {e}")
            return None
    
    @staticmethod
    def load_mask(mask_path: str, target_shape: Optional[Tuple[int, int]] = None) -> Optional[np.ndarray]:
        """加载掩码图像"""
        try:
            path = Path(mask_path)
            if not path.exists() or not path.is_file():
                logger.error(f"掩码文件不存在: {mask_path}")
                return None
            if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                logger.error("不支持的掩码格式: %s", path.suffix)
                return None

            with Image.open(path) as image:
                mask = np.array(image.convert("L"))
            
            # 调整掩码尺寸以匹配目标图像
            if target_shape is not None:
                h, w = target_shape[:2]
                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    logger.info(f"调整掩码尺寸到: {mask.shape}")
            
            # 确保掩码是二值的
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            logger.info(f"成功加载掩码: {mask_path}, 尺寸: {mask.shape}")
            return mask
        except Exception as e:
            logger.error(f"加载掩码时出错: {e}")
            return None
    
    @staticmethod
    def save_image(img: np.ndarray, output_path: str) -> bool:
        """保存图像"""
        try:
            # 确保输出目录存在
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                logger.error("不支持的输出图像格式: %s", output.suffix)
                return False
            
            # 处理图像数据范围
            if img is None or img.size == 0:
                logger.error("无法保存空图像")
                return False
            if img.max() <= 1.0:
                img = img * 255.0
            
            img = np.clip(img, 0, 255).astype(np.uint8)
            
            # 保存图像
            if len(img.shape) == 3:
                img_pil = Image.fromarray(img)
            else:
                img_pil = Image.fromarray(img, mode='L')
            
            img_pil.save(output)
            logger.info(f"图像已保存到: {output}")
            return True
        except Exception as e:
            logger.error(f"保存图像时出错: {e}")
            return False

class CRAModel:
    """CRA图像修复模型"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.is_loaded = False
    
    def load_model(self, checkpoint_path: str) -> bool:
        """加载CRA模型"""
        if not HAS_MINDSPORE:
            logger.error("无法加载 CRA 深度模型: MindSpore 未安装或不可用: %s", MINDSPORE_IMPORT_ERROR)
            return False
        try:
            # 导入CRA模型
            from CRA.src.models.inpainting_network import GatedGenerator
            
            # 创建模型参数
            args = self._create_args()
            
            # 创建模型
            self.model = GatedGenerator(args)
            
            # 加载检查点
            if os.path.exists(checkpoint_path):
                param_dict = load_checkpoint(checkpoint_path)
                load_param_into_net(self.model, param_dict)
                logger.info(f"CRA模型加载成功: {checkpoint_path}")
                self.is_loaded = True
                return True
            else:
                logger.error(f"CRA检查点文件不存在: {checkpoint_path}")
                return False
        except ModuleNotFoundError as e:
            logger.error(
                "加载CRA模型时出错: %s。请确认项目根目录在导入路径中（当前: %s）",
                e,
                PROJECT_ROOT,
            )
            return False
        except Exception as e:
            logger.error(f"加载CRA模型时出错: {e}")
            return False
    
    def _create_args(self):
        """创建CRA模型所需的参数"""
        class Args:
            def __init__(self, config):
                self.input_size = config.input_size
                self.times = config.times
                self.train_batchsize = config.train_batchsize
                self.attention_type = config.attention_type
                self.IMG_SHAPE = [512, 512, 3]
                
                # 添加CRA模型期望的所有必要参数
                self.sc = False
                self.dilation = 2
                self.seed = 2022
                self.nThreads = 1
                self.max_dataset_size = float("inf")
                self.gpu_ids = [0]
                self.dis_iter = 1
        
        return Args(self.config)
    
    def repair_image(self, img: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        """修复图像"""
        if not self.is_loaded:
            logger.error("CRA模型未加载")
            return None
            
        try:
            logger.info("开始CRA图像修复")
            
            # 预处理
            args = self._create_args()
            
            # 设置全局ApplyAttention2
            global ApplyAttention2
            if ApplyAttention2 is None:
                ApplyAttention2 = ApplyAttention2Impl
                logger.info("设置ApplyAttention2为自定义实现")
            
            # 转换为张量，注意掩码需要反转
            img_tensor = Tensor(img.astype(np.float32))
            mask_tensor = Tensor((255 - mask).astype(np.float32))
            
            # 执行完整的CRA推理
            outputs, _, _ = build_inference_net(img_tensor, mask_tensor, self.model, args)
            
            # 提取修复后的图像
            repaired_img = outputs[0].asnumpy()
            
            logger.info("CRA图像修复完成")
            return repaired_img
                
        except Exception as e:
            logger.error(f"CRA修复过程出错: {e}")
            logger.warning("返回原始图像作为备选结果")
            return img
    


class SRGANModel:
    """SRGAN超分辨率模型"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.is_loaded = False
        # 设置分块大小
        self.block_size = 512  # 基础块大小
        self.overlap = 32      # 重叠区域大小
    
    def load_model(self, checkpoint_path: str) -> bool:
        """加载SRGAN模型"""
        if not HAS_MINDSPORE:
            logger.error("无法加载 SRGAN 深度模型: MindSpore 未安装或不可用: %s", MINDSPORE_IMPORT_ERROR)
            return False
        try:
            # 创建生成器
            self.model = self._create_generator()
            
            # 加载检查点
            if os.path.exists(checkpoint_path):
                param_dict = load_checkpoint(checkpoint_path)
                load_param_into_net(self.model, param_dict)
                logger.info(f"SRGAN模型加载成功: {checkpoint_path}")
                self.is_loaded = True
                return True
            else:
                logger.error(f"SRGAN检查点文件不存在: {checkpoint_path}")
                return False
        except ModuleNotFoundError as e:
            logger.error(
                "加载SRGAN模型时出错: %s。请确认项目根目录在导入路径中（当前: %s）",
                e,
                PROJECT_ROOT,
            )
            return False
        except Exception as e:
            logger.error(f"加载SRGAN模型时出错: {e}")
            return False
    
    def _create_generator(self):
        """创建SRGAN生成器"""
        from srgan.src.model.generator import Generator
        return Generator(self.config.scale)
    
    def enhance_image(self, img: np.ndarray) -> Optional[np.ndarray]:
        """超分辨率增强图像"""
        if not self.is_loaded:
            logger.error("SRGAN模型未加载")
            return None
            
        try:
            logger.info("开始SRGAN超分辨率处理")
            
            # 获取图像尺寸
            h, w = img.shape[:2]
            
            # 如果图像尺寸小于块大小，直接处理
            if h <= self.block_size and w <= self.block_size:
                return self._process_single_block(img)
            
            # 分块处理
            logger.info(f"图像尺寸 ({h}x{w}) 超过块大小，进行分块处理")
            
            # 计算分块数量
            num_blocks_h = math.ceil(h / (self.block_size - self.overlap))
            num_blocks_w = math.ceil(w / (self.block_size - self.overlap))
            
            # 创建输出图像
            output_h = h * self.config.scale
            output_w = w * self.config.scale
            output_img = np.zeros((output_h, output_w, 3), dtype=np.float32)
            weight_map = np.zeros((output_h, output_w), dtype=np.float32)
            
            # 处理每个块
            for i in range(num_blocks_h):
                for j in range(num_blocks_w):
                    # 计算当前块的坐标
                    y_start = i * (self.block_size - self.overlap)
                    x_start = j * (self.block_size - self.overlap)
                    
                    # 确保不超出图像边界
                    y_end = min(y_start + self.block_size, h)
                    x_end = min(x_start + self.block_size, w)
                    
                    # 提取当前块
                    block = img[y_start:y_end, x_start:x_end]
                    
                    # 处理当前块
                    processed_block = self._process_single_block(block)
                    if processed_block is None:
                        logger.error(f"处理块 ({i},{j}) 失败")
                        continue
                    
                    # 计算输出块的坐标
                    out_y_start = y_start * self.config.scale
                    out_x_start = x_start * self.config.scale
                    out_y_end = y_end * self.config.scale
                    out_x_end = x_end * self.config.scale
                    
                    # 创建权重掩码
                    block_h, block_w = processed_block.shape[:2]
                    weight = np.ones((block_h, block_w), dtype=np.float32)
                    
                    # 在重叠区域应用渐变权重
                    if self.overlap > 0:
                        fade = np.linspace(0, 1, self.overlap * self.config.scale)
                        weight[:self.overlap * self.config.scale, :] *= fade[:, np.newaxis]
                        weight[-self.overlap * self.config.scale:, :] *= fade[::-1, np.newaxis]
                        weight[:, :self.overlap * self.config.scale] *= fade[np.newaxis, :]
                        weight[:, -self.overlap * self.config.scale:] *= fade[np.newaxis, ::-1]
                    
                    # 将处理后的块添加到输出图像
                    for c in range(3):
                        output_img[out_y_start:out_y_end, out_x_start:out_x_end, c] += \
                            processed_block[:, :, c] * weight
                    
                    # 更新权重图
                    weight_map[out_y_start:out_y_end, out_x_start:out_x_end] += weight
            
            # 归一化输出图像
            weight_map = np.maximum(weight_map, 1e-6)  # 避免除以零
            for c in range(3):
                output_img[:, :, c] = output_img[:, :, c] / weight_map
            
            logger.info("SRGAN超分辨率处理完成")
            return np.clip(output_img, 0, 255).astype(np.uint8)
            
        except Exception as e:
            logger.error(f"SRGAN处理过程出错: {e}")
            return None
    
    def _process_single_block(self, block: np.ndarray) -> Optional[np.ndarray]:
        """处理单个图像块"""
        try:
            # 预处理
            block_processed = self._preprocess(block)
            
            # 推理
            enhanced = self.model(block_processed)
            
            # 后处理
            result = self._postprocess(enhanced)
            
            return result
            
        except Exception as e:
            logger.error(f"处理图像块时出错: {e}")
            return None
    
    def _preprocess(self, img: np.ndarray) -> Tensor:
        """预处理图像"""
        # 归一化到[-1, 1]
        img = img.astype(np.float32) / 127.5 - 1.0
        
        # 调整维度顺序：HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        
        # 添加batch维度
        img = np.expand_dims(img, 0)
        
        return Tensor(img, dtype=mindspore.float32)
    
    def _postprocess(self, tensor: Tensor) -> np.ndarray:
        """后处理图像"""
        # 移除batch维度
        img = tensor.asnumpy()[0]
        
        # 限制到[-1, 1]
        img = np.clip(img, -1.0, 1.0)
        
        # 转换到[0, 1]
        img = (img + 1.0) / 2.0
        
        # 调整维度顺序：CHW -> HWC
        img = np.transpose(img, (1, 2, 0))
        
        # 转换到[0, 255]
        img = (img * 255.0).astype(np.uint8)
        
        return img


class ClassicRepairModel:
    """无需深度框架的 OpenCV 图像修复后端。"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.is_loaded = True

    def load_model(self, checkpoint_path: str = "") -> bool:
        logger.info("使用经典 OpenCV 修复后端，无需 CRA 检查点")
        return True

    def repair_image(self, img: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        try:
            if img.ndim != 3 or img.shape[2] != 3:
                logger.error("输入图像必须是 RGB 三通道")
                return None
            if mask.shape[:2] != img.shape[:2]:
                logger.error("掩码尺寸必须与输入图像一致")
                return None

            _, binary_mask = cv2.threshold(mask.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
            if np.count_nonzero(binary_mask) == 0:
                logger.warning("掩码没有白色修复区域，将直接返回原图")
                return img.copy()

            method = cv2.INPAINT_TELEA if self.config.inpaint_method == "telea" else cv2.INPAINT_NS
            repaired = cv2.inpaint(img.astype(np.uint8), binary_mask, self.config.inpaint_radius, method)
            logger.info("OpenCV 图像修复完成")
            return repaired
        except Exception as e:
            logger.error("OpenCV 图像修复失败: %s", e)
            return None


class ClassicSRModel:
    """无需深度框架的插值超分后端。"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.is_loaded = True

    def load_model(self, checkpoint_path: str = "") -> bool:
        logger.info("使用经典插值增强后端，无需 SRGAN 检查点")
        return True

    def enhance_image(self, img: np.ndarray) -> Optional[np.ndarray]:
        try:
            if img.ndim != 3 or img.shape[2] != 3:
                logger.error("输入图像必须是 RGB 三通道")
                return None

            scale = int(self.config.scale)
            if scale == 1:
                enhanced = img.copy()
            else:
                h, w = img.shape[:2]
                enhanced = cv2.resize(
                    img,
                    (w * scale, h * scale),
                    interpolation=cv2.INTER_CUBIC,
                )

            strength = float(self.config.sharpen_strength)
            if strength > 0:
                blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
                enhanced = cv2.addWeighted(enhanced, 1.0 + strength, blurred, -strength, 0)

            logger.info("经典超分增强完成，输出尺寸: %s", enhanced.shape)
            return np.clip(enhanced, 0, 255).astype(np.uint8)
        except Exception as e:
            logger.error("经典超分增强失败: %s", e)
            return None


class CombinedProcessor:
    """联合处理器"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.active_backend = "classic" if config.should_use_classic_backend() else "deep"
        self.cra_model = ClassicRepairModel(config) if self.active_backend == "classic" else CRAModel(config)
        self.srgan_model = ClassicSRModel(config) if self.active_backend == "classic" else SRGANModel(config)
        self.loaded_ckpts: Dict[str, str] = {"cra": "", "srgan": ""}
        self.setup_context()
    
    def setup_context(self):
        """设置MindSpore上下文"""
        if self.active_backend == "classic":
            if not HAS_MINDSPORE:
                logger.info("MindSpore 不可用，已启用 classic 后端: %s", MINDSPORE_IMPORT_ERROR)
            else:
                logger.info("已按配置启用 classic 后端")
            return
        try:
            if not HAS_MINDSPORE:
                raise RuntimeError(f"MindSpore 不可用: {MINDSPORE_IMPORT_ERROR}")
            context.set_context(
                mode=context.PYNATIVE_MODE,
                device_target=self.config.device_target
            )
            set_seed(2022)
            logger.info(f"MindSpore上下文设置完成: {self.config.device_target}")
        except Exception as e:
            logger.error(f"设置MindSpore上下文失败: {e}")
    
    @staticmethod
    def _build_load_order(primary_path: str, fallback_candidates: List[str]) -> List[str]:
        """按主路径优先，附加回退路径，构造加载顺序。"""
        merged: List[str] = []
        seen = set()

        for raw in [primary_path, *fallback_candidates]:
            if not raw:
                continue
            p = str(Path(raw))
            key = p.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(p)
        return merged

    def load_models(self, cra_ckpt: str, srgan_ckpt: str) -> Tuple[bool, bool]:
        """加载所有模型"""
        if self.active_backend == "classic":
            cra_success = self.cra_model.load_model("")
            srgan_success = self.srgan_model.load_model("")
            self.loaded_ckpts["cra"] = "classic-opencv"
            self.loaded_ckpts["srgan"] = "classic-interpolation"
            return cra_success, srgan_success

        cra_order = self._build_load_order(cra_ckpt, self.config.cra_ckpt_candidates)
        srgan_order = self._build_load_order(srgan_ckpt, self.config.srgan_ckpt_candidates)

        cra_success = False
        for p in cra_order:
            logger.info(f"尝试加载 CRA 检查点: {p}")
            if self.cra_model.load_model(p):
                self.loaded_ckpts["cra"] = p
                cra_success = True
                break

        if not cra_success:
            logger.error("CRA 模型加载失败，已尝试路径: %s", " | ".join(cra_order) if cra_order else "(无可用路径)")

        srgan_success = False
        for p in srgan_order:
            logger.info(f"尝试加载 SRGAN 检查点: {p}")
            if self.srgan_model.load_model(p):
                self.loaded_ckpts["srgan"] = p
                srgan_success = True
                break

        if not srgan_success:
            logger.error("SRGAN 模型加载失败，已尝试路径: %s", " | ".join(srgan_order) if srgan_order else "(无可用路径)")

        if (not cra_success or not srgan_success) and self.config.backend == "auto" and self.config.allow_classic_fallback:
            logger.warning("深度模型加载失败，auto 模式切换到 classic 后端")
            self.active_backend = "classic"
            self.cra_model = ClassicRepairModel(self.config)
            self.srgan_model = ClassicSRModel(self.config)
            self.loaded_ckpts["cra"] = "classic-opencv"
            self.loaded_ckpts["srgan"] = "classic-interpolation"
            return True, True

        return cra_success, srgan_success
    
    def process_image(self, input_path: str, mask_path: str, output_dir: str, 
                     callback=None) -> Tuple[bool, str]:
        """处理图像的完整流程"""
        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 加载图像和掩码
            if callback:
                callback("正在加载图像...")
            
            img = ImageProcessor.load_image(input_path)
            if img is None:
                return False, "无法加载输入图像"
            
            mask = ImageProcessor.load_mask(mask_path, img.shape)
            if mask is None:
                return False, "无法加载掩码图像"
            
            # 生成输出文件名
            base_name = Path(input_path).stem
            repaired_path = os.path.join(output_dir, f"{base_name}_repaired.png")
            enhanced_path = os.path.join(output_dir, f"{base_name}_enhanced.png")
            
            # 步骤1：CRA修复
            if callback:
                callback("正在进行图像修复...")
            
            repaired_img = self.cra_model.repair_image(img, mask)
            if repaired_img is None:
                return False, "图像修复失败"
            
            # 保存修复结果
            if not ImageProcessor.save_image(repaired_img, repaired_path):
                return False, "无法保存修复结果"
            
            # 步骤2：SRGAN超分辨率
            if callback:
                callback("正在进行超分辨率处理...")
            
            enhanced_img = self.srgan_model.enhance_image(repaired_img)
            if enhanced_img is None:
                logger.warning("超分辨率处理失败，使用修复图像作为最终结果")
                enhanced_img = repaired_img
            
            # 保存最终结果
            if not ImageProcessor.save_image(enhanced_img, enhanced_path):
                return False, "无法保存最终结果"
            
            if callback:
                callback("处理完成！")
            
            return True, enhanced_path
            
        except Exception as e:
            error_msg = f"处理过程中出错: {e}"
            logger.error(error_msg)
            return False, error_msg

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="CRA+SRGAN联合图像处理工具")
    parser.add_argument("--input_image", type=str, help="输入图像路径")
    parser.add_argument("--mask_image", type=str, help="掩码图像路径")
    parser.add_argument("--output_dir", type=str, help="输出目录")
    parser.add_argument("--cra_ckpt", type=str, help="CRA模型检查点路径")
    parser.add_argument("--srgan_ckpt", type=str, help="SRGAN模型检查点路径")
    parser.add_argument("--device_target", type=str,
                       choices=['GPU', 'CPU', 'Ascend'], help="运行设备")
    parser.add_argument("--backend", type=str, choices=["auto", "deep", "classic"],
                        help="处理后端：auto 自动选择，deep 使用 MindSpore 模型，classic 使用 OpenCV fallback")
    parser.add_argument("--scale", type=int, choices=[1, 2, 4, 8], help="输出放大倍数")
    parser.add_argument("--input_size", type=int, help="CRA 深度模型输入尺寸")
    parser.add_argument("--inpaint_radius", type=int, help="classic 后端修复半径")
    parser.add_argument("--inpaint_method", type=str, choices=["telea", "ns"], help="classic 后端修复算法")
    parser.add_argument("--sharpen_strength", type=float, help="classic 后端锐化强度，0-1")
    parser.add_argument("--config", type=str, help="JSON 配置文件路径")
    parser.add_argument("--env_file", type=str, help=".env 配置文件路径")
    parser.add_argument("--demo", action="store_true", help="生成内置示例并运行一次 classic/auto 流程")
    parser.add_argument("--gui", action="store_true", help="启动图形界面")
    parser.add_argument("--version", action="version", version=f"%(prog)s {APP_VERSION}")
    return parser.parse_args()


def apply_cli_overrides(config: ModelConfig, args: argparse.Namespace) -> None:
    """把命令行参数覆盖到配置对象。"""
    mapping = {
        "backend": args.backend,
        "device_target": args.device_target,
        "scale": args.scale,
        "input_size": args.input_size,
        "inpaint_radius": args.inpaint_radius,
        "inpaint_method": args.inpaint_method,
        "sharpen_strength": args.sharpen_strength,
    }
    for key, value in mapping.items():
        if value is not None:
            setattr(config, key, value)
    if args.output_dir:
        config.default_output_dir = args.output_dir
    if args.cra_ckpt:
        config.default_cra_ckpt = args.cra_ckpt
    if args.srgan_ckpt:
        config.default_srgan_ckpt = args.srgan_ckpt
    config.validate()


def create_demo_assets(output_dir: Union[str, Path]) -> Tuple[str, str]:
    """创建一组最小可验证输入图和掩码。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    image_path = out / "demo_input.png"
    mask_path = out / "demo_mask.png"

    width, height = 128, 96
    x = np.linspace(40, 220, width, dtype=np.uint8)
    y = np.linspace(30, 180, height, dtype=np.uint8)
    xv, yv = np.meshgrid(x, y)
    image = np.stack([xv, yv, np.full_like(xv, 120)], axis=2)
    cv2.rectangle(image, (42, 30), (86, 64), (20, 20, 20), thickness=-1)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(mask, (42, 30), (86, 64), 255, thickness=-1)

    ImageProcessor.save_image(image, str(image_path))
    ImageProcessor.save_image(mask, str(mask_path))
    return str(image_path), str(mask_path)

def main():
    """主函数"""
    args = parse_args()

    if args.env_file:
        load_environment(args.env_file)
    
    if args.gui:
        # 启动图形界面
        from combined_repair_sr_gui import launch_gui
        launch_gui()
    else:
        # 创建配置
        config = ModelConfig(args.config)
        apply_cli_overrides(config, args)

        output_dir = args.output_dir or config.default_output_dir
        input_image = args.input_image
        mask_image = args.mask_image

        if args.demo:
            input_image, mask_image = create_demo_assets(Path(output_dir) / "demo_assets")
            if args.backend is None:
                config.backend = "classic"
                config.validate()
            logger.info("已创建 demo 输入: %s", input_image)
            logger.info("已创建 demo 掩码: %s", mask_image)

        # 命令行模式
        if not all([input_image, mask_image]):
            logger.error("命令行模式至少需要提供 --input_image 和 --mask_image，或使用 --demo")
            raise SystemExit(2)
        
        # 创建处理器
        processor = CombinedProcessor(config)
        
        # 加载模型
        logger.info("正在加载模型...")
        selected_cra = config.default_cra_ckpt
        selected_sr = config.default_srgan_ckpt

        if processor.active_backend == "deep" and not selected_cra:
            logger.error("未找到可用 CRA 检查点，请通过 --cra_ckpt 手动指定")
            raise SystemExit(2)
        if processor.active_backend == "deep" and not selected_sr:
            logger.error("未找到可用 SRGAN 检查点，请通过 --srgan_ckpt 手动指定")
            raise SystemExit(2)

        cra_success, srgan_success = processor.load_models(selected_cra, selected_sr)
        
        if not cra_success:
            logger.error("CRA模型加载失败")
            raise SystemExit(1)
        if not srgan_success:
            logger.error("SRGAN模型加载失败")
            raise SystemExit(1)

        logger.info("当前后端: %s", processor.active_backend)
        logger.info(f"CRA检查点/后端: {processor.loaded_ckpts.get('cra', selected_cra)}")
        logger.info(f"SRGAN检查点/后端: {processor.loaded_ckpts.get('srgan', selected_sr)}")
        
        # 处理图像
        logger.info("开始处理图像...")
        success, result = processor.process_image(
            input_image, mask_image, output_dir
        )
        
        if success:
            logger.info(f"处理完成，结果保存在: {result}")
        else:
            logger.error(f"处理失败: {result}")
            raise SystemExit(1)

if __name__ == "__main__":
    main() 
