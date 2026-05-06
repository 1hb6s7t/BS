# 图像修复与超分辨率联合处理工具

本工具整合了CRA（图像修复）和SRGAN（超分辨率）两种深度学习技术，为用户提供完整的图像修复和增强解决方案。

> 当前版本新增 `auto/classic/deep` 后端。没有 MindSpore 或检查点时，`auto` 会自动使用 OpenCV classic 后端，保证命令行和 GUI 可以端到端运行；需要原始 CRA/SRGAN 深度推理时再配置 MindSpore 和 ckpt。

## 🚀 功能特点

- **智能图像修复**: 使用CRA技术自动修复图像中的缺失或损坏区域
- **超分辨率增强**: 使用SRGAN技术提升图像分辨率和清晰度
- **用户友好界面**: 提供直观的图形用户界面，无需编程基础
- **实时预览**: 支持输入图像和掩码的实时预览
- **自动检查点发现**: 自动检测 CRA/SRGAN 可用 ckpt，减少手动配置错误
- **批量处理**: 支持多种图像格式和灵活的参数配置
- **详细日志**: 提供完整的处理过程日志记录

## 📋 系统要求

### 基本要求
- Python 3.7 或更高版本
- 4GB+ 内存（推荐8GB+）
- 支持CUDA的GPU（推荐，CPU也可以但速度较慢）

### 依赖包
```
mindspore>=2.0.0
opencv-python>=4.5.0
pillow>=8.0.0
numpy>=1.19.0
tkinter (通常随Python安装)
```

## 🛠️ 安装步骤

### 1. 安装Python环境
确保已安装Python 3.7或更高版本。

### 2. 安装依赖包
```bash
pip install mindspore>=2.0.0
pip install opencv-python
pip install pillow
pip install numpy
```

### 3. 下载模型文件
需要准备以下模型文件：
- **CRA模型**: `cra.ckpt`
- **SRGAN模型**: `srgan_generator.ckpt`

将模型文件放在 `checkpoints/` 目录下。

### 4. 启动工具
双击 `start_gui.py` 文件或在命令行中运行：
```bash
python start_gui.py
```

## 📖 使用说明

### 图形界面使用

1. **启动程序**
   - 双击 `start_gui.py` 启动图形界面
   - 程序会自动检查依赖包是否完整

2. **选择文件**
   - **输入图像**: 选择需要修复的原始图像
   - **掩码图像**: 选择标记损坏区域的掩码图像（白色区域为需要修复的部分）
   - **输出目录**: 选择结果保存的目录

3. **配置模型**
   - 可点击**自动检测模型**，自动填充可用检查点路径
   - **CRA模型**: 选择CRA模型检查点文件（可手动覆盖）
   - **SRGAN模型**: 选择SRGAN模型检查点文件（可手动覆盖）
   - 点击"加载模型"按钮加载模型

4. **设置参数**
   - **运行设备**: 选择GPU、CPU或Ascend
   - **超分辨率倍数**: 选择2x、4x或8x
   - **输入尺寸**: 选择处理图像的尺寸

5. **开始处理**
   - 确认所有设置正确后，点击"开始处理"
   - 等待处理完成，结果会自动保存到输出目录
   - 处理成功后，预览区会显示输出结果缩略图，便于演示对比

### 命令行使用

```bash
python combined_repair_sr_optimized.py \
    --input_image path/to/input.jpg \
    --mask_image path/to/mask.png \
    --output_dir path/to/output \
    --cra_ckpt path/to/cra.ckpt \
    --srgan_ckpt path/to/srgan.ckpt \
    --device_target GPU
```

## 🎯 掩码图像制作

掩码图像用于标记需要修复的区域：

1. **格式要求**
   - 与输入图像相同的尺寸
   - 灰度图像或黑白图像
   - 白色(255)表示需要修复的区域
   - 黑色(0)表示保留的区域

2. **制作工具**
   - Photoshop: 使用画笔工具绘制白色区域
   - GIMP: 免费替代方案
   - 画图工具: Windows自带的简单工具

3. **制作技巧**
   - 确保掩码边缘平滑
   - 避免过小的孤立区域
   - 掩码区域不宜过大（建议不超过图像面积的50%）

## 📊 处理流程

```
输入图像 + 掩码图像
         ↓
    CRA图像修复
         ↓
   SRGAN超分辨率
         ↓
      输出结果
```

1. **预处理**: 加载和调整图像尺寸
2. **CRA修复**: 使用上下文注意力机制修复缺失区域
3. **SRGAN增强**: 提升图像分辨率和细节
4. **后处理**: 保存最终结果

## ⚙️ 参数说明

### 模型参数
- **input_size**: CRA处理的输入尺寸（默认512）
- **scale**: SRGAN超分辨率倍数（2, 4, 8）
- **device_target**: 运行设备（GPU, CPU, Ascend）

### 输出文件
- `*_repaired.png`: CRA修复后的结果
- `*_enhanced.png`: 最终超分辨率结果

## 🐛 常见问题

### Q1: 程序启动失败
**A1**: 检查以下几点：
- Python版本是否为3.7+
- 是否安装了所有依赖包
- MindSpore是否正确安装

### Q2: 内存不足错误
**A2**: 尝试以下解决方案：
- 降低输入图像尺寸
- 使用CPU而不是GPU
- 减少batch size参数

### Q3: GPU无法使用
**A3**: 确认：
- NVIDIA驱动已正确安装
- CUDA版本与MindSpore兼容
- 显存足够（建议4GB+）

### Q4: 处理速度慢
**A4**: 优化建议：
- 使用GPU而不是CPU
- 降低输入图像分辨率
- 关闭其他占用GPU的程序

### Q5: 修复效果不理想
**A5**: 改进方法：
- 检查掩码是否准确标记损坏区域
- 尝试不同的参数设置
- 确保模型文件完整且正确

## 📁 文件结构

```
combined_repair_sr2.0/
├── combined_repair_sr_optimized.py    # 核心处理模块
├── combined_repair_sr_gui.py          # 图形界面
├── start_gui.py                       # 快速启动脚本
├── README.md                          # 使用说明
├── requirements.txt                   # 依赖列表
├── checkpoints/                       # 模型文件目录
│   ├── cra.ckpt
│   └── srgan_generator.ckpt
├── examples/                          # 示例文件
│   ├── input_image.jpg
│   └── mask_image.png
└── output/                           # 输出目录
```

## 🔧 高级使用

### 自定义配置
可以修改 `ModelConfig` 类来调整默认参数：

```python
class ModelConfig:
    def __init__(self):
        self.input_size = 512      # 调整输入尺寸
        self.scale = 4             # 调整超分辨率倍数
        self.device_target = 'GPU' # 调整运行设备
```

### 批量处理
虽然GUI目前只支持单张处理，但可以通过命令行脚本实现批量处理：

```python
import os
from combined_repair_sr_optimized import CombinedProcessor, ModelConfig

# 批量处理示例
config = ModelConfig()
processor = CombinedProcessor(config)
processor.load_models("cra.ckpt", "srgan.ckpt")

for img_file in os.listdir("input_dir"):
    if img_file.endswith(('.jpg', '.png')):
        processor.process_image(
            f"input_dir/{img_file}",
            f"masks/{img_file}",
            "output_dir"
        )
```

## 📝 更新日志

### v2.0.0
- 全新的模块化架构
- 添加用户友好的图形界面
- 改进的错误处理和日志系统
- 支持实时图像预览
- 优化的内存使用

### v1.0.0
- 基本的CRA+SRGAN联合处理功能
- 命令行界面

## 📄 许可证

本项目基于Apache 2.0许可证开源。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个工具！

## 📞 联系支持

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

*感谢使用图像修复与超分辨率联合处理工具！* 
