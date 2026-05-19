# CRA + SRGAN 图像修复与超分辨率工具

本项目整合了 CRA 图像修复和 SRGAN 超分辨率代码，并提供一个可直接运行的工程入口。

当前默认使用 `auto` 后端：

- 如果本机可用 MindSpore 且检查点可加载，使用原始 CRA/SRGAN 深度模型。
- 如果 MindSpore 不可用，自动切换到 `classic` 后端，使用 OpenCV inpaint 完成修复，并使用双三次插值 + 轻量锐化完成超分增强。

这样即使没有深度学习运行环境和模型检查点，也能完成端到端运行验证。

## 目录结构

```text
combined_repair_sr2.0/
  combined_repair_sr_optimized.py   # 推荐 CLI/核心入口
  combined_repair_sr_gui.py         # Tkinter GUI
  start_gui.py                      # GUI 启动脚本
  batch_process_example.py          # 批处理示例
  requirements.txt                  # 核心依赖
CRA/                                # CRA 原始模型代码和示例资源
srgan/                              # SRGAN 原始模型代码
ckpt/                               # 已有检查点候选目录
tests/                              # 最小回归测试
```

## 安装

建议先创建虚拟环境：

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

可选深度模型依赖：

```bash
python -m pip install -r requirements-optional.txt
```

MindSpore 需要按你的系统、Python 版本和硬件从官方安装选择器安装。本机 Python 3.13 环境下 `pip index versions mindspore` 没有匹配发行版，所以本项目默认可通过 `classic` 后端运行。

### MindSpore 后端实测状态

本仓库已在 `.venv-ms` 中安装并验证 `mindspore==2.9.0`：

- Windows + Python 3.12.2 + MindSpore 2.9.0 的 CPU 后端可用，`mindspore.run_check()` 通过。
- 本机 RTX 4060 Laptop GPU 可被 NVIDIA/Docker 识别，但当前 Windows wheel 仅支持 CPU；设置 `device_target=GPU` 会提示当前 wheel 只支持 `['CPU']`。
- 项目 deep 后端在 CPU 上可加载 CRA/SRGAN 权重并跑通 SRGAN 输出；CRA 推理路径中的 `ExtractImagePatches` 算子不支持 CPU，因此完整 CRA deep 推理需要 GPU/Ascend 环境。

详细安装命令、真实运行记录和“满血”GPU/Ascend 重跑方案见 `MINDSPORE_SETUP.md`。

### 真实 GPU 性能补充（SRGAN 阶段）

由于当前 Windows 主机上的 MindSpore wheel 仅支持 CPU，本仓库额外提供了 **SRGAN 原始权重的 PyTorch CUDA 对照基准**，用于补充真实 NVIDIA GPU 超分阶段性能：

```bash
python .\tools\benchmark_srgan_cpu_gpu.py
```

当前本机实测：

- GPU：`NVIDIA GeForce RTX 4060 Laptop GPU`
- PyTorch CUDA：`12.8`
- 重点引用的真实加速比：`torch_cpu -> torch_cuda` 平均约 `11.24x`

结果文件位于：

- `paper_results/gpu_benchmark/README_gpu_benchmark.md`
- `paper_results/gpu_benchmark/metrics/srgan_cpu_gpu_benchmark.csv`
- `paper_results/gpu_benchmark/figures/srgan_cpu_gpu_latency.png`
- `paper_results/gpu_benchmark/figures/srgan_cpu_gpu_speedup.png`

### 完整 GPU 端到端论文结果（CUDA 补充路径）

当前 Windows MindSpore wheel 仍只支持 CPU，因此不能把本机结果写成“MindSpore GPU 端到端”。
为满足论文中的 GPU 效果展示与对比实验，仓库额外提供了真实 CUDA 端到端结果生成脚本：

```powershell
# 如尚未安装 CUDA 版 PyTorch，可先安装：
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision

.\.venv-ms\Scripts\python .\tools\generate_gpu_paper_results.py
```

该脚本会生成 3 组 GPU 效果图与 3 组对比数据图：

- `paper_results/gpu_full_pipeline/README_gpu_full_pipeline.md`
- `paper_results/gpu_full_pipeline/figures/case01_natural_texture_gpu_effect_figure.png`
- `paper_results/gpu_full_pipeline/figures/case02_indoor_structure_gpu_effect_figure.png`
- `paper_results/gpu_full_pipeline/figures/case03_scene_depth_gpu_effect_figure.png`
- `paper_results/gpu_full_pipeline/figures/gpu_quality_psnr_ssim.png`
- `paper_results/gpu_full_pipeline/figures/gpu_mask_repair_quality.png`
- `paper_results/gpu_full_pipeline/figures/gpu_runtime_memory.png`

当前本机实测：平均全图 PSNR 相比未修复双三次放大提升约 `7.29 dB`，掩码区 PSNR 提升约 `7.32 dB`，平均端到端耗时约 `77.6 ms`，峰值显存约 `38.3 MB`。

## 命令行运行

生成内置示例并跑通完整流程：

```bash
python .\combined_repair_sr2.0\combined_repair_sr_optimized.py --demo --backend classic --output_dir .\output_demo
```

处理自己的图片：

```bash
python .\combined_repair_sr2.0\combined_repair_sr_optimized.py ^
  --input_image .\path\input.png ^
  --mask_image .\path\mask.png ^
  --output_dir .\output ^
  --backend auto ^
  --scale 4
```

掩码规则：白色区域表示需要修复，黑色区域表示保留。

Note: `ckpt/pretrained_generator_epoch100000.ckpt` is an SRGAN x4 checkpoint. Use `--scale 4`; do not mix it with `--scale 2`, otherwise color drift and noise can appear.

## GUI 运行

```bash
python .\combined_repair_sr2.0\start_gui.py
```

如果没有 MindSpore，GUI 点击“加载模型”后会进入 `classic` 后端。
界面内可以点击“生成演示输入”快速填充示例图片和掩码，再点击“加载模型”和“开始处理”展示完整效果；运行日志和结果预览会直接显示在窗口内。

## 配置

复制 `.env.example` 为 `.env` 后可配置默认参数：

```text
CRSR_BACKEND=auto
CRSR_SCALE=4
CRSR_INPAINT_RADIUS=3
CRSR_INPAINT_METHOD=telea
CRSR_OUTPUT_DIR=combined_repair_sr2.0/output
```

也可使用 JSON 配置：

```json
{
  "backend": "classic",
  "scale": 4,
  "inpaint_radius": 3,
  "inpaint_method": "telea"
}
```

运行时指定：

```bash
python .\combined_repair_sr2.0\combined_repair_sr_optimized.py --config .\config.json --demo
```

## 测试

```bash
python -m compileall -q combined_repair_sr2.0 CRA srgan tests
python -m unittest discover -s tests
```

预期结果：编译命令无输出；单测输出 `OK`。

## Docker GPU 演示

如果你要在 Ubuntu / WSL2 / NVIDIA GPU 上演示真实推理，请看：

- `DOCKER_GPU_DEMO.md`
- `docker-compose.gpu.yml`
- `docker/run_gpu_demo.ps1`
