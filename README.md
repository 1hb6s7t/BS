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
  --scale 2
```

掩码规则：白色区域表示需要修复，黑色区域表示保留。

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
CRSR_SCALE=2
CRSR_INPAINT_RADIUS=3
CRSR_INPAINT_METHOD=telea
CRSR_OUTPUT_DIR=combined_repair_sr2.0/output
```

也可使用 JSON 配置：

```json
{
  "backend": "classic",
  "scale": 2,
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
