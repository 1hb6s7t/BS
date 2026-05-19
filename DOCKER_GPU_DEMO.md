# Docker + Ubuntu + GPU 一键演示

这套配置用于在 **Linux / WSL2 / Ubuntu + NVIDIA GPU** 环境中，复现本项目的真实 ckpt 推理，并自动生成答辩/论文可用的证据材料。

已验证环境：

- Base: `nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04`
- MindSpore: `2.0.0`
- Pillow: `9.5.0`
- OpenCV: `4.10.0.84`
- Demo image: `bs-mindspore-gpu-demo:2.0.0-cuda11.6`

## 1. 前置条件

- Docker
- NVIDIA 显卡驱动
- NVIDIA Container Toolkit
- 可用 NVIDIA GPU

先验证 Docker 能看到 GPU：

```powershell
docker run --rm --gpus all nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04 nvidia-smi
```

## 2. 一键运行

Windows PowerShell：

```powershell
docker\run_gpu_demo.ps1
```

Linux / WSL：

```bash
bash docker/run_gpu_demo.sh
```

等价命令：

```powershell
docker compose -f docker-compose.gpu.yml up --build
```

## 3. 自动生成的材料

默认输出目录：

```text
output/docker_gpu_demo/
```

会自动生成：

- `1_repaired.png`：CRA 修复阶段输出
- `1_enhanced.png`：SRGAN 超分输出
- `docker_gpu_evidence.png`：四宫格证据图，可直接放入论文/答辩
- `environment.json`：GPU、Python、MindSpore、OpenCV 等环境信息
- `run.log`：完整运行日志
- `docker_gpu_demo_report.md`：Markdown 报告
- `docker_gpu_demo_report.json`：结构化报告
- `docker_gpu_demo_export.zip`：一键打包导出文件

默认 ckpt：

- CRA：`ckpt/generator_epoch11_batch56358.ckpt`
- SRGAN：`ckpt/pretrained_generator_epoch100000.ckpt` (x4 checkpoint, default `--scale 4`)

> Note: `ckpt/pretrained_generator_epoch100000.ckpt` is an SRGAN x4 checkpoint. Docker demo defaults to `--scale 4`; do not mix it with `--scale 2`, otherwise color drift and noise can appear.

如果希望使用官方/独立 CRA 权重，也可以指定：

```powershell
docker compose -f docker-compose.gpu.yml run --rm gpu-demo gpu-demo `
  --cra_ckpt CRA/cra.ckpt `
  --output_dir output/docker_gpu_demo_cra_official
```

## 4. 自定义输入

```powershell
docker compose -f docker-compose.gpu.yml run --rm gpu-demo gpu-demo `
  --input_image CRA/test/images/2.png `
  --mask_image CRA/test/masks/2.png `
  --output_dir output/docker_gpu_demo_case2
```

查看参数：

```powershell
docker compose -f docker-compose.gpu.yml run --rm gpu-demo gpu-demo --help
```

## 5. 说明

- 该镜像是 **CLI 推理型**，适合答辩现场演示真实 GPU 推理和论文结果复现。
- GUI 建议继续在 Windows 本机运行；Docker 侧重点是证明 deep 后端、GPU、真实 ckpt 能跑。
- 当前默认使用 `--no_classic_fallback`，避免 deep 失败后悄悄退回 classic。

## 6. 答辩现场推荐说法

可以这样讲：

> Windows 本机主要用于 GUI 可视化演示；为了展示深度模型和 GPU 推理，我将同一套项目代码挂载到 Ubuntu + CUDA Docker 容器中运行。容器内会自动检测 GPU、加载真实训练好的 CRA 与 SRGAN ckpt，完成推理后生成结果图、运行日志、环境报告和导出压缩包。

如果日志中出现 GPU memory warning，可以说明：

> 当前 RTX 4060 Laptop GPU 显存有限，CRA 后处理阶段触发了 MindSpore 的显存告警；程序记录该情况后使用项目内的后处理降级路径继续生成修复结果，SRGAN 超分阶段正常完成，整体流程退出码为 0。报告中保留了完整日志和哈希，方便复现。
