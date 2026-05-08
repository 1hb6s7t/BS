# MindSpore 安装、满血验证与本项目真实运行记录

本文档记录本机（Windows + NVIDIA RTX 4060 Laptop GPU）对 MindSpore 后端的安装与验证结果，避免论文实验中把 classic 后端误写成深度模型“满血 GPU”结果。

## 1. 本机已完成安装

已创建隔离环境：

```powershell
py -3.12 -m venv .venv-ms
.\.venv-ms\Scripts\python -m pip install -U pip
.\.venv-ms\Scripts\python -m pip install mindspore==2.9.0
.\.venv-ms\Scripts\python -m pip install -r requirements.txt -r requirements-optional.txt
```

当前可用版本：

- Python：3.12.2
- MindSpore：2.9.0
- 设备：CPU 可用
- 本机 GPU：NVIDIA GeForce RTX 4060 Laptop GPU，驱动 595.79，NVIDIA-SMI 显示 CUDA Version 13.2

验证命令：

```powershell
@'
import mindspore as ms
print(ms.__version__)
ms.set_device(device_target="CPU")
ms.run_check()
'@ | .\.venv-ms\Scripts\python -
```

CPU 验证结果：`MindSpore has been installed on platform [CPU] successfully`。

## 2. 为什么当前 Windows 环境不能宣称 MindSpore GPU 满血

当前 `mindspore==2.9.0` Windows wheel 可成功导入并在 CPU 上运行，但切换 GPU 会报：

```text
Unsupported device target GPU. This process only supports one of the ['CPU'].
Please check whether the GPU environment is installed and configured correctly,
and check whether current mindspore wheel package was built with "-e GPU".
```

这说明当前 wheel 本身不是 GPU 构建。Docker + NVIDIA runtime 已验证能看到显卡，但公开 pip / 镜像组合在本机仍未跑通 MindSpore GPU 后端。因此本机真实深度后端结果只能如实标注为 **MindSpore CPU + SRGAN 成功，CRA CPU 算子受限**。

参考：

- MindSpore 官方安装选择器：<https://www.mindspore.cn/install/en>
- PyPI `mindspore` 发行页：<https://pypi.org/project/mindspore/>

## 3. 本项目 deep 后端真实运行记录

已执行：

```powershell
.\.venv-ms\Scripts\python .\combined_repair_sr2.0\combined_repair_sr_optimized.py `
  --input_image .\CRA\test\images\1.png `
  --mask_image .\CRA\test\masks\1.png `
  --output_dir .\output_mindspore_cpu_srgan_ckpt2 `
  --backend deep `
  --device_target CPU `
  --scale 2 `
  --srgan_ckpt .\ckpt\pretrained_generator_epoch100000.ckpt
```

结果：

- CRA 权重：`CRA/cra.ckpt` 能加载。
- SRGAN 权重：`ckpt/pretrained_generator_epoch100000.ckpt` 能加载。
- SRGAN 深度超分输出：`output_mindspore_cpu_srgan_ckpt2/1_enhanced.png`
  - 尺寸：4032 × 2268
  - SHA256：`64132c9b075ec925ebb87951a1f3a31a67fe90ca8144d0b2823e2ef5628b0a7d`
- CRA CPU 推理限制：MindSpore CPU 不支持 CRA 注意力路径中的 `ExtractImagePatches`，项目当前会记录错误并回退为原图再进入 SRGAN。

详细记录见：`output_mindspore_cpu_srgan_ckpt2/mindspore_cpu_run_summary.json`。

## 4. 论文实验图的可复现命令

默认生成 classic 后端论文图，属于真实运行 baseline，不冒充 GPU 深度模型：

```powershell
.\.venv-ms\Scripts\python .\tools\generate_paper_results.py --backend classic --device_target CPU --scale 2
```

如果后续在 Linux GPU/Ascend 环境中安装了对应 MindSpore 后端，可用同一脚本严格重跑：

```powershell
python tools\generate_paper_results.py `
  --backend deep `
  --device_target GPU `
  --cra_ckpt .\CRA\cra.ckpt `
  --srgan_ckpt .\ckpt\pretrained_generator_epoch100000.ckpt `
  --no_classic_fallback
```

`--no_classic_fallback` 用于防止 deep 失败后自动回退 classic，从而保证生成的数据确实来自深度后端。

## 5. 推荐的满血方案

若论文必须给出完整 CRA + SRGAN 深度模型加速性能，建议使用以下之一：

1. **Ascend 环境**：MindSpore 原生支持更完整，适合 CRA 中的注意力算子。
2. **Linux + 官方匹配的 MindSpore GPU 构建**：按官方安装选择器选择 Linux、Python、CUDA/cuDNN 组合，并先运行 `ms.run_check()` 与本项目 `--backend deep --no_classic_fallback`。

在没有上述环境前，论文中应明确区分：

- `classic`：OpenCV 修复 + 双三次超分 + 锐化，适合作为可复现工程 baseline。
- `deep CPU`：MindSpore CPU 能加载权重并运行 SRGAN，但 CRA 受 CPU 算子限制。
- `deep GPU/Ascend`：只有在后续真实跑通后才能写作完整深度模型性能。

## 6. 当前已补充的真实 GPU 性能数据

虽然当前主机无法直接跑通完整 MindSpore deep GPU 版，但已经补充了 **SRGAN 超分阶段** 的真实 NVIDIA GPU 性能：

```powershell
.\.venv-ms\Scripts\python .\tools\benchmark_srgan_cpu_gpu.py
```

该脚本做了三类基准：

1. `mindspore_cpu`：保留仓库当前原始实现的 CPU 参考性能；
2. `torch_cpu`：使用同一 SRGAN 原始 checkpoint 的 PyTorch CPU 推理；
3. `torch_cuda`：使用同一 SRGAN 原始 checkpoint 的 PyTorch CUDA 推理。

建议论文中重点引用 `torch_cpu -> torch_cuda` 的真实加速比，因为这两者使用完全相同的转换后权重，输出几乎一致。

当前本机结果摘要：

- 平均加速比：约 `11.24×`
- 小图（512×384）GPU 延迟：约 `139–147 ms`
- 大图 `CRA/test/images/1.png`（2016×1134）GPU 延迟：约 `2270.5 ms`
- 大图 `CRA/test/images/2.png`（2730×4096）GPU 延迟：约 `10087.0 ms`

结果见：

- `paper_results/gpu_benchmark/README_gpu_benchmark.md`
- `paper_results/gpu_benchmark/metrics/srgan_cpu_gpu_benchmark.csv`
- `paper_results/gpu_benchmark/metrics/srgan_cpu_gpu_comparison.csv`
