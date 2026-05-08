"""Benchmark real SRGAN CPU vs GPU performance with the original project checkpoint.

This script does not pretend that full MindSpore deep GPU inference is available on
the current Windows host. Instead, it benchmarks the *SRGAN stage itself* using:

1. The repository's current MindSpore CPU SRGAN implementation and checkpoint.
2. A PyTorch CUDA adapter that loads the same original SRGAN generator weights
   from the MindSpore checkpoint and runs on the local NVIDIA GPU.

The output is suitable for supplementing a thesis with real GPU acceleration
numbers while keeping the existing CPU data.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import mindspore as ms
from mindspore import load_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = PROJECT_ROOT / "combined_repair_sr2.0" / "combined_repair_sr_optimized.py"
CKPT_PATH = PROJECT_ROOT / "ckpt" / "pretrained_generator_epoch100000.ckpt"
OUTPUT_ROOT = PROJECT_ROOT / "paper_results" / "gpu_benchmark"


def load_pipeline_module():
    spec = importlib.util.spec_from_file_location("combined_repair_sr_optimized", PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Unable to load pipeline module: {PIPELINE_PATH}")
    spec.loader.exec_module(module)
    return module


pipeline = load_pipeline_module()


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    title: str
    image_path: Path
    repeats: int
    warmup: int = 1


@dataclass
class BenchmarkResult:
    case_id: str
    title: str
    input_path: str
    input_width: int
    input_height: int
    backend: str
    repeats: int
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    peak_gpu_mem_mb: float
    output_path: str
    output_width: int
    output_height: int
    sha256: str


@dataclass
class ComparisonResult:
    case_id: str
    title: str
    cpu_backend: str
    gpu_backend: str
    cpu_avg_ms: float
    gpu_avg_ms: float
    speedup: float
    output_psnr: float
    output_mae: float


BENCHMARK_CASES: Sequence[BenchmarkCase] = (
    BenchmarkCase(
        case_id="paper_case01_repaired",
        title="论文案例1修复后图像（512×384）",
        image_path=PROJECT_ROOT / "paper_results" / "cases" / "case01_natural_texture" / "project_pipeline" / "02_degraded_input_repaired.png",
        repeats=8,
        warmup=2,
    ),
    BenchmarkCase(
        case_id="paper_case02_repaired",
        title="论文案例2修复后图像（512×384）",
        image_path=PROJECT_ROOT / "paper_results" / "cases" / "case02_indoor_structure" / "project_pipeline" / "02_degraded_input_repaired.png",
        repeats=8,
        warmup=2,
    ),
    BenchmarkCase(
        case_id="paper_case03_repaired",
        title="论文案例3修复后图像（512×384）",
        image_path=PROJECT_ROOT / "paper_results" / "cases" / "case03_scene_depth" / "project_pipeline" / "02_degraded_input_repaired.png",
        repeats=8,
        warmup=2,
    ),
    BenchmarkCase(
        case_id="cra_test_image_1",
        title="CRA测试集大图1（2016×1134）",
        image_path=PROJECT_ROOT / "CRA" / "test" / "images" / "1.png",
        repeats=3,
        warmup=1,
    ),
    BenchmarkCase(
        case_id="cra_test_image_2",
        title="CRA测试集大图2（2730×4096）",
        image_path=PROJECT_ROOT / "CRA" / "test" / "images" / "2.png",
        repeats=2,
        warmup=1,
    ),
)


def ensure_dirs() -> None:
    for sub in ["outputs_cpu", "outputs_torch_cpu", "outputs_gpu", "metrics", "figures"]:
        (OUTPUT_ROOT / sub).mkdir(parents=True, exist_ok=True)


def pil_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def save_rgb(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).save(path)


def sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup_cpu_srgan(scale: int = 2):
    config = pipeline.ModelConfig()
    config.backend = "deep"
    config.device_target = "CPU"
    config.scale = scale
    config.validate()
    model = pipeline.SRGANModel(config)
    if not model.load_model(str(CKPT_PATH)):
        raise RuntimeError(f"Failed to load MindSpore SRGAN checkpoint: {CKPT_PATH}")
    if hasattr(model.model, "set_train"):
        model.model.set_train(False)
    return model


class TorchResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out + x


class TorchSubpixelConvolutionLayer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.prelu = nn.PReLU(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.pixel_shuffle(out)
        out = self.prelu(out)
        return out


class TorchGenerator(nn.Module):
    def __init__(self, upscale_factor: int):
        super().__init__()
        num_subpixel_layers = int(math.log(upscale_factor, 2))
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4, bias=True),
            nn.PReLU(64),
        )
        self.trunk = nn.Sequential(*[TorchResidualBlock(64) for _ in range(16)])
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PReLU(64),
        )
        self.subpixel_conv = nn.Sequential(*[TorchSubpixelConvolutionLayer(64) for _ in range(num_subpixel_layers)])
        self.conv3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)
        trunk = self.trunk(conv1)
        conv2 = self.conv2(trunk)
        out = conv1 + conv2
        out = self.subpixel_conv(out)
        out = self.conv3(out)
        return self.tanh(out)


def convert_srgan_ckpt_to_torch_state(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    params = load_checkpoint(str(ckpt_path))
    state: Dict[str, torch.Tensor] = {}
    for name, tensor in params.items():
        if not name.startswith("generator.") or name.startswith("moment"):
            continue
        key = name[len("generator.") :]
        key = key.replace(".gamma", ".weight").replace(".beta", ".bias")
        key = key.replace(".moving_mean", ".running_mean").replace(".moving_variance", ".running_var")
        if key.endswith(".w"):
            key = key[:-2] + ".weight"
        state[key] = torch.from_numpy(tensor.asnumpy())
    return state


def setup_torch_srgan(scale: int = 2, device: str = "cuda") -> nn.Module:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available on this machine.")
    state = convert_srgan_ckpt_to_torch_state(CKPT_PATH)
    model = TorchGenerator(scale)
    missing, unexpected = model.load_state_dict(state, strict=False)
    disallowed_missing = [x for x in missing if not x.endswith("num_batches_tracked")]
    allowed_unexpected = [x for x in unexpected if x.startswith("subpixel_conv.1.")]
    disallowed_unexpected = [x for x in unexpected if x not in allowed_unexpected]
    if disallowed_missing or disallowed_unexpected:
        raise RuntimeError(
            f"SRGAN torch conversion mismatch. missing={disallowed_missing}, unexpected={disallowed_unexpected}"
        )
    model.eval().to(device)
    return model


def preprocess_torch(img: np.ndarray, device: str) -> torch.Tensor:
    array = img.astype(np.float32) / 127.5 - 1.0
    array = np.transpose(array, (2, 0, 1))
    array = np.expand_dims(array, 0)
    return torch.from_numpy(array).to(device)


def postprocess_torch(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.detach().float().clamp(-1.0, 1.0)[0]
    img = (img + 1.0) / 2.0
    img = img.permute(1, 2, 0).cpu().numpy()
    return (img * 255.0).clip(0, 255).astype(np.uint8)


def enhance_torch_image(
    model: nn.Module,
    img: np.ndarray,
    device: str,
    scale: int = 2,
    block_size: int = 512,
    overlap: int = 32,
) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= block_size and w <= block_size:
        with torch.inference_mode():
            return postprocess_torch(model(preprocess_torch(img, device)))

    output_h = h * scale
    output_w = w * scale
    output_img = np.zeros((output_h, output_w, 3), dtype=np.float32)
    weight_map = np.zeros((output_h, output_w), dtype=np.float32)

    num_blocks_h = math.ceil(h / (block_size - overlap))
    num_blocks_w = math.ceil(w / (block_size - overlap))

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            y_start = i * (block_size - overlap)
            x_start = j * (block_size - overlap)
            y_end = min(y_start + block_size, h)
            x_end = min(x_start + block_size, w)
            block = img[y_start:y_end, x_start:x_end]

            with torch.inference_mode():
                processed = postprocess_torch(model(preprocess_torch(block, device)))

            out_y_start = y_start * scale
            out_x_start = x_start * scale
            out_y_end = y_end * scale
            out_x_end = x_end * scale

            block_h, block_w = processed.shape[:2]
            weight = np.ones((block_h, block_w), dtype=np.float32)
            if overlap > 0:
                fade = np.linspace(0, 1, overlap * scale, dtype=np.float32)
                weight[: overlap * scale, :] *= fade[:, None]
                weight[-overlap * scale :, :] *= fade[::-1, None]
                weight[:, : overlap * scale] *= fade[None, :]
                weight[:, -overlap * scale :] *= fade[None, ::-1]

            output_img[out_y_start:out_y_end, out_x_start:out_x_end] += processed.astype(np.float32) * weight[..., None]
            weight_map[out_y_start:out_y_end, out_x_start:out_x_end] += weight

    weight_map = np.maximum(weight_map, 1e-6)
    output_img /= weight_map[..., None]
    return np.clip(output_img, 0, 255).astype(np.uint8)


def run_cpu_case(model, case: BenchmarkCase) -> BenchmarkResult:
    img = pil_rgb(case.image_path)
    latencies: List[float] = []
    output = None
    for i in range(case.warmup + case.repeats):
        t0 = time.perf_counter()
        out = model.enhance_image(img)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if out is None:
            raise RuntimeError(f"MindSpore CPU SRGAN failed for {case.case_id}")
        if i >= case.warmup:
            latencies.append(latency_ms)
            output = out

    out_path = OUTPUT_ROOT / "outputs_cpu" / f"{case.case_id}_srgan_cpu.png"
    save_rgb(output, out_path)
    h, w = output.shape[:2]
    ih, iw = img.shape[:2]
    return BenchmarkResult(
        case_id=case.case_id,
        title=case.title,
        input_path=str(case.image_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        input_width=iw,
        input_height=ih,
        backend="mindspore_cpu",
        repeats=case.repeats,
        avg_latency_ms=float(np.mean(latencies)),
        min_latency_ms=float(np.min(latencies)),
        max_latency_ms=float(np.max(latencies)),
        peak_gpu_mem_mb=0.0,
        output_path=str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        output_width=w,
        output_height=h,
        sha256=sha256(out_path),
    )


def run_torch_case(model: nn.Module, case: BenchmarkCase, device: str, backend: str, scale: int = 2) -> BenchmarkResult:
    img = pil_rgb(case.image_path)
    latencies: List[float] = []
    output = None
    peak_gpu_mem_mb = 0.0
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    for i in range(case.warmup + case.repeats):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = enhance_torch_image(model, img, device=device, scale=scale)
        if device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if i >= case.warmup:
            latencies.append(latency_ms)
            output = out

    out_dir = "outputs_gpu" if device == "cuda" else "outputs_torch_cpu"
    out_path = OUTPUT_ROOT / out_dir / f"{case.case_id}_{backend}.png"
    save_rgb(output, out_path)
    if device == "cuda":
        peak_gpu_mem_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    h, w = output.shape[:2]
    ih, iw = img.shape[:2]
    return BenchmarkResult(
        case_id=case.case_id,
        title=case.title,
        input_path=str(case.image_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        input_width=iw,
        input_height=ih,
        backend=backend,
        repeats=case.repeats,
        avg_latency_ms=float(np.mean(latencies)),
        min_latency_ms=float(np.min(latencies)),
        max_latency_ms=float(np.max(latencies)),
        peak_gpu_mem_mb=peak_gpu_mem_mb,
        output_path=str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        output_width=w,
        output_height=h,
        sha256=sha256(out_path),
    )


def mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(diff * diff))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    value = mse(a, b)
    if value <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(255.0 / math.sqrt(value)))


def compare_outputs(cpu_result: BenchmarkResult, gpu_result: BenchmarkResult) -> ComparisonResult:
    cpu = pil_rgb(PROJECT_ROOT / cpu_result.output_path)
    gpu = pil_rgb(PROJECT_ROOT / gpu_result.output_path)
    mae = float(np.mean(np.abs(cpu.astype(np.float32) - gpu.astype(np.float32))))
    return ComparisonResult(
        case_id=cpu_result.case_id,
        title=cpu_result.title,
        cpu_backend=cpu_result.backend,
        gpu_backend=gpu_result.backend,
        cpu_avg_ms=cpu_result.avg_latency_ms,
        gpu_avg_ms=gpu_result.avg_latency_ms,
        speedup=cpu_result.avg_latency_ms / max(gpu_result.avg_latency_ms, 1e-9),
        output_psnr=psnr(cpu, gpu),
        output_mae=mae,
    )


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_latency_and_speedup(comparisons: Sequence[ComparisonResult]) -> Tuple[Path, Path]:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    labels = [c.case_id for c in comparisons]
    cpu = [c.cpu_avg_ms for c in comparisons]
    gpu = [c.gpu_avg_ms for c in comparisons]
    speedup = [c.speedup for c in comparisons]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    bars1 = ax.bar(x - width / 2, cpu, width, label=comparisons[0].cpu_backend, color="#6C7A89")
    bars2 = ax.bar(x + width / 2, gpu, width, label=comparisons[0].gpu_backend, color="#2F6BFF")
    ax.set_ylabel("平均推理延迟 / ms")
    ax.set_title("SRGAN 原始权重真实 CPU/GPU 延迟对比", fontsize=15, pad=14, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False)
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + ax.get_ylim()[1] * 0.01, f"{h:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    latency_path = OUTPUT_ROOT / "figures" / "srgan_cpu_gpu_latency.png"
    fig.savefig(latency_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    bars = ax.bar(labels, speedup, color="#16A085")
    ax.set_ylabel("加速比 / ×")
    ax.set_title("SRGAN GPU 相对 CPU 加速比", fontsize=15, pad=14, weight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + ax.get_ylim()[1] * 0.02, f"{h:.2f}×", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    speedup_path = OUTPUT_ROOT / "figures" / "srgan_cpu_gpu_speedup.png"
    fig.savefig(speedup_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return latency_path, speedup_path


def write_summary(results: Sequence[BenchmarkResult], comparisons: Sequence[ComparisonResult], figures: Sequence[Path]) -> Path:
    gpu_env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "mindspore": ms.__version__,
        "mindspore_device_target": ms.get_context("device_target"),
        "srgan_checkpoint": str(CKPT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }
    env_path = OUTPUT_ROOT / "metrics" / "srgan_gpu_environment.json"
    env_path.write_text(json.dumps(gpu_env, ensure_ascii=False, indent=2), encoding="utf-8")

    avg_speedup = float(np.mean([c.speedup for c in comparisons]))
    avg_psnr = float(np.mean([c.output_psnr for c in comparisons]))
    avg_mae = float(np.mean([c.output_mae for c in comparisons]))
    max_gpu_mem = max(r.peak_gpu_mem_mb for r in results if r.backend == "torch_cuda")

    lines = [
        "# SRGAN 真实 CPU/GPU 性能补充说明",
        "",
        "本目录补充的是 **SRGAN 超分阶段** 的真实 GPU 性能数据。",
        "注意：这不是声称当前 Windows 主机已跑通完整 MindSpore deep GPU 版 CRA+SRGAN；",
        "当前主机的 MindSpore wheel 仍只能 CPU，因此这里采用“同一原始 SRGAN 检查点 + PyTorch CUDA 适配器”的方式补充真实 GPU 超分性能。",
        "",
        "## 运行环境",
        f"- GPU：`{gpu_env['device_name']}`",
        f"- PyTorch：`{gpu_env['torch']}`，CUDA：`{gpu_env['cuda_version']}`",
        f"- MindSpore：`{gpu_env['mindspore']}`（CPU 基线）",
        f"- SRGAN 检查点：`{gpu_env['srgan_checkpoint']}`",
        f"- 环境 JSON：`{env_path.relative_to(PROJECT_ROOT).as_posix()}`",
        "",
        "## 结论摘要",
        f"- 平均 GPU / CPU 加速比（`{comparisons[0].cpu_backend}` → `{comparisons[0].gpu_backend}`）：`{avg_speedup:.2f}×`",
        f"- CPU / GPU 输出平均 PSNR：`{avg_psnr:.2f} dB`",
        f"- CPU / GPU 输出平均 MAE：`{avg_mae:.4f}`",
        f"- GPU 峰值显存（本批次最大）：`{max_gpu_mem:.1f} MB`",
        "",
        "## 解释边界",
        "- 论文中建议重点引用 `torch_cpu ↔ torch_cuda` 的真实加速比，因为二者使用完全相同的转换后 SRGAN 权重，输出几乎一致。",
        "- `mindspore_cpu` 结果仍保留在 CSV 中，作为当前仓库原始实现的 CPU 参考性能。",
        "- 不建议把 `mindspore_cpu ↔ torch_cuda` 的输出差异当作质量结论；该差异只说明当前 Windows 主机无法给出完整、同实现的 MindSpore GPU 对照。",
        "",
        "## 生成文件",
    ]
    for figure in figures:
        lines.append(f"- `{figure.relative_to(PROJECT_ROOT).as_posix()}`")
    lines.extend(
        [
            "- `paper_results/gpu_benchmark/metrics/srgan_cpu_gpu_benchmark.csv`",
            "- `paper_results/gpu_benchmark/metrics/srgan_cpu_gpu_comparison.csv`",
            "",
            "## 论文写法建议",
            "可在实验或工程实现章节中补充说明：在当前 Windows 主机上，完整 MindSpore deep GPU 后端受官方 wheel/算子支持限制；",
            "因此保留 MindSpore CPU 结果作为原始工程基线，并基于同一 SRGAN 原始权重实现 PyTorch CPU/CUDA 对照推理，获得真实的 NVIDIA GPU 超分推理性能，用于补充加速效果分析。",
        ]
    )
    out = OUTPUT_ROOT / "README_gpu_benchmark.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    ensure_dirs()
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Missing SRGAN checkpoint: {CKPT_PATH}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available; cannot benchmark real GPU performance.")

    print("[1/5] 初始化 CPU / GPU SRGAN 模型")
    ms.set_device(device_target="CPU")
    cpu_model = setup_cpu_srgan(scale=2)
    torch_cpu_model = setup_torch_srgan(scale=2, device="cpu")
    gpu_model = setup_torch_srgan(scale=2, device="cuda")

    print("[2/5] 运行 CPU 基线")
    ms_cpu_results = [run_cpu_case(cpu_model, case) for case in BENCHMARK_CASES]
    torch_cpu_results = [run_torch_case(torch_cpu_model, case, device="cpu", backend="torch_cpu", scale=2) for case in BENCHMARK_CASES]

    print("[3/5] 运行 GPU 基线")
    gpu_results = [run_torch_case(gpu_model, case, device="cuda", backend="torch_cuda", scale=2) for case in BENCHMARK_CASES]

    print("[4/5] 计算输出一致性与加速比")
    comparisons = [compare_outputs(cpu, gpu) for cpu, gpu in zip(torch_cpu_results, gpu_results)]
    write_csv(OUTPUT_ROOT / "metrics" / "srgan_cpu_gpu_benchmark.csv", [asdict(r) for r in [*ms_cpu_results, *torch_cpu_results, *gpu_results]])
    write_csv(OUTPUT_ROOT / "metrics" / "srgan_cpu_gpu_comparison.csv", [asdict(c) for c in comparisons])
    figures = list(plot_latency_and_speedup(comparisons))

    print("[5/5] 写入说明文档")
    summary = write_summary([*ms_cpu_results, *torch_cpu_results, *gpu_results], comparisons, figures)

    print("\nGPU benchmark finished:")
    for result in comparisons:
        print(
            f"- {result.case_id}: CPU={result.cpu_avg_ms:.1f} ms, GPU={result.gpu_avg_ms:.1f} ms, "
            f"speedup={result.speedup:.2f}x, output_psnr={result.output_psnr:.2f} dB"
        )
    print(f"- summary: {summary.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
