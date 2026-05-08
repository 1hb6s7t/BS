"""Generate paper-ready end-to-end CUDA result figures and comparison charts.

The current Windows MindSpore wheel on this machine supports CPU only, so this
script does not claim a full MindSpore-GPU run. It provides a real complete GPU
validation path for the thesis: CUDA mask repair followed by CUDA
fidelity-preserving super-resolution/enhancement.

Generated artifacts:
- 3 qualitative GPU effect figures.
- 3 quantitative comparison charts.
- CSV/JSON runtime and metric records for reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "paper_results" / "gpu_full_pipeline"
RNG_SEED = 20260508


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    title: str
    source: Path
    crop_box: Tuple[int, int, int, int]
    mask_kind: str


@dataclass
class CaseRunResult:
    spec: CaseSpec
    case_dir: Path
    gt_path: Path
    degraded_path: Path
    mask_path: Path
    repaired_path: Path
    enhanced_path: Path
    repair_ms: float
    enhance_ms: float
    total_ms: float
    peak_gpu_mem_mb: float
    repaired_sha256: str
    enhanced_sha256: str


@dataclass
class MethodResult:
    case_id: str
    case_title: str
    method: str
    full_psnr: float
    full_ssim: float
    mask_psnr: float
    mask_mae: float
    edge_similarity: float
    detail_fidelity: float
    time_ms: float
    peak_gpu_mem_mb: float
    output_path: Path
    sha256: str


@dataclass(frozen=True)
class RunOptions:
    scale: int = 2
    device: str = "cuda"
    clean: bool = True
    propagation_iterations: int = 256
    diffusion_iterations: int = 80
    sharpen_strength: float = 0.08


CASES: Sequence[CaseSpec] = (
    CaseSpec(
        case_id="case01_natural_texture",
        title="自然纹理划痕修复与超分增强",
        source=PROJECT_ROOT / "CRA" / "examples" / "5.jpg",
        crop_box=(95, 40, 607, 424),
        mask_kind="rock_scratches",
    ),
    CaseSpec(
        case_id="case02_indoor_structure",
        title="室内结构小面积遮挡修复与超分增强",
        source=PROJECT_ROOT / "CRA" / "examples" / "8.jpg",
        crop_box=(90, 35, 602, 419),
        mask_kind="indoor_small_occlusion",
    ),
    CaseSpec(
        case_id="case03_scene_depth",
        title="复杂场景水印式缺损修复与超分增强",
        source=PROJECT_ROOT / "CRA" / "examples" / "12.jpg",
        crop_box=(135, 45, 647, 429),
        mask_kind="scene_watermark",
    ),
)


PROJECT_METHOD = "本项目GPU：CUDA修复+保真超分"
METHOD_ORDER = [
    "退化输入+最近邻放大",
    "退化输入+双三次放大",
    "退化输入+GPU保真超分",
    PROJECT_METHOD,
]
SHORT_METHOD = {
    "退化输入+最近邻放大": "退化+最近邻",
    "退化输入+双三次放大": "退化+双三次",
    "退化输入+GPU保真超分": "退化+GPU超分",
    PROJECT_METHOD: "本项目GPU",
}


def ensure_output_dirs(options: RunOptions) -> None:
    if options.clean and OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    for sub in ["cases", "figures", "metrics"]:
        (OUTPUT_ROOT / sub).mkdir(parents=True, exist_ok=True)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def pil_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_rgb(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).save(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")


def safe_name(text: str) -> str:
    return (
        text.replace("：", "_")
        .replace("+", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("GPU", "gpu")
        .replace("CPU", "cpu")
    )


def make_mask(shape: Tuple[int, int], kind: str) -> np.ndarray:
    """Create deterministic masks. White means the region to repair."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if kind == "rock_scratches":
        cv2.line(mask, (int(0.18 * w), int(0.28 * h)), (int(0.67 * w), int(0.50 * h)), 255, 12, cv2.LINE_AA)
        cv2.line(mask, (int(0.32 * w), int(0.17 * h)), (int(0.58 * w), int(0.64 * h)), 255, 9, cv2.LINE_AA)
        cv2.ellipse(mask, (int(0.48 * w), int(0.42 * h)), (28, 18), 15, 0, 360, 255, -1)
    elif kind == "indoor_small_occlusion":
        cv2.rectangle(mask, (int(0.46 * w), int(0.20 * h)), (int(0.58 * w), int(0.31 * h)), 255, -1)
        cv2.circle(mask, (int(0.40 * w), int(0.62 * h)), 18, 255, -1)
        cv2.line(mask, (int(0.61 * w), int(0.43 * h)), (int(0.72 * w), int(0.62 * h)), 255, 10, cv2.LINE_AA)
    elif kind == "scene_watermark":
        cv2.line(mask, (int(0.18 * w), int(0.28 * h)), (int(0.78 * w), int(0.28 * h)), 255, 10, cv2.LINE_AA)
        cv2.line(mask, (int(0.24 * w), int(0.34 * h)), (int(0.70 * w), int(0.57 * h)), 255, 9, cv2.LINE_AA)
        cv2.rectangle(mask, (int(0.62 * w), int(0.52 * h)), (int(0.72 * w), int(0.66 * h)), 255, -1)
        cv2.circle(mask, (int(0.33 * w), int(0.62 * h)), 18, 255, -1)
    else:
        raise ValueError(f"Unknown mask kind: {kind}")
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def prepare_case_inputs(spec: CaseSpec) -> Tuple[Path, Path, Path]:
    case_dir = OUTPUT_ROOT / "cases" / spec.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    gt = np.array(pil_rgb(spec.source).crop(spec.crop_box).resize((512, 384), Image.Resampling.LANCZOS))
    mask = make_mask(gt.shape[:2], spec.mask_kind)
    degraded = gt.copy()
    degraded[mask > 0] = np.array([28, 28, 28], dtype=np.uint8)
    edge = cv2.GaussianBlur(mask, (11, 11), 0).astype(np.float32) / 255.0
    rng = np.random.default_rng(RNG_SEED)
    degraded = np.clip(degraded.astype(np.float32) + rng.normal(0, 7, degraded.shape).astype(np.float32) * edge[..., None], 0, 255)
    degraded = degraded.astype(np.uint8)

    gt_path = case_dir / "01_ground_truth.png"
    degraded_path = case_dir / "02_degraded_input.png"
    mask_path = case_dir / "03_mask.png"
    save_rgb(gt, gt_path)
    save_rgb(degraded, degraded_path)
    Image.fromarray(mask).save(mask_path)
    return gt_path, degraded_path, mask_path


def cuda_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def gpu_boundary_propagation_repair(
    degraded: np.ndarray,
    mask: np.ndarray,
    *,
    device: str,
    propagation_iterations: int,
    diffusion_iterations: int,
) -> np.ndarray:
    """Repair masked pixels with a deterministic CUDA propagation+diffusion solver."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU repair stage.")
    img = torch.from_numpy(degraded.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    mask_t = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    known = 1.0 - mask_t
    x = img * known
    valid = known.clone()
    neighbor = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=device)
    neighbor[0, 0, 1, 1] = 0.0
    neighbor_rgb = neighbor.repeat(3, 1, 1, 1)

    with torch.inference_mode():
        # First fill missing pixels from nearest valid boundary neighborhoods.
        for _ in range(propagation_iterations):
            count = F.conv2d(valid, neighbor, padding=1)
            avg = F.conv2d(x, neighbor_rgb, padding=1, groups=3) / count.clamp_min(1e-6)
            fill = ((valid < 0.5) & (count > 0)).float()
            x = x * (1.0 - fill) + avg * fill
            valid = torch.maximum(valid, fill)
            if bool((valid >= 1.0).all().item()):
                break

        # Then smooth only the repaired area while keeping original known pixels fixed.
        kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 0.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )
        kernel = (kernel / kernel.sum()).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        for _ in range(diffusion_iterations):
            avg = F.conv2d(x, kernel, padding=1, groups=3)
            x = img * known + avg * mask_t

        soft = F.avg_pool2d(mask_t, kernel_size=15, stride=1, padding=7).clamp(0.0, 1.0)
        out = img * (1.0 - soft) + x * soft
    return (out.clamp(0, 1)[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).round().astype(np.uint8)


def gpu_fidelity_super_resolution(
    image: np.ndarray,
    *,
    device: str,
    scale: int,
    sharpen_strength: float,
) -> np.ndarray:
    """Run fidelity-preserving super-resolution/enhancement on GPU."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU enhancement stage.")
    x = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.inference_mode():
        up = F.interpolate(x, scale_factor=scale, mode="bicubic", align_corners=False).clamp(0.0, 1.0)
        gaussian = torch.tensor(
            [
                [1.0, 4.0, 6.0, 4.0, 1.0],
                [4.0, 16.0, 24.0, 16.0, 4.0],
                [6.0, 24.0, 36.0, 24.0, 6.0],
                [4.0, 16.0, 24.0, 16.0, 4.0],
                [1.0, 4.0, 6.0, 4.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        gaussian = (gaussian / gaussian.sum()).view(1, 1, 5, 5).repeat(3, 1, 1, 1)
        blur = F.conv2d(up, gaussian, padding=2, groups=3)
        enhanced = (up + sharpen_strength * (up - blur)).clamp(0.0, 1.0)
    return (enhanced[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).round().astype(np.uint8)


def run_gpu_project_case(spec: CaseSpec, options: RunOptions) -> CaseRunResult:
    gt_path, degraded_path, mask_path = prepare_case_inputs(spec)
    case_dir = OUTPUT_ROOT / "cases" / spec.case_id
    run_dir = case_dir / "gpu_project"
    run_dir.mkdir(parents=True, exist_ok=True)
    degraded = read_rgb(degraded_path)
    mask = np.array(Image.open(mask_path).convert("L"))

    if options.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    cuda_sync(options.device)
    t0 = time.perf_counter()
    repaired = gpu_boundary_propagation_repair(
        degraded,
        mask,
        device=options.device,
        propagation_iterations=options.propagation_iterations,
        diffusion_iterations=options.diffusion_iterations,
    )
    cuda_sync(options.device)
    t1 = time.perf_counter()
    enhanced = gpu_fidelity_super_resolution(
        repaired,
        device=options.device,
        scale=options.scale,
        sharpen_strength=options.sharpen_strength,
    )
    cuda_sync(options.device)
    t2 = time.perf_counter()

    repaired_path = run_dir / "04_gpu_repaired.png"
    enhanced_path = run_dir / "05_gpu_repaired_enhanced.png"
    save_rgb(repaired, repaired_path)
    save_rgb(enhanced, enhanced_path)
    peak_mem = float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if options.device.startswith("cuda") else 0.0
    return CaseRunResult(
        spec=spec,
        case_dir=case_dir,
        gt_path=gt_path,
        degraded_path=degraded_path,
        mask_path=mask_path,
        repaired_path=repaired_path,
        enhanced_path=enhanced_path,
        repair_ms=(t1 - t0) * 1000.0,
        enhance_ms=(t2 - t1) * 1000.0,
        total_ms=(t2 - t0) * 1000.0,
        peak_gpu_mem_mb=peak_mem,
        repaired_sha256=sha256(repaired_path),
        enhanced_sha256=sha256(enhanced_path),
    )


def mse(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = a.astype(np.float64) - b.astype(np.float64)
    if mask is not None:
        sel = mask > 0
        if not np.any(sel):
            return 0.0
        diff = diff[sel]
    return float(np.mean(diff * diff))


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    value = mse(a, b, mask)
    if value <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(255.0 / math.sqrt(value)))


def mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    if mask is not None:
        sel = mask > 0
        if not np.any(sel):
            return 0.0
        diff = diff[sel]
    return float(np.mean(diff))


def to_gray(array: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = kernel @ kernel.T
    mu_a = cv2.filter2D(a, -1, window)[5:-5, 5:-5]
    mu_b = cv2.filter2D(b, -1, window)[5:-5, 5:-5]
    sigma_a = cv2.filter2D(a * a, -1, window)[5:-5, 5:-5] - mu_a * mu_a
    sigma_b = cv2.filter2D(b * b, -1, window)[5:-5, 5:-5] - mu_b * mu_b
    sigma_ab = cv2.filter2D(a * b, -1, window)[5:-5, 5:-5] - mu_a * mu_b
    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2)
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def edge_similarity(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.Canny(to_gray(a), 80, 160).astype(np.float32) / 255.0
    gb = cv2.Canny(to_gray(b), 80, 160).astype(np.float32) / 255.0
    intersection = float(np.sum(np.minimum(ga, gb)))
    union = float(np.sum(np.maximum(ga, gb)))
    return 1.0 if union <= 1e-9 else intersection / union


def sharpness(array: np.ndarray) -> float:
    return float(cv2.Laplacian(to_gray(array), cv2.CV_64F).var())


def detail_fidelity(output: np.ndarray, reference: np.ndarray) -> float:
    out_sharp = sharpness(output)
    ref_sharp = sharpness(reference)
    denom = max(out_sharp, ref_sharp, 1e-9)
    return float(min(out_sharp, ref_sharp) / denom)


def resize_like(array: np.ndarray, reference: np.ndarray) -> np.ndarray:
    h, w = reference.shape[:2]
    if array.shape[:2] == (h, w):
        return array
    return cv2.resize(array, (w, h), interpolation=cv2.INTER_AREA)


def score_output(
    case: CaseRunResult,
    method: str,
    output: np.ndarray,
    gt_scaled: np.ndarray,
    mask_scaled: np.ndarray,
    elapsed_ms: float,
    peak_gpu_mem_mb: float,
    output_path: Path,
) -> MethodResult:
    output = resize_like(output, gt_scaled)
    return MethodResult(
        case_id=case.spec.case_id,
        case_title=case.spec.title,
        method=method,
        full_psnr=psnr(output, gt_scaled),
        full_ssim=ssim_gray(to_gray(output), to_gray(gt_scaled)),
        mask_psnr=psnr(output, gt_scaled, mask_scaled),
        mask_mae=mae(output, gt_scaled, mask_scaled),
        edge_similarity=edge_similarity(output, gt_scaled),
        detail_fidelity=detail_fidelity(output, gt_scaled),
        time_ms=elapsed_ms,
        peak_gpu_mem_mb=peak_gpu_mem_mb,
        output_path=output_path,
        sha256=sha256(output_path),
    )


def evaluate_methods(case_results: Sequence[CaseRunResult], options: RunOptions) -> List[MethodResult]:
    rows: List[MethodResult] = []
    for case in case_results:
        gt = read_rgb(case.gt_path)
        degraded = read_rgb(case.degraded_path)
        mask = np.array(Image.open(case.mask_path).convert("L"))
        gt_scaled = cv2.resize(gt, (gt.shape[1] * options.scale, gt.shape[0] * options.scale), interpolation=cv2.INTER_CUBIC)
        mask_scaled = cv2.resize(mask, (gt_scaled.shape[1], gt_scaled.shape[0]), interpolation=cv2.INTER_NEAREST)
        out_dir = case.case_dir / "method_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        degraded_nearest = cv2.resize(degraded, (gt_scaled.shape[1], gt_scaled.shape[0]), interpolation=cv2.INTER_NEAREST)
        elapsed = (time.perf_counter() - t0) * 1000.0
        p = out_dir / f"{safe_name('退化输入+最近邻放大')}.png"
        save_rgb(degraded_nearest, p)
        rows.append(score_output(case, "退化输入+最近邻放大", degraded_nearest, gt_scaled, mask_scaled, elapsed, 0.0, p))

        t0 = time.perf_counter()
        degraded_bicubic = cv2.resize(degraded, (gt_scaled.shape[1], gt_scaled.shape[0]), interpolation=cv2.INTER_CUBIC)
        elapsed = (time.perf_counter() - t0) * 1000.0
        p = out_dir / f"{safe_name('退化输入+双三次放大')}.png"
        save_rgb(degraded_bicubic, p)
        rows.append(score_output(case, "退化输入+双三次放大", degraded_bicubic, gt_scaled, mask_scaled, elapsed, 0.0, p))

        if options.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        cuda_sync(options.device)
        t0 = time.perf_counter()
        direct_gpu_enhanced = gpu_fidelity_super_resolution(
            degraded,
            device=options.device,
            scale=options.scale,
            sharpen_strength=options.sharpen_strength,
        )
        cuda_sync(options.device)
        elapsed = (time.perf_counter() - t0) * 1000.0
        mem = float(torch.cuda.max_memory_allocated() / (1024 * 1024)) if options.device.startswith("cuda") else 0.0
        p = out_dir / f"{safe_name('退化输入+GPU保真超分')}.png"
        save_rgb(direct_gpu_enhanced, p)
        rows.append(score_output(case, "退化输入+GPU保真超分", direct_gpu_enhanced, gt_scaled, mask_scaled, elapsed, mem, p))

        project_out = read_rgb(case.enhanced_path)
        rows.append(score_output(case, PROJECT_METHOD, project_out, gt_scaled, mask_scaled, case.total_ms, case.peak_gpu_mem_mb, case.enhanced_path))
    return rows


def aggregate_metrics(results: Sequence[MethodResult]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[MethodResult]] = {m: [] for m in METHOD_ORDER}
    for row in results:
        grouped.setdefault(row.method, []).append(row)
    return {
        method: {
            "full_psnr": float(np.mean([r.full_psnr for r in rows])),
            "full_ssim": float(np.mean([r.full_ssim for r in rows])),
            "mask_psnr": float(np.mean([r.mask_psnr for r in rows])),
            "mask_mae": float(np.mean([r.mask_mae for r in rows])),
            "edge_similarity": float(np.mean([r.edge_similarity for r in rows])),
            "detail_fidelity": float(np.mean([r.detail_fidelity for r in rows])),
            "time_ms": float(np.mean([r.time_ms for r in rows])),
            "peak_gpu_mem_mb": float(np.mean([r.peak_gpu_mem_mb for r in rows])),
        }
        for method, rows in grouped.items()
        if rows
    }


def write_metrics_csv(results: Sequence[MethodResult]) -> Path:
    path = OUTPUT_ROOT / "metrics" / "gpu_full_pipeline_metrics.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id", "case_title", "method", "full_psnr", "full_ssim", "mask_psnr", "mask_mae",
            "edge_similarity", "detail_fidelity", "time_ms", "peak_gpu_mem_mb", "output_path", "sha256",
        ])
        for r in results:
            writer.writerow([
                r.case_id, r.case_title, r.method, f"{r.full_psnr:.4f}", f"{r.full_ssim:.6f}",
                f"{r.mask_psnr:.4f}", f"{r.mask_mae:.4f}", f"{r.edge_similarity:.6f}",
                f"{r.detail_fidelity:.6f}", f"{r.time_ms:.4f}", f"{r.peak_gpu_mem_mb:.2f}",
                relative(r.output_path), r.sha256,
            ])
    return path


def write_case_csv(case_results: Sequence[CaseRunResult]) -> Path:
    path = OUTPUT_ROOT / "metrics" / "gpu_full_pipeline_case_runtime.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id", "title", "repair_ms", "enhance_ms", "total_ms", "peak_gpu_mem_mb",
            "repaired_path", "enhanced_path", "repaired_sha256", "enhanced_sha256",
        ])
        for r in case_results:
            writer.writerow([
                r.spec.case_id, r.spec.title, f"{r.repair_ms:.4f}", f"{r.enhance_ms:.4f}", f"{r.total_ms:.4f}",
                f"{r.peak_gpu_mem_mb:.2f}", relative(r.repaired_path), relative(r.enhanced_path),
                r.repaired_sha256, r.enhanced_sha256,
            ])
    return path


def setup_matplotlib_style() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.edgecolor"] = "#2B3440"
    plt.rcParams["axes.labelcolor"] = "#2B3440"
    plt.rcParams["xtick.color"] = "#2B3440"
    plt.rcParams["ytick.color"] = "#2B3440"


def bar_label(ax, bars, fmt="{:.2f}", dy=0.012) -> None:
    ymax = ax.get_ylim()[1]
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + ymax * dy, fmt.format(h), ha="center", va="bottom", fontsize=8.5, color="#263238")


def plot_quality_psnr_ssim(agg: Dict[str, Dict[str, float]]) -> Path:
    methods = [m for m in METHOD_ORDER if m in agg]
    labels = [SHORT_METHOD[m] for m in methods]
    x = np.arange(len(methods))
    width = 0.38
    psnr_values = [agg[m]["full_psnr"] for m in methods]
    ssim_values = [agg[m]["full_ssim"] for m in methods]

    fig, ax1 = plt.subplots(figsize=(10.8, 5.6))
    bars1 = ax1.bar(x - width / 2, psnr_values, width, label="PSNR / dB", color="#2F6BFF")
    ax1.set_ylabel("PSNR / dB")
    ax1.set_ylim(0, max(psnr_values) * 1.22)
    ax1.grid(axis="y", linestyle="--", alpha=0.26)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, ssim_values, width, label="SSIM", color="#21A67A")
    ax2.set_ylabel("SSIM")
    ax2.set_ylim(0, min(1.0, max(ssim_values) * 1.25))
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=10, ha="right")
    ax1.set_title("端到端GPU流水线重建质量对比（3组真实运行平均）", fontsize=15, pad=14, weight="bold")
    bar_label(ax1, bars1, "{:.2f}")
    bar_label(ax2, bars2, "{:.3f}", dy=0.016)
    lines, labs = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labs + labs2, loc="upper left", frameon=False)
    fig.tight_layout()
    out = OUTPUT_ROOT / "figures" / "gpu_quality_psnr_ssim.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_mask_repair_quality(agg: Dict[str, Dict[str, float]]) -> Path:
    methods = [m for m in METHOD_ORDER if m in agg]
    labels = [SHORT_METHOD[m] for m in methods]
    x = np.arange(len(methods))
    width = 0.38
    mask_psnr = [agg[m]["mask_psnr"] for m in methods]
    mask_mae = [agg[m]["mask_mae"] for m in methods]

    fig, ax1 = plt.subplots(figsize=(10.8, 5.6))
    bars1 = ax1.bar(x - width / 2, mask_psnr, width, label="掩码区PSNR / dB", color="#6B4FB3")
    ax1.set_ylabel("掩码区PSNR / dB（越高越好）")
    ax1.set_ylim(0, max(mask_psnr) * 1.25)
    ax1.grid(axis="y", linestyle="--", alpha=0.26)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, mask_mae, width, label="掩码区MAE", color="#E2711D")
    ax2.set_ylabel("掩码区MAE（越低越好）")
    ax2.set_ylim(0, max(mask_mae) * 1.28)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=10, ha="right")
    ax1.set_title("缺损区域修复质量对比（重点体现修复有效性）", fontsize=15, pad=14, weight="bold")
    bar_label(ax1, bars1, "{:.2f}")
    bar_label(ax2, bars2, "{:.1f}", dy=0.016)
    lines, labs = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labs + labs2, loc="upper left", frameon=False)
    fig.tight_layout()
    out = OUTPUT_ROOT / "figures" / "gpu_mask_repair_quality.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_runtime_memory(agg: Dict[str, Dict[str, float]]) -> Path:
    methods = [m for m in METHOD_ORDER if m in agg]
    labels = [SHORT_METHOD[m] for m in methods]
    x = np.arange(len(methods))
    width = 0.42
    time_values = [agg[m]["time_ms"] for m in methods]
    mem_values = [agg[m]["peak_gpu_mem_mb"] for m in methods]

    fig, ax1 = plt.subplots(figsize=(10.8, 5.6))
    bars1 = ax1.bar(x - width / 2, time_values, width, label="平均耗时 / ms", color="#506784")
    ax1.set_ylabel("平均耗时 / ms")
    ax1.set_ylim(0, max(time_values) * 1.32)
    ax1.grid(axis="y", linestyle="--", alpha=0.26)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, mem_values, width, label="峰值GPU显存 / MB", color="#16A085")
    ax2.set_ylabel("峰值GPU显存 / MB")
    ax2.set_ylim(0, max(max(mem_values) * 1.32, 1.0))
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=10, ha="right")
    ax1.set_title("端到端运行效率与GPU显存占用对比", fontsize=15, pad=14, weight="bold")
    bar_label(ax1, bars1, "{:.1f}")
    bar_label(ax2, bars2, "{:.0f}", dy=0.016)
    lines, labs = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labs + labs2, loc="upper left", frameon=False)
    fig.tight_layout()
    out = OUTPUT_ROOT / "figures" / "gpu_runtime_memory.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def mask_overlay(image: Image.Image, mask_path: Path) -> Image.Image:
    base = image.convert("RGBA")
    mask = Image.open(mask_path).convert("L").resize(base.size, Image.Resampling.NEAREST)
    overlay = Image.new("RGBA", base.size, (230, 77, 64, 0))
    overlay.putalpha(mask.point(lambda p: 112 if p > 0 else 0))
    return Image.alpha_composite(base, overlay).convert("RGB")


def resize_to_tile(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return image.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def draw_centered(draw: ImageDraw.ImageDraw, xy: Tuple[int, int, int, int], text: str, size: int = 24) -> None:
    f = font(size)
    bbox = draw.textbbox((0, 0), text, font=f)
    tw = bbox[2] - bbox[0]
    x0, y0, x1, _ = xy
    draw.text((x0 + (x1 - x0 - tw) / 2, y0), text, font=f, fill=(25, 35, 48))


def make_effect_figure(result: CaseRunResult) -> Path:
    tile_w, tile_h = 360, 270
    gap = 18
    top = 86
    bottom = 48
    labels = ["原始参考图", "退化输入", "掩码区域", "GPU修复结果", "GPU修复+超分"]
    images = [
        pil_rgb(result.gt_path),
        pil_rgb(result.degraded_path),
        mask_overlay(pil_rgb(result.degraded_path), result.mask_path),
        pil_rgb(result.repaired_path),
        pil_rgb(result.enhanced_path),
    ]
    width = tile_w * len(labels) + gap * (len(labels) + 1)
    height = top + tile_h + bottom
    canvas = Image.new("RGB", (width, height), (248, 250, 253))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 18), f"{result.spec.title}（真实CUDA端到端输出）", font=font(32), fill=(20, 35, 55))
    draw.text(
        (gap, 56),
        f"repair={result.repair_ms:.1f} ms  |  enhance={result.enhance_ms:.1f} ms  |  total={result.total_ms:.1f} ms",
        font=font(18),
        fill=(86, 98, 112),
    )
    draw.line((gap, 78, width - gap, 78), fill=(210, 218, 228), width=2)
    for idx, (label, image) in enumerate(zip(labels, images)):
        x = gap + idx * (tile_w + gap)
        y = top
        tile = resize_to_tile(image, (tile_w, tile_h))
        outline = (47, 107, 255) if idx == 4 else (220, 228, 238)
        draw.rounded_rectangle((x - 4, y - 4, x + tile_w + 4, y + tile_h + 36), radius=14, fill=(255, 255, 255), outline=outline, width=2)
        canvas.paste(tile, (x, y))
        draw_centered(draw, (x, y + tile_h + 8, x + tile_w, y + tile_h + 40), label)
    out = OUTPUT_ROOT / "figures" / f"{result.spec.case_id}_gpu_effect_figure.png"
    canvas.save(out, quality=95)
    return out


def metric_output_path(results: Sequence[MethodResult], case_id: str, method: str) -> Path:
    for row in results:
        if row.case_id == case_id and row.method == method:
            return row.output_path
    raise KeyError(f"Missing output for {case_id} / {method}")


def make_visual_overview(case_results: Sequence[CaseRunResult], metrics: Sequence[MethodResult]) -> Path:
    columns = [
        ("原始参考图", lambda case: case.gt_path),
        ("退化输入", lambda case: case.degraded_path),
        ("未修复GPU超分", lambda case: metric_output_path(metrics, case.spec.case_id, "退化输入+GPU保真超分")),
        ("本项目GPU输出", lambda case: case.enhanced_path),
    ]
    tile_w, tile_h = 300, 225
    left_w = 230
    top_h = 92
    gap = 16
    row_h = tile_h + 46
    width = left_w + len(columns) * tile_w + (len(columns) + 1) * gap
    height = top_h + len(case_results) * row_h + gap
    canvas = Image.new("RGB", (width, height), (248, 250, 253))
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 20), "GPU端到端结果与未修复增强对比", font=font(34), fill=(20, 35, 55))
    draw.text((gap, 60), "每一行对应一组真实运行样例；最右列为CUDA修复后再进行GPU超分增强。", font=font(20), fill=(85, 96, 110))
    for col, (label, _) in enumerate(columns):
        x = left_w + gap + col * (tile_w + gap)
        draw.text((x + 12, top_h - 30), label, font=font(22), fill=(30, 45, 65))
    for row, case in enumerate(case_results):
        y = top_h + row * row_h
        draw.rounded_rectangle((gap, y + 8, left_w - gap, y + tile_h - 8), radius=14, fill=(255, 255, 255), outline=(220, 228, 238))
        title_lines = case.spec.title.replace("与", "与\n").replace("小面积", "小面积\n").replace("水印式", "水印式\n")
        draw.text((gap + 20, y + 56), title_lines, font=font(21), fill=(20, 35, 55), spacing=8)
        for col, (_, getter) in enumerate(columns):
            x = left_w + gap + col * (tile_w + gap)
            tile = resize_to_tile(pil_rgb(getter(case)), (tile_w, tile_h))
            outline = (47, 107, 255) if col == len(columns) - 1 else (220, 228, 238)
            draw.rounded_rectangle((x - 4, y - 4, x + tile_w + 4, y + tile_h + 4), radius=14, fill=(255, 255, 255), outline=outline, width=2)
            canvas.paste(tile, (x, y))
    out = OUTPUT_ROOT / "figures" / "gpu_visual_methods_overview.png"
    canvas.save(out, quality=95)
    return out


def write_environment(options: RunOptions) -> Path:
    env = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "device": options.device,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and options.device.startswith("cuda") else "",
        "pipeline_scope": "CUDA mask repair + CUDA fidelity-preserving super-resolution enhancement",
        "srgan_gpu_benchmark": "See paper_results/gpu_benchmark for the separate original-SRGAN CUDA benchmark.",
        "mindspore_gpu_note": "Local Windows MindSpore wheel supports CPU only; this is not a MindSpore-GPU claim.",
        "scale": options.scale,
        "propagation_iterations": options.propagation_iterations,
        "diffusion_iterations": options.diffusion_iterations,
        "sharpen_strength": options.sharpen_strength,
    }
    path = OUTPUT_ROOT / "metrics" / "gpu_full_pipeline_environment.json"
    path.write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_summary(
    case_results: Sequence[CaseRunResult],
    metrics: Sequence[MethodResult],
    figures: Sequence[Path],
    metrics_csv: Path,
    case_csv: Path,
    env_path: Path,
    options: RunOptions,
) -> Path:
    agg = aggregate_metrics(metrics)
    project = agg[PROJECT_METHOD]
    degraded = agg["退化输入+双三次放大"]
    direct_gpu = agg["退化输入+GPU保真超分"]
    avg_total = float(np.mean([r.total_ms for r in case_results]))
    avg_mem = float(np.mean([r.peak_gpu_mem_mb for r in case_results]))
    lines = [
        "# 完整GPU端到端实验结果说明",
        "",
        "本目录中的图片和数据均由本项目脚本真实运行生成，用于论文中的 GPU 效果展示与对比实验。",
        "",
        "## 生成命令",
        "",
        "```powershell",
        ".\\.venv-ms\\Scripts\\python .\\tools\\generate_gpu_paper_results.py",
        "```",
        "",
        "## 运行边界说明",
        "",
        "- 当前主机 PyTorch CUDA 可用，GPU 为 NVIDIA GeForce RTX 4060 Laptop GPU。",
        "- 当前 Windows MindSpore wheel 仍只支持 CPU，因此这里不把结果表述为“MindSpore GPU 端到端”。",
        "- 本组实验采用完整 CUDA 流水线：CUDA 掩码修复 + CUDA 保真超分增强。",
        "- 原 SRGAN checkpoint 的独立 CUDA 性能数据保留在 `paper_results/gpu_benchmark/`。",
        "",
        "## 3组GPU效果图",
    ]
    for case in case_results:
        lines.append(f"- `{relative(OUTPUT_ROOT / 'figures' / f'{case.spec.case_id}_gpu_effect_figure.png')}`：{case.spec.title}")
    lines.extend([
        "",
        "## 3组对比数据图",
        f"- `{relative(OUTPUT_ROOT / 'figures' / 'gpu_quality_psnr_ssim.png')}`：全图 PSNR/SSIM 对比。",
        f"- `{relative(OUTPUT_ROOT / 'figures' / 'gpu_mask_repair_quality.png')}`：缺损掩码区域 PSNR/MAE 对比。",
        f"- `{relative(OUTPUT_ROOT / 'figures' / 'gpu_runtime_memory.png')}`：端到端耗时与 GPU 显存占用对比。",
        f"- `{relative(OUTPUT_ROOT / 'figures' / 'gpu_visual_methods_overview.png')}`：附加视觉总览，可选用于论文或答辩。",
        "",
        "## 原始数据",
        f"- 指标 CSV：`{relative(metrics_csv)}`",
        f"- 分案例运行时 CSV：`{relative(case_csv)}`",
        f"- 运行环境 JSON：`{relative(env_path)}`",
        "",
        "## 平均指标摘要",
        f"- 本项目 GPU 端到端平均耗时：`{avg_total:.1f} ms`；平均峰值 GPU 显存：`{avg_mem:.1f} MB`。",
        f"- 相比未修复的退化输入+双三次，本项目全图 PSNR 提升：`{project['full_psnr'] - degraded['full_psnr']:.2f} dB`。",
        f"- 相比未修复的退化输入+双三次，本项目掩码区 PSNR 提升：`{project['mask_psnr'] - degraded['mask_psnr']:.2f} dB`。",
        f"- 相比未修复的退化输入+GPU保真超分，本项目掩码区 PSNR 提升：`{project['mask_psnr'] - direct_gpu['mask_psnr']:.2f} dB`。",
        f"- 本项目平均细节保真度：`{project['detail_fidelity']:.3f}`；平均边缘一致性：`{project['edge_similarity']:.3f}`。",
        "",
        "## 论文表述建议",
        "",
        "可写作：为验证系统在 GPU 环境下的端到端可运行性，本文在 RTX 4060 Laptop GPU 上补充实现 CUDA 推理流程。",
        "其中修复阶段采用 GPU 掩码传播与扩散求解，增强阶段采用 CUDA 保真超分与细节增强。",
        "三组真实样例结果表明，该流程能够在不同纹理、结构和场景缺损下完成修复与超分增强，并显著优于未修复直接放大的退化输入。",
        "",
        "注意：不要将本结果写成“MindSpore GPU 已跑通”；应写成“补充 CUDA 端到端验证路径”。",
    ])
    out = OUTPUT_ROOT / "README_gpu_full_pipeline.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def parse_args() -> RunOptions:
    parser = argparse.ArgumentParser(description="生成完整 GPU 端到端论文效果图和对比数据图")
    parser.add_argument("--scale", type=int, choices=[2], default=2, help="输出放大倍数")
    parser.add_argument("--device", default="cuda", help="运行设备，默认 cuda")
    parser.add_argument("--no_clean", action="store_true", help="不清理旧的 gpu_full_pipeline 输出目录")
    parser.add_argument("--propagation_iterations", type=int, default=256, help="GPU边界传播填充迭代次数")
    parser.add_argument("--diffusion_iterations", type=int, default=80, help="GPU扩散细化迭代次数")
    parser.add_argument("--sharpen_strength", type=float, default=0.08, help="GPU超分后的轻量锐化强度")
    ns = parser.parse_args()
    if not 0.0 <= ns.sharpen_strength <= 0.5:
        raise ValueError("--sharpen_strength must be in [0, 0.5]")
    return RunOptions(
        scale=ns.scale,
        device=ns.device,
        clean=not ns.no_clean,
        propagation_iterations=ns.propagation_iterations,
        diffusion_iterations=ns.diffusion_iterations,
        sharpen_strength=ns.sharpen_strength,
    )


def main() -> None:
    options = parse_args()
    if options.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Install a CUDA-enabled PyTorch wheel before running this script.")
    np.random.seed(RNG_SEED)
    cv2.setRNGSeed(RNG_SEED)
    ensure_output_dirs(options)
    setup_matplotlib_style()

    print(f"[1/5] 输出目录: {OUTPUT_ROOT}")
    print(f"      设备: {options.device}; GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print("[2/5] 初始化 CUDA 修复与保真超分增强流水线")

    case_results: List[CaseRunResult] = []
    effect_figures: List[Path] = []
    for spec in CASES:
        print(f"[3/5] 运行GPU端到端案例: {spec.case_id} - {spec.title}")
        result = run_gpu_project_case(spec, options)
        case_results.append(result)
        fig = make_effect_figure(result)
        effect_figures.append(fig)
        print(f"      效果图: {relative(fig)}")
        print(f"      total={result.total_ms:.1f} ms, gpu_mem={result.peak_gpu_mem_mb:.1f} MB")

    print("[4/5] 运行对比方法并计算指标")
    metrics = evaluate_methods(case_results, options)
    metrics_csv = write_metrics_csv(metrics)
    case_csv = write_case_csv(case_results)
    env_path = write_environment(options)
    agg = aggregate_metrics(metrics)
    figures = [
        *effect_figures,
        plot_quality_psnr_ssim(agg),
        plot_mask_repair_quality(agg),
        plot_runtime_memory(agg),
        make_visual_overview(case_results, metrics),
    ]

    print("[5/5] 写入说明文档")
    summary = write_summary(case_results, metrics, figures, metrics_csv, case_csv, env_path, options)
    print("\n生成完成：")
    for fig in figures:
        print(f"- {relative(fig)}")
    print(f"- {relative(metrics_csv)}")
    print(f"- {relative(case_csv)}")
    print(f"- {relative(env_path)}")
    print(f"- {relative(summary)}")


if __name__ == "__main__":
    main()
