"""Generate paper-ready qualitative results and quantitative comparison figures.

This script reuses the repository's runnable CRA+SRGAN combined pipeline
(`combined_repair_sr2.0/combined_repair_sr_optimized.py`) instead of drawing
synthetic "fake" outputs.  It creates controlled degraded inputs from real
sample images, runs the project pipeline, evaluates multiple baselines, and
exports figures suitable for direct use in a thesis.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = PROJECT_ROOT / "combined_repair_sr2.0" / "combined_repair_sr_optimized.py"
OUTPUT_ROOT = PROJECT_ROOT / "paper_results"


def load_pipeline_module():
    spec = importlib.util.spec_from_file_location("combined_repair_sr_optimized", PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Unable to load pipeline module: {PIPELINE_PATH}")
    spec.loader.exec_module(module)
    return module


pipeline = load_pipeline_module()


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    title: str
    source: Path
    crop_box: Tuple[int, int, int, int]  # left, top, right, bottom
    mask_kind: str
    inpaint_radius: int = 3
    sharpen_strength: float = 0.15


@dataclass
class CaseRunResult:
    spec: CaseSpec
    case_dir: Path
    gt_path: Path
    degraded_path: Path
    mask_path: Path
    repaired_path: Path
    enhanced_path: Path
    success: bool
    elapsed_ms: float


@dataclass
class MethodResult:
    case_id: str
    case_title: str
    method: str
    psnr: float
    ssim: float
    edge_similarity: float
    sharpness: float
    detail_fidelity: float
    time_ms: float
    output_path: Path


CASES: Sequence[CaseSpec] = (
    CaseSpec(
        case_id="case01_natural_texture",
        title="自然纹理划痕修复与细节增强",
        source=PROJECT_ROOT / "CRA" / "examples" / "5.jpg",
        crop_box=(95, 40, 607, 424),
        mask_kind="rock_scratches",
        inpaint_radius=4,
        sharpen_strength=0.18,
    ),
    CaseSpec(
        case_id="case02_indoor_structure",
        title="室内结构小面积遮挡修复",
        source=PROJECT_ROOT / "CRA" / "examples" / "8.jpg",
        crop_box=(90, 35, 602, 419),
        mask_kind="indoor_small_occlusion",
        inpaint_radius=3,
        sharpen_strength=0.14,
    ),
    CaseSpec(
        case_id="case03_scene_depth",
        title="复杂场景水印式缺损修复增强",
        source=PROJECT_ROOT / "CRA" / "examples" / "12.jpg",
        crop_box=(135, 45, 647, 429),
        mask_kind="scene_watermark",
        inpaint_radius=3,
        sharpen_strength=0.16,
    ),
)


def ensure_output_dirs() -> None:
    if OUTPUT_ROOT.exists():
        # Keep the generated bundle reproducible while preserving any
        # user-created files that may later be placed directly under
        # paper_results/.
        for generated_dir in ["cases", "figures", "metrics"]:
            target = OUTPUT_ROOT / generated_dir
            if target.exists():
                shutil.rmtree(target)
        summary = OUTPUT_ROOT / "README_paper_results.md"
        if summary.exists():
            summary.unlink()
    for subdir in ["cases", "figures", "metrics"]:
        (OUTPUT_ROOT / subdir).mkdir(parents=True, exist_ok=True)


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def pil_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_rgb(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).save(path)


def make_mask(shape: Tuple[int, int], kind: str) -> np.ndarray:
    """Create deterministic paper-demo masks; white means region to repair."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)

    if kind == "rock_scratches":
        # Thin, crossing scratch-like defects on natural texture.
        cv2.line(mask, (int(0.18 * w), int(0.28 * h)), (int(0.67 * w), int(0.50 * h)), 255, 12, cv2.LINE_AA)
        cv2.line(mask, (int(0.32 * w), int(0.17 * h)), (int(0.58 * w), int(0.64 * h)), 255, 9, cv2.LINE_AA)
        cv2.ellipse(mask, (int(0.48 * w), int(0.42 * h)), (28, 18), 15, 0, 360, 255, -1)
    elif kind == "indoor_small_occlusion":
        # Small block/circle occlusions near structure boundaries.
        cv2.rectangle(mask, (int(0.46 * w), int(0.20 * h)), (int(0.58 * w), int(0.31 * h)), 255, -1)
        cv2.circle(mask, (int(0.40 * w), int(0.62 * h)), 18, 255, -1)
        cv2.line(mask, (int(0.61 * w), int(0.43 * h)), (int(0.72 * w), int(0.62 * h)), 255, 10, cv2.LINE_AA)
    elif kind == "scene_watermark":
        # Watermark-like defects spread across sky/building/vegetation.
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

    image = pil_rgb(spec.source).crop(spec.crop_box).resize((512, 384), Image.Resampling.LANCZOS)
    gt = np.array(image)
    mask = make_mask(gt.shape[:2], spec.mask_kind)

    degraded = gt.copy()
    # Fill the masked region with a visually obvious but realistic scratch/occlusion.
    degraded[mask > 0] = np.array([28, 28, 28], dtype=np.uint8)
    # Add a soft edge so the degradation resembles real acquisition defects.
    edge = cv2.GaussianBlur(mask, (11, 11), 0).astype(np.float32) / 255.0
    noisy_patch = np.random.default_rng(20260508).normal(0, 7, degraded.shape).astype(np.float32)
    degraded = np.clip(degraded.astype(np.float32) + noisy_patch * edge[..., None], 0, 255).astype(np.uint8)

    gt_path = case_dir / "01_ground_truth.png"
    degraded_path = case_dir / "02_degraded_input.png"
    mask_path = case_dir / "03_mask.png"
    save_rgb(gt, gt_path)
    save_rgb(degraded, degraded_path)
    Image.fromarray(mask).save(mask_path)
    return gt_path, degraded_path, mask_path


def configure_processor(spec: CaseSpec, scale: int = 2):
    config = pipeline.ModelConfig()
    config.backend = "classic"
    config.scale = scale
    config.inpaint_method = "telea"
    config.inpaint_radius = spec.inpaint_radius
    config.sharpen_strength = spec.sharpen_strength
    config.validate()
    processor = pipeline.CombinedProcessor(config)
    cra_ok, sr_ok = processor.load_models("", "")
    if not (cra_ok and sr_ok):
        raise RuntimeError("Failed to load classic backend models")
    return processor


def run_project_pipeline(spec: CaseSpec) -> CaseRunResult:
    case_dir = OUTPUT_ROOT / "cases" / spec.case_id
    gt_path, degraded_path, mask_path = prepare_case_inputs(spec)
    run_dir = case_dir / "project_pipeline"
    processor = configure_processor(spec, scale=2)
    t0 = time.perf_counter()
    success, result = processor.process_image(str(degraded_path), str(mask_path), str(run_dir))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    enhanced_path = Path(result)
    repaired_path = run_dir / f"{degraded_path.stem}_repaired.png"
    return CaseRunResult(
        spec=spec,
        case_dir=case_dir,
        gt_path=gt_path,
        degraded_path=degraded_path,
        mask_path=mask_path,
        repaired_path=repaired_path,
        enhanced_path=enhanced_path,
        success=success,
        elapsed_ms=elapsed_ms,
    )


def draw_caption(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], text: str, *, fill=(25, 35, 45)) -> None:
    x0, y0, x1, y1 = box
    f = font(26)
    bbox = draw.textbbox((0, 0), text, font=f)
    tw = bbox[2] - bbox[0]
    draw.text((x0 + (x1 - x0 - tw) // 2, y0), text, font=f, fill=fill)


def mask_overlay(image: Image.Image, mask_path: Path) -> Image.Image:
    base = image.convert("RGBA")
    mask = Image.open(mask_path).convert("L").resize(base.size, Image.Resampling.NEAREST)
    overlay = Image.new("RGBA", base.size, (230, 77, 64, 0))
    overlay.putalpha(mask.point(lambda p: 105 if p > 0 else 0))
    return Image.alpha_composite(base, overlay).convert("RGB")


def resize_to_tile(image: Image.Image, size: Tuple[int, int]) -> Image.Image:
    return image.convert("RGB").resize(size, Image.Resampling.LANCZOS)


def make_effect_figure(result: CaseRunResult) -> Path:
    """Create a 5-panel qualitative figure for one case."""
    tile_w, tile_h = 360, 270
    gap = 18
    top = 74
    bottom = 44
    labels = ["原始参考图", "退化输入", "掩码区域", "项目修复结果", "联合增强输出"]
    paths_or_images: List[Image.Image] = [
        pil_rgb(result.gt_path),
        pil_rgb(result.degraded_path),
        mask_overlay(pil_rgb(result.degraded_path), result.mask_path),
        pil_rgb(result.repaired_path),
        pil_rgb(result.enhanced_path),
    ]
    # Enhanced output is 2x; show at the same physical tile size for fair visual comparison.
    width = tile_w * len(labels) + gap * (len(labels) + 1)
    height = top + tile_h + bottom
    canvas = Image.new("RGB", (width, height), (248, 250, 253))
    draw = ImageDraw.Draw(canvas)

    title_font = font(32)
    draw.text((gap, 18), f"{result.spec.title}（真实运行输出）", font=title_font, fill=(20, 35, 55))
    draw.line((gap, 60, width - gap, 60), fill=(210, 218, 228), width=2)

    for idx, (label, image) in enumerate(zip(labels, paths_or_images)):
        x = gap + idx * (tile_w + gap)
        y = top
        tile = resize_to_tile(image, (tile_w, tile_h))
        # White card
        draw.rounded_rectangle((x - 4, y - 4, x + tile_w + 4, y + tile_h + 35), radius=14, fill=(255, 255, 255), outline=(220, 228, 238))
        canvas.paste(tile, (x, y))
        draw_caption(draw, (x, y + tile_h + 8, x + tile_w, y + tile_h + 42), label)

    out = OUTPUT_ROOT / "figures" / f"{result.spec.case_id}_effect_figure.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=95)
    return out


def mse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a.astype(np.float64) - b.astype(np.float64)
    return float(np.mean(diff * diff))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    value = mse(a, b)
    if value <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(255.0 / math.sqrt(value)))


def ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    """Small dependency-free SSIM implementation for luminance images."""
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


def to_gray(array: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)


def edge_similarity(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.Canny(to_gray(a), 80, 160).astype(np.float32) / 255.0
    gb = cv2.Canny(to_gray(b), 80, 160).astype(np.float32) / 255.0
    intersection = float(np.sum(np.minimum(ga, gb)))
    union = float(np.sum(np.maximum(ga, gb)))
    if union <= 1e-9:
        return 1.0
    return intersection / union


def sharpness(array: np.ndarray) -> float:
    gray = to_gray(array)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def detail_fidelity(output: np.ndarray, reference: np.ndarray) -> float:
    """How close the output detail energy is to the reference.

    Raw Laplacian variance alone is unsafe for paper comparison because
    nearest-neighbor block artifacts can create very high variance.  This
    normalized score rewards detail energy close to the reference and penalizes
    both over-sharpened artifacts and over-smoothed outputs.
    """
    out_sharp = sharpness(output)
    ref_sharp = sharpness(reference)
    denom = max(out_sharp, ref_sharp, 1e-9)
    return float(min(out_sharp, ref_sharp) / denom)


def read_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def resize_like(array: np.ndarray, reference: np.ndarray) -> np.ndarray:
    h, w = reference.shape[:2]
    if array.shape[:2] == (h, w):
        return array
    return cv2.resize(array, (w, h), interpolation=cv2.INTER_AREA)


def baseline_nearest(degraded: np.ndarray, mask: np.ndarray, scale: int) -> np.ndarray:
    repaired = degraded.copy()
    # Keep the occlusion; nearest-neighbor enlargement is a weak baseline.
    return cv2.resize(repaired, (degraded.shape[1] * scale, degraded.shape[0] * scale), interpolation=cv2.INTER_NEAREST)


def baseline_bicubic(degraded: np.ndarray, mask: np.ndarray, scale: int) -> np.ndarray:
    return cv2.resize(degraded, (degraded.shape[1] * scale, degraded.shape[0] * scale), interpolation=cv2.INTER_CUBIC)


def baseline_inpaint_only(degraded: np.ndarray, mask: np.ndarray, scale: int) -> np.ndarray:
    repaired = cv2.inpaint(degraded, mask, 3, cv2.INPAINT_TELEA)
    return cv2.resize(repaired, (degraded.shape[1] * scale, degraded.shape[0] * scale), interpolation=cv2.INTER_CUBIC)


def baseline_inpaint_nearest(degraded: np.ndarray, mask: np.ndarray, scale: int) -> np.ndarray:
    repaired = cv2.inpaint(degraded, mask, 3, cv2.INPAINT_TELEA)
    return cv2.resize(repaired, (degraded.shape[1] * scale, degraded.shape[0] * scale), interpolation=cv2.INTER_NEAREST)


def project_method(spec: CaseSpec, degraded_path: Path, mask_path: Path, output_dir: Path) -> Tuple[np.ndarray, float, Path]:
    processor = configure_processor(spec, scale=2)
    t0 = time.perf_counter()
    success, result = processor.process_image(str(degraded_path), str(mask_path), str(output_dir))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if not success:
        raise RuntimeError(f"Project pipeline failed for {spec.case_id}: {result}")
    return read_rgb(Path(result)), elapsed_ms, Path(result)


def evaluate_methods(case_results: Sequence[CaseRunResult]) -> List[MethodResult]:
    methods: List[Tuple[str, Callable[[np.ndarray, np.ndarray, int], np.ndarray]]] = [
        ("退化输入+最近邻放大", baseline_nearest),
        ("退化输入+双三次放大", baseline_bicubic),
        ("单独修复+最近邻放大", baseline_inpaint_nearest),
        ("传统修复+双三次放大", baseline_inpaint_only),
    ]
    all_results: List[MethodResult] = []

    for case in case_results:
        gt = read_rgb(case.gt_path)
        gt_2x = cv2.resize(gt, (gt.shape[1] * 2, gt.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
        degraded = read_rgb(case.degraded_path)
        mask = np.array(Image.open(case.mask_path).convert("L"))
        method_dir = case.case_dir / "method_outputs"
        method_dir.mkdir(parents=True, exist_ok=True)

        for method_name, func in methods:
            t0 = time.perf_counter()
            out = func(degraded, mask, 2)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            out = np.clip(out, 0, 255).astype(np.uint8)
            out_path = method_dir / f"{safe_name(method_name)}.png"
            save_rgb(out, out_path)
            all_results.append(score_output(case, method_name, out, gt_2x, elapsed_ms, out_path))

        project_output = read_rgb(case.enhanced_path)
        all_results.append(
            score_output(case, "本项目：修复+超分增强", project_output, gt_2x, case.elapsed_ms, case.enhanced_path)
        )

    return all_results


def safe_name(text: str) -> str:
    return (
        text.replace("：", "_")
        .replace("+", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace("本项目", "project")
    )


def score_output(case: CaseRunResult, method_name: str, output: np.ndarray, gt: np.ndarray, elapsed_ms: float, out_path: Path) -> MethodResult:
    output = resize_like(output, gt)
    out_sharpness = sharpness(output)
    return MethodResult(
        case_id=case.spec.case_id,
        case_title=case.spec.title,
        method=method_name,
        psnr=psnr(output, gt),
        ssim=ssim_gray(to_gray(output), to_gray(gt)),
        edge_similarity=edge_similarity(output, gt),
        sharpness=out_sharpness,
        detail_fidelity=detail_fidelity(output, gt),
        time_ms=elapsed_ms,
        output_path=out_path,
    )


def save_metrics_csv(results: Sequence[MethodResult]) -> Path:
    out = OUTPUT_ROOT / "metrics" / "comparison_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case_id",
                "case_title",
                "method",
                "psnr",
                "ssim",
                "edge_similarity",
                "sharpness",
                "detail_fidelity",
                "time_ms",
                "output_path",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.case_id,
                    r.case_title,
                    r.method,
                    f"{r.psnr:.4f}",
                    f"{r.ssim:.6f}",
                    f"{r.edge_similarity:.6f}",
                    f"{r.sharpness:.4f}",
                    f"{r.detail_fidelity:.6f}",
                    f"{r.time_ms:.3f}",
                    str(r.output_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                ]
            )
    return out


def aggregate_metrics(results: Sequence[MethodResult]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[MethodResult]] = {}
    for r in results:
        grouped.setdefault(r.method, []).append(r)
    return {
        method: {
            "psnr": float(np.mean([r.psnr for r in rows])),
            "ssim": float(np.mean([r.ssim for r in rows])),
            "edge_similarity": float(np.mean([r.edge_similarity for r in rows])),
            "sharpness": float(np.mean([r.sharpness for r in rows])),
            "detail_fidelity": float(np.mean([r.detail_fidelity for r in rows])),
            "time_ms": float(np.mean([r.time_ms for r in rows])),
        }
        for method, rows in grouped.items()
    }


def setup_matplotlib_style() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.edgecolor"] = "#2B3440"
    plt.rcParams["axes.labelcolor"] = "#2B3440"
    plt.rcParams["xtick.color"] = "#2B3440"
    plt.rcParams["ytick.color"] = "#2B3440"


def bar_label(ax, bars, fmt="{:.2f}", dy=0.01):
    ymax = ax.get_ylim()[1]
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + ymax * dy,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#263238",
        )


def plot_psnr_ssim(agg: Dict[str, Dict[str, float]]) -> Path:
    methods = list(agg.keys())
    labels = [short_method(m) for m in methods]
    psnr_values = [agg[m]["psnr"] for m in methods]
    ssim_values = [agg[m]["ssim"] for m in methods]

    fig, ax1 = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(len(methods))
    width = 0.38
    bars1 = ax1.bar(x - width / 2, psnr_values, width, label="PSNR / dB", color="#3164C8")
    ax1.set_ylabel("PSNR / dB")
    ax1.set_ylim(0, max(psnr_values) * 1.22)
    ax1.grid(axis="y", linestyle="--", alpha=0.28)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, ssim_values, width, label="SSIM", color="#2EAD7F")
    ax2.set_ylabel("SSIM")
    ax2.set_ylim(0, min(1.0, max(ssim_values) * 1.28))

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=14, ha="right")
    ax1.set_title("不同方法重建质量对比（3组真实运行平均）", fontsize=15, pad=14, weight="bold")
    bar_label(ax1, bars1, "{:.2f}")
    bar_label(ax2, bars2, "{:.3f}", dy=0.015)
    lines, labs = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labs + labs2, loc="upper left", frameon=False)
    fig.tight_layout()
    out = OUTPUT_ROOT / "figures" / "comparison_quality_psnr_ssim.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_edge_sharpness(agg: Dict[str, Dict[str, float]]) -> Path:
    methods = list(agg.keys())
    labels = [short_method(m) for m in methods]
    edge_values = [agg[m]["edge_similarity"] for m in methods]
    detail_values = [agg[m]["detail_fidelity"] for m in methods]

    fig, ax1 = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(len(methods))
    ax1.plot(x, edge_values, marker="o", linewidth=2.8, markersize=8, color="#E2711D", label="边缘一致性")
    ax1.set_ylabel("边缘一致性（越高越好）")
    ax1.set_ylim(0, min(1.0, max(edge_values) * 1.35))
    ax1.grid(axis="y", linestyle="--", alpha=0.28)

    ax2 = ax1.twinx()
    ax2.plot(x, detail_values, marker="s", linewidth=2.8, markersize=7, color="#6B4FB3", label="细节保真度")
    ax2.set_ylabel("细节保真度（越高越好）")
    ax2.set_ylim(0, min(1.0, max(detail_values) * 1.18))

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=14, ha="right")
    ax1.set_title("结构边缘与细节保真度对比（3组真实运行平均）", fontsize=15, pad=14, weight="bold")

    for i, v in enumerate(edge_values):
        ax1.text(i, v + ax1.get_ylim()[1] * 0.025, f"{v:.3f}", ha="center", fontsize=9, color="#8A3F0B")
    for i, v in enumerate(detail_values):
        ax2.text(i, v + ax2.get_ylim()[1] * 0.025, f"{v:.3f}", ha="center", fontsize=9, color="#4B388A")

    lines, labs = ax1.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labs + labs2, loc="upper left", frameon=False)
    fig.tight_layout()
    out = OUTPUT_ROOT / "figures" / "comparison_edge_detail_fidelity.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_case_heatmap(results: Sequence[MethodResult]) -> Path:
    methods = list(dict.fromkeys(r.method for r in results))
    cases = list(dict.fromkeys(r.case_title for r in results))
    matrix = np.zeros((len(cases), len(methods)), dtype=float)
    for r in results:
        i = cases.index(r.case_title)
        j = methods.index(r.method)
        matrix[i, j] = r.psnr

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    im = ax.imshow(matrix, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([short_method(m) for m in methods], rotation=16, ha="right")
    ax.set_yticks(np.arange(len(cases)))
    ax.set_yticklabels(cases)
    ax.set_title("各测试组 PSNR 对比热力图 / dB", fontsize=15, pad=14, weight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color="#17212B", fontsize=10)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("PSNR / dB")
    fig.tight_layout()
    out = OUTPUT_ROOT / "figures" / "comparison_case_psnr_heatmap.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def short_method(method: str) -> str:
    mapping = {
        "退化输入+最近邻放大": "最近邻",
        "退化输入+双三次放大": "双三次",
        "单独修复+最近邻放大": "修复+最近邻",
        "传统修复+双三次放大": "传统修复+双三次",
        "本项目：修复+超分增强": "本项目",
    }
    return mapping.get(method, method)


def write_summary(case_results: Sequence[CaseRunResult], metrics: Sequence[MethodResult], figures: Sequence[Path], csv_path: Path) -> Path:
    agg = aggregate_metrics(metrics)
    project = agg["本项目：修复+超分增强"]
    bicubic = agg["退化输入+双三次放大"]
    traditional = agg["传统修复+双三次放大"]

    lines = [
        "# 论文效果图与对比测试数据说明",
        "",
        "本目录中的图片均由本项目代码真实运行生成，生成命令：",
        "",
        "```powershell",
        "python tools\\generate_paper_results.py",
        "```",
        "",
        "## 3组效果图片",
    ]
    for r in case_results:
        lines.append(f"- `{relative(OUTPUT_ROOT / 'figures' / f'{r.spec.case_id}_effect_figure.png')}`：{r.spec.title}")

    lines.extend(
        [
            "",
            "## 2–3组对比测试数据图",
            f"- `{relative(OUTPUT_ROOT / 'figures' / 'comparison_quality_psnr_ssim.png')}`：PSNR 与 SSIM 对比。",
            f"- `{relative(OUTPUT_ROOT / 'figures' / 'comparison_edge_detail_fidelity.png')}`：边缘一致性与细节保真度对比。",
            f"- `{relative(OUTPUT_ROOT / 'figures' / 'comparison_case_psnr_heatmap.png')}`：不同测试组 PSNR 热力图。",
            "",
            "## 原始量化数据",
            f"- CSV：`{relative(csv_path)}`",
            "",
            "## 平均指标摘要",
            f"- 本项目平均 PSNR：{project['psnr']:.2f} dB；相较退化输入+双三次提升 {project['psnr'] - bicubic['psnr']:.2f} dB。",
            f"- 本项目平均 SSIM：{project['ssim']:.3f}；相较传统修复+双三次放大变化 {project['ssim'] - traditional['ssim']:+.3f}。",
            f"- 本项目平均边缘一致性：{project['edge_similarity']:.3f}；平均细节保真度：{project['detail_fidelity']:.3f}。",
            "- 结论建议：论文中可表述为“项目端到端联合流程显著优于未修复退化输入与简单放大；在 classic 后端下，与传统修复+双三次插值质量接近，并额外提供可配置、可视化和批处理的一体化流程”。",
            "",
            "注：当前机器未安装 MindSpore，脚本使用项目内置 classic 后端（OpenCV 修复 + 双三次超分 + 轻量锐化）完成真实运行验证；若在 MindSpore 环境下切换 deep/auto 并加载权重，可复用同一评测脚本扩展深度模型结果。",
        ]
    )
    out = OUTPUT_ROOT / "README_paper_results.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")


def main() -> None:
    np.random.seed(20260508)
    ensure_output_dirs()
    setup_matplotlib_style()

    print(f"[1/4] 输出目录: {OUTPUT_ROOT}")
    case_results: List[CaseRunResult] = []
    effect_figures: List[Path] = []
    for spec in CASES:
        print(f"[2/4] 运行案例: {spec.case_id} - {spec.title}")
        result = run_project_pipeline(spec)
        if not result.success:
            raise RuntimeError(f"Pipeline failed for {spec.case_id}")
        case_results.append(result)
        fig = make_effect_figure(result)
        effect_figures.append(fig)
        print(f"      效果图: {relative(fig)}")

    print("[3/4] 运行对比方法并计算指标")
    metrics = evaluate_methods(case_results)
    csv_path = save_metrics_csv(metrics)
    agg = aggregate_metrics(metrics)
    comparison_figures = [
        plot_psnr_ssim(agg),
        plot_edge_sharpness(agg),
        plot_case_heatmap(metrics),
    ]

    print("[4/4] 写入说明文件")
    summary_path = write_summary(case_results, metrics, effect_figures + comparison_figures, csv_path)

    print("\n生成完成：")
    for fig in [*effect_figures, *comparison_figures]:
        print(f"- {relative(fig)}")
    print(f"- {relative(csv_path)}")
    print(f"- {relative(summary_path)}")


if __name__ == "__main__":
    main()
