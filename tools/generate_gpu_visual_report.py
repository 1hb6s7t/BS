#!/usr/bin/env python3
"""Generate a browser-based visual report for the Docker GPU demo.

The report is intentionally dependency-free so it can run with the system
Python on Windows after the Docker GPU inference finishes.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


KEY_PATTERNS = (
    "MindSpore上下文设置完成",
    "CRA模型加载成功",
    "SRGAN model loaded successfully",
    "当前后端",
    "CRA检查点/后端",
    "SRGAN检查点/后端",
    "Applied mask-aware SRGAN artifact suppression",
    "处理完成",
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def rel(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def image_card(title: str, path: Path, base: Path, note: str = "") -> str:
    if not path.exists():
        return f"""
        <section class="card missing">
          <h3>{html.escape(title)}</h3>
          <p>文件不存在：<code>{html.escape(str(path))}</code></p>
        </section>
        """
    return f"""
    <section class="card">
      <h3>{html.escape(title)}</h3>
      <a href="{html.escape(rel(path, base))}" target="_blank">
        <img src="{html.escape(rel(path, base))}" alt="{html.escape(title)}" loading="lazy" />
      </a>
      <p>{html.escape(note)}</p>
      <small>{html.escape(path.name)} · {path.stat().st_size / 1024 / 1024:.2f} MB</small>
    </section>
    """


def collect_key_logs(log_text: str) -> list[str]:
    lines: list[str] = []
    for line in log_text.splitlines():
        if any(pattern in line for pattern in KEY_PATTERNS):
            lines.append(line)
    return lines[-30:]


def generate(output_dir: Path, input_image: str, mask_image: str) -> Path:
    output_dir = output_dir.resolve()
    stem = Path(input_image).stem
    env = read_json(output_dir / "environment.json")
    report = read_json(output_dir / "docker_gpu_demo_report.json")
    log_text = read_text(output_dir / "run.log")
    key_logs = collect_key_logs(log_text)

    input_path = (Path.cwd() / input_image).resolve()
    mask_path = (Path.cwd() / mask_image).resolve()
    repaired_path = output_dir / f"{stem}_repaired.png"
    enhanced_path = output_dir / f"{stem}_enhanced.png"
    evidence_path = output_dir / "docker_gpu_evidence.png"

    optional_cards = []
    for filename, title in (
        ("repair_region_pipeline_preview.png", "修复区域流程预览"),
        ("final_v2_preview.png", "修复区域稳定化对比"),
        ("final_v2_left_bottom_preview.png", "左下角区域稳定化对比"),
        ("stable_noise_comparison.png", "噪点/色彩稳定对比"),
    ):
        path = output_dir / filename
        if path.exists():
            optional_cards.append(image_card(title, path, output_dir))

    nvidia = env.get("nvidia_smi", {}).get("output", "").strip()
    if not nvidia:
        nvidia = "未在 environment.json 中检测到 nvidia-smi 输出"

    cmd = ""
    if isinstance(report.get("command"), list):
        cmd = " ".join(str(x) for x in report["command"])

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CRA + SRGAN GPU 可视化演示报告</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --panel: rgba(255,255,255,0.92);
      --text: #172033;
      --muted: #607089;
      --blue: #1f6feb;
      --green: #1b7f4c;
      --border: #d9e1ee;
      --shadow: 0 18px 45px rgba(20, 35, 70, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, #dfeaff 0, transparent 32rem),
        linear-gradient(135deg, #f8fbff 0%, var(--bg) 48%, #eef4ff 100%);
    }}
    header {{
      padding: 42px min(5vw, 72px) 24px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(28px, 4vw, 48px);
      letter-spacing: -0.04em;
    }}
    .subtitle {{ color: var(--muted); font-size: 18px; line-height: 1.6; }}
    main {{ padding: 0 min(5vw, 72px) 56px; }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 12px; margin-top: 22px; }}
    .badge {{
      padding: 10px 14px;
      border-radius: 999px;
      background: var(--panel);
      border: 1px solid var(--border);
      box-shadow: 0 8px 20px rgba(20, 35, 70, 0.08);
      font-weight: 600;
    }}
    .badge.ok {{ color: var(--green); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 22px;
      margin-top: 24px;
    }}
    .card, .info {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 24px;
      padding: 18px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .card h3, .info h2 {{ margin: 0 0 12px; }}
    .card img {{
      width: 100%;
      max-height: 520px;
      object-fit: contain;
      border-radius: 16px;
      background: #111827;
      border: 1px solid #e6edf7;
    }}
    .card p {{ color: var(--muted); min-height: 1.3em; }}
    code, pre {{
      font-family: Consolas, "Cascadia Code", monospace;
      background: #0f172a;
      color: #dbeafe;
      border-radius: 14px;
    }}
    code {{ padding: 2px 6px; }}
    pre {{
      padding: 16px;
      white-space: pre-wrap;
      word-break: break-word;
      overflow: auto;
      line-height: 1.5;
    }}
    .info {{ margin-top: 24px; }}
    .split {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 22px;
    }}
    @media (max-width: 980px) {{
      .grid, .split {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>CRA + SRGAN 真实 GPU 可视化演示</h1>
    <div class="subtitle">
      本页面由现场 Docker GPU 推理自动生成，用于答辩展示：输入图像 → 掩码 → CRA 修复 → SRGAN x4 增强 → GPU 环境与日志证据。
    </div>
    <div class="badges">
      <span class="badge ok">MindSpore: {html.escape(str(env.get("mindspore", "unknown")))}</span>
      <span class="badge ok">GPU: {html.escape(nvidia)}</span>
      <span class="badge">Exit code: {html.escape(str(report.get("exit_code", "unknown")))}</span>
      <span class="badge">Elapsed: {html.escape(str(report.get("elapsed_sec", "unknown")))}s</span>
    </div>
  </header>
  <main>
    <section class="grid">
      {image_card("1. 原始输入图", input_path, output_dir, "待修复的原始图像。")}
      {image_card("2. 掩码图", mask_path, output_dir, "白色区域表示需要修复，黑色区域保留。")}
      {image_card("3. CRA 修复结果", repaired_path, output_dir, "真实加载 CRA ckpt 后的修复阶段输出。")}
      {image_card("4. SRGAN x4 增强结果", enhanced_path, output_dir, "真实加载 SRGAN ckpt 后的最终增强输出，已加入色彩与修复区稳定化。")}
      {image_card("5. GPU 运行证据图", evidence_path, output_dir, "自动拼接的论文/答辩证据图。")}
      {''.join(optional_cards)}
    </section>

    <section class="split">
      <div class="info">
        <h2>关键运行日志</h2>
        <pre>{html.escape("\\n".join(key_logs) if key_logs else "未找到关键日志，请查看 run.log。")}</pre>
      </div>
      <div class="info">
        <h2>环境信息</h2>
        <pre>{html.escape(json.dumps(env, ensure_ascii=False, indent=2))}</pre>
      </div>
    </section>

    <section class="info">
      <h2>实际执行命令</h2>
      <pre>{html.escape(cmd or "命令信息见 docker_gpu_demo_report.md")}</pre>
    </section>
  </main>
</body>
</html>
"""
    report_path = output_dir / "visual_gpu_demo.html"
    report_path.write_text(html_doc, encoding="utf-8")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate visual HTML report for GPU demo outputs.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--input_image", default="CRA/test/images/1.png")
    parser.add_argument("--mask_image", default="CRA/test/masks/1.png")
    args = parser.parse_args()
    path = generate(Path(args.output_dir), args.input_image, args.mask_image)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
