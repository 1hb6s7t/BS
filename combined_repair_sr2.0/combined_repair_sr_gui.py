#!/usr/bin/env python3
"""
图像修复与超分辨率联合处理工具 - 图形用户界面
基于tkinter的用户友好界面
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import threading
import os
import sys
from pathlib import Path
from PIL import Image, ImageTk
import numpy as np
import logging
from datetime import datetime

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from combined_repair_sr_optimized import (
        CombinedProcessor,
        ModelConfig,
        ImageProcessor,
        create_demo_assets,
        discover_checkpoint_paths,
    )
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保 combined_repair_sr_optimized.py 文件存在")

    def discover_checkpoint_paths(*args, **kwargs):
        return {
            "cra_default": "",
            "srgan_default": "",
            "cra_candidates": [],
            "srgan_candidates": [],
        }

    def create_demo_assets(*args, **kwargs):
        raise RuntimeError("无法导入 create_demo_assets")


class TextLogHandler(logging.Handler):
    """把日志安全转发到 Tk 文本控件。"""

    def __init__(self, widget):
        super().__init__()
        self.widget = widget

    def emit(self, record):
        msg = self.format(record)
        self.widget.after(0, self._append, msg)

    def _append(self, msg):
        self.widget.configure(state="normal")
        self.widget.insert("end", msg + "\n")
        self.widget.see("end")
        self.widget.configure(state="disabled")

class ImageRepairGUI:
    """图像修复与超分辨率GUI应用"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.setup_fonts()
        self.setup_window()
        self.setup_variables()
        self.setup_widgets()
        self.setup_layout()
        self.processor = None
        self.models_loaded = False
        
        # 设置日志
        self.setup_logging()
    
    def setup_fonts(self):
        """设置字体配置"""
        # 配置字体大小，适合全屏显示
        self.fonts = {
            'title': ('Arial', 20, 'bold'),          # 标题字体
            'heading': ('Arial', 12, 'bold'),        # 区域标题字体
            'label': ('Arial', 11),                  # 标签字体
            'entry': ('Arial', 11),                  # 输入框字体
            'button': ('Arial', 11),                 # 按钮字体
            'log': ('Consolas', 10),                 # 日志字体
            'status': ('Arial', 11),                 # 状态字体
        }
        
        # 配置ttk样式
        style = ttk.Style()
        style.configure('Title.TLabel', font=self.fonts['title'])
        style.configure('Heading.TLabelframe.Label', font=self.fonts['heading'])
        style.configure('Large.TLabel', font=self.fonts['label'])
        style.configure('Large.TEntry', font=self.fonts['entry'])
        style.configure('Large.TButton', font=self.fonts['button'], padding=(10, 8))
        style.configure('Large.TCombobox', font=self.fonts['entry'])
        
        # 配置进度条样式
        style.configure('Large.Horizontal.TProgressbar', 
                       thickness=20)  # 增加进度条高度
        
        # 配置LabelFrame标题字体
        style.configure('Heading.TLabelframe', relief='solid', borderwidth=1)
        style.configure('Heading.TLabelframe.Label', font=self.fonts['heading'])
    
    def setup_window(self):
        """设置主窗口"""
        self.root.title("图像修复与超分辨率处理工具")
        
        # 获取屏幕尺寸并设置更大的窗口
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # 设置窗口为屏幕的82%大小，并限制在常见屏幕可容纳范围内
        window_width = min(max(int(screen_width * 0.82), 1100), screen_width - 80)
        window_height = min(max(int(screen_height * 0.82), 760), screen_height - 80)
        
        # 计算居中位置
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.resizable(True, True)
        
        # 设置最小窗口大小
        self.root.minsize(1040, 720)
        
        # 设置图标（如果有的话）
        try:
            # self.root.iconbitmap("icon.ico")
            pass
        except:
            pass
    
    def setup_variables(self):
        """设置变量"""
        ckpt_info = discover_checkpoint_paths()

        # 文件路径变量
        self.input_image_var = tk.StringVar()
        self.mask_image_var = tk.StringVar()
        self.output_dir_var = tk.StringVar(value="./output")
        self.cra_ckpt_var = tk.StringVar(
            value=ckpt_info.get("cra_default") or "./checkpoints/cra.ckpt"
        )
        self.srgan_ckpt_var = tk.StringVar(
            value=ckpt_info.get("srgan_default") or "./checkpoints/srgan_generator.ckpt"
        )
        self.latest_result_path = ""
        
        # 参数变量
        self.backend_var = tk.StringVar(value="auto")
        self.device_var = tk.StringVar(value="CPU")
        self.scale_var = tk.IntVar(value=2)
        self.input_size_var = tk.IntVar(value=512)
        
        # 状态变量
        self.status_var = tk.StringVar(value="就绪")
        self.progress_var = tk.DoubleVar()
        self.is_processing = False
    
    def setup_logging(self):
        """设置日志显示"""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        handler = TextLogHandler(self.log_text)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%H:%M:%S'))
        root_logger = logging.getLogger()
        if not any(isinstance(h, TextLogHandler) for h in root_logger.handlers):
            root_logger.addHandler(handler)
        self.append_log("GUI 已启动，auto 后端会在无 MindSpore 时自动使用 classic。")

    def append_log(self, message: str):
        """向日志区域追加一行文本。"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{timestamp} - {message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
    
    def setup_widgets(self):
        """设置所有控件"""
        # 创建主要框架 - 增加padding以适应更大字体
        self.main_frame = ttk.Frame(self.root, padding="15")
        
        # 标题
        title_label = ttk.Label(self.main_frame, text="图像修复与超分辨率处理工具", 
                               style="Title.TLabel")
        
        # 文件选择区域
        self.create_file_selection_frame()
        
        # 模型配置区域
        self.create_model_config_frame()
        
        # 参数配置区域
        self.create_parameter_frame()
        
        # 图像预览区域
        self.create_preview_frame()
        
        # 控制按钮区域
        self.create_control_frame()
        
        # 进度和状态区域
        self.create_status_frame()

        # 日志区域
        self.create_log_frame()
        
        # 保存控件引用
        self.title_label = title_label
    
    def create_file_selection_frame(self):
        """创建文件选择区域"""
        self.file_frame = ttk.LabelFrame(self.main_frame, text="文件选择", padding="8", 
                                        style="Heading.TLabelframe")
        
        # 输入图像选择
        ttk.Label(self.file_frame, text="输入图像:", style="Large.TLabel").grid(
            row=0, column=0, sticky="w", padx=8, pady=4)
        self.input_entry = ttk.Entry(self.file_frame, textvariable=self.input_image_var, 
                                    width=60, style="Large.TEntry")
        self.input_entry.grid(row=0, column=1, padx=8, pady=4)
        ttk.Button(self.file_frame, text="浏览", style="Large.TButton",
                  command=self.browse_input_image).grid(row=0, column=2, padx=8, pady=4)
        
        # 掩码图像选择
        ttk.Label(self.file_frame, text="掩码图像:", style="Large.TLabel").grid(
            row=1, column=0, sticky="w", padx=8, pady=4)
        self.mask_entry = ttk.Entry(self.file_frame, textvariable=self.mask_image_var, 
                                   width=60, style="Large.TEntry")
        self.mask_entry.grid(row=1, column=1, padx=8, pady=4)
        ttk.Button(self.file_frame, text="浏览", style="Large.TButton",
                  command=self.browse_mask_image).grid(row=1, column=2, padx=8, pady=4)
        
        # 输出目录选择
        ttk.Label(self.file_frame, text="输出目录:", style="Large.TLabel").grid(
            row=2, column=0, sticky="w", padx=8, pady=4)
        self.output_entry = ttk.Entry(self.file_frame, textvariable=self.output_dir_var, 
                                     width=60, style="Large.TEntry")
        self.output_entry.grid(row=2, column=1, padx=8, pady=4)
        ttk.Button(self.file_frame, text="浏览", style="Large.TButton",
                  command=self.browse_output_dir).grid(row=2, column=2, padx=8, pady=4)
        self.file_frame.columnconfigure(1, weight=1)
    
    def create_model_config_frame(self):
        """创建模型配置区域"""
        self.model_frame = ttk.LabelFrame(self.main_frame, text="模型配置", padding="8",
                                         style="Heading.TLabelframe")
        
        # CRA模型路径
        ttk.Label(self.model_frame, text="CRA模型:", style="Large.TLabel").grid(
            row=0, column=0, sticky="w", padx=8, pady=4)
        self.cra_entry = ttk.Entry(self.model_frame, textvariable=self.cra_ckpt_var, 
                                  width=60, style="Large.TEntry")
        self.cra_entry.grid(row=0, column=1, padx=8, pady=4)
        self.cra_browse_btn = ttk.Button(self.model_frame, text="浏览", style="Large.TButton",
                                         command=self.browse_cra_model)
        self.cra_browse_btn.grid(row=0, column=2, padx=8, pady=4)
        
        # SRGAN模型路径
        ttk.Label(self.model_frame, text="SRGAN模型:", style="Large.TLabel").grid(
            row=1, column=0, sticky="w", padx=8, pady=4)
        self.srgan_entry = ttk.Entry(self.model_frame, textvariable=self.srgan_ckpt_var, 
                                    width=60, style="Large.TEntry")
        self.srgan_entry.grid(row=1, column=1, padx=8, pady=4)
        self.srgan_browse_btn = ttk.Button(self.model_frame, text="浏览", style="Large.TButton",
                                           command=self.browse_srgan_model)
        self.srgan_browse_btn.grid(row=1, column=2, padx=8, pady=4)
        
        # 自动检测模型按钮
        self.detect_models_btn = ttk.Button(
            self.model_frame,
            text="自动检测模型",
            style="Large.TButton",
            command=self.auto_detect_model_paths
        )
        self.detect_models_btn.grid(row=2, column=0, pady=12)

        # 加载模型按钮
        self.load_models_btn = ttk.Button(self.model_frame, text="加载模型", style="Large.TButton",
                                         command=self.load_models)
        self.load_models_btn.grid(row=2, column=1, pady=12)
        
        # 模型状态指示
        self.model_status_label = ttk.Label(self.model_frame, text="模型未加载", 
                                           style="Large.TLabel", foreground="red")
        self.model_status_label.grid(row=2, column=2, padx=8, pady=12)
        self.model_frame.columnconfigure(1, weight=1)

    def auto_detect_model_paths(self):
        """自动检测模型路径并填充到输入框"""
        try:
            ckpt_info = discover_checkpoint_paths()
            cra = ckpt_info.get("cra_default") or ""
            sr = ckpt_info.get("srgan_default") or ""

            if cra:
                self.cra_ckpt_var.set(cra)
            if sr:
                self.srgan_ckpt_var.set(sr)

            if cra and sr:
                self.status_var.set("已自动检测到 CRA 与 SRGAN 检查点")
                self.append_log("已自动检测到 CRA 与 SRGAN 检查点")
            elif cra or sr:
                self.status_var.set("已检测到部分检查点，请确认路径")
                self.append_log("已检测到部分检查点")
            else:
                self.status_var.set("未检测到检查点，请手动选择")
                self.append_log("未检测到检查点")
                messagebox.showwarning("提示", "未自动检测到可用检查点，请手动选择模型文件。")
        except Exception as e:
            messagebox.showerror("错误", f"自动检测模型路径失败: {e}")

    def on_backend_changed(self, *args):
        """根据后端调整模型输入区状态。"""
        backend = self.backend_var.get()
        state = "disabled" if backend == "classic" else "normal"
        for widget in (self.cra_entry, self.srgan_entry, self.cra_browse_btn, self.srgan_browse_btn, self.detect_models_btn):
            widget.configure(state=state)
        if backend == "classic":
            self.model_status_label.config(text="classic 后端无需模型", foreground="green")
        else:
            self.model_status_label.config(text="模型未加载", foreground="red")
        self.models_loaded = False
    
    def create_parameter_frame(self):
        """创建参数配置区域"""
        self.param_frame = ttk.LabelFrame(self.main_frame, text="参数配置", padding="8",
                                         style="Heading.TLabelframe")
        
        # 设备选择
        ttk.Label(self.param_frame, text="处理后端:", style="Large.TLabel").grid(
            row=0, column=0, sticky="w", padx=8, pady=4)
        backend_combo = ttk.Combobox(self.param_frame, textvariable=self.backend_var,
                                     values=["auto", "classic", "deep"], state="readonly",
                                     style="Large.TCombobox", width=12)
        backend_combo.grid(row=0, column=1, padx=8, pady=4)

        # 设备选择
        ttk.Label(self.param_frame, text="运行设备:", style="Large.TLabel").grid(
            row=0, column=2, sticky="w", padx=8, pady=4)
        device_combo = ttk.Combobox(self.param_frame, textvariable=self.device_var, 
                                   values=["GPU", "CPU", "Ascend"], state="readonly",
                                   style="Large.TCombobox", width=12)
        device_combo.grid(row=0, column=3, padx=8, pady=4)
        
        # 超分辨率倍数
        ttk.Label(self.param_frame, text="超分辨率倍数:", style="Large.TLabel").grid(
            row=1, column=0, sticky="w", padx=8, pady=4)
        scale_combo = ttk.Combobox(self.param_frame, textvariable=self.scale_var, 
                                  values=[1, 2, 4, 8], state="readonly",
                                  style="Large.TCombobox", width=8)
        scale_combo.grid(row=1, column=1, padx=8, pady=4)
        
        # 输入尺寸
        ttk.Label(self.param_frame, text="输入尺寸:", style="Large.TLabel").grid(
            row=1, column=2, sticky="w", padx=8, pady=4)
        size_combo = ttk.Combobox(self.param_frame, textvariable=self.input_size_var, 
                                 values=[256, 512, 1024], state="readonly",
                                 style="Large.TCombobox", width=12)
        size_combo.grid(row=1, column=3, padx=8, pady=4)
    
    def create_preview_frame(self):
        """创建图像预览区域"""
        self.preview_frame = ttk.LabelFrame(self.main_frame, text="图像预览", padding="8",
                                           style="Heading.TLabelframe")
        
        # 创建预览画布，增加高度以适应更大的字体
        self.preview_canvas = tk.Canvas(self.preview_frame, width=800, height=250, bg="white")
        self.preview_canvas.pack(fill="both", expand=True)
        
        # 预览标签
        self.preview_info_label = ttk.Label(self.preview_frame, text="选择图像后将显示预览",
                                           style="Large.TLabel")
        self.preview_info_label.pack(pady=8)
    
    def create_control_frame(self):
        """创建控制按钮区域"""
        self.control_frame = ttk.Frame(self.main_frame)

        self.demo_btn = ttk.Button(self.control_frame, text="生成演示输入",
                                  command=self.create_demo_inputs,
                                  style="Large.TButton")
        self.demo_btn.pack(side="left", padx=8, pady=5)

        self.mask_check_btn = ttk.Button(self.control_frame, text="检查掩码",
                                        command=self.check_mask,
                                        style="Large.TButton")
        self.mask_check_btn.pack(side="left", padx=8, pady=5)
        
        # 开始处理按钮
        self.process_btn = ttk.Button(self.control_frame, text="开始处理", 
                                     command=self.start_processing, 
                                     style="Large.TButton")
        self.process_btn.pack(side="left", padx=8, pady=5)
        
        # 停止按钮
        self.stop_btn = ttk.Button(self.control_frame, text="停止", 
                                  command=self.stop_processing, 
                                  state="disabled", style="Large.TButton")
        self.stop_btn.pack(side="left", padx=8, pady=5)
        
        # 清空日志按钮
        self.clear_log_btn = ttk.Button(self.control_frame, text="清空日志", 
                                       command=self.clear_log, style="Large.TButton")
        self.clear_log_btn.pack(side="left", padx=8, pady=5)
        
        # 打开输出目录按钮
        self.open_output_btn = ttk.Button(self.control_frame, text="打开输出目录", 
                                         command=self.open_output_directory, style="Large.TButton")
        self.open_output_btn.pack(side="left", padx=8, pady=5)
    
    def create_status_frame(self):
        """创建状态和进度区域"""
        self.status_frame = ttk.Frame(self.main_frame)
        
        # 状态标签
        ttk.Label(self.status_frame, text="状态:", style="Large.TLabel").pack(side="left")
        self.status_label = ttk.Label(self.status_frame, textvariable=self.status_var,
                                     style="Large.TLabel")
        self.status_label.pack(side="left", padx=8)
        
        # 进度条 - 增加高度以适应更大的字体
        self.progress_bar = ttk.Progressbar(self.status_frame, mode="indeterminate",
                                           style="Large.Horizontal.TProgressbar")
        self.progress_bar.pack(side="right", fill="x", expand=True, padx=12)

    def create_log_frame(self):
        """创建可见日志区域"""
        self.log_frame = ttk.LabelFrame(self.main_frame, text="运行日志", padding="8",
                                       style="Heading.TLabelframe")
        self.log_text = ScrolledText(
            self.log_frame,
            height=8,
            font=self.fonts["log"],
            state="disabled",
            wrap="word",
        )
        self.log_text.pack(fill="both", expand=True)
    
    def setup_layout(self):
        """设置布局"""
        self.main_frame.pack(fill="both", expand=True)
        
        # 标题 - 增加更多间距
        self.title_label.pack(pady=15)
        
        # 文件选择区域 - 增加间距
        self.file_frame.pack(fill="x", pady=8)
        
        # 模型配置区域
        self.model_frame.pack(fill="x", pady=8)
        
        # 参数配置区域
        self.param_frame.pack(fill="x", pady=8)
        
        # 预览区域
        self.preview_frame.pack(fill="x", pady=8)
        
        # 控制按钮区域
        self.control_frame.pack(fill="x", pady=8)
        
        # 状态区域
        self.status_frame.pack(fill="x", pady=8)

        # 日志区域
        self.log_frame.pack(fill="both", expand=True, pady=8)
        
        # 绑定事件
        self.input_image_var.trace("w", self.on_image_selected)
        self.mask_image_var.trace("w", self.on_image_selected)
        self.backend_var.trace("w", self.on_backend_changed)
        self.on_backend_changed()
    
    def browse_input_image(self):
        """浏览输入图像"""
        filename = filedialog.askopenfilename(
            title="选择输入图像",
            filetypes=[
                ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("所有文件", "*.*")
            ]
        )
        if filename:
            self.input_image_var.set(filename)
    
    def browse_mask_image(self):
        """浏览掩码图像"""
        filename = filedialog.askopenfilename(
            title="选择掩码图像",
            filetypes=[
                ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("所有文件", "*.*")
            ]
        )
        if filename:
            self.mask_image_var.set(filename)
    
    def browse_output_dir(self):
        """浏览输出目录"""
        dirname = filedialog.askdirectory(title="选择输出目录")
        if dirname:
            self.output_dir_var.set(dirname)
    
    def browse_cra_model(self):
        """浏览CRA模型文件"""
        filename = filedialog.askopenfilename(
            title="选择CRA模型文件",
            filetypes=[
                ("检查点文件", "*.ckpt"),
                ("所有文件", "*.*")
            ]
        )
        if filename:
            self.cra_ckpt_var.set(filename)
    
    def browse_srgan_model(self):
        """浏览SRGAN模型文件"""
        filename = filedialog.askopenfilename(
            title="选择SRGAN模型文件",
            filetypes=[
                ("检查点文件", "*.ckpt"),
                ("所有文件", "*.*")
            ]
        )
        if filename:
            self.srgan_ckpt_var.set(filename)

    def build_config(self):
        """从界面状态构造配置。"""
        config = ModelConfig()
        config.backend = self.backend_var.get()
        config.device_target = self.device_var.get()
        config.scale = self.scale_var.get()
        config.input_size = self.input_size_var.get()
        config.validate()
        return config

    def create_demo_inputs(self):
        """生成内置演示图像和掩码，方便展示 GUI 效果。"""
        try:
            output_dir = Path(self.output_dir_var.get() or "./output")
            demo_dir = output_dir / "gui_demo_assets"
            input_path, mask_path = create_demo_assets(demo_dir)
            self.input_image_var.set(input_path)
            self.mask_image_var.set(mask_path)
            self.backend_var.set("classic")
            self.status_var.set("演示输入已生成")
            self.append_log(f"已生成演示输入: {input_path}")
            self.append_log(f"已生成演示掩码: {mask_path}")
        except Exception as e:
            messagebox.showerror("错误", f"生成演示输入失败: {e}")

    def check_mask(self):
        """检查掩码尺寸和白色区域占比。"""
        input_path = self.input_image_var.get()
        mask_path = self.mask_image_var.get()
        if not mask_path or not os.path.exists(mask_path):
            messagebox.showerror("错误", "请先选择有效的掩码图像")
            return

        try:
            image_shape = None
            if input_path and os.path.exists(input_path):
                img = ImageProcessor.load_image(input_path)
                if img is not None:
                    image_shape = img.shape

            mask = ImageProcessor.load_mask(mask_path, image_shape)
            if mask is None:
                messagebox.showerror("错误", "无法读取掩码图像")
                return

            white_pixels = int(np.count_nonzero(mask == 255))
            total_pixels = int(mask.size)
            ratio = white_pixels / total_pixels * 100 if total_pixels else 0
            message = f"掩码尺寸: {mask.shape[1]} x {mask.shape[0]}\n白色修复区域: {ratio:.2f}%"
            if ratio < 0.1:
                message += "\n提示: 修复区域过小，效果可能不明显。"
            elif ratio > 50:
                message += "\n提示: 修复区域过大，质量风险较高。"
            else:
                message += "\n掩码比例正常。"
            self.append_log(message.replace("\n", " | "))
            messagebox.showinfo("掩码检查", message)
        except Exception as e:
            messagebox.showerror("错误", f"掩码检查失败: {e}")
    
    def on_image_selected(self, *args):
        """当图像被选择时更新预览"""
        self.update_preview()
    
    def update_preview(self):
        """更新图像预览"""
        try:
            input_path = self.input_image_var.get()
            mask_path = self.mask_image_var.get()
            result_path = self.latest_result_path
            
            self.preview_canvas.delete("all")
            
            x_offset = 10
            y_offset = 10
            max_height = 180
            
            # 显示输入图像
            if input_path and os.path.exists(input_path):
                img = Image.open(input_path)
                # 调整大小
                ratio = min(max_height / img.height, 200 / img.width)
                new_size = (int(img.width * ratio), int(img.height * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                self.input_photo = ImageTk.PhotoImage(img)
                self.preview_canvas.create_image(x_offset, y_offset, anchor="nw", 
                                               image=self.input_photo)
                self.preview_canvas.create_text(x_offset + new_size[0]//2, y_offset + new_size[1] + 8, 
                                              text="输入图像", anchor="n", 
                                              font=self.fonts['label'])
                x_offset += new_size[0] + 20
            
            # 显示掩码图像
            if mask_path and os.path.exists(mask_path):
                mask_img = Image.open(mask_path)
                # 调整大小
                ratio = min(max_height / mask_img.height, 200 / mask_img.width)
                new_size = (int(mask_img.width * ratio), int(mask_img.height * ratio))
                mask_img = mask_img.resize(new_size, Image.Resampling.LANCZOS)
                
                self.mask_photo = ImageTk.PhotoImage(mask_img)
                self.preview_canvas.create_image(x_offset, y_offset, anchor="nw", 
                                               image=self.mask_photo)
                self.preview_canvas.create_text(x_offset + new_size[0]//2, y_offset + new_size[1] + 8, 
                                              text="掩码图像", anchor="n",
                                              font=self.fonts['label'])
                x_offset += new_size[0] + 20

            # 显示结果图像（若存在）
            if result_path and os.path.exists(result_path):
                result_img = Image.open(result_path)
                ratio = min(max_height / result_img.height, 200 / result_img.width)
                new_size = (int(result_img.width * ratio), int(result_img.height * ratio))
                result_img = result_img.resize(new_size, Image.Resampling.LANCZOS)

                self.result_photo = ImageTk.PhotoImage(result_img)
                self.preview_canvas.create_image(x_offset, y_offset, anchor="nw",
                                               image=self.result_photo)
                self.preview_canvas.create_text(x_offset + new_size[0]//2, y_offset + new_size[1] + 8,
                                              text="处理结果", anchor="n",
                                              font=self.fonts['label'])
            
            if input_path or mask_path:
                self.preview_info_label.config(text="")
            else:
                self.preview_info_label.config(text="选择图像后将显示预览")
                
        except Exception as e:
            self.preview_info_label.config(text=f"预览错误: {e}")
    
    def load_models(self):
        """加载模型"""
        self.status_var.set("正在准备后端...")
        self.progress_bar.start()
        self.load_models_btn.config(state="disabled")
        self.detect_models_btn.config(state="disabled")
        self.append_log("开始初始化处理后端")

        def load_in_thread():
            try:
                config = self.build_config()
                self.processor = CombinedProcessor(config)
                
                cra_ckpt = self.cra_ckpt_var.get()
                srgan_ckpt = self.srgan_ckpt_var.get()
                cra_success, srgan_success = self.processor.load_models(cra_ckpt, srgan_ckpt)
                
                if cra_success and srgan_success:
                    loaded_cra = self.processor.loaded_ckpts.get("cra") or cra_ckpt
                    loaded_sr = self.processor.loaded_ckpts.get("srgan") or srgan_ckpt
                    backend = self.processor.active_backend
                    self.root.after(0, lambda: self._finish_model_load_success(backend, loaded_cra, loaded_sr))
                else:
                    error_msgs = []
                    if not cra_success:
                        error_msgs.append("CRA模型加载失败")
                    if not srgan_success:
                        error_msgs.append("SRGAN模型加载失败")
                    raise Exception("; ".join(error_msgs))
                    
            except Exception as e:
                self.root.after(0, lambda err=e: self._finish_model_load_error(err))
            finally:
                self.root.after(0, self._restore_model_buttons)
        
        threading.Thread(target=load_in_thread, daemon=True).start()

    def _finish_model_load_success(self, backend, loaded_cra, loaded_sr):
        self.models_loaded = True
        self.cra_ckpt_var.set(loaded_cra)
        self.srgan_ckpt_var.set(loaded_sr)
        self.model_status_label.config(text=f"{backend} 后端就绪", foreground="green")
        self.status_var.set(f"{backend} 后端已就绪")
        self.append_log(f"处理后端已就绪: {backend}")
        messagebox.showinfo("成功", f"处理后端已就绪！\n后端: {backend}\nCRA: {loaded_cra}\nSRGAN: {loaded_sr}")

    def _finish_model_load_error(self, error):
        self.models_loaded = False
        self.model_status_label.config(text="模型加载失败", foreground="red")
        self.status_var.set("模型加载失败")
        self.append_log(f"模型加载失败: {error}")
        messagebox.showerror("错误", f"模型加载失败: {error}")

    def _restore_model_buttons(self):
        self.progress_bar.stop()
        self.load_models_btn.config(state="normal")
        state = "disabled" if self.backend_var.get() == "classic" else "normal"
        self.detect_models_btn.config(state=state)
    
    def start_processing(self):
        """开始处理"""
        # 验证输入
        if not self.models_loaded:
            messagebox.showerror("错误", "请先加载模型")
            return
        
        input_path = self.input_image_var.get()
        mask_path = self.mask_image_var.get()
        output_dir = self.output_dir_var.get()
        
        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("错误", "请选择有效的输入图像")
            return
        
        if not mask_path or not os.path.exists(mask_path):
            messagebox.showerror("错误", "请选择有效的掩码图像")
            return
        
        if not output_dir:
            messagebox.showerror("错误", "请选择输出目录")
            return

        self.latest_result_path = ""
        self.is_processing = True
        self.process_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress_bar.start()
        self.append_log(f"开始处理: {input_path}")
        
        def process_in_thread():
            try:
                def status_callback(message):
                    self.root.after(0, lambda msg=message: self.status_var.set(msg))
                    self.root.after(0, lambda msg=message: self.append_log(msg))
                
                success, result = self.processor.process_image(
                    input_path, mask_path, output_dir, status_callback
                )
                
                if success:
                    self.latest_result_path = result
                    self.root.after(0, lambda: self.status_var.set("处理完成"))
                    self.root.after(0, lambda res=result: self.append_log(f"处理完成: {res}"))
                    self.root.after(0, self.update_preview)
                    self.root.after(0, lambda res=result: messagebox.showinfo("成功", f"处理完成！\n结果保存在: {res}"))
                else:
                    self.root.after(0, lambda: self.status_var.set("处理失败"))
                    self.root.after(0, lambda res=result: self.append_log(f"处理失败: {res}"))
                    self.root.after(0, lambda res=result: messagebox.showerror("错误", f"处理失败: {res}"))
                    
            except Exception as e:
                self.root.after(0, lambda: self.status_var.set("处理出错"))
                self.root.after(0, lambda err=e: self.append_log(f"处理过程中出错: {err}"))
                self.root.after(0, lambda err=e: messagebox.showerror("错误", f"处理过程中出错: {err}"))
            finally:
                self.is_processing = False
                self.root.after(0, lambda: self.progress_bar.stop())
                self.root.after(0, lambda: self.process_btn.config(state="normal"))
                self.root.after(0, lambda: self.stop_btn.config(state="disabled"))
        
        # 在单独线程中处理
        threading.Thread(target=process_in_thread, daemon=True).start()
    
    def stop_processing(self):
        """停止处理"""
        self.status_var.set("当前任务会在本轮处理结束后释放界面")
        self.append_log("当前后端暂不支持安全中断，等待本轮处理结束")
        messagebox.showinfo("信息", "当前处理步骤暂不支持安全中断，请等待本轮处理结束。")
    
    def clear_log(self):
        """清空日志"""
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
    
    def open_output_directory(self):
        """打开输出目录"""
        output_dir = self.output_dir_var.get()
        if os.path.exists(output_dir):
            import subprocess
            import platform
            
            system = platform.system()
            if system == "Windows":
                os.startfile(output_dir)
            elif system == "Darwin":  # macOS
                subprocess.Popen(["open", output_dir])
            else:  # Linux
                subprocess.Popen(["xdg-open", output_dir])
        else:
            messagebox.showwarning("警告", "输出目录不存在")
    
    def run(self):
        """运行GUI"""
        # 设置日志处理器
        self.setup_logging()
        
        # 显示欢迎信息
        welcome_msg = """
欢迎使用图像修复与超分辨率处理工具！

使用步骤：
1. 可点击"生成演示输入"快速展示效果
2. 或选择输入图像和掩码图像
3. 选择处理后端并点击"加载模型"
4. 点击"开始处理"并在预览区查看结果

注意事项：
- classic 后端无需模型文件
- 掩码图像中白色区域表示需要修复的部分
- 处理时间取决于图像大小和硬件性能
"""
        print(welcome_msg)
        
        # 启动主循环
        self.root.mainloop()

def launch_gui():
    """启动GUI应用"""
    try:
        app = ImageRepairGUI()
        app.run()
    except Exception as e:
        print(f"启动GUI失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    launch_gui() 
