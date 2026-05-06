#!/usr/bin/env python3
import os
import graphviz
import shutil
from pathlib import Path

OUTPUT_DIR = str(Path(__file__).resolve().parents[1] / "模型流程图")

def create_cra_inference_diagram():
    """创建CRA模型推理流程图"""
    try:
        # 确保输出目录存在
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        dot = graphviz.Digraph('CRA模型推理流程图', format='png')
        # 设置更高的DPI和图形尺寸以提高清晰度
        dot.attr(rankdir='LR', size='16,12', ratio='fill', fontname='SimHei', dpi='600')
        
        # 设置节点样式
        dot.attr('node', shape='box', style='filled,rounded', fontname='SimHei', fontsize='16', 
                 margin='0.3,0.2', width='2.2', height='1.2')
        
        # 定义颜色方案
        colors = {
            'input': '#E0F7FA',     # 浅青色 - 输入数据
            'model': '#F3E5F5',     # 浅紫色 - 模型组件
            'process': '#FFF3E0',   # 浅橙色 - 处理步骤
            'output': '#E8F5E9',    # 浅绿色 - 输出数据
            'arrow': '#37474F'      # 深灰色 - 箭头
        }
        
        # 设置边的样式
        dot.attr('edge', color=colors['arrow'], fontname='SimHei', fontsize='14', 
                 penwidth='1.5')
        
        # 创建输入节点
        with dot.subgraph(name='cluster_input') as c:
            c.attr(label='输入处理', style='filled', fillcolor='#F0F0F0', fontsize='18', fontname='SimHei')
            
            c.node('input_img', '输入损坏图像\n含有缺失或损坏区域', 
                   fillcolor=colors['input'], style='filled,rounded', shape='folder')
            c.node('input_mask', '损坏区域掩码\n指示需要修复的区域', 
                   fillcolor=colors['input'], style='filled,rounded', shape='folder')
            c.node('load_img', '加载图像\n读取并解码图像', 
                   fillcolor=colors['process'])
            c.node('load_mask', '加载/生成掩码\n读取或交互式生成掩码', 
                   fillcolor=colors['process'])
        
        # 创建预处理节点
        with dot.subgraph(name='cluster_preprocess') as c:
            c.attr(label='预处理', style='filled', fillcolor='#F8F8F8', fontsize='18', fontname='SimHei')
            
            c.node('resize', '调整尺寸\n将图像和掩码调整为模型输入尺寸', 
                   fillcolor=colors['process'])
            c.node('normalize', '归一化\n将像素值归一化到[-1,1]', 
                   fillcolor=colors['process'])
            c.node('to_tensor', '转换为张量\n从NumPy数组转换为MindSpore张量', 
                   fillcolor=colors['process'])
            c.node('expand_dims', '扩展维度\n添加批次维度', 
                   fillcolor=colors['process'])
        
        # 创建模型加载和推理节点
        with dot.subgraph(name='cluster_model') as c:
            c.attr(label='模型推理', style='filled', fillcolor='#F0F8FF', fontsize='18', fontname='SimHei')
            
            c.node('load_model', '加载CRA模型\n从检查点文件加载参数', 
                   fillcolor=colors['model'])
            c.node('checkpoint', 'CRA模型检查点\n预训练的模型参数', 
                   fillcolor=colors['model'], shape='note')
            c.node('build_net', '构建推理网络\n创建门控卷积网络结构', 
                   fillcolor=colors['model'])
            c.node('gated_conv', '门控卷积\n对输入特征进行处理', 
                   fillcolor=colors['model'])
            c.node('context_attn', '上下文注意力\n关注未损坏区域特征', 
                   fillcolor=colors['model'])
            c.node('inference', '执行推理\n前向传播生成修复结果', 
                   fillcolor=colors['model'])
        
        # 创建后处理和输出节点
        with dot.subgraph(name='cluster_output') as c:
            c.attr(label='后处理与输出', style='filled', fillcolor='#F0FFF0', fontsize='18', fontname='SimHei')
            
            c.node('denormalize', '反归一化\n将值范围恢复到[0,255]', 
                   fillcolor=colors['process'])
            c.node('resize_back', '恢复原始尺寸\n调整回输入图像尺寸', 
                   fillcolor=colors['process'])
            c.node('blending', '图像融合\n将修复区域与原始图像融合', 
                   fillcolor=colors['process'])
            c.node('repaired_img', '修复后的图像\nCRA模型的最终输出', 
                   fillcolor=colors['output'], style='filled,rounded', shape='folder')
        
        # 连接输入节点
        dot.edge('input_img', 'load_img')
        dot.edge('input_mask', 'load_mask')
        
        # 连接预处理节点
        dot.edge('load_img', 'resize')
        dot.edge('load_mask', 'resize')
        dot.edge('resize', 'normalize')
        dot.edge('normalize', 'to_tensor')
        dot.edge('to_tensor', 'expand_dims')
        
        # 连接模型加载和推理节点
        dot.edge('checkpoint', 'load_model')
        dot.edge('load_model', 'build_net')
        dot.edge('build_net', 'gated_conv')
        dot.edge('gated_conv', 'context_attn')
        dot.edge('context_attn', 'inference')
        dot.edge('expand_dims', 'inference')
        
        # 连接后处理和输出节点
        dot.edge('inference', 'denormalize')
        dot.edge('denormalize', 'resize_back')
        dot.edge('resize_back', 'blending')
        dot.edge('blending', 'repaired_img')
        
        # 渲染和保存图形
        output_filename = os.path.join(OUTPUT_DIR, 'CRA模型推理流程图')
        dot.render(output_filename, cleanup=True)
        print(f"CRA模型推理流程图已生成: {os.path.abspath(output_filename + '.png')}")
        
        return output_filename + '.png'
    except ImportError:
        print("错误: 未找到graphviz库。请执行 'pip install graphviz' 并确保系统已安装graphviz。")
        return None
    except Exception as e:
        print(f"生成CRA模型推理流程图时出错: {e}")
        return None

def create_srgan_inference_diagram():
    """创建SRGAN模型推理流程图"""
    try:
        # 确保输出目录存在
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        dot = graphviz.Digraph('SRGAN模型推理流程图', format='png')
        # 设置更高的DPI和图形尺寸以提高清晰度
        dot.attr(rankdir='LR', size='16,12', ratio='fill', fontname='SimHei', dpi='600')
        
        # 设置节点样式
        dot.attr('node', shape='box', style='filled,rounded', fontname='SimHei', fontsize='16', 
                 margin='0.3,0.2', width='2.2', height='1.2')
        
        # 定义颜色方案
        colors = {
            'input': '#E0F7FA',     # 浅青色 - 输入数据
            'model': '#F3E5F5',     # 浅紫色 - 模型组件
            'process': '#FFF3E0',   # 浅橙色 - 处理步骤
            'output': '#E8F5E9',    # 浅绿色 - 输出数据
            'arrow': '#37474F'      # 深灰色 - 箭头
        }
        
        # 设置边的样式
        dot.attr('edge', color=colors['arrow'], fontname='SimHei', fontsize='14', 
                 penwidth='1.5')
        
        # 创建输入节点
        with dot.subgraph(name='cluster_input') as c:
            c.attr(label='输入处理', style='filled', fillcolor='#F0F0F0', fontsize='18', fontname='SimHei')
            
            c.node('input_img', '输入低分辨率图像\nCRA修复后的图像或其他低分辨率图像', 
                   fillcolor=colors['input'], style='filled,rounded', shape='folder')
            c.node('load_img', '加载图像\n读取并解码图像', 
                   fillcolor=colors['process'])
        
        # 创建预处理节点
        with dot.subgraph(name='cluster_preprocess') as c:
            c.attr(label='预处理', style='filled', fillcolor='#F8F8F8', fontsize='18', fontname='SimHei')
            
            c.node('resize', '调整尺寸\n确保尺寸满足上采样要求', 
                   fillcolor=colors['process'])
            c.node('normalize', '归一化\n将像素值归一化到[-1,1]', 
                   fillcolor=colors['process'])
            c.node('to_tensor', '转换为张量\n从NumPy数组转换为MindSpore张量', 
                   fillcolor=colors['process'])
            c.node('expand_dims', '扩展维度\n添加批次维度', 
                   fillcolor=colors['process'])
        
        # 创建模型加载和推理节点
        with dot.subgraph(name='cluster_model') as c:
            c.attr(label='模型推理', style='filled', fillcolor='#F0F8FF', fontsize='18', fontname='SimHei')
            
            c.node('load_model', '加载SRGAN生成器\n从检查点文件加载参数', 
                   fillcolor=colors['model'])
            c.node('checkpoint', 'SRGAN模型检查点\n预训练的模型参数', 
                   fillcolor=colors['model'], shape='note')
            c.node('build_net', '构建生成器网络\n创建残差块和上采样网络', 
                   fillcolor=colors['model'])
            c.node('residual_blocks', '残差块处理\n提取深层特征', 
                   fillcolor=colors['model'])
            c.node('upsampling', '上采样层\n增加空间分辨率', 
                   fillcolor=colors['model'])
            c.node('inference', '执行推理\n前向传播生成高分辨率结果', 
                   fillcolor=colors['model'])
        
        # 创建后处理和输出节点
        with dot.subgraph(name='cluster_output') as c:
            c.attr(label='后处理与输出', style='filled', fillcolor='#F0FFF0', fontsize='18', fontname='SimHei')
            
            c.node('denormalize', '反归一化\n将值范围恢复到[0,255]', 
                   fillcolor=colors['process'])
            c.node('post_process', '后处理\n应用可选的锐化或色彩调整', 
                   fillcolor=colors['process'])
            c.node('sr_img', '超分辨率图像\nSRGAN模型的最终输出', 
                   fillcolor=colors['output'], style='filled,rounded', shape='folder')
        
        # 连接输入节点
        dot.edge('input_img', 'load_img')
        
        # 连接预处理节点
        dot.edge('load_img', 'resize')
        dot.edge('resize', 'normalize')
        dot.edge('normalize', 'to_tensor')
        dot.edge('to_tensor', 'expand_dims')
        
        # 连接模型加载和推理节点
        dot.edge('checkpoint', 'load_model')
        dot.edge('load_model', 'build_net')
        dot.edge('build_net', 'residual_blocks')
        dot.edge('residual_blocks', 'upsampling')
        dot.edge('upsampling', 'inference')
        dot.edge('expand_dims', 'inference')
        
        # 连接后处理和输出节点
        dot.edge('inference', 'denormalize')
        dot.edge('denormalize', 'post_process')
        dot.edge('post_process', 'sr_img')
        
        # 渲染和保存图形
        output_filename = os.path.join(OUTPUT_DIR, 'SRGAN模型推理流程图')
        dot.render(output_filename, cleanup=True)
        print(f"SRGAN模型推理流程图已生成: {os.path.abspath(output_filename + '.png')}")
        
        return output_filename + '.png'
    except ImportError:
        print("错误: 未找到graphviz库。请执行 'pip install graphviz' 并确保系统已安装graphviz。")
        return None
    except Exception as e:
        print(f"生成SRGAN模型推理流程图时出错: {e}")
        return None

def create_combined_inference_diagram():
    """创建CRA+SRGAN联合推理流程图"""
    try:
        # 确保输出目录存在
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        dot = graphviz.Digraph('CRA和SRGAN联合推理流程图', format='png')
        # 设置更高的DPI和图形尺寸以提高清晰度
        dot.attr(rankdir='LR', size='20,12', ratio='fill', fontname='SimHei', dpi='600')
        
        # 设置节点样式
        dot.attr('node', shape='box', style='filled,rounded', fontname='SimHei', fontsize='16', 
                 margin='0.3,0.2', width='2.2', height='1.2')
        
        # 定义颜色方案
        colors = {
            'input': '#E0F7FA',     # 浅青色 - 输入数据
            'cra': '#FFECB3',       # 浅黄色 - CRA相关
            'srgan': '#E8F5E9',     # 浅绿色 - SRGAN相关
            'process': '#F3E5F5',   # 浅紫色 - 处理步骤
            'output': '#FBE9E7',    # 浅红色 - 输出数据
            'arrow': '#37474F'      # 深灰色 - 箭头
        }
        
        # 设置边的样式
        dot.attr('edge', color=colors['arrow'], fontname='SimHei', fontsize='14', 
                 penwidth='1.5')
        
        # 创建输入节点
        with dot.subgraph(name='cluster_input') as c:
            c.attr(label='输入处理', style='filled', fillcolor='#F0F0F0', fontsize='18', fontname='SimHei')
            
            c.node('input_img', '输入损坏图像', 
                   fillcolor=colors['input'], style='filled,rounded', shape='folder')
            c.node('input_mask', '损坏区域掩码\n(可选)', 
                   fillcolor=colors['input'], style='filled,rounded', shape='folder')
            c.node('load_img', '加载图像', 
                   fillcolor=colors['process'])
            c.node('load_mask', '加载/生成掩码', 
                   fillcolor=colors['process'])
        
        # 创建CRA处理节点
        with dot.subgraph(name='cluster_cra') as c:
            c.attr(label='CRA修复流程', style='filled', fillcolor='#FFF8E1', fontsize='18', fontname='SimHei')
            
            c.node('cra_preprocess', 'CRA预处理\n调整尺寸和归一化', 
                   fillcolor=colors['cra'])
            c.node('cra_model', 'CRA模型\n门控卷积网络与上下文注意力', 
                   fillcolor=colors['cra'])
            c.node('cra_inference', 'CRA推理过程\n图像缺失区域修复', 
                   fillcolor=colors['cra'])
            c.node('cra_postprocess', 'CRA后处理\n图像融合与尺寸恢复', 
                   fillcolor=colors['cra'])
            c.node('repaired_img', '修复后图像', 
                   fillcolor=colors['cra'], style='filled,rounded', shape='folder')
        
        # 创建SRGAN处理节点
        with dot.subgraph(name='cluster_srgan') as c:
            c.attr(label='SRGAN超分辨率流程', style='filled', fillcolor='#E8F5E9', fontsize='18', fontname='SimHei')
            
            c.node('srgan_preprocess', 'SRGAN预处理\n调整尺寸和归一化', 
                   fillcolor=colors['srgan'])
            c.node('srgan_model', 'SRGAN模型\n残差块与上采样网络', 
                   fillcolor=colors['srgan'])
            c.node('srgan_inference', 'SRGAN推理过程\n图像超分辨率生成', 
                   fillcolor=colors['srgan'])
            c.node('srgan_postprocess', 'SRGAN后处理\n后期锐化和色彩调整', 
                   fillcolor=colors['srgan'])
            c.node('sr_img', '超分辨率图像', 
                   fillcolor=colors['srgan'], style='filled,rounded', shape='folder')
        
        # 创建输出节点
        with dot.subgraph(name='cluster_output') as c:
            c.attr(label='最终输出', style='filled', fillcolor='#FBE9E7', fontsize='18', fontname='SimHei')
            
            c.node('metrics_calc', '评估指标计算\nPSNR和SSIM (可选)', 
                   fillcolor=colors['output'])
            c.node('save_result', '保存结果\n保存为图片文件', 
                   fillcolor=colors['output'])
            c.node('final_output', '最终输出图像\n修复+超分辨率', 
                   fillcolor=colors['output'], style='filled,rounded', shape='folder')
        
        # 连接输入节点
        dot.edge('input_img', 'load_img')
        dot.edge('input_mask', 'load_mask')
        
        # 连接CRA处理节点
        dot.edge('load_img', 'cra_preprocess')
        dot.edge('load_mask', 'cra_preprocess')
        dot.edge('cra_preprocess', 'cra_model')
        dot.edge('cra_model', 'cra_inference')
        dot.edge('cra_inference', 'cra_postprocess')
        dot.edge('cra_postprocess', 'repaired_img')
        
        # 连接SRGAN处理节点
        dot.edge('repaired_img', 'srgan_preprocess')
        dot.edge('srgan_preprocess', 'srgan_model')
        dot.edge('srgan_model', 'srgan_inference')
        dot.edge('srgan_inference', 'srgan_postprocess')
        dot.edge('srgan_postprocess', 'sr_img')
        
        # 连接输出节点
        dot.edge('sr_img', 'metrics_calc', style='dashed')
        dot.edge('sr_img', 'save_result')
        dot.edge('save_result', 'final_output')
        
        # 添加组合系统节点和连接
        dot.node('combined_system', 'CRA+SRGAN 组合系统', shape='component', 
                 style='filled', fillcolor='#F5F5F5', fontname='SimHei')
        dot.edge('combined_system', 'cra_model', style='dashed')
        dot.edge('combined_system', 'srgan_model', style='dashed')
        
        # 添加MindSpore推理框架节点和连接
        dot.node('mindspore_infer', 'MindSpore 推理框架', shape='component', 
                 style='filled', fillcolor='#F5F5F5', fontname='SimHei')
        dot.edge('mindspore_infer', 'cra_inference', style='dashed')
        dot.edge('mindspore_infer', 'srgan_inference', style='dashed')
        
        # 渲染和保存图形
        output_filename = os.path.join(OUTPUT_DIR, 'CRA和SRGAN联合推理流程图')
        dot.render(output_filename, cleanup=True)
        print(f"CRA和SRGAN联合推理流程图已生成: {os.path.abspath(output_filename + '.png')}")
        
        return output_filename + '.png'
    except ImportError:
        print("错误: 未找到graphviz库。请执行 'pip install graphviz' 并确保系统已安装graphviz。")
        return None
    except Exception as e:
        print(f"生成CRA和SRGAN联合推理流程图时出错: {e}")
        return None

if __name__ == "__main__":
    print("生成CRA和SRGAN的模型推理流程图...")
    create_cra_inference_diagram()
    create_srgan_inference_diagram()
    create_combined_inference_diagram()
    print("模型推理流程图生成完成!") 
