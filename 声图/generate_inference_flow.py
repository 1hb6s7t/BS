#!/usr/bin/env python3
import graphviz
import os

def create_inference_flow_diagram():
    """创建模型推理流程图"""
    # 检查graphviz是否安装
    try:
        dot = graphviz.Digraph('模型推理流程图', format='png')
        # 设置更高的DPI以提高清晰度
        dot.attr(rankdir='LR', size='14,10', ratio='fill', fontname='SimHei', dpi='300')
        
        # 设置节点样式
        dot.attr('node', shape='box', style='filled,rounded', fontname='SimHei', fontsize='14')
        
        # 定义颜色方案
        colors = {
            'input': '#E6F2FF',        # 浅蓝色 - 输入相关
            'cra': '#FFE6E6',          # 浅红色 - CRA相关
            'srgan': '#E6FFE6',        # 浅绿色 - SRGAN相关
            'output': '#F2E6FF',       # 浅紫色 - 输出相关
            'process': '#FFFDE6'       # 浅黄色 - 处理步骤
        }
        
        # 创建输入处理子图
        with dot.subgraph(name='cluster_input') as c:
            c.attr(label='输入处理', style='filled', fillcolor='#F0F8FF', fontsize='16', fontname='SimHei')
            
            # 输入节点
            c.node('user_input', '用户输入', fillcolor=colors['input'])
            c.node('input_image', '输入图像', fillcolor=colors['input'])
            c.node('input_mask', '输入掩码 (可选)', fillcolor=colors['input'])
            c.node('interactive_mask', '交互式创建掩码', fillcolor=colors['input'])
            
            c.node('load_image', '加载图像\n读取并预处理图像', fillcolor=colors['process'])
            c.node('load_mask', '加载掩码\n读取并预处理掩码', fillcolor=colors['process'])
            
            # 连接节点
            c.edges([
                ('user_input', 'input_image'),
                ('user_input', 'input_mask'),
                ('user_input', 'interactive_mask'),
                ('input_image', 'load_image'),
                ('input_mask', 'load_mask'),
                ('interactive_mask', 'load_mask')
            ])
        
        # 创建CRA推理处理子图
        with dot.subgraph(name='cluster_cra_inference') as c:
            c.attr(label='CRA 图像修复推理', style='filled', fillcolor='#FFF0F0', fontsize='16', fontname='SimHei')
            
            # CRA推理节点
            c.node('load_cra_model', '加载CRA模型\n从检查点加载模型参数', fillcolor=colors['cra'])
            c.node('preprocess_cra_input', '预处理输入\n调整大小、标准化', fillcolor=colors['cra'])
            c.node('cra_inference', 'CRA模型推理\n对损坏区域进行修复', fillcolor=colors['cra'])
            c.node('gated_conv', '门控卷积层处理', fillcolor=colors['cra'])
            c.node('attention_mechanism', '上下文注意力机制', fillcolor=colors['cra'])
            c.node('postprocess_cra_output', '后处理输出\n从模型输出生成修复图像', fillcolor=colors['cra'])
            c.node('repaired_image', 'CRA修复图像', fillcolor=colors['cra'])
            
            # 连接节点
            c.edges([
                ('load_cra_model', 'cra_inference'),
                ('preprocess_cra_input', 'cra_inference'),
                ('cra_inference', 'gated_conv'),
                ('gated_conv', 'attention_mechanism'),
                ('attention_mechanism', 'postprocess_cra_output'),
                ('postprocess_cra_output', 'repaired_image')
            ])
        
        # 创建SRGAN推理处理子图
        with dot.subgraph(name='cluster_srgan_inference') as c:
            c.attr(label='SRGAN 超分辨率推理', style='filled', fillcolor='#F0FFF0', fontsize='16', fontname='SimHei')
            
            # SRGAN推理节点
            c.node('load_srgan_model', '加载SRGAN模型\n从检查点加载模型参数', fillcolor=colors['srgan'])
            c.node('preprocess_srgan_input', '预处理输入\n调整大小、标准化', fillcolor=colors['srgan'])
            c.node('srgan_inference', 'SRGAN模型推理\n提高图像分辨率', fillcolor=colors['srgan'])
            c.node('residual_blocks', '残差块处理', fillcolor=colors['srgan'])
            c.node('upsampling', '上采样层处理', fillcolor=colors['srgan'])
            c.node('postprocess_srgan_output', '后处理输出\n从模型输出生成高清图像', fillcolor=colors['srgan'])
            c.node('enhanced_image', 'SRGAN增强图像', fillcolor=colors['srgan'])
            
            # 连接节点
            c.edges([
                ('load_srgan_model', 'srgan_inference'),
                ('preprocess_srgan_input', 'srgan_inference'),
                ('srgan_inference', 'residual_blocks'),
                ('residual_blocks', 'upsampling'),
                ('upsampling', 'postprocess_srgan_output'),
                ('postprocess_srgan_output', 'enhanced_image')
            ])
        
        # 创建输出处理子图
        with dot.subgraph(name='cluster_output') as c:
            c.attr(label='输出处理', style='filled', fillcolor='#F5F0FF', fontsize='16', fontname='SimHei')
            
            # 输出处理节点
            c.node('final_result', '最终结果\n修复+增强图像', fillcolor=colors['output'])
            c.node('save_result', '保存结果\n将处理后的图像保存到磁盘', fillcolor=colors['output'])
            c.node('calculate_metrics', '计算评估指标\nPSNR、SSIM (可选)', fillcolor=colors['process'])
            
            # 连接节点
            c.edges([
                ('final_result', 'save_result'),
                ('final_result', 'calculate_metrics')
            ])
        
        # 连接主要过程
        dot.edge('load_image', 'preprocess_cra_input')
        dot.edge('load_mask', 'preprocess_cra_input')
        dot.edge('repaired_image', 'preprocess_srgan_input')
        dot.edge('enhanced_image', 'final_result')
        
        # 添加组合系统
        dot.node('combined_system', 'CRA+SRGAN 组合系统', shape='component', style='filled', fillcolor='#F2F2F2', fontname='SimHei')
        dot.edge('combined_system', 'cra_inference', style='dashed')
        dot.edge('combined_system', 'srgan_inference', style='dashed')
        
        # 添加MindSpore推理框架
        dot.node('mindspore_infer', 'MindSpore 推理框架', shape='component', style='filled', fillcolor='#F2F2F2', fontname='SimHei')
        dot.edge('mindspore_infer', 'cra_inference', style='dashed')
        dot.edge('mindspore_infer', 'srgan_inference', style='dashed')
        
        # 渲染图形
        output_filename = '模型推理流程图'
        dot.render(output_filename, cleanup=True)
        print(f"模型推理流程图已生成: {os.path.abspath(output_filename + '.png')}")
        
        # 保存英文文件名版本作为备份
        en_output_filename = 'inference_flow'
        if os.path.exists(output_filename + '.png'):
            import shutil
            shutil.copy2(output_filename + '.png', en_output_filename + '.png')
        
        return output_filename + '.png'
    except ImportError:
        print("错误: 未找到graphviz库。请执行 'pip install graphviz' 并确保系统已安装graphviz。")
        return None

# 如果直接运行此脚本
if __name__ == "__main__":
    create_inference_flow_diagram() 