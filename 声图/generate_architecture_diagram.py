#!/usr/bin/env python3
import graphviz as gv
import os

def create_architecture_diagram():
    """创建项目总体架构图"""
    # 创建有向图
    dot = gv.Digraph('Architecture', comment='CRA与SRGAN结合的图像修复与超分辨率架构')
    
    # 设置图像属性
    dot.attr(rankdir='TB', size='12,10', ratio='fill', fontname='SimHei')
    dot.attr('node', shape='box', style='filled', fontname='SimHei')
    
    # 定义节点颜色
    colors = {
        'input': '#FFECB3',    # 浅黄色
        'cra': '#E1F5FE',      # 浅蓝色
        'srgan': '#E8F5E9',    # 浅绿色 
        'output': '#FFCCBC',   # 浅橙色
        'module': '#F3E5F5',   # 浅紫色
        'data': '#EEEEEE'      # 浅灰色
    }
    
    # 添加子图 - 输入模块
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='输入模块', style='filled', color=colors['input'], fontcolor='black')
        c.node('input_image', '输入图像', color='#FFA000', fontcolor='white')
        c.node('input_mask', '掩码图像\n(可选)', color='#FFA000', fontcolor='white')
        c.node('mask_creation', '交互式掩码创建', color='#FFC107')
        
        c.edge('input_image', 'image_preprocessing')
        c.edge('input_mask', 'mask_preprocessing')
        c.edge('input_image', 'mask_creation', style='dashed')
        c.edge('mask_creation', 'mask_preprocessing', style='dashed')
        
        c.node('image_preprocessing', '图像预处理')
        c.node('mask_preprocessing', '掩码预处理')
    
    # 添加子图 - CRA模块
    with dot.subgraph(name='cluster_cra') as c:
        c.attr(label='CRA图像修复模块', style='filled', color=colors['cra'], fontcolor='black')
        
        # 添加CRA模型核心组件
        c.node('gated_generator', 'GatedGenerator\n门控生成器', color='#0288D1', fontcolor='white')
        c.node('contextual_attention', 'ApplyAttention2\n上下文注意力', color='#039BE5')
        c.node('build_inference', 'build_inference_net\n推理构建', color='#29B6F6')
        c.node('post_processing', 'post_processing\n后处理', color='#4FC3F7')
        
        # 添加CRA处理流程
        c.edge('gated_generator', 'contextual_attention')
        c.edge('contextual_attention', 'build_inference')
        c.edge('build_inference', 'post_processing')
        c.edge('post_processing', 'repaired_image')
        
        # 添加结果节点
        c.node('repaired_image', '修复后的图像', shape='box', color='#01579B', fontcolor='white')
    
    # 添加子图 - SRGAN模块
    with dot.subgraph(name='cluster_srgan') as c:
        c.attr(label='SRGAN超分辨率模块', style='filled', color=colors['srgan'], fontcolor='black')
        
        # 添加SRGAN模型核心组件
        c.node('generator', 'Generator\n生成器', color='#2E7D32', fontcolor='white')
        c.node('residual_blocks', 'ResidualBlock x16\n残差块', color='#388E3C')
        c.node('subpixel_conv', 'SubpixelConvolution\n子像素卷积层', color='#43A047')
        c.node('sr_post_processing', '超分后处理', color='#66BB6A')
        
        # 添加SRGAN处理流程
        c.edge('generator', 'residual_blocks')
        c.edge('residual_blocks', 'subpixel_conv')
        c.edge('subpixel_conv', 'sr_post_processing')
        c.edge('sr_post_processing', 'enhanced_image')
        
        # 添加结果节点
        c.node('enhanced_image', '超分辨率图像', shape='box', color='#1B5E20', fontcolor='white')

    # 添加子图 - 输出模块
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='输出模块', style='filled', color=colors['output'], fontcolor='black')
        c.node('save_results', '保存结果', color='#E64A19', fontcolor='white')
        c.node('visualization', '结果可视化', color='#F4511E', fontcolor='white')
        c.node('metrics', '质量评估指标\n(PSNR, SSIM)', color='#FF5722')
    
    # 添加整体数据流
    dot.edge('image_preprocessing', 'gated_generator')
    dot.edge('mask_preprocessing', 'gated_generator')
    dot.edge('repaired_image', 'generator')
    dot.edge('enhanced_image', 'save_results')
    dot.edge('enhanced_image', 'visualization')
    dot.edge('enhanced_image', 'metrics')
    
    # 添加联合系统节点
    dot.node('combined_system', 'combined_repair_sr.py\n联合处理系统', shape='box', style='filled,rounded', 
            color='#6A1B9A', fontcolor='white')
    
    # 核心模型参数
    dot.node('cra_params', 'CRA模型参数\ninput_size, times, etc.', shape='note', color=colors['data'])
    dot.node('srgan_params', 'SRGAN模型参数\nupscale_factor, etc.', shape='note', color=colors['data'])
    
    # 添加模型参数连接
    dot.edge('cra_params', 'gated_generator', style='dashed')
    dot.edge('srgan_params', 'generator', style='dashed')
    
    # 添加MindSpore框架
    dot.node('mindspore', 'MindSpore框架', shape='component', style='filled', color='#D1C4E9')
    dot.edge('mindspore', 'combined_system', style='dashed', constraint='false')
    
    # 渲染图像
    dot.render('project_architecture', format='png', cleanup=True)
    print(f"已生成架构图: {os.path.abspath('project_architecture.png')}")
    
    return dot

if __name__ == "__main__":
    try:
        # 检查是否安装了graphviz
        import graphviz
    except ImportError:
        print("未找到graphviz库。请先手动安装系统 Graphviz，并运行: python -m pip install graphviz")
        exit(1)
    
    # 生成架构图
    create_architecture_diagram() 
