#!/usr/bin/env python3
import graphviz as gv
import os

def create_data_flow_diagram():
    """创建项目数据流架构图，分别展示训练和推理流程"""
    # 创建有向图
    dot = gv.Digraph('DataFlow', comment='CRA与SRGAN的训练与推理数据流')
    
    # 设置图像属性
    dot.attr(rankdir='LR', size='14,10', ratio='fill', fontname='SimHei')
    dot.attr('node', shape='box', style='filled', fontname='SimHei')
    
    # 定义节点颜色
    colors = {
        'dataset': '#E8EAF6',   # 数据集颜色
        'model': '#E3F2FD',     # 模型颜色
        'loss': '#FFECB3',      # 损失函数颜色
        'train': '#E0F2F1',     # 训练颜色
        'infer': '#F9FBE7',     # 推理颜色
        'output': '#FFCCBC',    # 输出颜色
        'process': '#F3E5F5'    # 处理颜色
    }
    
    # ===== 训练流程 =====
    with dot.subgraph(name='cluster_train') as c:
        c.attr(label='训练流程', style='filled', color=colors['train'], fontcolor='black')
        
        # CRA训练流程
        with c.subgraph(name='cluster_cra_train') as cra:
            cra.attr(label='CRA训练', style='filled', color='#B3E5FC', fontcolor='black')
            
            # 数据集
            cra.node('cra_dataset', 'Places365/Paris StreetView\n训练数据集', shape='cylinder', color='#0D47A1', fontcolor='white')
            
            # 随机掩码生成
            cra.node('cra_mask_gen', '随机掩码生成器', color='#29B6F6')
            
            # 模型组件
            cra.node('cra_gen', 'GatedGenerator', color='#0288D1', fontcolor='white')
            cra.node('cra_disc', 'Discriminator', color='#0288D1', fontcolor='white')
            
            # 损失函数
            cra.node('cra_l1_loss', 'L1 Loss', shape='diamond', color='#FFC107')
            cra.node('cra_adv_loss', 'Adversarial Loss', shape='diamond', color='#FFC107')
            cra.node('cra_perc_loss', 'Perceptual Loss', shape='diamond', color='#FFC107')
            
            # 优化器
            cra.node('cra_optimizer', '优化器 (Adam)', color='#26A69A')
            
            # 数据流
            cra.edge('cra_dataset', 'cra_mask_gen')
            cra.edge('cra_mask_gen', 'cra_gen')
            cra.edge('cra_gen', 'cra_disc')
            cra.edge('cra_gen', 'cra_l1_loss')
            cra.edge('cra_disc', 'cra_adv_loss')
            cra.edge('cra_gen', 'cra_perc_loss')
            cra.edge('cra_l1_loss', 'cra_optimizer')
            cra.edge('cra_adv_loss', 'cra_optimizer')
            cra.edge('cra_perc_loss', 'cra_optimizer')
            cra.edge('cra_optimizer', 'cra_gen', label='更新参数')
            cra.edge('cra_optimizer', 'cra_disc', label='更新参数')
            
            # 检查点保存
            cra.node('cra_ckpt', 'CRA检查点', shape='note', color='#01579B', fontcolor='white')
            cra.edge('cra_gen', 'cra_ckpt', style='dashed')
            
        # SRGAN训练流程
        with c.subgraph(name='cluster_srgan_train') as srgan:
            srgan.attr(label='SRGAN训练', style='filled', color='#C8E6C9', fontcolor='black')
            
            # 数据集
            srgan.node('srgan_dataset', 'DIV2K/ImageNet\n训练数据集', shape='cylinder', color='#1B5E20', fontcolor='white')
            
            # 下采样处理
            srgan.node('srgan_downsample', '图像下采样\n(生成低分辨率)', color='#66BB6A')
            
            # 模型组件
            srgan.node('srgan_gen', 'Generator', color='#2E7D32', fontcolor='white')
            srgan.node('srgan_disc', 'Discriminator', color='#2E7D32', fontcolor='white')
            
            # 损失函数
            srgan.node('srgan_pixel_loss', 'Pixel Loss', shape='diamond', color='#FFC107')
            srgan.node('srgan_adv_loss', 'Adversarial Loss', shape='diamond', color='#FFC107')
            srgan.node('srgan_perc_loss', 'Perceptual Loss', shape='diamond', color='#FFC107')
            
            # 优化器
            srgan.node('srgan_optimizer', '优化器 (Adam)', color='#26A69A')
            
            # 数据流
            srgan.edge('srgan_dataset', 'srgan_downsample')
            srgan.edge('srgan_downsample', 'srgan_gen')
            srgan.edge('srgan_gen', 'srgan_disc')
            srgan.edge('srgan_gen', 'srgan_pixel_loss')
            srgan.edge('srgan_disc', 'srgan_adv_loss')
            srgan.edge('srgan_gen', 'srgan_perc_loss')
            srgan.edge('srgan_pixel_loss', 'srgan_optimizer')
            srgan.edge('srgan_adv_loss', 'srgan_optimizer')
            srgan.edge('srgan_perc_loss', 'srgan_optimizer')
            srgan.edge('srgan_optimizer', 'srgan_gen', label='更新参数')
            srgan.edge('srgan_optimizer', 'srgan_disc', label='更新参数')
            
            # 检查点保存
            srgan.node('srgan_ckpt', 'SRGAN检查点', shape='note', color='#1B5E20', fontcolor='white')
            srgan.edge('srgan_gen', 'srgan_ckpt', style='dashed')
    
    # ===== 推理流程 =====
    with dot.subgraph(name='cluster_infer') as c:
        c.attr(label='推理流程', style='filled', color=colors['infer'], fontcolor='black')
        
        # 输入处理
        c.node('input_image', '输入图像', color='#F57F17', fontcolor='white')
        c.node('input_mask', '输入掩码\n(或交互式创建)', color='#F57F17', fontcolor='white')
        
        # CRA推理流程
        with c.subgraph(name='cluster_cra_infer') as cra:
            cra.attr(label='CRA推理', style='filled', color='#B3E5FC', fontcolor='black')
            
            # 加载模型
            cra.node('cra_load_ckpt', '加载CRA检查点', color='#01579B', fontcolor='white')
            cra.edge('cra_ckpt', 'cra_load_ckpt', style='dashed')
            
            # 预处理
            cra.node('cra_preprocess', '图像预处理\n(调整大小，归一化)', color='#29B6F6')
            
            # 推理过程
            cra.node('cra_inference', '执行CRA推理', color='#0288D1', fontcolor='white')
            cra.node('cra_attention', '应用上下文注意力', color='#0288D1', fontcolor='white')
            cra.node('cra_postprocess', '后处理', color='#29B6F6')
            
            # 修复结果
            cra.node('cra_output', '修复后的图像', color='#01579B', fontcolor='white')
            
            # 数据流
            cra.edge('cra_load_ckpt', 'cra_inference')
            cra.edge('cra_preprocess', 'cra_inference')
            cra.edge('cra_inference', 'cra_attention')
            cra.edge('cra_attention', 'cra_postprocess')
            cra.edge('cra_postprocess', 'cra_output')
        
        # SRGAN推理流程
        with c.subgraph(name='cluster_srgan_infer') as srgan:
            srgan.attr(label='SRGAN推理', style='filled', color='#C8E6C9', fontcolor='black')
            
            # 加载模型
            srgan.node('srgan_load_ckpt', '加载SRGAN检查点', color='#1B5E20', fontcolor='white')
            srgan.edge('srgan_ckpt', 'srgan_load_ckpt', style='dashed')
            
            # 预处理
            srgan.node('srgan_preprocess', '图像预处理\n(归一化)', color='#66BB6A')
            
            # 推理过程
            srgan.node('srgan_inference', '执行SRGAN推理', color='#2E7D32', fontcolor='white')
            srgan.node('srgan_postprocess', '后处理', color='#66BB6A')
            
            # 超分结果
            srgan.node('srgan_output', '超分辨率图像', color='#1B5E20', fontcolor='white')
            
            # 数据流
            srgan.edge('srgan_load_ckpt', 'srgan_inference')
            srgan.edge('srgan_preprocess', 'srgan_inference')
            srgan.edge('srgan_inference', 'srgan_postprocess')
            srgan.edge('srgan_postprocess', 'srgan_output')
        
        # 结果处理
        c.node('save_results', '保存结果', color='#E64A19', fontcolor='white')
        c.node('evaluate', '评估质量\n(PSNR, SSIM)', color='#F4511E', fontcolor='white')
        
        # 连接输入到推理流程
        c.edge('input_image', 'cra_preprocess')
        c.edge('input_mask', 'cra_preprocess')
        c.edge('cra_output', 'srgan_preprocess')
        c.edge('srgan_output', 'save_results')
        c.edge('srgan_output', 'evaluate')
    
    # 连接框架
    dot.node('combined_system', 'combined_repair_sr.py\n联合处理系统', shape='component', color='#6A1B9A', fontcolor='white')
    dot.edge('input_image', 'combined_system', style='dashed')
    dot.edge('input_mask', 'combined_system', style='dashed')
    dot.edge('combined_system', 'save_results', style='dashed')
    
    dot.node('mindspore', 'MindSpore框架', shape='component', color='#D1C4E9')
    dot.edge('mindspore', 'combined_system', style='dashed')
    
    # 渲染图像
    dot.render('data_flow_diagram', format='png', cleanup=True)
    print(f"已生成数据流图: {os.path.abspath('data_flow_diagram.png')}")
    
    return dot

if __name__ == "__main__":
    try:
        # 检查是否安装了graphviz
        import graphviz
    except ImportError:
        print("未找到graphviz库。请先手动安装系统 Graphviz，并运行: python -m pip install graphviz")
        exit(1)
    
    # 生成数据流图
    create_data_flow_diagram() 
