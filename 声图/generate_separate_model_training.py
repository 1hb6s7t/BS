#!/usr/bin/env python3
import os
import graphviz
import shutil
from pathlib import Path

OUTPUT_DIR = str(Path(__file__).resolve().parents[1] / "模型流程图")

def create_cra_training_diagram():
    """创建CRA模型训练流程图"""
    try:
        # 确保输出目录存在
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        dot = graphviz.Digraph('CRA模型训练流程图', format='png')
        # 设置更高的DPI和图形尺寸以提高清晰度
        dot.attr(rankdir='TB', size='18,14', ratio='fill', fontname='SimHei', dpi='600')
        
        # 设置节点样式
        dot.attr('node', shape='box', style='filled,rounded', fontname='SimHei', fontsize='16', 
                 margin='0.3,0.2', width='2.2', height='1.2')
        
        # 定义颜色方案
        colors = {
            'data': '#E0F7FA',      # 浅青色 - 数据相关
            'model': '#F3E5F5',     # 浅紫色 - 模型组件
            'train': '#FFF3E0',     # 浅橙色 - 训练步骤
            'loss': '#FFEBEE',      # 浅红色 - 损失函数
            'output': '#E8F5E9',    # 浅绿色 - 输出结果
            'arrow': '#37474F'      # 深灰色 - 箭头
        }
        
        # 设置边的样式
        dot.attr('edge', color=colors['arrow'], fontname='SimHei', fontsize='14', 
                 penwidth='1.5')
        
        # 创建数据加载和预处理节点
        with dot.subgraph(name='cluster_data') as c:
            c.attr(label='数据准备', style='filled', fillcolor='#F0F0F0', fontsize='18', fontname='SimHei')
            
            c.node('dataset', '训练数据集\n损坏/未损坏的图像对', 
                   fillcolor=colors['data'], style='filled,rounded', shape='folder')
            c.node('create_dataset', '创建数据集对象\n包装为MindSpore数据集', 
                   fillcolor=colors['data'])
            c.node('data_loader', '数据加载器\n批量加载并预处理数据', 
                   fillcolor=colors['data'])
            c.node('data_augment', '数据增强\n旋转、翻转、调整大小等', 
                   fillcolor=colors['data'])
            c.node('normalize', '数据归一化\n将像素值缩放到[-1,1]', 
                   fillcolor=colors['data'])
        
        # 创建模型初始化节点
        with dot.subgraph(name='cluster_model_init') as c:
            c.attr(label='模型初始化', style='filled', fillcolor='#F8F8F8', fontsize='18', fontname='SimHei')
            
            c.node('create_model', '创建CRA模型\n初始化门控卷积网络结构', 
                   fillcolor=colors['model'])
            c.node('gated_conv_init', '门控卷积层初始化\n处理输入特征', 
                   fillcolor=colors['model'])
            c.node('attn_init', '上下文注意力初始化\n关注未损坏区域特征', 
                   fillcolor=colors['model'])
            c.node('init_weights', '参数初始化\n使用Xavier或Normal初始化权重', 
                   fillcolor=colors['model'])
        
        # 创建训练循环节点
        with dot.subgraph(name='cluster_training') as c:
            c.attr(label='训练循环', style='filled', fillcolor='#FFF8E1', fontsize='18', fontname='SimHei')
            
            c.node('train_loop', '训练循环\n迭代数据集进行训练', 
                   fillcolor=colors['train'])
            c.node('forward_pass', '前向传播\n计算修复结果', 
                   fillcolor=colors['train'])
            c.node('calc_loss', '计算损失\n逐像素损失和感知损失', 
                   fillcolor=colors['loss'])
            c.node('backward_pass', '反向传播\n计算梯度', 
                   fillcolor=colors['train'])
            c.node('update_params', '更新参数\n使用Adam优化器', 
                   fillcolor=colors['train'])
        
        # 创建损失计算节点
        with dot.subgraph(name='cluster_loss') as c:
            c.attr(label='损失计算', style='filled', fillcolor='#FFEBEE', fontsize='18', fontname='SimHei')
            
            c.node('l1_loss', 'L1损失\n修复结果与真实图像的绝对差', 
                   fillcolor=colors['loss'])
            c.node('perceptual_loss', '感知损失\n使用VGG特征的差异', 
                   fillcolor=colors['loss'])
            c.node('style_loss', '风格损失\n保持图像风格一致性', 
                   fillcolor=colors['loss'])
            c.node('total_loss', '总损失\n加权合并各损失项', 
                   fillcolor=colors['loss'])
        
        # 创建评估和保存节点
        with dot.subgraph(name='cluster_output') as c:
            c.attr(label='评估与保存', style='filled', fillcolor='#E8F5E9', fontsize='18', fontname='SimHei')
            
            c.node('validation', '验证评估\n在验证集上评估模型', 
                   fillcolor=colors['output'])
            c.node('metrics', '计算指标\nPSNR和SSIM', 
                   fillcolor=colors['output'])
            c.node('save_ckpt', '保存检查点\n保存模型参数', 
                   fillcolor=colors['output'])
            c.node('viz_results', '可视化结果\n显示修复效果', 
                   fillcolor=colors['output'])
        
        # 连接数据节点
        dot.edge('dataset', 'create_dataset')
        dot.edge('create_dataset', 'data_loader')
        dot.edge('data_loader', 'data_augment')
        dot.edge('data_augment', 'normalize')
        
        # 连接模型初始化节点
        dot.edge('create_model', 'gated_conv_init')
        dot.edge('create_model', 'attn_init')
        dot.edge('gated_conv_init', 'init_weights')
        dot.edge('attn_init', 'init_weights')
        
        # 连接训练循环节点
        dot.edge('normalize', 'train_loop')
        dot.edge('init_weights', 'train_loop')
        dot.edge('train_loop', 'forward_pass')
        dot.edge('forward_pass', 'calc_loss')
        dot.edge('calc_loss', 'backward_pass')
        dot.edge('backward_pass', 'update_params')
        dot.edge('update_params', 'train_loop', label='迭代下一批次')
        
        # 连接损失计算节点
        dot.edge('calc_loss', 'l1_loss')
        dot.edge('calc_loss', 'perceptual_loss')
        dot.edge('calc_loss', 'style_loss')
        dot.edge('l1_loss', 'total_loss')
        dot.edge('perceptual_loss', 'total_loss')
        dot.edge('style_loss', 'total_loss')
        
        # 连接评估和保存节点
        dot.edge('train_loop', 'validation', label='每N个epoch')
        dot.edge('validation', 'metrics')
        dot.edge('metrics', 'save_ckpt', label='如果指标提升')
        dot.edge('metrics', 'viz_results')
        
        # 添加训练框架和环境节点
        dot.node('mindspore', 'MindSpore\n训练框架', shape='component', 
                 style='filled', fillcolor='#E1F5FE', fontname='SimHei')
        dot.edge('mindspore', 'train_loop', style='dashed')
        dot.edge('mindspore', 'backward_pass', style='dashed')
        
        # 添加训练配置节点
        dot.node('train_config', '训练配置\n学习率, batch_size, epochs等', 
                 shape='note', style='filled', fillcolor='#E0F2F1', fontname='SimHei')
        dot.edge('train_config', 'train_loop', style='dashed')
        dot.edge('train_config', 'create_model', style='dashed')
        
        # 渲染和保存图形
        output_filename = os.path.join(OUTPUT_DIR, 'CRA模型训练流程图')
        dot.render(output_filename, cleanup=True)
        print(f"CRA模型训练流程图已生成: {os.path.abspath(output_filename + '.png')}")
        
        return output_filename + '.png'
    except ImportError:
        print("错误: 未找到graphviz库。请执行 'pip install graphviz' 并确保系统已安装graphviz。")
        return None
    except Exception as e:
        print(f"生成CRA模型训练流程图时出错: {e}")
        return None

def create_srgan_training_diagram():
    """创建SRGAN模型训练流程图"""
    try:
        # 确保输出目录存在
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        dot = graphviz.Digraph('SRGAN模型训练流程图', format='png')
        # 设置更高的DPI和图形尺寸以提高清晰度
        dot.attr(rankdir='TB', size='20,16', ratio='fill', fontname='SimHei', dpi='600')
        
        # 设置节点样式
        dot.attr('node', shape='box', style='filled,rounded', fontname='SimHei', fontsize='16', 
                 margin='0.3,0.2', width='2.2', height='1.2')
        
        # 定义颜色方案
        colors = {
            'data': '#E0F7FA',      # 浅青色 - 数据相关
            'generator': '#F3E5F5',  # 浅紫色 - 生成器相关
            'discriminator': '#FFF3E0', # 浅橙色 - 判别器相关
            'train': '#E3F2FD',     # 浅蓝色 - 训练步骤
            'loss': '#FFEBEE',      # 浅红色 - 损失函数
            'output': '#E8F5E9',    # 浅绿色 - 输出结果
            'arrow': '#37474F'      # 深灰色 - 箭头
        }
        
        # 设置边的样式
        dot.attr('edge', color=colors['arrow'], fontname='SimHei', fontsize='14', 
                 penwidth='1.5')
        
        # 创建数据加载和预处理节点
        with dot.subgraph(name='cluster_data') as c:
            c.attr(label='数据准备', style='filled', fillcolor='#F0F0F0', fontsize='18', fontname='SimHei')
            
            c.node('dataset', '训练数据集\n高/低分辨率图像对', 
                   fillcolor=colors['data'], style='filled,rounded', shape='folder')
            c.node('create_dataset', '创建数据集对象\n包装为MindSpore数据集', 
                   fillcolor=colors['data'])
            c.node('data_loader', '数据加载器\n批量加载并预处理数据', 
                   fillcolor=colors['data'])
            c.node('data_augment', '数据增强\n随机裁剪、翻转等', 
                   fillcolor=colors['data'])
            c.node('normalize', '数据归一化\n将像素值缩放到[-1,1]', 
                   fillcolor=colors['data'])
        
        # 创建生成器初始化节点
        with dot.subgraph(name='cluster_generator') as c:
            c.attr(label='生成器初始化', style='filled', fillcolor='#F3E5F5', fontsize='18', fontname='SimHei')
            
            c.node('create_generator', '创建生成器模型\n残差网络结构', 
                   fillcolor=colors['generator'])
            c.node('residual_blocks', '残差块初始化\n提取深层特征', 
                   fillcolor=colors['generator'])
            c.node('upsampling_init', '上采样层初始化\n增加空间分辨率', 
                   fillcolor=colors['generator'])
            c.node('g_init_weights', '生成器参数初始化\n使用正态分布初始化', 
                   fillcolor=colors['generator'])
        
        # 创建判别器初始化节点
        with dot.subgraph(name='cluster_discriminator') as c:
            c.attr(label='判别器初始化', style='filled', fillcolor='#FFF3E0', fontsize='18', fontname='SimHei')
            
            c.node('create_discriminator', '创建判别器模型\n卷积神经网络', 
                   fillcolor=colors['discriminator'])
            c.node('conv_layers', '卷积层初始化\n层次化特征提取', 
                   fillcolor=colors['discriminator'])
            c.node('dense_layers', '全连接层初始化\n二分类输出', 
                   fillcolor=colors['discriminator'])
            c.node('d_init_weights', '判别器参数初始化\n使用正态分布初始化', 
                   fillcolor=colors['discriminator'])
        
        # 创建训练循环节点
        with dot.subgraph(name='cluster_training') as c:
            c.attr(label='训练循环', style='filled', fillcolor='#E3F2FD', fontsize='18', fontname='SimHei')
            
            c.node('train_loop', '训练循环\n迭代数据集进行训练', 
                   fillcolor=colors['train'])
            c.node('g_forward', '生成器前向传播\n生成超分辨率图像', 
                   fillcolor=colors['generator'])
            c.node('d_forward_real', '判别器前向传播 (真实)\n真实高分辨率图像的判别', 
                   fillcolor=colors['discriminator'])
            c.node('d_forward_fake', '判别器前向传播 (生成)\n生成高分辨率图像的判别', 
                   fillcolor=colors['discriminator'])
            c.node('g_loss_calc', '计算生成器损失\n对抗损失和内容损失', 
                   fillcolor=colors['loss'])
            c.node('d_loss_calc', '计算判别器损失\n二分类交叉熵损失', 
                   fillcolor=colors['loss'])
            c.node('g_backward', '生成器反向传播\n计算梯度', 
                   fillcolor=colors['generator'])
            c.node('d_backward', '判别器反向传播\n计算梯度', 
                   fillcolor=colors['discriminator'])
            c.node('g_update', '更新生成器参数\n使用Adam优化器', 
                   fillcolor=colors['generator'])
            c.node('d_update', '更新判别器参数\n使用Adam优化器', 
                   fillcolor=colors['discriminator'])
        
        # 创建损失计算节点
        with dot.subgraph(name='cluster_loss') as c:
            c.attr(label='损失计算', style='filled', fillcolor='#FFEBEE', fontsize='18', fontname='SimHei')
            
            c.node('content_loss', '内容损失\nMSE或VGG特征的差异', 
                   fillcolor=colors['loss'])
            c.node('adversarial_loss', '对抗损失\n生成器欺骗判别器的能力', 
                   fillcolor=colors['loss'])
            c.node('total_g_loss', '总生成器损失\n加权合并各损失项', 
                   fillcolor=colors['loss'])
            c.node('real_d_loss', '真实样本判别损失\n二分类损失', 
                   fillcolor=colors['loss'])
            c.node('fake_d_loss', '生成样本判别损失\n二分类损失', 
                   fillcolor=colors['loss'])
            c.node('total_d_loss', '总判别器损失\n真实和生成样本损失之和', 
                   fillcolor=colors['loss'])
        
        # 创建评估和保存节点
        with dot.subgraph(name='cluster_output') as c:
            c.attr(label='评估与保存', style='filled', fillcolor='#E8F5E9', fontsize='18', fontname='SimHei')
            
            c.node('validation', '验证评估\n在验证集上评估模型', 
                   fillcolor=colors['output'])
            c.node('metrics', '计算指标\nPSNR和SSIM', 
                   fillcolor=colors['output'])
            c.node('save_g_ckpt', '保存生成器检查点\n保存生成器参数', 
                   fillcolor=colors['output'])
            c.node('save_d_ckpt', '保存判别器检查点\n保存判别器参数', 
                   fillcolor=colors['output'])
            c.node('viz_results', '可视化结果\n显示超分辨率效果', 
                   fillcolor=colors['output'])
        
        # 连接数据节点
        dot.edge('dataset', 'create_dataset')
        dot.edge('create_dataset', 'data_loader')
        dot.edge('data_loader', 'data_augment')
        dot.edge('data_augment', 'normalize')
        
        # 连接生成器初始化节点
        dot.edge('create_generator', 'residual_blocks')
        dot.edge('create_generator', 'upsampling_init')
        dot.edge('residual_blocks', 'g_init_weights')
        dot.edge('upsampling_init', 'g_init_weights')
        
        # 连接判别器初始化节点
        dot.edge('create_discriminator', 'conv_layers')
        dot.edge('conv_layers', 'dense_layers')
        dot.edge('dense_layers', 'd_init_weights')
        
        # 连接训练循环节点
        dot.edge('normalize', 'train_loop')
        dot.edge('g_init_weights', 'train_loop')
        dot.edge('d_init_weights', 'train_loop')
        dot.edge('train_loop', 'g_forward')
        dot.edge('g_forward', 'd_forward_fake')
        dot.edge('normalize', 'd_forward_real')
        dot.edge('d_forward_real', 'd_loss_calc')
        dot.edge('d_forward_fake', 'd_loss_calc')
        dot.edge('g_forward', 'g_loss_calc')
        dot.edge('g_loss_calc', 'g_backward')
        dot.edge('d_loss_calc', 'd_backward')
        dot.edge('g_backward', 'g_update')
        dot.edge('d_backward', 'd_update')
        dot.edge('g_update', 'train_loop', label='迭代下一批次')
        dot.edge('d_update', 'train_loop', label='迭代下一批次')
        
        # 连接损失计算节点
        dot.edge('g_loss_calc', 'content_loss')
        dot.edge('g_loss_calc', 'adversarial_loss')
        dot.edge('content_loss', 'total_g_loss')
        dot.edge('adversarial_loss', 'total_g_loss')
        dot.edge('d_loss_calc', 'real_d_loss')
        dot.edge('d_loss_calc', 'fake_d_loss')
        dot.edge('real_d_loss', 'total_d_loss')
        dot.edge('fake_d_loss', 'total_d_loss')
        
        # 连接评估和保存节点
        dot.edge('train_loop', 'validation', label='每N个epoch')
        dot.edge('validation', 'metrics')
        dot.edge('metrics', 'save_g_ckpt', label='如果指标提升')
        dot.edge('metrics', 'save_d_ckpt', label='如果指标提升')
        dot.edge('metrics', 'viz_results')
        
        # 添加训练框架和环境节点
        dot.node('mindspore', 'MindSpore\n训练框架', shape='component', 
                 style='filled', fillcolor='#E1F5FE', fontname='SimHei')
        dot.edge('mindspore', 'train_loop', style='dashed')
        dot.edge('mindspore', 'g_backward', style='dashed')
        dot.edge('mindspore', 'd_backward', style='dashed')
        
        # 添加训练配置节点
        dot.node('train_config', '训练配置\n学习率, batch_size, epochs等', 
                 shape='note', style='filled', fillcolor='#E0F2F1', fontname='SimHei')
        dot.edge('train_config', 'train_loop', style='dashed')
        dot.edge('train_config', 'g_update', style='dashed')
        dot.edge('train_config', 'd_update', style='dashed')
        
        # 渲染和保存图形
        output_filename = os.path.join(OUTPUT_DIR, 'SRGAN模型训练流程图')
        dot.render(output_filename, cleanup=True)
        print(f"SRGAN模型训练流程图已生成: {os.path.abspath(output_filename + '.png')}")
        
        return output_filename + '.png'
    except ImportError:
        print("错误: 未找到graphviz库。请执行 'pip install graphviz' 并确保系统已安装graphviz。")
        return None
    except Exception as e:
        print(f"生成SRGAN模型训练流程图时出错: {e}")
        return None

if __name__ == "__main__":
    print("生成CRA和SRGAN的模型训练流程图...")
    create_cra_training_diagram()
    create_srgan_training_diagram()
    print("模型训练流程图生成完成!") 
