#!/usr/bin/env python3
import os
import graphviz

def create_data_preprocessing_training_diagram():
    """创建数据预处理和模型训练流程图"""
    # 检查graphviz是否安装
    try:
        dot = graphviz.Digraph('数据预处理和训练流程图', format='png')
        # 设置更高的DPI以提高清晰度
        dot.attr(rankdir='TB', size='14,10', ratio='fill', fontname='SimHei', dpi='300')
        
        # 设置节点样式
        dot.attr('node', shape='box', style='filled,rounded', fontname='SimHei', fontsize='14')
        
        # 定义颜色方案
        colors = {
            'data': '#E6F2FF',           # 浅蓝色 - 数据相关
            'preprocessing': '#FFE6E6',  # 浅红色 - 预处理相关
            'model': '#E6FFE6',          # 浅绿色 - 模型相关
            'training': '#FFFDE6',       # 浅黄色 - 训练相关
            'output': '#F2E6FF',         # 浅紫色 - 输出相关
            'framework': '#F2F2F2'       # 浅灰色 - 框架组件
        }
        
        # 创建CRA训练子图
        with dot.subgraph(name='cluster_cra_training') as c:
            c.attr(label='CRA 模型训练流程', style='filled', fillcolor='#F0F8FF', fontsize='16', fontname='SimHei')
            
            # 数据加载和预处理
            c.node('cra_data', '损坏图像数据集', fillcolor=colors['data'])
            c.node('cra_masks', '损坏区域掩码', fillcolor=colors['data'])
            c.node('cra_gt', '原始图像（GT）', fillcolor=colors['data'])
            
            c.node('cra_data_loading', '数据加载\n读取损坏图像、掩码和GT', fillcolor=colors['preprocessing'])
            c.node('cra_data_augment', '数据增强\n随机裁剪、翻转、旋转', fillcolor=colors['preprocessing'])
            c.node('cra_batch_gen', '批处理生成\nMindSpore DataLoader', fillcolor=colors['preprocessing'])
            
            # 模型初始化和训练
            c.node('cra_model_init', 'CRA模型初始化\n创建门控卷积网络和上下文注意力模块', fillcolor=colors['model'])
            c.node('cra_optimizer', '优化器初始化\nAdam优化器', fillcolor=colors['model'])
            
            c.node('cra_training_loop', 'CRA训练循环', fillcolor=colors['training'])
            c.node('cra_forward', '前向传播\n生成修复图像', fillcolor=colors['training'])
            c.node('cra_loss_calc', '损失计算\n重建损失 + 对抗损失', fillcolor=colors['training'])
            c.node('cra_backward', '反向传播\n计算梯度', fillcolor=colors['training'])
            c.node('cra_optimize', '参数更新\n应用梯度', fillcolor=colors['training'])
            c.node('cra_checkpoint', '保存检查点\n定期保存模型参数', fillcolor=colors['output'])
            
            # 连接节点
            c.edges([
                ('cra_data', 'cra_data_loading'),
                ('cra_masks', 'cra_data_loading'),
                ('cra_gt', 'cra_data_loading'),
                ('cra_data_loading', 'cra_data_augment'),
                ('cra_data_augment', 'cra_batch_gen'),
                ('cra_batch_gen', 'cra_training_loop'),
                ('cra_model_init', 'cra_training_loop'),
                ('cra_optimizer', 'cra_training_loop'),
                ('cra_training_loop', 'cra_forward'),
                ('cra_forward', 'cra_loss_calc'),
                ('cra_loss_calc', 'cra_backward'),
                ('cra_backward', 'cra_optimize'),
                ('cra_optimize', 'cra_checkpoint'),
                ('cra_optimize', 'cra_training_loop')  # 回到训练循环
            ])
        
        # 创建SRGAN训练子图
        with dot.subgraph(name='cluster_srgan_training') as c:
            c.attr(label='SRGAN 模型训练流程', style='filled', fillcolor='#F0FFF0', fontsize='16', fontname='SimHei')
            
            # 数据加载和预处理
            c.node('srgan_lr_data', '低分辨率图像数据集', fillcolor=colors['data'])
            c.node('srgan_hr_data', '高分辨率图像（GT）', fillcolor=colors['data'])
            
            c.node('srgan_data_loading', '数据加载\n读取LR和HR图像对', fillcolor=colors['preprocessing'])
            c.node('srgan_data_augment', '数据增强\n随机裁剪、翻转、旋转', fillcolor=colors['preprocessing'])
            c.node('srgan_batch_gen', '批处理生成\nMindSpore DataLoader', fillcolor=colors['preprocessing'])
            
            # 模型初始化和训练
            c.node('srgan_g_init', 'SRGAN生成器初始化\n残差块和上采样层', fillcolor=colors['model'])
            c.node('srgan_d_init', 'SRGAN判别器初始化\n卷积层和LeakyReLU', fillcolor=colors['model'])
            c.node('srgan_g_optimizer', '生成器优化器\nAdam优化器', fillcolor=colors['model'])
            c.node('srgan_d_optimizer', '判别器优化器\nAdam优化器', fillcolor=colors['model'])
            
            c.node('srgan_training_loop', 'SRGAN训练循环', fillcolor=colors['training'])
            
            # 判别器训练
            c.node('srgan_d_train', '判别器训练', fillcolor=colors['training'])
            c.node('srgan_g_forward', '生成器前向传播\n生成SR图像', fillcolor=colors['training'])
            c.node('srgan_d_forward', '判别器前向传播\n预测真假', fillcolor=colors['training'])
            c.node('srgan_d_loss', '判别器损失计算\n二分类损失', fillcolor=colors['training'])
            c.node('srgan_d_backward', '判别器反向传播', fillcolor=colors['training'])
            c.node('srgan_d_optimize', '判别器参数更新', fillcolor=colors['training'])
            
            # 生成器训练
            c.node('srgan_g_train', '生成器训练', fillcolor=colors['training'])
            c.node('srgan_g_loss', '生成器损失计算\n内容损失 + 对抗损失', fillcolor=colors['training'])
            c.node('srgan_g_backward', '生成器反向传播', fillcolor=colors['training'])
            c.node('srgan_g_optimize', '生成器参数更新', fillcolor=colors['training'])
            
            c.node('srgan_checkpoint', '保存检查点\n定期保存模型参数', fillcolor=colors['output'])
            
            # 连接节点
            c.edges([
                ('srgan_lr_data', 'srgan_data_loading'),
                ('srgan_hr_data', 'srgan_data_loading'),
                ('srgan_data_loading', 'srgan_data_augment'),
                ('srgan_data_augment', 'srgan_batch_gen'),
                ('srgan_batch_gen', 'srgan_training_loop'),
                ('srgan_g_init', 'srgan_training_loop'),
                ('srgan_d_init', 'srgan_training_loop'),
                ('srgan_g_optimizer', 'srgan_training_loop'),
                ('srgan_d_optimizer', 'srgan_training_loop'),
                
                # 判别器训练流程
                ('srgan_training_loop', 'srgan_d_train'),
                ('srgan_d_train', 'srgan_g_forward'),
                ('srgan_g_forward', 'srgan_d_forward'),
                ('srgan_d_forward', 'srgan_d_loss'),
                ('srgan_d_loss', 'srgan_d_backward'),
                ('srgan_d_backward', 'srgan_d_optimize'),
                ('srgan_d_optimize', 'srgan_g_train'),
                
                # 生成器训练流程
                ('srgan_g_train', 'srgan_g_forward'),
                ('srgan_g_forward', 'srgan_d_forward'),
                ('srgan_d_forward', 'srgan_g_loss'),
                ('srgan_g_loss', 'srgan_g_backward'),
                ('srgan_g_backward', 'srgan_g_optimize'),
                ('srgan_g_optimize', 'srgan_checkpoint'),
                ('srgan_g_optimize', 'srgan_training_loop')  # 回到训练循环
            ])
        
        # 添加共享组件
        dot.node('mindspore', 'MindSpore 框架\n深度学习训练框架', fillcolor=colors['framework'], shape='component', fontname='SimHei')
        
        # 连接共享组件
        dot.edge('mindspore', 'cra_model_init', style='dashed')
        dot.edge('mindspore', 'cra_optimizer', style='dashed')
        dot.edge('mindspore', 'srgan_g_init', style='dashed')
        dot.edge('mindspore', 'srgan_d_init', style='dashed')
        dot.edge('mindspore', 'srgan_g_optimizer', style='dashed')
        dot.edge('mindspore', 'srgan_d_optimizer', style='dashed')
        
        # 渲染图形
        output_filename = '数据预处理和训练流程图'
        dot.render(output_filename, cleanup=True)
        print(f"数据预处理和训练流程图已生成: {os.path.abspath(output_filename + '.png')}")
        
        # 保存英文文件名版本作为备份
        en_output_filename = 'data_preprocessing_training'
        if os.path.exists(output_filename + '.png'):
            import shutil
            shutil.copy2(output_filename + '.png', en_output_filename + '.png')
        
        return output_filename + '.png'
    except ImportError:
        print("错误: 未找到graphviz库。请执行 'pip install graphviz' 并确保系统已安装graphviz。")
        return None

# 如果直接运行此脚本
if __name__ == "__main__":
    create_data_preprocessing_training_diagram() 