#!/usr/bin/env python3
import os
import graphviz
import shutil
from pathlib import Path

OUTPUT_DIR = str(Path(__file__).resolve().parents[1] / "模型流程图")

def create_cra_data_processing_diagram():
    """创建CRA模型数据处理流程图"""
    try:
        # 确保输出目录存在
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        dot = graphviz.Digraph('CRA数据处理流程图', format='png')
        # 设置更高的DPI和图形尺寸以提高清晰度
        dot.attr(rankdir='TB', size='16,14', ratio='fill', fontname='SimHei', dpi='600')
        
        # 设置节点样式
        dot.attr('node', shape='box', style='filled,rounded', fontname='SimHei', fontsize='16', 
                 margin='0.3,0.2', width='2.2', height='1.2')
        
        # 定义颜色方案
        colors = {
            'data': '#E0F7FA',      # 浅青色 - 原始数据
            'mask': '#FFECB3',      # 浅黄色 - 掩码生成
            'preproc': '#F3E5F5',   # 浅紫色 - 预处理
            'augment': '#FFF3E0',   # 浅橙色 - 数据增强
            'output': '#E8F5E9',    # 浅绿色 - 输出数据
            'arrow': '#37474F'      # 深灰色 - 箭头
        }
        
        # 设置边的样式
        dot.attr('edge', color=colors['arrow'], fontname='SimHei', fontsize='14', 
                 penwidth='1.5')
        
        # 创建数据收集节点
        with dot.subgraph(name='cluster_data_collection') as c:
            c.attr(label='数据收集', style='filled', fillcolor='#F0F0F0', fontsize='18', fontname='SimHei')
            
            c.node('raw_data', '原始图像数据集\n高质量完整图像', 
                   fillcolor=colors['data'], style='filled,rounded', shape='folder')
            c.node('data_selection', '数据筛选\n选择符合条件的图像', 
                   fillcolor=colors['data'])
            c.node('data_splitting', '数据集划分\n训练集、验证集、测试集', 
                   fillcolor=colors['data'])
        
        # 创建掩码生成节点
        with dot.subgraph(name='cluster_mask_gen') as c:
            c.attr(label='掩码生成', style='filled', fillcolor='#FFF8E1', fontsize='18', fontname='SimHei')
            
            c.node('mask_gen', '掩码生成\n模拟图像损坏区域', 
                   fillcolor=colors['mask'])
            c.node('random_mask', '随机掩码\n随机形状和大小', 
                   fillcolor=colors['mask'])
            c.node('edge_mask', '边缘掩码\n图像边缘缺失', 
                   fillcolor=colors['mask'])
            c.node('center_mask', '中心掩码\n图像中心区域缺失', 
                   fillcolor=colors['mask'])
            c.node('custom_mask', '自定义掩码\n特定区域缺失模式', 
                   fillcolor=colors['mask'])
        
        # 创建图像处理节点
        with dot.subgraph(name='cluster_preprocessing') as c:
            c.attr(label='数据预处理', style='filled', fillcolor='#F3F5F7', fontsize='18', fontname='SimHei')
            
            c.node('image_load', '图像加载\n读取原始图像', 
                   fillcolor=colors['preproc'])
            c.node('resize', '尺寸调整\n统一为模型输入尺寸', 
                   fillcolor=colors['preproc'])
            c.node('color_convert', '颜色空间转换\nRGB标准化', 
                   fillcolor=colors['preproc'])
            c.node('normalize', '数据归一化\n像素值缩放到[-1,1]', 
                   fillcolor=colors['preproc'])
            c.node('apply_mask', '应用掩码\n创建受损图像', 
                   fillcolor=colors['preproc'])
        
        # 创建数据增强节点
        with dot.subgraph(name='cluster_augmentation') as c:
            c.attr(label='数据增强', style='filled', fillcolor='#FFF3E0', fontsize='18', fontname='SimHei')
            
            c.node('augmentation', '数据增强\n扩充训练数据集', 
                   fillcolor=colors['augment'])
            c.node('flip', '翻转\n水平/垂直翻转', 
                   fillcolor=colors['augment'])
            c.node('rotate', '旋转\n随机角度旋转', 
                   fillcolor=colors['augment'])
            c.node('crop', '裁剪\n随机区域裁剪', 
                   fillcolor=colors['augment'])
            c.node('color_jitter', '颜色抖动\n亮度/对比度/饱和度变化', 
                   fillcolor=colors['augment'])
        
        # 创建数据加载节点
        with dot.subgraph(name='cluster_data_loading') as c:
            c.attr(label='数据加载', style='filled', fillcolor='#E8F5E9', fontsize='18', fontname='SimHei')
            
            c.node('batch_generation', '批次生成\n生成训练批次', 
                   fillcolor=colors['output'])
            c.node('pair_creation', '创建训练对\n(损坏图像,掩码,原始图像)', 
                   fillcolor=colors['output'])
            c.node('data_loader', '数据加载器\nMindSpore数据加载', 
                   fillcolor=colors['output'])
            c.node('to_tensor', '张量转换\n转换为MindSpore张量', 
                   fillcolor=colors['output'])
            c.node('shuffle', '数据打乱\n随机化训练样本顺序', 
                   fillcolor=colors['output'])
        
        # 连接数据收集节点
        dot.edge('raw_data', 'data_selection')
        dot.edge('data_selection', 'data_splitting')
        
        # 连接掩码生成节点
        dot.edge('mask_gen', 'random_mask')
        dot.edge('mask_gen', 'edge_mask')
        dot.edge('mask_gen', 'center_mask')
        dot.edge('mask_gen', 'custom_mask')
        
        # 连接图像处理节点
        dot.edge('data_splitting', 'image_load')
        dot.edge('image_load', 'resize')
        dot.edge('resize', 'color_convert')
        dot.edge('color_convert', 'normalize')
        dot.edge('normalize', 'apply_mask')
        dot.edge('random_mask', 'apply_mask')
        dot.edge('edge_mask', 'apply_mask')
        dot.edge('center_mask', 'apply_mask')
        dot.edge('custom_mask', 'apply_mask')
        
        # 连接数据增强节点
        dot.edge('apply_mask', 'augmentation')
        dot.edge('augmentation', 'flip')
        dot.edge('augmentation', 'rotate')
        dot.edge('augmentation', 'crop')
        dot.edge('augmentation', 'color_jitter')
        
        # 连接数据加载节点
        dot.edge('flip', 'pair_creation')
        dot.edge('rotate', 'pair_creation')
        dot.edge('crop', 'pair_creation')
        dot.edge('color_jitter', 'pair_creation')
        dot.edge('apply_mask', 'pair_creation', label='无增强')
        dot.edge('pair_creation', 'to_tensor')
        dot.edge('to_tensor', 'batch_generation')
        dot.edge('batch_generation', 'shuffle')
        dot.edge('shuffle', 'data_loader')
        
        # 添加MindSpore节点
        dot.node('mindspore_dataset', 'MindSpore Dataset API', shape='component', 
                 style='filled', fillcolor='#E1F5FE', fontname='SimHei')
        dot.edge('mindspore_dataset', 'batch_generation', style='dashed')
        dot.edge('mindspore_dataset', 'data_loader', style='dashed')
        
        # 添加配置信息节点
        dot.node('dataset_config', '数据集配置\nbatch_size、shuffle、repeat等', 
                 shape='note', style='filled', fillcolor='#E0F2F1', fontname='SimHei')
        dot.edge('dataset_config', 'data_loader', style='dashed')
        
        # 渲染和保存图形
        output_filename = os.path.join(OUTPUT_DIR, 'CRA数据处理流程图')
        dot.render(output_filename, cleanup=True)
        print(f"CRA数据处理流程图已生成: {os.path.abspath(output_filename + '.png')}")
        
        return output_filename + '.png'
    except ImportError:
        print("错误: 未找到graphviz库。请执行 'pip install graphviz' 并确保系统已安装graphviz。")
        return None
    except Exception as e:
        print(f"生成CRA数据处理流程图时出错: {e}")
        return None

def create_srgan_data_processing_diagram():
    """创建SRGAN模型数据处理流程图"""
    try:
        # 确保输出目录存在
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        dot = graphviz.Digraph('SRGAN数据处理流程图', format='png')
        # 设置更高的DPI和图形尺寸以提高清晰度
        dot.attr(rankdir='TB', size='16,14', ratio='fill', fontname='SimHei', dpi='600')
        
        # 设置节点样式
        dot.attr('node', shape='box', style='filled,rounded', fontname='SimHei', fontsize='16', 
                 margin='0.3,0.2', width='2.2', height='1.2')
        
        # 定义颜色方案
        colors = {
            'data': '#E0F7FA',      # 浅青色 - 原始数据
            'hr': '#FFECB3',        # 浅黄色 - 高分辨率处理
            'lr': '#F3E5F5',        # 浅紫色 - 低分辨率处理
            'augment': '#FFF3E0',   # 浅橙色 - 数据增强
            'output': '#E8F5E9',    # 浅绿色 - 输出数据
            'arrow': '#37474F'      # 深灰色 - 箭头
        }
        
        # 设置边的样式
        dot.attr('edge', color=colors['arrow'], fontname='SimHei', fontsize='14', 
                 penwidth='1.5')
        
        # 创建数据收集节点
        with dot.subgraph(name='cluster_data_collection') as c:
            c.attr(label='数据收集', style='filled', fillcolor='#F0F0F0', fontsize='18', fontname='SimHei')
            
            c.node('raw_data', '原始高清图像数据集\nDIV2K、Flickr、ImageNet等', 
                   fillcolor=colors['data'], style='filled,rounded', shape='folder')
            c.node('data_selection', '数据筛选\n选择高质量、高分辨率图像', 
                   fillcolor=colors['data'])
            c.node('data_splitting', '数据集划分\n训练集、验证集、测试集', 
                   fillcolor=colors['data'])
        
        # 创建高分辨率处理节点
        with dot.subgraph(name='cluster_hr_processing') as c:
            c.attr(label='高分辨率图像处理', style='filled', fillcolor='#FFF8E1', fontsize='18', fontname='SimHei')
            
            c.node('hr_loading', 'HR图像加载\n读取高分辨率图像', 
                   fillcolor=colors['hr'])
            c.node('hr_resize', 'HR尺寸调整\n统一为标准高分辨率尺寸', 
                   fillcolor=colors['hr'])
            c.node('hr_color', 'HR颜色空间转换\nRGB标准化', 
                   fillcolor=colors['hr'])
            c.node('hr_crop', 'HR裁剪\n裁剪为固定大小块', 
                   fillcolor=colors['hr'])
            c.node('hr_normalize', 'HR归一化\n像素值缩放到[-1,1]', 
                   fillcolor=colors['hr'])
        
        # 创建低分辨率处理节点
        with dot.subgraph(name='cluster_lr_processing') as c:
            c.attr(label='低分辨率图像处理', style='filled', fillcolor='#F3F5F7', fontsize='18', fontname='SimHei')
            
            c.node('lr_resize', 'LR降采样\n降低分辨率(×2/×4倍)', 
                   fillcolor=colors['lr'])
            c.node('lr_blur', 'LR模糊处理\n添加高斯模糊', 
                   fillcolor=colors['lr'])
            c.node('lr_compress', 'LR压缩\n模拟JPEG压缩伪影', 
                   fillcolor=colors['lr'])
            c.node('lr_noise', 'LR添加噪声\n高斯噪声或椒盐噪声', 
                   fillcolor=colors['lr'])
            c.node('lr_normalize', 'LR归一化\n像素值缩放到[-1,1]', 
                   fillcolor=colors['lr'])
        
        # 创建数据增强节点
        with dot.subgraph(name='cluster_augmentation') as c:
            c.attr(label='数据增强', style='filled', fillcolor='#FFF3E0', fontsize='18', fontname='SimHei')
            
            c.node('augmentation', '数据增强\n扩充训练数据集', 
                   fillcolor=colors['augment'])
            c.node('flip', '翻转\n水平/垂直翻转', 
                   fillcolor=colors['augment'])
            c.node('rotate', '旋转\n90°/180°/270°旋转', 
                   fillcolor=colors['augment'])
            c.node('color_jitter', '颜色抖动\n亮度/对比度变化', 
                   fillcolor=colors['augment'])
        
        # 创建数据加载节点
        with dot.subgraph(name='cluster_data_loading') as c:
            c.attr(label='数据加载', style='filled', fillcolor='#E8F5E9', fontsize='18', fontname='SimHei')
            
            c.node('pair_creation', '创建LR-HR对\n(低分辨率,高分辨率)图像对', 
                   fillcolor=colors['output'])
            c.node('to_tensor', '张量转换\n转换为MindSpore张量', 
                   fillcolor=colors['output'])
            c.node('batch_generation', '批次生成\n生成训练批次', 
                   fillcolor=colors['output'])
            c.node('shuffle', '数据打乱\n随机化训练样本顺序', 
                   fillcolor=colors['output'])
            c.node('data_loader', '数据加载器\nMindSpore数据加载', 
                   fillcolor=colors['output'])
        
        # 连接数据收集节点
        dot.edge('raw_data', 'data_selection')
        dot.edge('data_selection', 'data_splitting')
        
        # 连接高分辨率处理节点
        dot.edge('data_splitting', 'hr_loading')
        dot.edge('hr_loading', 'hr_resize')
        dot.edge('hr_resize', 'hr_color')
        dot.edge('hr_color', 'hr_crop')
        dot.edge('hr_crop', 'hr_normalize')
        
        # 连接低分辨率处理节点
        dot.edge('hr_crop', 'lr_resize')
        dot.edge('lr_resize', 'lr_blur', style='dashed')
        dot.edge('lr_blur', 'lr_compress', style='dashed')
        dot.edge('lr_compress', 'lr_noise', style='dashed')
        dot.edge('lr_noise', 'lr_normalize')
        dot.edge('lr_resize', 'lr_normalize', label='直接降采样')
        
        # 连接数据增强节点
        dot.edge('hr_normalize', 'augmentation')
        dot.edge('lr_normalize', 'augmentation')
        dot.edge('augmentation', 'flip')
        dot.edge('augmentation', 'rotate')
        dot.edge('augmentation', 'color_jitter')
        
        # 连接数据加载节点
        dot.edge('flip', 'pair_creation')
        dot.edge('rotate', 'pair_creation')
        dot.edge('color_jitter', 'pair_creation')
        dot.edge('hr_normalize', 'pair_creation', label='无增强')
        dot.edge('lr_normalize', 'pair_creation', label='无增强')
        dot.edge('pair_creation', 'to_tensor')
        dot.edge('to_tensor', 'batch_generation')
        dot.edge('batch_generation', 'shuffle')
        dot.edge('shuffle', 'data_loader')
        
        # 添加MindSpore节点
        dot.node('mindspore_dataset', 'MindSpore Dataset API', shape='component', 
                 style='filled', fillcolor='#E1F5FE', fontname='SimHei')
        dot.edge('mindspore_dataset', 'batch_generation', style='dashed')
        dot.edge('mindspore_dataset', 'data_loader', style='dashed')
        
        # 添加配置信息节点
        dot.node('dataset_config', '数据集配置\nbatch_size、缩放因子等', 
                 shape='note', style='filled', fillcolor='#E0F2F1', fontname='SimHei')
        dot.edge('dataset_config', 'lr_resize', style='dashed')
        dot.edge('dataset_config', 'data_loader', style='dashed')
        
        # 渲染和保存图形
        output_filename = os.path.join(OUTPUT_DIR, 'SRGAN数据处理流程图')
        dot.render(output_filename, cleanup=True)
        print(f"SRGAN数据处理流程图已生成: {os.path.abspath(output_filename + '.png')}")
        
        return output_filename + '.png'
    except ImportError:
        print("错误: 未找到graphviz库。请执行 'pip install graphviz' 并确保系统已安装graphviz。")
        return None
    except Exception as e:
        print(f"生成SRGAN数据处理流程图时出错: {e}")
        return None

if __name__ == "__main__":
    print("生成CRA和SRGAN的数据处理流程图...")
    create_cra_data_processing_diagram()
    create_srgan_data_processing_diagram()
    print("数据处理流程图生成完成!") 
