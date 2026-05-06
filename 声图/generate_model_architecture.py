#!/usr/bin/env python3
import graphviz as gv
import os

def create_cra_model_architecture():
    """创建CRA模型架构图"""
    dot = gv.Digraph('CRAModelArchitecture', comment='CRA模型架构')
    
    # 设置图像属性
    dot.attr(rankdir='TB', size='12,18', ratio='fill', fontname='SimHei')
    dot.attr('node', shape='box', style='filled', fontname='SimHei')
    
    # 定义节点颜色
    cra_color = '#E1F5FE'  # 浅蓝色
    
    # 创建CRA模型架构
    with dot.subgraph(name='cluster_cra_model') as c:
        c.attr(label='CRA模型架构 (GatedGenerator)', style='filled', color=cra_color, fontcolor='black')
        
        # 输入层
        c.node('input_layer', '输入\n[损坏图像, 掩码]', color='#0D47A1', fontcolor='white')
        
        # 编码器部分
        c.node('encoder1', '编码器层1\n门控卷积 + PReLU', color='#1976D2')
        c.node('encoder2', '编码器层2\n门控卷积 + PReLU', color='#1976D2')
        c.node('encoder3', '编码器层3\n门控卷积 + PReLU', color='#1976D2')
        c.node('encoder4', '编码器层4\n门控卷积 + PReLU', color='#1976D2')
        
        # 特征提取部分
        c.node('features', '特征提取\n门控卷积块', color='#1E88E5')
        
        # 注意力机制
        c.node('attention', 'ApplyAttention2\n上下文注意力模块', color='#2196F3', shape='box', style='filled')
        c.node('attention_detail1', 'Unfold操作\n[核大小=(3,3)]', color='#64B5F6')
        c.node('attention_detail2', '相关性计算\nBatchMatMul', color='#64B5F6')
        c.node('attention_detail3', '特征聚合', color='#64B5F6')
        
        # 解码器部分
        c.node('decoder4', '解码器层4\n门控卷积 + PReLU', color='#42A5F5')
        c.node('decoder3', '解码器层3\n门控卷积 + PReLU', color='#42A5F5')
        c.node('decoder2', '解码器层2\n门控卷积 + PReLU', color='#42A5F5')
        c.node('decoder1', '解码器层1\n门控卷积 + PReLU', color='#42A5F5')
        
        # 输出层
        c.node('output_layer', '输出层\n卷积 + Tanh', color='#0D47A1', fontcolor='white')
        c.node('final_output', '修复后的图像', color='#01579B', fontcolor='white')
        
        # 残差连接
        c.node('residual1', '残差连接1', shape='point', color='#1976D2')
        c.node('residual2', '残差连接2', shape='point', color='#1976D2')
        c.node('residual3', '残差连接3', shape='point', color='#1976D2')
        c.node('residual4', '残差连接4', shape='point', color='#1976D2')
        
        # 数据流
        c.edge('input_layer', 'encoder1')
        c.edge('encoder1', 'encoder2')
        c.edge('encoder2', 'encoder3')
        c.edge('encoder3', 'encoder4')
        c.edge('encoder4', 'features')
        
        # 注意力模块细节
        c.edge('features', 'attention')
        c.edge('attention', 'attention_detail1', style='dashed')
        c.edge('attention_detail1', 'attention_detail2', style='dashed')
        c.edge('attention_detail2', 'attention_detail3', style='dashed')
        c.edge('attention_detail3', 'attention', style='dashed')
        
        # 解码器流程
        c.edge('attention', 'decoder4')
        c.edge('decoder4', 'decoder3')
        c.edge('decoder3', 'decoder2')
        c.edge('decoder2', 'decoder1')
        c.edge('decoder1', 'output_layer')
        c.edge('output_layer', 'final_output')
        
        # 残差连接
        c.edge('encoder1', 'residual1')
        c.edge('residual1', 'decoder1', style='dashed')
        c.edge('encoder2', 'residual2')
        c.edge('residual2', 'decoder2', style='dashed')
        c.edge('encoder3', 'residual3')
        c.edge('residual3', 'decoder3', style='dashed')
        c.edge('encoder4', 'residual4')
        c.edge('residual4', 'decoder4', style='dashed')
    
    # 保存图像
    dot.render('cra_model_architecture', format='png', cleanup=True)
    print(f"已生成CRA模型架构图: {os.path.abspath('cra_model_architecture.png')}")
    
    return dot

def create_srgan_model_architecture():
    """创建SRGAN模型架构图"""
    dot = gv.Digraph('SRGANModelArchitecture', comment='SRGAN模型架构')
    
    # 设置图像属性
    dot.attr(rankdir='TB', size='12,15', ratio='fill', fontname='SimHei')
    dot.attr('node', shape='box', style='filled', fontname='SimHei')
    
    # 定义节点颜色
    srgan_color = '#E8F5E9'  # 浅绿色
    
    # 创建SRGAN生成器模型架构
    with dot.subgraph(name='cluster_srgan_gen') as c:
        c.attr(label='SRGAN生成器架构', style='filled', color=srgan_color, fontcolor='black')
        
        # 输入层
        c.node('input_layer', '低分辨率图像输入', color='#1B5E20', fontcolor='white')
        
        # 初始特征提取
        c.node('initial_conv', '初始卷积层\nConv2d + PReLU', color='#2E7D32', fontcolor='white')
        
        # 残差块
        c.node('res_blocks', '16个残差块', color='#388E3C')
        
        # 残差块细节
        with c.subgraph(name='cluster_res_block') as res:
            res.attr(label='残差块详细结构', style='filled', color='#C8E6C9')
            res.node('rb_conv1', 'Conv2d', color='#66BB6A')
            res.node('rb_bn1', 'BatchNorm2d', color='#66BB6A')
            res.node('rb_prelu', 'PReLU', color='#66BB6A')
            res.node('rb_conv2', 'Conv2d', color='#66BB6A')
            res.node('rb_bn2', 'BatchNorm2d', color='#66BB6A')
            res.node('rb_add', '残差相加', color='#66BB6A', shape='point')
            
            # 残差块内部连接
            res.edge('rb_conv1', 'rb_bn1')
            res.edge('rb_bn1', 'rb_prelu')
            res.edge('rb_prelu', 'rb_conv2')
            res.edge('rb_conv2', 'rb_bn2')
            res.edge('rb_bn2', 'rb_add')
            res.edge('rb_conv1', 'rb_add', style='dashed')
        
        # 残差后卷积
        c.node('post_res_conv', '后残差卷积层\nConv2d + BatchNorm2d', color='#43A047')
        
        # 全局残差连接
        c.node('global_residual', '全局残差连接', shape='point', color='#43A047')
        
        # 子像素卷积上采样
        c.node('upsample', '子像素卷积上采样', color='#4CAF50')
        
        # 子像素卷积细节
        with c.subgraph(name='cluster_subpixel') as sub:
            sub.attr(label='子像素卷积详细结构', style='filled', color='#C8E6C9')
            sub.node('sp_conv', 'Conv2d\n(channels*4)', color='#66BB6A')
            sub.node('sp_depth2space', 'DepthToSpace\n(r=2)', color='#66BB6A')
            sub.node('sp_prelu', 'PReLU', color='#66BB6A')
            
            # 子像素卷积内部连接
            sub.edge('sp_conv', 'sp_depth2space')
            sub.edge('sp_depth2space', 'sp_prelu')
        
        # 输出层
        c.node('final_conv', '最终卷积层\nConv2d', color='#2E7D32', fontcolor='white')
        c.node('tanh', 'Tanh激活', color='#1B5E20', fontcolor='white')
        c.node('output', '高分辨率图像输出', color='#1B5E20', fontcolor='white')
        
        # 连接所有组件
        c.edge('input_layer', 'initial_conv')
        c.edge('initial_conv', 'res_blocks')
        c.edge('res_blocks', 'post_res_conv')
        c.edge('post_res_conv', 'global_residual')
        c.edge('initial_conv', 'global_residual', style='dashed')
        c.edge('global_residual', 'upsample')
        c.edge('upsample', 'final_conv')
        c.edge('final_conv', 'tanh')
        c.edge('tanh', 'output')
    
    # 保存图像
    dot.render('srgan_model_architecture', format='png', cleanup=True)
    print(f"已生成SRGAN模型架构图: {os.path.abspath('srgan_model_architecture.png')}")
    
    return dot

def create_combined_architecture():
    """创建组合架构总览图"""
    dot = gv.Digraph('CombinedArchitecture', comment='CRA与SRGAN组合架构')
    
    # 设置图像属性
    dot.attr(rankdir='LR', size='14,8', ratio='fill', fontname='SimHei')
    dot.attr('node', shape='box', style='filled', fontname='SimHei')
    
    # 添加组合系统节点
    dot.node('combined_system', 'combined_repair_sr.py\n联合处理系统', shape='component', 
           color='#6A1B9A', fontcolor='white')
    
    # 添加主要模块
    dot.node('cra_model', 'CRA修复模型\n(GatedGenerator)', color='#0288D1', fontcolor='white')
    dot.node('srgan_model', 'SRGAN超分模型\n(Generator)', color='#2E7D32', fontcolor='white')
    
    # 添加处理流程
    dot.node('input', '输入图像\n+ 掩码', color='#FFA000', fontcolor='white')
    dot.node('repaired', '修复图像', color='#01579B', fontcolor='white')
    dot.node('enhanced', '超分辨率图像', color='#1B5E20', fontcolor='white')
    
    # 添加连接
    dot.edge('input', 'cra_model')
    dot.edge('cra_model', 'repaired')
    dot.edge('repaired', 'srgan_model')
    dot.edge('srgan_model', 'enhanced')
    
    # 添加MindSpore框架
    dot.node('mindspore', 'MindSpore框架', shape='component', color='#D1C4E9')
    
    # 添加关键模块
    dot.node('attention_module', 'ApplyAttention2\n上下文注意力模块', color='#039BE5')
    dot.node('residual_module', 'ResidualBlock\n残差学习模块', color='#388E3C')
    dot.node('subpixel_module', 'SubpixelConvolution\n子像素卷积模块', color='#43A047')
    
    # 连接关键模块
    dot.edge('cra_model', 'attention_module', style='dashed')
    dot.edge('srgan_model', 'residual_module', style='dashed')
    dot.edge('srgan_model', 'subpixel_module', style='dashed')
    dot.edge('mindspore', 'combined_system', style='dashed')
    
    # 保存图像
    dot.render('combined_architecture', format='png', cleanup=True)
    print(f"已生成组合架构总览图: {os.path.abspath('combined_architecture.png')}")
    
    return dot

def main():
    """生成所有架构图"""
    try:
        # 检查是否安装了graphviz
        import graphviz
    except ImportError:
        print("未找到graphviz库。请先手动安装系统 Graphviz，并运行: python -m pip install graphviz")
        exit(1)
    
    # 生成各种架构图
    create_cra_model_architecture()
    create_srgan_model_architecture()
    create_combined_architecture()
    
    print("所有架构图生成完成")

if __name__ == "__main__":
    main() 
