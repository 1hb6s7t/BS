#!/usr/bin/env python3
import os
import graphviz

def create_model_call_relationships_diagram():
    """创建模型调用关系图"""
    # 检查graphviz是否安装
    try:
        dot = graphviz.Digraph('模型调用关系图', format='png')
        # 设置更高的DPI以提高清晰度
        dot.attr(rankdir='TB', size='14,10', ratio='fill', fontname='SimHei', dpi='300')
        
        # 设置节点样式
        dot.attr('node', shape='box', style='filled,rounded', fontname='SimHei', fontsize='14')
        
        # 定义颜色方案
        colors = {
            'main': '#E6F2FF',         # 浅蓝色 - 主要组件
            'cra': '#FFE6E6',          # 浅红色 - CRA组件
            'srgan': '#E6FFE6',        # 浅绿色 - SRGAN组件
            'util': '#FFFDE6',         # 浅黄色 - 工具函数
            'extern': '#F2F2F2'        # 浅灰色 - 外部依赖
        }
        
        # 创建主程序和入口点子图
        with dot.subgraph(name='cluster_main') as c:
            c.attr(label='主程序和入口点', style='filled', fillcolor='#F0F8FF', fontsize='16', fontname='SimHei')
            
            c.node('main', 'main()', fillcolor=colors['main'])
            c.node('parse_args', 'parse_args()', fillcolor=colors['main'])
            c.node('setup_context', 'setup_context()', fillcolor=colors['main'])
            c.node('repair_image', 'repair_image()', fillcolor=colors['main'])
            c.node('enhance_image', 'enhance_image()', fillcolor=colors['main'])
            c.node('save_result', 'save_result()', fillcolor=colors['main'])
            
            # 连接节点
            c.edges([
                ('main', 'parse_args'),
                ('main', 'setup_context'),
                ('main', 'repair_image'),
                ('main', 'enhance_image'),
                ('main', 'save_result')
            ])
        
        # 创建CRA组件子图
        with dot.subgraph(name='cluster_cra') as c:
            c.attr(label='CRA组件', style='filled', fillcolor='#FFF0F0', fontsize='16', fontname='SimHei')
            
            c.node('gated_generator', 'GatedGenerator', fillcolor=colors['cra'])
            c.node('build_cra_net', 'build_inference_net()', fillcolor=colors['cra'])
            c.node('apply_attention', 'apply_attention()', fillcolor=colors['cra'])
            c.node('gated_conv', 'gated_conv()', fillcolor=colors['cra'])
            c.node('cra_infer', 'infer()', fillcolor=colors['cra'])
            c.node('cra_postproc', 'post_processing()', fillcolor=colors['cra'])
            
            # 连接节点
            c.edges([
                ('build_cra_net', 'gated_generator'),
                ('build_cra_net', 'apply_attention'),
                ('gated_generator', 'gated_conv'),
                ('gated_generator', 'cra_infer'),
                ('apply_attention', 'cra_postproc')
            ])
        
        # 创建SRGAN组件子图
        with dot.subgraph(name='cluster_srgan') as c:
            c.attr(label='SRGAN组件', style='filled', fillcolor='#F0FFF0', fontsize='16', fontname='SimHei')
            
            c.node('generator', 'Generator', fillcolor=colors['srgan'])
            c.node('residual_block', 'ResidualBlock', fillcolor=colors['srgan'])
            c.node('pixel_shuffle', 'PixelShuffle', fillcolor=colors['srgan'])
            c.node('srgan_infer', 'infer()', fillcolor=colors['srgan'])
            
            # 连接节点
            c.edges([
                ('generator', 'residual_block'),
                ('generator', 'pixel_shuffle'),
                ('generator', 'srgan_infer')
            ])
        
        # 创建工具函数子图
        with dot.subgraph(name='cluster_utils') as c:
            c.attr(label='工具函数', style='filled', fillcolor='#FFFEF0', fontsize='16', fontname='SimHei')
            
            c.node('load_image', 'load_image()', fillcolor=colors['util'])
            c.node('load_mask', 'load_mask()', fillcolor=colors['util'])
            c.node('create_mask', 'create_mask_interactively()', fillcolor=colors['util'])
            
            # 连接节点
            c.edge('load_mask', 'create_mask', style='dashed')
        
        # 创建外部依赖子图
        with dot.subgraph(name='cluster_extern') as c:
            c.attr(label='外部依赖', style='filled', fillcolor='#F8F8F8', fontsize='16', fontname='SimHei')
            
            c.node('mindspore', 'MindSpore框架', fillcolor=colors['extern'])
            c.node('context', 'ms.context', fillcolor=colors['extern'])
            c.node('ops', 'ms.ops', fillcolor=colors['extern'])
            c.node('cv2', 'OpenCV (cv2)', fillcolor=colors['extern'])
            c.node('numpy', 'NumPy (np)', fillcolor=colors['extern'])
            c.node('pil', 'PIL (Image)', fillcolor=colors['extern'])
            
            # 连接节点
            c.edges([
                ('mindspore', 'context'),
                ('mindspore', 'ops')
            ])
        
        # 连接主程序与CRA组件
        dot.edge('repair_image', 'build_cra_net')
        dot.edge('repair_image', 'gated_generator')
        dot.edge('repair_image', 'apply_attention')
        
        # 连接主程序与SRGAN组件
        dot.edge('enhance_image', 'generator')
        dot.edge('enhance_image', 'srgan_infer')
        
        # 连接主程序与工具函数
        dot.edge('repair_image', 'load_image')
        dot.edge('repair_image', 'load_mask')
        dot.edge('load_mask', 'create_mask', style='dashed')
        
        # 连接主程序与外部依赖
        dot.edge('setup_context', 'context')
        dot.edge('repair_image', 'numpy')
        dot.edge('enhance_image', 'numpy')
        dot.edge('load_image', 'cv2')
        dot.edge('load_mask', 'cv2')
        dot.edge('create_mask', 'cv2')
        dot.edge('save_result', 'cv2')
        dot.edge('save_result', 'pil')
        
        # 渲染图形
        output_filename = '模型调用关系图'
        dot.render(output_filename, cleanup=True)
        print(f"模型调用关系图已生成: {os.path.abspath(output_filename + '.png')}")
        
        # 保存英文文件名版本作为备份
        en_output_filename = 'model_call_relationships'
        if os.path.exists(output_filename + '.png'):
            import shutil
            shutil.copy2(output_filename + '.png', en_output_filename + '.png')
        
        # 生成更详细的函数关系图
        create_detailed_function_relationships_diagram()
        
        return output_filename + '.png'
    except ImportError:
        print("错误: 未找到graphviz库。请执行 'pip install graphviz' 并确保系统已安装graphviz。")
        return None

def create_detailed_function_relationships_diagram():
    """创建详细的函数调用关系图"""
    try:
        dot = graphviz.Digraph('函数调用详细关系图', format='png')
        # 设置更高的DPI以提高清晰度
        dot.attr(rankdir='TB', size='14,12', ratio='fill', fontname='SimHei', dpi='300')
        
        # 设置节点样式
        dot.attr('node', shape='box', style='filled,rounded', fontname='SimHei', fontsize='14')
        
        # 定义颜色方案
        colors = {
            'main': '#E6F2FF',         # 浅蓝色 - 主要组件
            'io': '#F2E6FF',           # 浅紫色 - 输入输出
            'cra': '#FFE6E6',          # 浅红色 - CRA组件
            'srgan': '#E6FFE6',        # 浅绿色 - SRGAN组件
            'data': '#FFFDE6',         # 浅黄色 - 数据流
            'extern': '#F2F2F2'        # 浅灰色 - 外部依赖
        }
        
        # 主函数和参数解析
        dot.node('main', 'main()', fillcolor=colors['main'])
        dot.node('parse_args', 'parse_args()\n解析命令行参数', fillcolor=colors['main'])
        dot.node('args', '命令行参数\n输入图像、掩码、输出路径等', fillcolor=colors['data'])
        
        # 上下文设置
        dot.node('setup_context', 'setup_context()\n设置MindSpore上下文', fillcolor=colors['main'])
        dot.node('context', 'ms.context\n设置运行模式和设备', fillcolor=colors['extern'])
        
        # 输入输出处理函数
        with dot.subgraph(name='cluster_io') as c:
            c.attr(label='输入/输出处理', style='filled', fillcolor='#F5F0FF', fontsize='16', fontname='SimHei')
            
            c.node('load_image', 'load_image()\n加载并预处理输入图像', fillcolor=colors['io'])
            c.node('load_mask', 'load_mask()\n加载或创建掩码', fillcolor=colors['io'])
            c.node('create_mask', 'create_mask_interactively()\n交互式创建掩码', fillcolor=colors['io'])
            c.node('save_result', 'save_result()\n保存处理结果', fillcolor=colors['io'])
            
            c.node('input_image', '输入图像', fillcolor=colors['data'])
            c.node('input_mask', '输入掩码', fillcolor=colors['data'])
            c.node('output_path', '输出路径', fillcolor=colors['data'])
        
        # CRA修复函数
        with dot.subgraph(name='cluster_cra_funcs') as c:
            c.attr(label='CRA图像修复函数', style='filled', fillcolor='#FFF0F0', fontsize='16', fontname='SimHei')
            
            c.node('repair_image', 'repair_image()\n使用CRA模型修复图像', fillcolor=colors['cra'])
            c.node('load_cra_model', 'load_model()\n加载CRA模型参数', fillcolor=colors['cra'])
            c.node('build_cra_net', 'build_inference_net()\n构建CRA推理网络', fillcolor=colors['cra'])
            c.node('apply_attention', 'apply_attention()\n应用上下文注意力', fillcolor=colors['cra'])
            c.node('cra_postprocess', 'post_processing()\n处理CRA输出', fillcolor=colors['cra'])
            
            c.node('cra_input', 'CRA输入\n损坏图像和掩码', fillcolor=colors['data'])
            c.node('cra_output', 'CRA输出\n修复的图像', fillcolor=colors['data'])
        
        # SRGAN增强函数
        with dot.subgraph(name='cluster_srgan_funcs') as c:
            c.attr(label='SRGAN图像增强函数', style='filled', fillcolor='#F0FFF0', fontsize='16', fontname='SimHei')
            
            c.node('enhance_image', 'enhance_image()\n使用SRGAN模型增强图像', fillcolor=colors['srgan'])
            c.node('load_srgan_model', 'load_model()\n加载SRGAN模型参数', fillcolor=colors['srgan'])
            c.node('srgan_preprocess', 'preprocess()\n预处理SRGAN输入', fillcolor=colors['srgan'])
            c.node('srgan_infer', 'infer()\nSRGAN模型推理', fillcolor=colors['srgan'])
            c.node('srgan_postprocess', 'postprocess()\n处理SRGAN输出', fillcolor=colors['srgan'])
            
            c.node('srgan_input', 'SRGAN输入\n修复后的图像', fillcolor=colors['data'])
            c.node('srgan_output', 'SRGAN输出\n增强的图像', fillcolor=colors['data'])
        
        # 连接主要函数流
        dot.edges([
            ('main', 'parse_args'),
            ('parse_args', 'args'),
            ('main', 'setup_context'),
            ('setup_context', 'context'),
            ('args', 'load_image'),
            ('args', 'load_mask'),
            ('args', 'output_path')
        ])
        
        # 连接输入/输出处理
        dot.edges([
            ('load_image', 'input_image'),
            ('load_mask', 'input_mask'),
            ('load_mask', 'create_mask'),
            ('save_result', 'output_path')
        ])
        
        # 连接CRA修复流程
        dot.edges([
            ('main', 'repair_image'),
            ('input_image', 'cra_input'),
            ('input_mask', 'cra_input'),
            ('cra_input', 'repair_image'),
            ('repair_image', 'load_cra_model'),
            ('repair_image', 'build_cra_net'),
            ('build_cra_net', 'apply_attention'),
            ('apply_attention', 'cra_postprocess'),
            ('cra_postprocess', 'cra_output')
        ])
        
        # 连接SRGAN增强流程
        dot.edges([
            ('main', 'enhance_image'),
            ('cra_output', 'srgan_input'),
            ('srgan_input', 'enhance_image'),
            ('enhance_image', 'load_srgan_model'),
            ('enhance_image', 'srgan_preprocess'),
            ('srgan_preprocess', 'srgan_infer'),
            ('srgan_infer', 'srgan_postprocess'),
            ('srgan_postprocess', 'srgan_output')
        ])
        
        # 连接输出
        dot.edges([
            ('cra_output', 'save_result'),
            ('srgan_output', 'save_result')
        ])
        
        # 渲染图形
        output_filename = '函数调用详细关系图'
        dot.render(output_filename, cleanup=True)
        print(f"详细函数调用关系图已生成: {os.path.abspath(output_filename + '.png')}")
        
        # 保存英文文件名版本作为备份
        en_output_filename = 'detailed_function_relationships'
        if os.path.exists(output_filename + '.png'):
            import shutil
            shutil.copy2(output_filename + '.png', en_output_filename + '.png')
        
        return output_filename + '.png'
    except ImportError:
        print("错误: 未找到graphviz库。请执行 'pip install graphviz' 并确保系统已安装graphviz。")
        return None

# 如果直接运行此脚本
if __name__ == "__main__":
    create_model_call_relationships_diagram() 