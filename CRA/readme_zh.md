# 上下文残差聚合(CRA)

传统的图像修复方法只能处理低分辨率的输入图像，而简单地对低分辨率修复结果进行上采样只会产生模糊的结果。将高频残差图像添加到模糊的大图像上可以生成清晰的结果，丰富细节和纹理。CRA将上下文聚合残差添加到上采样的神经网络修复结果中，输出最终结果。通过注意力传输模块(ATM)，掩码区域中的聚合残差是根据上下文残差和注意力分数计算得出的。通过构建生成对抗网络来预测低分辨率图像，很好地抑制了内存和计算时间的成本，使超高分辨率图像恢复更加有效和高质量。

## 预训练模型

MindSpore训练的模型：

|  数据集  |  ckpt  |
| :-----: | ------ |
| places | [ckpt](https://download.mindspore.cn/vision/cra/cra.ckpt) |

## 训练参数说明

| 参数 | 默认值 | 描述 |
|:----:|:--------:|:-------:|
| image_dir | ../places | 训练输入数据的图像路径 |
| mask_template_dir | ../mask_template | 训练输入数据的掩码路径 |
| save_folder |../ckpt_out| 训练中存储检查点文件的文件路径 |
| device_target | GPU | 训练设备 |
| device_id | 0 | 获取设备id |
| device_num | 1 | 获取设备数量 |
| IMG_SHAPE | [512, 512, 3] | 网络输入张量的所需维度 |
| attention_type | SOFT | 计算注意力类型 |
| coarse_alpha | 1.2 | 损失计算中粗略输出的比例 |
| gan_with_mask | False | 计算对抗损失时是否连接掩码 |
| gan_loss_alpha | 0.001 | 生成器对抗损失的比例 |
| in_hole_alpha | 1.2 | 掩码区域中生成结果对损失值的影响 |
| context_alpha | 1.2 | 掩码区域外生成结果对损失值的影响 |
| wgan_gp_lambda | 10 | WGAN-GP损失对判别器损失值的影响 |
| learning_rate | 1e-4 | 初始学习率 |
| lr_decrease_epoch | 2 | 衰减所需的轮数 |
| lr_decrease_factor | 0.5 | 衰减率 |
| run_distribute | False | 是否运行分布式 |
| train_batchsize | 4 | 训练的批量大小 |
| epochs | 15 | 训练的轮数 |
| dis_iter | 1 | 训练dis_iter次判别器时训练一次生成器 |

## 示例

以下介绍如何使用CRA模型。

### 数据集

首先，您需要自己下载数据集。支持Places2数据集。

注意，对于Places2数据集，您需要下载高分辨率图像训练数据集，其中包含443个场景类别，包括超过180万张1024 * 1024的图片。

此外，我们的工作中还提供了掩码数据和测试数据。

下载链接如下：

Places2：http://places2.csail.mit.edu/download.html。

mask_templates：https://github.com/duxingren14/Hifill-tensorflow/tree/master/mask_templates。

test：https://github.com/duxingren14/Hifill-tensorflow/tree/master/data/test。

获取数据集后，确保您的路径如下：

```text
  CRA
   ├── places
           ├── a
               ├── auto_showroom
                         ├── 00000001.jpg
                         ├── 00000002.jpg
                         ├── 00000003.jpg
                         └── ......
               ├── auto_factory
               ├── ......
               ├── airplane_cabin
               └── airfield
           ├── b
           ├── c
           ├── ......
           ├── y
           └── z
    ├── mask_templates
           ├── 0.png
           ├── ......
           └── 99.png
    └── test
           ├──images
              ├── 0.png
              └── 1.png
           └──masks
              ├── 0.png
              └── 1.png
```

### 训练

以下配置使用1个GPU进行训练。训练15个轮次，批量大小为4。

```shell
python train.py --image_dir ../places --mask_template_dir ../mask_templates --save_folder ../ckpt_out --device_target GPU --device_id 0 --device_num 1 --run_distribute False --train_batchsize 4 --epochs 15
```

以下配置是针对八张GPU卡的分布式并行训练。

```shell
mpirun -n 8 python train.py --image_dir ../places --mask_template_dir ../mask_templates --save_folder ../ckpt_out --device_target GPU --device_id 0 --device_num 8 --run_distribute True --train_batchsize 4 --epochs 15
```

输出：

```text
epoch1/15, batch1/56358, d_loss is 1091.4999, g_loss is 1.3412, time is 0.5120
epoch1/15, batch1/56358, d_loss is 1238.4945, g_loss is 1.6735, time is 0.5127
epoch1/15, batch1/56358, d_loss is 1082.4247, g_loss is 1.8266, time is 0.5117
epoch1/15, batch1/56358, d_loss is 971.5017, g_loss is 1.8454, time is 0.5126
epoch1/15, batch1/56358, d_loss is 1157.3241, g_loss is 1.7420, time is 0.5127
epoch1/15, batch1/56358, d_loss is 1068.8934, g_loss is 1.5067, time is 0.5129
epoch1/15, batch1/56358, d_loss is 1284.8508, g_loss is 1.8697, time is 0.5120
epoch1/15, batch2/56358, d_loss is 987.3273, g_loss is 1.5855, time is 0.5125
epoch1/15, batch2/56358, d_loss is 1002.3116, g_loss is 1.6405, time is 0.4966
epoch1/15, batch2/56358, d_loss is 937.8546, g_loss is 1.3261, time is 0.4965
epoch1/15, batch2/56358, d_loss is 1288.6157, g_loss is 1.6953, time is 0.4973
epoch1/15, batch2/56358, d_loss is 1130.4807, g_loss is 1.6920, time is 0.4969
epoch1/15, batch2/56358, d_loss is 1203.1342, g_loss is 1.4811, time is 0.4973
epoch1/15, batch2/56358, d_loss is 1124.6455, g_loss is 1.4844, time is 0.4966
epoch1/15, batch2/56358, d_loss is 983.5717, g_loss is 1.3907, time is 0.4972
···
```

以下程序以Ascend：8 * Ascend-910(32GB) | ARM：192核768GB环境来训练places2数据集为例，训练运行如下。

```shell
python train.py --image_dir ../places --mask_template_dir ../mask_templates --save_folder ../ckpt_out --device_target Ascend --device_id 0 --device_num 8 --run_distribute True --train_batchsize 4 --epochs 15
```

### 推理

以下配置用于推理。

```shell
python test.py --image_dir ../test/images --mask_dir ../test/masks --output_dir ../output --checkpoint_dir ../ckpt_out/generator_epoch15_batch56358.ckpt
```

#### 结果

![1.jpg](attachment:1.jpg)

![0.jpg](attachment:0.jpg) 