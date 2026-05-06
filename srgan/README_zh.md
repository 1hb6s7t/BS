# SRGAN

尽管使用更快、更深的卷积神经网络在单图像超分辨率的准确性和速度方面取得了突破，但一个核心问题仍然未得到解决：当我们在大尺度因子上进行超分辨率时，如何恢复更精细的纹理细节？基于优化的超分辨率方法的行为主要由目标函数的选择驱动。最近的工作主要集中在最小化均方重构误差上。虽然得到的估计具有高的峰值信噪比，但它们往往缺乏高频细节，而且在知觉上不令人满意，因为它们无法匹配在更高分辨率下预期的保真度。在本文中，我们提出了SRGAN，一个用于图像超分辨率(SR)的生成对抗网络(GAN)。据我们所知，这是第一个能够推断4倍放大因子的照片级真实自然图像的框架。为了实现这一点，我们提出了一个感知损失函数，它由对抗损失和内容损失组成。对抗损失通过一个训练用于区分超分辨率图像和原始照片级真实图像的判别器网络，将我们的解决方案推向自然图像流形。此外，我们使用基于感知相似性而非像素空间相似性的内容损失。我们的深度残差网络能够从公共基准上的严重下采样图像中恢复照片级真实的纹理。

[论文](https://arxiv.org/pdf/1609.04802.pdf)：Christian Ledig, Lucas thesis, Ferenc Huszar, Jose Caballero, Andrew Cunningham, Alejandro Acosta, Andrew Aitken, Alykhan Tejani, Johannes Totz, Zehan Wang, Wenzhe Shi Twitter.

## 预训练模型

MindSpore训练的模型，训练SRGAN的过程需要一个基于Imagenet的[预训练VGG19](https://download.mindspore.cn/model_zoo/converted_pretrained/vgg/)。

|  类型  |  ckpt  |
| ------- | ------ |
| srgan |[ckpt](https://download.mindspore.cn/vision/srgan/) |

## 数据集

有3个数据集用于训练、验证和评估。

注意，对于DIV2K数据集，您只需要'训练数据(HR图像)'和'训练数据轨道1双三次下采样x4(LR图像)'这两个数据集。

|  类型  |  资源  |
| ------- |  ----------  |
| DIV2K | [链接](https://data.vision.ee.ethz.ch/cvl/DIV2K/) |
| Set5 |[链接](https://github.com/jbhuang0604/SelfExSR) |
| Set14 |[链接](https://github.com/jbhuang0604/SelfExSR) |

确保您的文件组织如下：

```markdown
.datasets/
    └── DIV2K
    |    ├── HR
    |    └── LR
    └── Set5
    |    ├── HR
    |    └── LR
    └── Set14
         ├── HR
         └── LR
```

## 训练参数描述

| 参数                 | 默认值                       | 描述                   |
| -------------------------- | ---------------------------------|---------------------------------------------|
| train_LR_path | None    | 训练集的低分辨率图像路径 |
| train_GT_path | None    | 训练集的高分辨率图像路径 |
| val_LR_path   | None    | 验证集的低分辨率图像路径 |
| val_GT_path   | None    | 验证集的高分辨率图像路径 |
| vgg_ckpt      | None    | 预训练vgg19模型的路径 |
| image_size    | 96    | 高分辨率图像的图像大小 |
| train_batch_size  | 16    | 训练阶段的批量大小 |
| val_batch_size     | 1    | 验证阶段的批量大小 |
| psnr_epochs       | 2000    | psnr训练阶段的轮数 |
| gan_epochs       | 1000    | gan训练阶段的轮数 |
| init_type        | normal    | 网络初始化中使用的方法 |
| platform | Ascend    | 训练阶段使用的平台 |
| run_distribute | 0    | 分布式训练 |
| device_num | 1    | 设备数量 |

## 性能

### 训练性能

| 数据集 | 资源 | 速度 |
|:------- |:---------|:------|
| DIV2K |Ascend 910|1pc: 540 ms/step; 8pcs: 1500 ms/step|
| DIV2K |NVIDIA GeForce RTX 3090|1pc: 350 ms/step|

### 评估性能

| 数据集 | 资源 | PSNR | PSNR(论文)|
| :-------------------|---------------------------------|--------------------------------|--------------------------------|
| Set5  | Ascend 910 | 31.00 | 29.40 |
| Set14 | Ascend 910  | 27.93 | 26.02 |

### 推理性能

| 数据集 | 资源 | 速度 |
| :------------------ | -------------------------|---------------------------------|
| Set5  | Ascend 910 | 1pc: 7 ms/step |
| Set14 | Ascend 910 | 1pc: 10 ms/step |

## 示例

### 训练

使用DIV2K数据集进行训练，Set5数据集进行验证。您可以根据需要更改路径。

训练结果将保存在'./ckpt'中。

  ```shell
  python -m src.train.train --train_LR_path './datasets/DIV2K/LR' --train_GT_path './datasets/DIV2K/HR' --val_LR_path './datasets/Set5/LR' --val_GT_path './datasets/Set5/HR' --vgg_ckpt './src/vgg19/vgg19.ckpt'
  ```

**输出**：

  ```text
  ...
   training 999 epoch
  per step needs time:356ms
  D_loss:
  0.6385005
  G_loss:
  0.009284813
   999/1000 epoch finished
  training 1000 epoch
  per step needs time:351ms
  D_loss:
  0.6633582
  G_loss:
  0.016534843
  saving ckpt
   1000/1000 epoch finished
  ```

### 分布式训练

您可以运行'./src/train/run_distribute_train.sh'在分布式环境中训练您的模型。

  ```shell
  sh ./src/train/run_distribute_train.sh 8 1 /home/user/work/srgan_p/datasets/DIV2K/LR /home/user/work/srgan_p/datasets/DIV2K/HR /home/user/work/srgan_p/vgg19.ckpt /home/user/work/srgan_p/datasets/Set5/LR /home/user/work/srgan_p/datasets/Set5/HR
  ```

### 评估

您可以运行'./src/eval.py'评估您的模型的性能。

  ```shell
  python -m src.eval --test_LR_path './datasets/Set5/LR' --test_GT_path './datasets/Set5/HR' --generator_path './ckpt/G_model_1000.ckpt'
  ```

**输出**：

  ```text
======load checkpoint
[WARNING] ME(3178:281473299786304,MainProcess):2022-08-28-14:22:01.705.760 [mindspore/train/serialization.py:674] For 'load_param_into_net', remove parameter generator.conv1.0.weight's prefix name: generator., continue to load it to net parameter conv1.0.weight.
=======starting test=====
avg PSNR: 30.997722005197154
  ```

### 推理

推理结果将保存在'./output'中。

  ```shell
  python -m src.infer --test_LR_path './datasets/Set14/LR' --generator_path './ckpt/G_model_1000.ckpt'
  ```

**输出**：

  ```text
======load checkpoint
[WARNING] ME(3178:281473299786304,MainProcess):2022-08-28-14:22:01.705.760 [mindspore/train/serialization.py:674] For 'load_param_into_net', remove parameter generator.conv1.0.weight's prefix name: generator., continue to load it to net parameter conv1.0.weight.
=======starting test=====
Total 14 images need 134ms, per image needs 10ms.
Inference End.
  ```

**结果**：

![result](./images/infer_result.png) 