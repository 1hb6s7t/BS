# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train scripts"""

import sys
import os
import argparse
import time

import numpy as np
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio
import mindspore.nn as nn
from mindspore.communication.management import init, get_rank
from mindspore import context
from mindspore import load_checkpoint, save_checkpoint, load_param_into_net
from mindspore.context import ParallelMode
import mindspore.ops as ops
from mindspore.common import set_seed

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(project_root)

from src.model.generator import get_generator
from src.model.discriminator import get_discriminator
from src.dataset.create_loader import create_train_dataloader, create_test_dataloader
from src.loss.psnr_loss import PSNRLoss
from src.loss.gan_loss import DiscriminatorLoss, GeneratorLoss
from src.vgg19.define import vgg19
from src.train.train_psnr import TrainOnestepPSNR
from src.train.train_gan import TrainOneStepD, TrainOnestepG

def main(args):
    """Training process"""
    set_seed(2022)
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target=args.platform)

    if args.run_distribute == 1:
        if args.platform == 'Ascend':
            if args.device_id == 0:
                context.set_context(device_id=int(os.getenv("DEVICE_ID", "0")))
            else:
                context.set_context(device_id=args.device_id)
        device_num = args.device_num
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
        init()
        rank = get_rank()
    else:
        if args.platform in ['GPU', 'Ascend']:
            rank = 0
            if args.device_id == 0:
                if os.getenv("DEVICE_ID", "not_set").isdigit():
                    context.set_context(device_id=int(os.getenv("DEVICE_ID", "0")))
            else:
                context.set_context(device_id=args.device_id)

    # create dataset
    train_ds = create_train_dataloader(args.train_batch_size, args.train_LR_path, args.train_GT_path, rank,
                                       args.device_num)
    test_ds = create_test_dataloader(args.val_batch_size, args.val_LR_path, args.val_GT_path)
    train_data_loader = train_ds.create_dict_iterator()
    test_data_loader = test_ds.create_dict_iterator()

    # definition of network
    generator = get_generator(4, 0.02)

    # 加载预训练的生成器模型
    if args.pretrained_generator and os.path.exists(args.pretrained_generator):
        print(f"Loading pretrained generator from {args.pretrained_generator}")
        params = load_checkpoint(args.pretrained_generator)
        load_param_into_net(generator, params)
        
        # 从文件名提取起始周期
        try:
            basename = os.path.basename(args.pretrained_generator)
            if "epoch" in basename:
                epoch_str = basename.split("epoch")[1].split(".")[0]
                args.start_psnr_epoch = int(epoch_str)
                print(f"从第 {args.start_psnr_epoch} 个周期继续训练")
        except Exception as e:
            print(f"无法从文件名提取周期信息: {e}")

    # network with loss
    psnr_loss = PSNRLoss(generator)

    # optimizer with Adam optimizer, β1 = 0.9
    psnr_optimizer = nn.Adam(generator.trainable_params(), 
                            learning_rate=args.lr, 
                            beta1=args.beta1)

    # operation for testing
    op = ops.ReduceSum(keep_dims=False)

    # trainonestep
    train_psnr = TrainOnestepPSNR(psnr_loss, psnr_optimizer)
    train_psnr.set_train()

    if not os.path.exists("./ckpt"):
        os.makedirs("./ckpt")

    # SRResNet pre-training (Generator pre-training)
    print("Starting SRResNet pre-training...")
    for epoch in range(args.start_psnr_epoch, args.psnr_epochs):
        print(f"Pre-training epoch: {epoch + 1}/{args.psnr_epochs}")
        time_begin = time.time()
        total_mse_loss = 0.0
        batch_count = 0
        
        for data in train_data_loader:
            lr = data['LR']
            hr = data['HR']
            mse_loss = train_psnr(hr, lr)
            total_mse_loss += float(mse_loss)
            batch_count += 1
            
        avg_mse_loss = total_mse_loss / batch_count if batch_count > 0 else 0
        steps = train_ds.get_dataset_size()
        time_elapsed = time.time() - time_begin
        step_time = time_elapsed / steps
        
        print('Epoch [{}/{}], Average MSE Loss: {:.6f}'.format(epoch + 1, args.psnr_epochs, avg_mse_loss))
        print('Per step time: {:.0f}ms, Total time: {:.2f}s'.format(step_time * 1000, time_elapsed))

        # Save checkpoint every 500 epochs and final epoch
        if (epoch + 1) % 500 == 0 or epoch + 1 == args.psnr_epochs:
            # 确保ckpt目录存在
            os.makedirs("./ckpt", exist_ok=True)
            
            if args.run_distribute == 0:
                save_checkpoint(train_psnr, f"./ckpt/pretrained_generator_epoch{epoch + 1}.ckpt")
            else:
                if rank == 0:
                    save_checkpoint(train_psnr, f"./ckpt/pretrained_generator_epoch{epoch + 1}.ckpt")
            print(f"检查点已保存: ./ckpt/pretrained_generator_epoch{epoch + 1}.ckpt")

        print(f"{epoch + 1}/{args.psnr_epochs} epoch finished")

    print("\nPre-training completed.")

    # GAN training
    print("\nStarting SRGAN training...")
    generator = get_generator(4, 0.02)
    discriminator = get_discriminator(96, 0.02)
    
    # Load pre-trained generator
    pretrained_path = "./ckpt/pretrained_generator_epoch{}.ckpt".format(args.psnr_epochs)
    if args.pretrained_generator:
        pretrained_path = args.pretrained_generator
        
    print(f"加载预训练生成器: {pretrained_path}")
    params = load_checkpoint(pretrained_path)
    load_param_into_net(generator, params)
    
    # Set up GAN training with VGG loss
    discriminator_loss = DiscriminatorLoss(discriminator, generator)
    
    # 检查VGG模型路径
    if not args.vgg_ckpt or not os.path.exists(args.vgg_ckpt):
        print(f"警告: VGG检查点路径不存在 '{args.vgg_ckpt}'，请检查参数设置")
        exit(1)
        
    print(f"加载VGG模型: {args.vgg_ckpt}")
    vgg = vgg19(args.vgg_ckpt)
    generator_loss = GeneratorLoss(discriminator, generator, vgg)
    
    # Optimizers with Adam, β1 = 0.9
    generator_optimizer = nn.Adam(generator.trainable_params(), 
                                learning_rate=args.lr, 
                                beta1=args.beta1)
    discriminator_optimizer = nn.Adam(discriminator.trainable_params(), 
                                    learning_rate=args.lr, 
                                    beta1=args.beta1)
    train_discriminator = TrainOneStepD(discriminator_loss, discriminator_optimizer)
    train_generator = TrainOnestepG(generator_loss, generator_optimizer)

    # Train GAN
    for epoch in range(args.start_gan_epoch, args.gan_epochs):
        print(f"GAN training epoch: {epoch + 1}/{args.gan_epochs}")
        train_begin = time.time()
        total_d_loss = 0.0
        total_g_loss = 0.0
        batch_count = 0
        
        for data in train_data_loader:
            lr = data['LR']
            hr = data['HR']
            d_loss = train_discriminator(hr, lr)
            g_loss = train_generator(hr, lr)
            total_d_loss += float(d_loss.mean())
            total_g_loss += float(g_loss.mean())
            batch_count += 1
            
        avg_d_loss = total_d_loss / batch_count if batch_count > 0 else 0
        avg_g_loss = total_g_loss / batch_count if batch_count > 0 else 0
        
        time_elapsed = time.time() - train_begin
        steps = train_ds.get_dataset_size()
        step_time = time_elapsed / steps
        
        print('Epoch [{}/{}], D_loss: {:.6f}, G_loss: {:.6f}'.format(
            epoch + 1, args.gan_epochs, avg_d_loss, avg_g_loss))
        print('Per step time: {:.0f}ms, Total time: {:.2f}s'.format(step_time * 1000, time_elapsed))

        # Save checkpoints every 500 epochs and final epoch
        if (epoch + 1) % 500 == 0 or epoch + 1 == args.gan_epochs:
            # 确保ckpt目录存在
            os.makedirs("./ckpt", exist_ok=True)
            
            if args.run_distribute == 0:
                save_checkpoint(train_generator, f'./ckpt/generator_epoch{epoch + 1}.ckpt')
                save_checkpoint(train_discriminator, f'./ckpt/discriminator_epoch{epoch + 1}.ckpt')
            else:
                if rank == 0:
                    save_checkpoint(train_generator, f'./ckpt/generator_epoch{epoch + 1}.ckpt')
                    save_checkpoint(train_discriminator, f'./ckpt/discriminator_epoch{epoch + 1}.ckpt')
            print(f"检查点已保存: ./ckpt/generator_epoch{epoch + 1}.ckpt 和 ./ckpt/discriminator_epoch{epoch + 1}.ckpt")
        
        print(f"Epoch {epoch + 1}/{args.gan_epochs} completed")

    print("\nTraining completed successfully!")

def parse_args():
    """Add argument"""
    parser = argparse.ArgumentParser(description="SRGAN train")
    parser.add_argument("--train_LR_path", default=os.path.join(project_root, 'datasets', 'DIV2K', 'LR'), type=str)
    parser.add_argument("--train_GT_path", default=os.path.join(project_root, 'datasets', 'DIV2K', 'HR'), type=str)
    parser.add_argument("--val_LR_path", default=os.path.join(project_root, 'datasets', 'Set5', 'LR'), type=str)
    parser.add_argument("--val_GT_path", default=os.path.join(project_root, 'datasets', 'Set5', 'HR'), type=str)
    parser.add_argument("--vgg_ckpt", type=str, default=os.path.join(project_root, 'src', 'vgg19', 'vgg19.ckpt'),
                       help="VGG19预训练模型路径")
    parser.add_argument("--pretrained_generator", type=str, default='',
                        help="预训练生成器模型的路径")
    parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8],
                        help='super resolution upscale factor')
    parser.add_argument("--image_size", type=int, default=96,
                        help="Image size of high resolution image. (default: 96)")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        metavar="N", help="batch size for training")
    parser.add_argument("--val_batch_size", default=1, type=int,
                        metavar="N", help="batch size for testing")
    parser.add_argument("--psnr_epochs", default=100000, type=int, metavar="N",
                        help="Number of total psnr epochs to run. (default: 200000)")
    parser.add_argument("--gan_epochs", default=200000, type=int, metavar="N",
                        help="Number of total gan epochs to run. (default: 200000)")
    parser.add_argument("--start_psnr_epoch", default=0, type=int, metavar='N',
                        help="Manual psnr epoch number (useful on restarts).")
    parser.add_argument("--start_gan_epoch", default=0, type=int, metavar='N',
                        help="Manual gan epoch number (useful on restarts).")
    parser.add_argument("--init_type", type=str, default='normal', choices=("normal", "xavier"),
                        help="network initialization, default is normal.")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--platform", type=str, default='GPU', choices=('Ascend', 'GPU', 'CPU'))
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default: 0.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--content_weight", type=float, default=1.0, help="Content loss weight")
    parser.add_argument("--adversarial_weight", type=float, default=1e-3, help="Adversarial loss weight")
    parser.add_argument("--run_distribute", type=int, default=0, help="Run distribute, default: 0.")
    parser.add_argument("--device_num", type=int, default=1, help="number of device, default: 1.")
    return parser.parse_args()

if __name__ == '__main__':
    args_list = parse_args()
    main(args_list)
