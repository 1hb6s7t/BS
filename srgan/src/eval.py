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
"""file for evaluating"""

import argparse
import os
import sys

import numpy as np
from skimage.color import rgb2ycbcr
from skimage.metrics import peak_signal_noise_ratio
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore import context
import mindspore.ops as ops

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from src.model.generator import get_generator
from src.dataset.create_loader import create_test_dataloader

def main(args):
    """Evaluating process"""
    set_seed(1)
    context.set_context(mode=context.GRAPH_MODE, device_id=args.device_id, save_graphs=False)
    test_ds = create_test_dataloader(1, args.test_LR_path, args.test_GT_path)
    test_data_loader = test_ds.create_dict_iterator()
    generator = get_generator(4, 0.02)
    params = load_checkpoint(args.generator_path)
    print("======load checkpoint")
    load_param_into_net(generator, params)
    op = ops.ReduceSum(keep_dims=False)
    psnr_list = []
    print("=======starting test=====")
    for data in test_data_loader:
        lr = data['LR']
        gt = data['HR']
        _, _, h, w = lr.shape[:4]
        gt = gt[:, :, : h * args.scale, : w *args.scale]

        output = generator(lr)
        output = op(output, 0)
        output = output.asnumpy()
        output = np.clip(output, -1.0, 1.0)
        gt = op(gt, 0)

        output = (output + 1.0) / 2.0
        gt = (gt + 1.0) / 2.0

        output = output.transpose(1, 2, 0)
        gt = gt.asnumpy()
        gt = gt.transpose(1, 2, 0)
        
        # 打印图像尺寸，辅助调试
        print("Output shape:", output.shape)
        print("GT shape:", gt.shape)
        
        # 确保两个图像具有相同的尺寸
        min_h = min(output.shape[0], gt.shape[0])
        min_w = min(output.shape[1], gt.shape[1])
        
        # 对齐尺寸
        output = output[:min_h, :min_w, :]
        gt = gt[:min_h, :min_w, :]
        
        # 再次打印修正后的图像尺寸
        print("Adjusted output shape:", output.shape)
        print("Adjusted GT shape:", gt.shape)
        
        # 截取Y通道并应用边界裁剪
        y_output = rgb2ycbcr(output)
        y_gt = rgb2ycbcr(gt)
        
        # 确保裁剪后仍有足够的区域计算PSNR
        crop_size = args.scale
        if min_h > 2*crop_size and min_w > 2*crop_size:
            y_output = y_output[crop_size:-crop_size, crop_size:-crop_size, :1]
            y_gt = y_gt[crop_size:-crop_size, crop_size:-crop_size, :1]
        else:
            # 如果图像太小，减少裁剪区域
            crop_size = max(1, crop_size // 2)
            y_output = y_output[crop_size:-crop_size, crop_size:-crop_size, :1]
            y_gt = y_gt[crop_size:-crop_size, crop_size:-crop_size, :1]

        # 再次检查尺寸是否匹配
        print("Final Y output shape:", y_output.shape)
        print("Final Y GT shape:", y_gt.shape)
        
        psnr = peak_signal_noise_ratio(y_output / 255.0, y_gt / 255.0, data_range=1.0)
        psnr_list.append(psnr)
        print("Image PSNR:", psnr)
        
    print("avg PSNR:", np.mean(psnr_list))

def parse_args():
    """Add argument"""
    parser = argparse.ArgumentParser(description="SRGAN eval")
    parser.add_argument("--test_LR_path", type=str)
    parser.add_argument("--test_GT_path", type=str)
    parser.add_argument("--res_num", type=int, default=16)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--generator_path", type=str)
    parser.add_argument("--device_id", type=int, default=0, help="device id, default: 0.")
    parser.add_argument('--platform', type=str, default='Ascend', choices=('Ascend', 'GPU', 'CPU'))
    return parser.parse_args()

if __name__ == '__main__':
    args_list = parse_args()
    main(args_list)
