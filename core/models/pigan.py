"""
Discriminator and Generator implementation from DCGAN paper
"""
import torch
import torch.nn as nn
import math
from collections import OrderedDict

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d,
            norm="batch_norm", img_size=64, final_sigmoid=True):
        super(Discriminator, self).__init__()
        self.norm = norm
        n_blocks = int(math.log2(img_size//8))
        block_list = [
            (f'block{i}', self._block(features_d*(2**(i-1)), features_d*(2**i),
                4, 2, 1))
            for i in range(1,n_blocks+1)]
        full_list = [
            ('conv_in', nn.Conv2d(
                channels_img, features_d, kernel_size=4,
                stride=2, padding=1, bias=False
            )),
            ('leaky_relu', nn.LeakyReLU(0.2)),
            *block_list,
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            ('conv_out', nn.Conv2d(features_d * (2**n_blocks), 1, kernel_size=4,
                stride=2, padding=0, bias=False)),
            ('sigmoid', nn.Sigmoid()) if final_sigmoid\
                    else ('identity', nn.Identity())
            ]
        self.disc = nn.Sequential(OrderedDict(full_list))

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
                )),
            ('batch_norm', nn.BatchNorm2d(out_channels))\
                    if self.norm=='batch_norm' else\
                    ('instance_norm2d', nn.InstanceNorm2d(out_channels, affine=True))\
                    if self.norm=='instance_norm2d' else\
                    ('identity', nn.Identity()),
            ('leaky_relu', nn.LeakyReLU(0.2)),
            ]))

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g, img_size=64):
        super(Generator, self).__init__()
        n_blocks = int(math.log2(img_size/4))
        block_list = [
            ('block1', self._block(channels_noise, features_g * (2**n_blocks), 4, 1, 0)),  # img: 4x4
            ]
        block_list.extend([
        (f'block{a}', self._block(features_g * 2**b, features_g * 2**(b-1), 4, 2, 1))
                for (a,b) in zip(range(1,n_blocks+1), range(n_blocks+1,1,-1))][1:])

        full_list = [
            # Input: N x channels_noise x 1 x 1
            *block_list,
            ('transpose_conv_out', nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4,
                stride=2, padding=1, bias=False)),
            # Output: N x channels_img x img_size x img_size
            ('tanh', nn.Tanh()),
            ]
        self.net = nn.Sequential(OrderedDict(full_list))

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(OrderedDict([
            ('transpose_conv', nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
                )),
            ('batch_norm', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU()),
            ]))

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        return self.net(x)
