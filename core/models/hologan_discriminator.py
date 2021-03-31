"""
Discriminator and Generator implementation from DCGAN paper
"""
import torch
import torch.nn as nn
from collections import OrderedDict

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d,
            norm="batch_norm", final_sigmoid=True):
        super(Discriminator, self).__init__()
        self.norm = norm
        self.disc = nn.Sequential(OrderedDict([
            # input: N x channels_img x 64 x 64
            ('conv_in', nn.Conv2d(
                channels_img, features_d, kernel_size=4,
                stride=2, padding=1, bias=False
            )),
            ('leaky_relu', nn.LeakyReLU(0.2)),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            ('block1', self._block(features_d, features_d * 2, 4, 2, 1)),
            ('block2', self._block(features_d * 2, features_d * 4, 4, 2, 1)),
            ('block3', self._block(features_d * 4, features_d * 8, 4, 2, 1)),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            ('conv_out', nn.Conv2d(features_d * 8, 1, kernel_size=4,
                stride=2, padding=0, bias=False)),
            ('sigmoid', nn.Sigmoid()) if final_sigmoid\
                    else ('identity', nn.Identity())
            ]))

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
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(OrderedDict([
            # Input: N x channels_noise x 1 x 1
            ('block1', self._block(channels_noise, features_g * 16, 4, 1, 0)),  # img: 4x4
            ('block2', self._block(features_g * 16, features_g * 8, 4, 2, 1)),  # img: 8x8
            ('block3', self._block(features_g * 8, features_g * 4, 4, 2, 1)),  # img: 16x16
            ('block4', self._block(features_g * 4, features_g * 2, 4, 2, 1)),  # img: 32x32
            ('transpose_conv_out', nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4,
                stride=2, padding=1, bias=False)),
            # Output: N x channels_img x 64 x 64
            ('tanh', nn.Tanh()),
            ]))

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
        return self.net(x)
