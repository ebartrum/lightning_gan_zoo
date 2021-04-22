"""
Discriminator and Generator implementation from DCGAN paper
"""
import torch
import torch.nn as nn
import math
from collections import OrderedDict
from core.nerf.nerf_renderer import RadianceFieldRenderer
from core.nerf.utils import sample_full_xys
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    look_at_view_transform,
)
import numpy as np

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
    def __init__(self, channels_noise, channels_img, features_g,
            nerf_cfg, azimuth_low, azimuth_high, img_size=64):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.azimuth_low = azimuth_low
        self.azimuth_high = azimuth_high
        self.nerf_renderer = RadianceFieldRenderer(
            n_pts_per_ray=nerf_cfg.n_pts_per_ray,
            n_pts_per_ray_fine=nerf_cfg.n_pts_per_ray_fine,
            min_depth=0.1,
            max_depth=3.0,
            stratified=True,
            stratified_test=False,
            chunk_size=6000,
            n_harmonic_functions_xyz=10,
            n_harmonic_functions_dir=4,
            n_hidden_neurons_xyz=256,
            n_hidden_neurons_dir=128,
            n_layers_xyz=8,
            density_noise_std=0.0,
        )
    def forward(self, z, sample_res=None):
        if sample_res is None:
            sample_res = self.img_size
        rays_xy = sample_full_xys(batch_size=len(z),
                img_size=sample_res).to(z.device)
        batch_size = len(z)
        azimuth_samples = np.random.randint(self.azimuth_low, self.azimuth_high,
                                  (batch_size)).astype(np.float)
        R, T = look_at_view_transform(dist=2.7, elev=batch_size*[0],
                azim=azimuth_samples)
        cameras = FoVOrthographicCameras(
            R = R, 
            T = T, 
            device = z.device,
        )
        rgb_out = self.nerf_renderer(cameras, rays_xy)
        out = rgb_out.permute(0,3,1,2)
        return out
