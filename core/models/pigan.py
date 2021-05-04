"""
Discriminator and Generator implementation from DCGAN paper
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
from core.nerf.nerf_renderer import RadianceFieldRenderer
from core.nerf.utils import sample_full_xys
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    look_at_view_transform,
)
import numpy as np
from core.utils.coordconv import CoordConv

class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g,
            nerf_cfg, view_args, img_size=64):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.azimuth_low = view_args.azimuth_low
        self.azimuth_high = view_args.azimuth_high
        self.camera_dist = view_args.camera_dist
        self.nerf_renderer = RadianceFieldRenderer(
            n_pts_per_ray=nerf_cfg.n_pts_per_ray,
            n_pts_per_ray_fine=nerf_cfg.n_pts_per_ray_fine,
            min_depth=nerf_cfg.min_depth,
            max_depth=nerf_cfg.max_depth,
            stratified=nerf_cfg.stratified,
            stratified_test=nerf_cfg.stratified_test,
            chunk_size=nerf_cfg.chunk_size,
            siren_dim_hidden=nerf_cfg.siren_dim_hidden,
            siren_num_layers=nerf_cfg.siren_num_layers,
            density_noise_std=nerf_cfg.density_noise_std,
            latent_z_dim=nerf_cfg.latent_z_dim,
            white_bg=nerf_cfg.white_bg,
            single_shape=nerf_cfg.single_shape
        )

    def pose_to_cameras(self, view_in, device):
        azimuth_samples = view_in[:,0]*180/math.pi
        elevation_samples = torch.zeros_like(azimuth_samples) 
        R, T = look_at_view_transform(dist=self.camera_dist,
                elev=elevation_samples,
                azim=azimuth_samples)
        cameras = FoVOrthographicCameras(
            R = R, 
            T = T, 
            device = device,
        )
        return cameras
        

    def sample_cameras(self, batch_size, device):
        azimuth_samples = np.random.randint(self.azimuth_low,
                self.azimuth_high, (batch_size)).astype(np.float)
        R, T = look_at_view_transform(dist=self.camera_dist, elev=batch_size*[0],
                azim=azimuth_samples)
        cameras = FoVOrthographicCameras(
            R = R, 
            T = T, 
            device = device,
        )
        return cameras

    def forward(self, z, sample_res=None, cameras=None, ray_scale=None,
            deformation_parameters=None, deformed_verts=None):
        if sample_res is None:
            sample_res = self.img_size
        rays_xy = sample_full_xys(batch_size=len(z),
                img_size=sample_res).to(z.device)
        if ray_scale is not None:
            rays_xy /= ray_scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        batch_size = len(z)

        if cameras is None:
            cameras = self.sample_cameras(batch_size, device=z.device)

        rgba_out = self.nerf_renderer(z, cameras, rays_xy,
                deformation_parameters=deformation_parameters,
                deformed_verts=deformed_verts)
        rgba_out = rgba_out.permute(0,3,1,2)
        return rgba_out

def leaky_relu(p = 0.2):
    return nn.LeakyReLU(p)

class DiscriminatorBlock(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.res = CoordConv(dim, dim_out, kernel_size = 1, stride = 2)

        self.net = nn.Sequential(
            CoordConv(dim, dim_out, kernel_size = 3, padding = 1),
            leaky_relu(),
            CoordConv(dim_out, dim_out, kernel_size = 3, padding = 1),
            leaky_relu()
        )

        self.down = nn.AvgPool2d(2)

    def forward(self, x):
        res = self.res(x)
        x = self.net(x)
        x = self.down(x)
        x = x + res
        return x

class Discriminator(nn.Module):
    def __init__(
        self,
        img_size,
        init_chan = 64,
        max_chan = 400,
        init_resolution = 32,
        add_layer_iters = 10000,
        final_sigmoid = False
    ):
        super().__init__()
        resolutions = math.log2(img_size)
        assert resolutions.is_integer(), 'image size must be a power of 2'
        assert math.log2(init_resolution).is_integer(), 'initial resolution must be power of 2'

        resolutions = int(resolutions)
        layers = resolutions - 1

        chans = list(reversed(list(map(lambda t: 2 ** (11 - t), range(layers)))))
        chans = list(map(lambda n: min(max_chan, n), chans))
        chans = [init_chan, *chans]
        final_chan = chans[-1]

        self.from_rgb_layers = nn.ModuleList([])
        self.layers = nn.ModuleList([])
        self.img_size = img_size
        self.resolutions = list(map(lambda t: 2 ** (resolutions - t), range(layers)))

        for resolution, in_chan, out_chan in zip(self.resolutions, chans[:-1], chans[1:]):

            from_rgb_layer = nn.Sequential(
                CoordConv(3, in_chan, kernel_size = 1),
                leaky_relu()
            ) if resolution >= init_resolution else None

            self.from_rgb_layers.append(from_rgb_layer)

            self.layers.append(DiscriminatorBlock(
                dim = in_chan,
                dim_out = out_chan
            ))

        self.final_conv = CoordConv(final_chan, 1, kernel_size = 2)
        if final_sigmoid:
            self.final_sigmoid = nn.Sigmoid()
        else:
            self.final_sigmoid = None

        self.add_layer_iters = add_layer_iters
        self.register_buffer('alpha', torch.tensor(0.))
        self.register_buffer('resolution', torch.tensor(init_resolution))
        self.register_buffer('iterations', torch.tensor(0.))

    def increase_resolution_(self):
        if self.resolution >= self.img_size:
            return

        self.alpha += self.alpha + (1 - self.alpha)
        self.iterations.fill_(0.)
        self.resolution *= 2

    def update_iter_(self):
        self.iterations += 1
        self.alpha = self.alpha - (1 / self.add_layer_iters)
        self.alpha.clamp_(min = 0.)

    def forward(self, img):
        x = img

        for resolution, from_rgb, layer in zip(self.resolutions, self.from_rgb_layers, self.layers):
            if self.resolution < resolution:
                continue

            if self.resolution == resolution:
                x = from_rgb(x)

            if bool(resolution == (self.resolution // 2)) and bool(self.alpha > 0):
                x_down = F.interpolate(img, scale_factor = 0.5)
                x = x * (1 - self.alpha) + from_rgb(x_down) * self.alpha

            x = layer(x)

        out = self.final_conv(x)
        if self.final_sigmoid:
            out = self.final_sigmoid(out)
        return out
