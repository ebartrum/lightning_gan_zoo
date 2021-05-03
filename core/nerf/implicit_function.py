# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
from typing import List
import torch
from pytorch3d.renderer import RayBundle, ray_bundle_to_ray_points, FoVOrthographicCameras
from .harmonic_embedding import HarmonicEmbedding
from torch import nn
from torch.nn import functional as F
import math
from core.submodules.tps_deformation.tps import functions as tps_functions

def leaky_relu(p = 0.2):
    return nn.LeakyReLU(p)

def exists(val):
    return val is not None

class Siren(nn.Module):
    def __init__(self, dim_in, dim_out, w0 = 1., c = 6., is_first = False, use_bias = True, activation = None):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x, gamma = None, beta = None):
        out =  F.linear(x, self.weight, self.bias)
        # FiLM modulation
        if exists(gamma):
            if len(out.shape) == 3:
                gamma=gamma.unsqueeze(1)
            elif len(out.shape) == 4:
                gamma=gamma.unsqueeze(1).unsqueeze(1)
            out = out * gamma

        if exists(beta):
            if len(out.shape) == 3:
                beta=beta.unsqueeze(1)
            elif len(out.shape) == 4:
                beta=beta.unsqueeze(1).unsqueeze(1)
            out = out + beta

        out = self.activation(out)
        return out

class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

# class EqualLinear(nn.Module):
#     def __init__(self, in_dim, out_dim, lr_mul = 0.1, bias = True):
#         super().__init__()
#         self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
#         if bias:
#             self.bias = nn.Parameter(torch.zeros(out_dim))
#         self.lr_mul = lr_mul
#     def forward(self, input):
#         return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)

class MappingNetwork(nn.Module):
    def __init__(self, dim, dim_out, n_heads=1, depth = 3):
        super().__init__()

        layers = [nn.Linear(dim, dim*n_heads), leaky_relu()]
        for i in range(depth-1):
            layers.extend([nn.Linear(dim*n_heads, dim*n_heads), leaky_relu()])

        self.n_heads = n_heads
        self.dim = dim
        self.dim_out = dim_out
        self.net = nn.Sequential(*layers)

        self.to_gamma = nn.Linear(dim*n_heads, dim_out*n_heads)
        self.to_beta = nn.Linear(dim*n_heads, dim_out*n_heads)

    def forward(self, x):
        x = F.normalize(x, dim = -1)
        x = self.net(x)
        gammas, betas = self.to_gamma(x), self.to_beta(x)
        gammas = gammas.reshape(-1, self.n_heads, self.dim_out)
        betas = betas.reshape(-1, self.n_heads, self.dim_out)
        return gammas, betas

class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0 = 1., w0_initial = 30., use_bias = True, final_activation = None):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            self.layers.append(Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first
            ))

        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

    def forward(self, x, gammas=None, betas=None):
        for i, layer in enumerate(self.layers):
            if gammas is None and betas is None:
                x = layer(x)
            else:
                gamma, beta = gammas[:,i], betas[:,i]
                x = layer(x, gamma, beta)
        return self.last_layer(x)

class SirenRadianceField(torch.nn.Module):
    def __init__(
        self,
        latent_z_dim: int,
        num_layers: int,
        dim_hidden:int
    ):
        super().__init__()

        self.mapping = MappingNetwork(
            dim = latent_z_dim,
            dim_out = dim_hidden,
            n_heads = num_layers
        )

        self.rgb_mapping = MappingNetwork(
            dim = latent_z_dim,
            dim_out = dim_hidden,
            n_heads = 1 
        )

        self.siren = SirenNet(
            dim_in = 3,
            dim_hidden = dim_hidden,
            dim_out = dim_hidden,
            num_layers = num_layers
        )

        self.to_alpha = nn.Linear(dim_hidden, 1)
        self.to_rgb_siren = Siren(
                dim_in = dim_hidden+3,
                dim_out = dim_hidden
            )

        self.to_rgb = nn.Linear(dim_hidden, 3)

    def forward(
        self,
        ray_bundle: RayBundle,
        z: torch.Tensor,
        density_noise_std: float = 0.0,
        **kwargs,
    ):
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        ray_directions = torch.nn.functional.normalize(ray_bundle.directions, dim=-1)
        ray_directions = ray_directions.unsqueeze(2).expand(
                rays_points_world.shape)
        

        gammas, betas = self.mapping(z)
        rgb_gamma, rgb_beta = self.rgb_mapping(z)

        x = self.siren(rays_points_world, gammas, betas)
        alpha = self.to_alpha(x)
        x = torch.cat((x,ray_directions), -1)
        x = self.to_rgb_siren(x, rgb_gamma[:,0], rgb_beta[:,0])
        rgb = self.to_rgb(x)

        rays_densities = torch.sigmoid(alpha)
        rays_colors = torch.sigmoid(rgb)

        return rays_densities, rays_colors

class SirenSingleShape(torch.nn.Module):
    def __init__(
        self,
        latent_z_dim: int,
        num_layers: int,
        dim_hidden:int
    ):
        super().__init__()

        self.mapping = MappingNetwork(
            dim = latent_z_dim,
            dim_out = dim_hidden,
            n_heads = num_layers
        )

        self.rgb_mapping = MappingNetwork(
            dim = latent_z_dim,
            dim_out = dim_hidden,
            n_heads = 1 
        )

        self.siren = SirenNet(
            dim_in = 3,
            dim_hidden = dim_hidden,
            dim_out = dim_hidden,
            num_layers = num_layers
        )

        self.alpha_siren = SirenNet(
            dim_in = 3,
            dim_hidden = dim_hidden,
            dim_out = dim_hidden,
            num_layers = num_layers
        )

        self.to_alpha = nn.Linear(dim_hidden, 1)

        self.to_rgb_siren = Siren(
                dim_in = dim_hidden+3,
                dim_out = dim_hidden
            )

        self.to_rgb = nn.Linear(dim_hidden, 3)

    def forward(
        self,
        ray_bundle: RayBundle,
        z: torch.Tensor,
        density_noise_std: float = 0.0,
        **kwargs,
    ):
        tps_coefficient, deformed_verts = kwargs['deformation_field'],\
                kwargs['deformed_verts']
        
        rays_points_world = ray_bundle_to_ray_points(ray_bundle)
        ray_directions = torch.nn.functional.normalize(ray_bundle.directions, dim=-1)
        ray_directions = ray_directions.unsqueeze(2).expand(
                rays_points_world.shape)
        

        gammas, betas = self.mapping(z)
        rgb_gamma, rgb_beta = self.rgb_mapping(z)

        if tps_coefficient is not None:
            rays_input_shape = rays_points_world.shape
            rays_2d = rays_points_world.flatten(start_dim=1, end_dim=-2)
            rays_points_deformed = tps_functions.transform(rays_2d,
                deformed_verts, tps_coefficient)
            rays_points_deformed = rays_points_deformed.reshape(
                    rays_input_shape)
        else:
            rays_points_deformed = rays_points_world

        x = self.siren(rays_points_deformed, gammas, betas)
        x = torch.cat((x,ray_directions), -1)
        x = self.to_rgb_siren(x, rgb_gamma[:,0], rgb_beta[:,0])
        rgb = self.to_rgb(x)

        alpha = self.to_alpha(self.alpha_siren(rays_points_deformed))

        rays_densities = torch.sigmoid(alpha)
        rays_colors = torch.sigmoid(rgb)

        return rays_densities, rays_colors
