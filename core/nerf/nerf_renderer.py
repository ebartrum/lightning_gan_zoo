# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
from typing import List, Optional, Tuple

import torch
from pytorch3d.renderer import ray_bundle_to_ray_points,\
        FoVOrthographicCameras, ImplicitRenderer, RayBundle
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import plot_scene
from visdom import Visdom

from .implicit_function import SirenRadianceField
from .raymarcher import EmissionAbsorptionNeRFRaymarcher
from .raysampler import NeRFRaysampler, ProbabilisticRaysampler
from .utils import calc_mse, calc_psnr
import math

class RadianceFieldRenderer(torch.nn.Module):
    def __init__(
        self,
        n_pts_per_ray: int,
        n_pts_per_ray_fine: int,
        min_depth: float,
        max_depth: float,
        stratified: bool,
        stratified_test: bool,
        chunk_size: int,
        latent_z_dim: int,
        siren_dim_hidden: int,
        siren_num_layers: int,
        density_noise_std: float = 0.0,
    ):
        super().__init__()

        # The renderers and implicit functions are stored under the fine/coarse
        # keys in ModuleDict PyTorch modules.
        self._renderer = torch.nn.ModuleDict()
        self._implicit_function = torch.nn.ModuleDict()

        # Init the EA raymarcher used by both passes.
        raymarcher = EmissionAbsorptionNeRFRaymarcher()

        for render_pass in ("coarse", "fine"):
            if render_pass == "coarse":
                # Initialize the coarse raysampler.
                raysampler = NeRFRaysampler(
                    n_pts_per_ray=n_pts_per_ray,
                    min_depth=min_depth,
                    max_depth=max_depth,
                    stratified=stratified,
                    stratified_test=stratified_test,
                )
            elif render_pass == "fine":
                # Initialize the fine raysampler.
                raysampler = ProbabilisticRaysampler(
                    n_pts_per_ray=n_pts_per_ray_fine,
                    stratified=stratified,
                    stratified_test=stratified_test,
                )
            else:
                raise ValueError(f"No such rendering pass {render_pass}")

            # Initialize the fine/coarse renderer.
            self._renderer[render_pass] = ImplicitRenderer(
                raysampler=raysampler,
                raymarcher=raymarcher,
            )

            # Instantiate the fine/coarse SirenRadianceField module.
            self._implicit_function[render_pass] = SirenRadianceField(
                latent_z_dim=latent_z_dim,
                num_layers=siren_num_layers,
                dim_hidden=siren_dim_hidden
            )

        self._density_noise_std = density_noise_std
        self.chunk_size = chunk_size
        self.min_depth=min_depth
        self.max_depth=max_depth
        self.n_pts_per_ray=n_pts_per_ray

    def _process_ray_chunk(
        self,
        z: torch.Tensor,
        camera: CamerasBase,
        chunk_idx: int,
        rays_xy: torch.Tensor
    ) -> dict:

        """
        Samples and renders a chunk of rays.

        Args:
            camera: A batch of cameras from which the scene is rendered.
            chunk_idx: The index of the currently rendered ray chunk.
        """
        # Initialize the outputs of the coarse rendering to None.
        coarse_ray_bundle = None
        coarse_weights = None

        # First evaluate the coarse rendering pass, then the fine one.
        for renderer_pass in ("coarse", "fine"):
            (rgb, weights), ray_bundle_out = self._renderer[renderer_pass](
                z=z,
                cameras=camera,
                volumetric_function=self._implicit_function[renderer_pass],
                chunksize=self.chunk_size,
                chunk_idx=chunk_idx,
                density_noise_std=(self._density_noise_std if self.training else 0.0),
                input_ray_bundle=coarse_ray_bundle,
                ray_weights=coarse_weights,
                rays_xy=rays_xy
            )

            if renderer_pass == "coarse":
                rgb_coarse = rgb
                # Store the weights and the rays of the first rendering pass
                # for the ensuing importance ray-sampling of the fine render.
                coarse_ray_bundle = ray_bundle_out
                coarse_weights = weights

            elif renderer_pass == "fine":
                rgb_fine = rgb

        return {
            "rgb_fine": rgb_fine,
            "rgb_coarse": rgb_coarse,
            "coarse_rays": type(coarse_ray_bundle)(
                *[v.detach().cpu() for k, v in coarse_ray_bundle._asdict().items()]
            ),
            "coarse_weights": coarse_weights.detach().cpu(),
        }

    def concat_rays(self, ray_bundle_list, dim=0):
        origin_list = [rb.origins for rb in ray_bundle_list]
        directions_list = [rb.directions for rb in ray_bundle_list]
        lengths_list = [rb.lengths for rb in ray_bundle_list]
        xys_list = [rb.xys for rb in ray_bundle_list]

        return RayBundle(
            origins=torch.cat(origin_list, dim=dim),
            directions=torch.cat(directions_list, dim=dim),
            lengths=torch.cat(lengths_list, dim=dim),
            xys=torch.cat(xys_list, dim=dim))

    def forward(
        self,
        z: torch.Tensor,
        camera: CamerasBase,
        rays_xy: torch.Tensor
    ) -> torch.Tensor:

        batch_size, device = len(z), z.device
        n_rays = rays_xy.shape[1:-1].numel()
        spatial_output_shape = rays_xy.shape[1:-1]+tuple([3])
        n_chunks = int(math.ceil((n_rays * batch_size) / self.chunk_size))

        # Process the chunks of rays.
        chunk_outputs = [
            self._process_ray_chunk(
                z,
                camera,
                chunk_idx,
                rays_xy
            )
            for chunk_idx in range(n_chunks)
        ]

        out = {
            k: torch.cat(
                [ch_o[k] for ch_o in chunk_outputs], dim=1)
            for k in ("rgb_fine", "rgb_coarse", "coarse_weights")
            }

        for k in ("rgb_fine", "rgb_coarse"):
            out[k] =  out[k].view(-1,*spatial_output_shape)
        out["coarse_rays"] = self.concat_rays(
                [ch_o["coarse_rays"] for ch_o in chunk_outputs], dim=1)
            
        return out['rgb_fine']
