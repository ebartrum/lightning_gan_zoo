from core.models import pigan
from core.nerf.nerf_renderer import RadianceFieldRenderer
from core.anigan.deformer import TPSDeformer
from hydra.utils import instantiate

class Generator(pigan.Generator):
    def __init__(self, channels_noise, channels_img, features_g,
            nerf_cfg, view_args, img_size=64):
        super(Generator, self).__init__(channels_noise, channels_img,
                features_g, nerf_cfg, view_args, img_size=64)
        self.deformer = instantiate(nerf_cfg.deformer)
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
            single_shape=nerf_cfg.single_shape,
            deformer=self.deformer,
            siren_input_channels=45
        )
