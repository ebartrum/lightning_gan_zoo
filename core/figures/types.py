from abc import ABC, abstractmethod
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from itertools import product
import imageio
from pytorch_lightning.callbacks import Callback
from PIL import Image
from core.utils import interpolate_sphere
from copy import deepcopy
from tqdm import tqdm
from core.submodules.graf.graf.utils import color_depth_map

class Figure(Callback):
    def __init__(self, cfg, parent_dir, monitor=None):
       self.save_dir = os.path.join(parent_dir, cfg.dir)
       self.filename = cfg.filename if cfg.filename else\
               f"{self.__class__.__name__}.png"
       if not os.path.exists(self.save_dir):
           os.makedirs(self.save_dir)
       self.monitor = monitor
       self.current_best_metric = np.inf

    @abstractmethod
    def draw(self, pl_module):
        """
        Draw figure as a numpy array. Type should be float or double.
        Range should be in [0,1]. Dim should be (H,W,3)
        """
        pass

    def save(self, array):
        assert array.min()>=0 and array.max()<=1,\
                "Figure array should lie in [0,1]"
        array = (array*255).astype(int)
        imageio.imwrite(f"{self.save_dir}/{self.filename}", array)

    def draw_and_save(self, pl_module):
        fig_array = self.draw(pl_module)
        self.save(fig_array)

    def on_validation_end(self, trainer, pl_module):
        current_metrics = deepcopy(
                trainer.logger_connector.logged_metrics)
        current_monitor = current_metrics[self.monitor]
        if current_monitor < self.current_best_metric:
            self.current_best_metric = current_monitor
            print(f"Drawing & saving {self.filename}...")
            self.draw_and_save(pl_module)
        else:
            print(f"Current metric {current_monitor} is worse than current best {self.current_best_metric}. Skipping figures")

class AnimationFigure(Figure):
    def __init__(self, cfg, parent_dir, monitor):
       super(AnimationFigure, self).__init__(cfg, parent_dir, monitor)
       self.filename = cfg.filename if cfg.filename else\
               f"{self.__class__.__name__}.gif"

    @abstractmethod
    def draw(self, pl_module):
        """
        Draw animation figure as a list of numpy arrays representing frames.
        Type should be float or double. Range should be in [0,1].
        Dim of each frame should be (H,W,3)
        """
        pass

    def save(self, array_list):
        pil_list = []
        for array in array_list:
            assert array.min()>=0 and array.max()<=1,\
                    "Figure frames arrays should lie in [0,1]"
            array = (array*255).astype('uint8')
            pil_list.append(Image.fromarray(array, 'RGB'))
        pil_list[0].save(f"{self.save_dir}/{self.filename}",
                       save_all=True, append_images=pil_list[1:],
                       optimize=False,
                       duration=40,
                       loop=0)

    def draw_and_save(self, pl_module):
        array_list = self.draw(pl_module)
        self.save(array_list)

class Grid(Figure):
    def __init__(self, cfg, parent_dir, monitor, ncol=4):
        super(Grid, self).__init__(cfg, parent_dir, monitor)
        self.ncol = ncol

    @torch.no_grad()
    def draw(self, pl_module):
        grid = torchvision.utils.make_grid(torch.cat(
            list(self.create_rows(pl_module)),dim=0),
            nrow=self.ncol)
        grid = grid.permute(1,2,0)
        grid = torch.clamp(grid, 0, 1)
        fig_array = grid.detach().cpu().numpy()
        return fig_array

class AnimationGrid(AnimationFigure):
    def __init__(self, cfg, parent_dir, monitor, ncol=4):
        super(AnimationGrid, self).__init__(cfg, parent_dir, monitor)
        self.ncol = ncol

    def draw(self, pl_module):
        pass

class SampleGrid(Grid):
    def __init__(self, cfg, parent_dir, monitor, ncol=4):
        super(SampleGrid, self).__init__(cfg, parent_dir, monitor, ncol)

    @torch.no_grad()
    def create_rows(self, pl_module):
        noise = torch.randn(16,
                pl_module.cfg.train.noise_dim, 1, 1).to(pl_module.device)
        fake = pl_module.generator(noise)
        rows = fake[:4], fake[4:8], fake[8:12], fake[12:16]
        return rows

class Interpolation(AnimationGrid):
    def __init__(self, cfg, parent_dir, monitor):
        super(Interpolation, self).__init__(cfg, parent_dir, monitor)

    def draw(self, pl_module):
        n_frames = 40
        z1 = torch.randn(16,
                pl_module.cfg.train.noise_dim, 1, 1).to(pl_module.device)
        z2 = torch.randn(16,
                pl_module.cfg.train.noise_dim, 1, 1).to(pl_module.device)
        ts = np.linspace(0, 1, n_frames)
        
        frame_list = []
        for t in ts: 
            z = interpolate_sphere(z1, z2, float(t))
            rows = self.create_rows(pl_module, z)
            grid = self.make_grid(rows)
            frame_list.append(grid)
        frame_list.extend(frame_list[::-1]) #forwards then backwards
        return frame_list

    def create_rows(self, pl_module, z):
        samples = pl_module.generator(z)
        rows = samples[:4], samples[4:8], samples[8:12], samples[12:16]
        return rows

    def make_grid(self, rows):
        grid = torchvision.utils.make_grid(torch.cat(
            list(rows),dim=0),
            nrow=self.ncol)
        grid = grid.permute(1,2,0)
        grid = torch.clamp(grid, 0, 1)
        fig_array = grid.detach().cpu().numpy()
        return fig_array

class GrafSampleGrid(Grid):
    def __init__(self, cfg, parent_dir, monitor, ncol=4):
        super(GrafSampleGrid, self).__init__(cfg, parent_dir, monitor, ncol)
        self.ntest = 16
        self.ptest = None
        self.ztest = torch.randn(self.ntest, cfg.noise_dim)
        self.batch_size = cfg.batch_size

    @torch.no_grad()
    def create_rows(self, pl_module):
        if self.ptest is None:
            self.ptest = torch.stack([pl_module.generator.sample_pose() for i in range(self.ntest)])
            
        rgb, depth, acc = self.create_samples(pl_module.generator,
                self.ztest.to(pl_module.device), poses=self.ptest)
        rows = rgb[:4], rgb[4:8], rgb[8:12], rgb[12:16]
        return rows

    @torch.no_grad()
    def create_samples(self, generator, z, poses=None):
        generator.eval()

        N_samples = len(z)
        device = generator.device

        z = z.to(device).split(self.batch_size)
        if poses is None:
            rays = [None] * len(z)
        else:
            rays = torch.stack([self.get_rays(generator, poses[i].to(device))
                for i in range(N_samples)])
            rays = rays.split(self.batch_size)

        rgb, disp, acc = [], [], []
        with torch.no_grad():
            for z_i, rays_i in tqdm(zip(z, rays), total=len(z), desc='Create samples...'):
                bs = len(z_i)
                if rays_i is not None:
                    rays_i = rays_i.permute(1, 0, 2, 3).flatten(1, 2)       # Bx2x(HxW)xC -> 2x(BxHxW)x3

                generator.use_test_kwargs = True #TODO: remove this hack.
                rgb_i, disp_i, acc_i, _ = generator(z_i, rays=rays_i)
                generator.use_test_kwargs = False

                reshape = lambda x: x.view(bs, generator.H, generator.W, x.shape[1]).permute(0, 3, 1, 2)  # (NxHxW)xC -> NxCxHxW
                rgb.append(reshape(rgb_i).cpu())
                disp.append(reshape(disp_i).cpu())
                acc.append(reshape(acc_i).cpu())

        rgb = torch.cat(rgb)
        disp = torch.cat(disp)
        acc = torch.cat(acc)

        depth = self.disp_to_cdepth(generator, disp)

        return rgb, depth, acc

    @torch.no_grad()
    def get_rays(self, generator, pose):
        return generator.val_ray_sampler(generator.H, generator.W,
                                              generator.focal, pose)[0]

    @torch.no_grad()
    def disp_to_cdepth(self, generator, disps):
        """Convert depth to color values"""
        if (disps == 2e10).all():           # no values predicted
            return torch.ones_like(disps)

        near, far = generator.render_kwargs_test['near'], generator.render_kwargs_test['far']

        disps = disps / 2 + 0.5  # [-1, 1] -> [0, 1]

        depth = 1. / torch.max(1e-10 * torch.ones_like(disps), disps)  # disparity -> depth
        depth[disps == 1e10] = far  # set undefined values to far plane

        # scale between near, far plane for better visualization
        depth = (depth - near) / (far - near)

        depth = np.stack([color_depth_map(d) for d in depth[:, 0].detach().cpu().numpy()])  # convert to color
        depth = (torch.from_numpy(depth).permute(0, 3, 1, 2) / 255.) * 2 - 1  # [0, 255] -> [-1, 1]

        return depth
