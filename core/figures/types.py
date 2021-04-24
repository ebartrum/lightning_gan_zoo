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
from core.utils.utils import interpolate_sphere
from copy import deepcopy
import math

class Figure(Callback):
    def __init__(self, cfg, parent_dir, monitor=None):
       self.save_dir = os.path.join(parent_dir, cfg.dir)
       self.filename = cfg.filename if cfg.filename else\
               f"{self.__class__.__name__}.png"
       if not os.path.exists(self.save_dir):
           os.makedirs(self.save_dir)
       self.monitor = monitor
       self.current_best_metric = np.inf
       self.save_all = cfg.save_all

    @abstractmethod
    def draw(self, pl_module):
        """
        Draw figure as a numpy array. Type should be float or double.
        Range should be in [0,1]. Dim should be (H,W,3)
        """
        pass

    def save(self, array, timestep=None):
        assert array.min()>=0 and array.max()<=1,\
                "Figure array should lie in [0,1]"
        array = (array*255).astype(int)
        if timestep:
           if not os.path.exists(f"{self.save_dir}/{timestep}"):
               os.makedirs(f"{self.save_dir}/{timestep}")
               imageio.imwrite(f"{self.save_dir}/{timestep}/{self.filename}", array)
        else:
            imageio.imwrite(f"{self.save_dir}/{self.filename}", array)

    def draw_and_save(self, pl_module):
        fig_array = self.draw(pl_module)
        timestep = f"epoch_{pl_module.current_epoch}" if self.save_all else None
        self.save(fig_array, timestep=timestep)

    def on_validation_end(self, trainer, pl_module):
        if self.monitor:
            current_metrics = deepcopy(
                    trainer.logger_connector.logged_metrics)
            current_monitor = current_metrics[self.monitor]
            if current_monitor < self.current_best_metric:
                self.current_best_metric = current_monitor
                print(f"Drawing & saving {self.filename}...")
                self.draw_and_save(pl_module)
            else:
                print(f"Current metric {current_monitor} is worse than current best {self.current_best_metric}. Skipping figures")
        else:
            print(f"Drawing & saving {self.filename}...")
            self.draw_and_save(pl_module)

class AnimationFigure(Figure):
    def __init__(self, cfg, parent_dir, monitor=None, n_frames=40):
       super(AnimationFigure, self).__init__(cfg, parent_dir, monitor)
       self.n_frames = n_frames
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
    def __init__(self, cfg, parent_dir, monitor=None, ncol=4):
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
    def __init__(self, cfg, parent_dir, monitor=None, ncol=4):
        super(AnimationGrid, self).__init__(cfg, parent_dir, monitor)
        self.ncol = ncol

    def draw(self, pl_module):
        pass

    def make_grid(self, rows):
        grid = torchvision.utils.make_grid(torch.cat(
            list(rows),dim=0),
            nrow=self.ncol)
        grid = grid.permute(1,2,0)
        grid = torch.clamp(grid, 0, 1)
        fig_array = grid.detach().cpu().numpy()
        return fig_array

class SampleGrid(Grid):
    def __init__(self, cfg, parent_dir, monitor=None, ncol=4):
        super(SampleGrid, self).__init__(cfg, parent_dir, monitor, ncol)
        self.ncol = ncol

    @torch.no_grad()
    def create_rows(self, pl_module):
        noise = pl_module.noise_distn.sample((self.ncol**2, pl_module.cfg.model.noise_dim)
                ).to(pl_module.device)
        fake = pl_module.generator(noise)
        rows = [fake[self.ncol*i:self.ncol*(i+1)] for i in range(self.ncol)]
        return rows

class AzimuthStep(Grid):
    def __init__(self, cfg, parent_dir, monitor=None, n_steps=8, n_objs=4):
        super(AzimuthStep, self).__init__(cfg, parent_dir, monitor, ncol=n_steps)
        self.n_steps = n_steps
        self.n_objs = n_objs

    @torch.no_grad()
    def create_rows(self, pl_module):
        z = pl_module.noise_distn.sample((self.n_objs, pl_module.cfg.model.noise_dim)
                ).to(pl_module.device)

        azimuth_low = pl_module.cfg.generator.view_args.azimuth_low
        azimuth_high = pl_module.cfg.generator.view_args.azimuth_high
        fixed_elevation = (pl_module.cfg.generator.view_args.elevation_high +
                pl_module.cfg.generator.view_args.elevation_low)/2

        columns = []
        for i in torch.linspace(azimuth_low, azimuth_high, self.n_steps):
            view_in = torch.tensor([i*math.pi/180, fixed_elevation*math.pi/180,
                1.0, 0, 0, 0])
            view_in = view_in.repeat(self.n_objs, 1)
            columns.append(pl_module.generator(z, view_in=view_in))
        rows = torch.stack(columns).permute(1,0,2,3,4)
        return rows

class ElevationStep(Grid):
    def __init__(self, cfg, parent_dir, monitor=None, n_steps=8, n_objs=4):
        super(ElevationStep, self).__init__(cfg, parent_dir, monitor, ncol=n_steps)
        self.n_steps = n_steps
        self.n_objs = n_objs

    @torch.no_grad()
    def create_rows(self, pl_module):
        z = pl_module.noise_distn.sample((self.n_objs, pl_module.cfg.model.noise_dim)
                ).to(pl_module.device)

        elevation_low = pl_module.cfg.generator.view_args.elevation_low
        elevation_high = pl_module.cfg.generator.view_args.elevation_high
        fixed_azimuth = (pl_module.cfg.generator.view_args.azimuth_high +
                pl_module.cfg.generator.view_args.azimuth_low)/2

        columns = []
        for i in torch.linspace(elevation_low, elevation_high, self.n_steps):
            view_in = torch.tensor([fixed_azimuth*math.pi/180, i*math.pi/180, 1.0, 0, 0, 0])
            view_in = view_in.repeat(self.n_objs, 1)
            columns.append(pl_module.generator(z, view_in=view_in))
        rows = torch.stack(columns).permute(1,0,2,3,4)
        return rows

class Interpolation(AnimationGrid):
    def __init__(self, cfg, parent_dir, monitor=None):
        super(Interpolation, self).__init__(cfg, parent_dir, monitor)

    def draw(self, pl_module):
        z1 = pl_module.noise_distn.sample((16, pl_module.cfg.model.noise_dim)
                ).to(pl_module.device)
        z2 = pl_module.noise_distn.sample((16, pl_module.cfg.model.noise_dim)
                ).to(pl_module.device)
        ts = np.linspace(0, 1, self.n_frames)
        
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

class Interpolation3d(AnimationGrid):
    def __init__(self, cfg, parent_dir, monitor=None):
        super(Interpolation3d, self).__init__(cfg, parent_dir, monitor)

    def draw(self, pl_module):
        z1 = pl_module.noise_distn.sample((16, pl_module.cfg.model.noise_dim)
                ).to(pl_module.device)
        z2 = pl_module.noise_distn.sample((16, pl_module.cfg.model.noise_dim)
                ).to(pl_module.device)
        p1 = pl_module.generator.sample_view(16)
        p2 = pl_module.generator.sample_view(16)
        
        ts = np.linspace(0, 1, self.n_frames)
        
        frame_list = []
        for t in ts: 
            z = interpolate_sphere(z1, z2, float(t))
            p = p2*t + p1*(1-t)
            rows = self.create_rows(pl_module, z, p)
            grid = self.make_grid(rows)
            frame_list.append(grid)
        frame_list.extend(frame_list[::-1]) #forwards then backwards
        return frame_list

    def create_rows(self, pl_module, z, p):
        samples = pl_module.generator(z, view_in=p)
        rows = samples[:4], samples[4:8], samples[8:12], samples[12:16]
        return rows

class ElevationGif(AnimationGrid):
    def __init__(self, cfg, parent_dir, num_objs=16, monitor=None):
        super(ElevationGif, self).__init__(cfg, parent_dir, monitor)
        self.num_objs = num_objs

    def draw(self, pl_module):
        z = pl_module.noise_distn.sample((self.num_objs, pl_module.cfg.model.noise_dim)
                ).to(pl_module.device)
        elevation_low = pl_module.cfg.generator.view_args.elevation_low
        elevation_high = pl_module.cfg.generator.view_args.elevation_high
        fixed_azimuth = (pl_module.cfg.generator.view_args.azimuth_high +
                pl_module.cfg.generator.view_args.azimuth_low)/2
        
        frame_list = []
        for i in torch.linspace(elevation_low, elevation_high, self.n_frames):
            view_in = torch.tensor(
                    [fixed_azimuth*math.pi/180, i*math.pi/180, 1.0, 0, 0, 0])
            view_in = view_in.repeat(self.num_objs, 1).to(pl_module.device)
            rows = self.create_rows(pl_module, z, view_in)
            grid = self.make_grid(rows)
            frame_list.append(grid)
        frame_list.extend(frame_list[::-1]) #forwards then backwards
        return frame_list

    def create_rows(self, pl_module, z, view_in):
        samples = pl_module.generator(z, view_in=view_in)
        rows = samples[:4], samples[4:8], samples[8:12], samples[12:16]
        return rows

class AzimuthGif(AnimationGrid):
    def __init__(self, cfg, parent_dir, num_objs=16, monitor=None):
        super(AzimuthGif, self).__init__(cfg, parent_dir, monitor)
        self.num_objs = num_objs

    def draw(self, pl_module):
        z = pl_module.noise_distn.sample((self.num_objs, pl_module.cfg.model.noise_dim)
                ).to(pl_module.device)
        azimuth_low = pl_module.cfg.generator.view_args.azimuth_low
        azimuth_high = pl_module.cfg.generator.view_args.azimuth_high
        fixed_elevation = (pl_module.cfg.generator.view_args.elevation_high +
                pl_module.cfg.generator.view_args.elevation_low)/2
        
        frame_list = []
        for i in torch.linspace(azimuth_low, azimuth_high, self.n_frames):
            view_in = torch.tensor(
                    [i*math.pi/180, fixed_elevation*math.pi/180, 1.0, 0, 0, 0])
            view_in = view_in.repeat(self.num_objs, 1).to(pl_module.device)
            rows = self.create_rows(pl_module, z, view_in)
            grid = self.make_grid(rows)
            frame_list.append(grid)
        frame_list.extend(frame_list[::-1]) #forwards then backwards
        return frame_list

    def create_rows(self, pl_module, z, view_in):
        samples = pl_module.generator(z, view_in=view_in)
        rows = samples[:4], samples[4:8], samples[8:12], samples[12:16]
        return rows
