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

class Figure(Callback):
    def __init__(self, cfg, parent_dir):
       self.save_dir = os.path.join(parent_dir, cfg.dir)
       self.filename = cfg.filename if cfg.filename else\
               f"{self.__class__.__name__}.png"
       if not os.path.exists(self.save_dir):
           os.makedirs(self.save_dir)

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
        print(f"Drawing & saving {self.filename}...")
        self.draw_and_save(pl_module)

class AnimationFigure(Figure):
    def __init__(self, cfg, parent_dir):
       super(AnimationFigure, self).__init__(cfg, parent_dir)
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
                       optimize=False, duration=len(array_list), loop=0)

    def draw_and_save(self, pl_module):
        array_list = self.draw(pl_module)
        self.save(array_list)

class Grid(Figure):
    def __init__(self, cfg, parent_dir, ncol=4):
        super(Grid, self).__init__(cfg, parent_dir)
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
    def __init__(self, cfg, parent_dir, ncol=4):
        super(AnimationGrid, self).__init__(cfg, parent_dir)
        self.ncol = ncol

    def draw(self, pl_module):
        pass

class SampleGrid(Grid):
    def __init__(self, cfg, parent_dir, ncol=4):
        super(SampleGrid, self).__init__(cfg, parent_dir, ncol)

    @torch.no_grad()
    def create_rows(self, pl_module):
        noise = torch.randn(16,
                pl_module.cfg.train.noise_dim, 1, 1).to(pl_module.device)
        fake = pl_module.generator(noise)
        rows = fake[:4], fake[4:8], fake[8:12], fake[12:16]
        return rows

class Interpolation(AnimationGrid):
    def __init__(self, cfg, parent_dir):
        super(Interpolation, self).__init__(cfg, parent_dir)

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
