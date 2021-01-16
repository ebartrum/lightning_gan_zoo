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

class RainbowSquare(Figure):
    def __init__(self, cfg, parent_dir):
        super(RainbowSquare, self).__init__(cfg, parent_dir)

    def draw(self, pl_module):
        fig_array = np.random.random((512,512,3))
        return fig_array

class MiniRainbowSquare(Figure):
    def __init__(self, cfg, parent_dir):
        super(MiniRainbowSquare, self).__init__(cfg, parent_dir)

    def draw(self, pl_module):
        fig_array = np.random.random((128,128,3))
        return fig_array
