from torchvision.datasets import ImageFolder
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from os import path
import numpy as np
import torch

class AnimalAnalysisFolder(ImageFolder):
    def __init__(
                self,
                root,
                analysis_root,
                transform = None):
        super(AnimalAnalysisFolder, self).__init__(
                root=root, transform=transform)
        self.analysis_root = analysis_root

    def __getitem__(self, index):
        sample, target = super(AnimalAnalysisFolder, self).__getitem__(index)
        sample_path, _ = self.samples[index]
        image_filename = sample_path.split("/")[-1]
        shape_analysis_filename = image_filename.replace("png", "npz")
        shape_analysis_path = path.join(
                self.analysis_root, shape_analysis_filename)
        shape_analysis_data_np = np.load(shape_analysis_path)
        shape_analysis_data = {k: torch.from_numpy(shape_analysis_data_np[k])\
                for k in shape_analysis_data_np}

        return sample, target, shape_analysis_data
