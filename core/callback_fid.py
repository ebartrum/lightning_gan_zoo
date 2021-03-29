import pytorch_lightning as pl
from tqdm import tqdm
import torch
import os
import imageio
from pytorch_fid.fid_score import calculate_activation_statistics,\
        calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
import numpy as np
import pathlib
import glob

def compute_statistics_of_path(path, model, batch_size, dims, device):
    IMAGE_EXTENSIONS = ['bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp']

    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        files = sorted(list(filter(lambda f: f.endswith(tuple(IMAGE_EXTENSIONS)),
            glob.iglob(path+'**/**', recursive=True))))
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, device)
    return m, s

def calculate_inception_statistics_on_paths(paths, batch_size, device, dims):
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    m1, s1 = compute_statistics_of_path(paths[0], model, batch_size,
                                        dims, device)
    m2, s2 = compute_statistics_of_path(paths[1], model, batch_size,
                                        dims, device)
    return m1, s1, m2, s2

class FIDCallback(pl.callbacks.base.Callback):
    def __init__(self, real_img_dir, fake_img_dir, fid_name, cfg,
            data_transform=None, n_samples=5000, batch_size=16):
        self.real_img_dir = real_img_dir
        if not os.path.exists(fake_img_dir):
            os.makedirs(fake_img_dir)

        cache_files = list(
                filter(lambda s: ".npz" in s, os.listdir(self.real_img_dir)))
        self.real_statistics_cache =\
                os.path.join(self.real_img_dir, cache_files[0])\
                if len(cache_files) else None

        self.fake_img_dir = fake_img_dir
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.fid_name = fid_name
        self.cfg = cfg
        self.z_samples = torch.split(
                torch.randn(n_samples, self.cfg.train.noise_dim,1,1),
                batch_size)

    def clear_fake_img_dir(self):
        img_filepath = os.path.join(self.fake_img_dir, "*.png")
        cached_filepath = os.path.join(self.fake_img_dir, "*.npz")
        if os.path.exists(img_filepath):
            os.remove(img_filepath)
        if os.path.exists(cached_filepath):
            os.remove(cached_filepath)

    def cache_statistics(self, mu, sigma):
        cache_filename = os.path.join(self.real_img_dir, "inception_cache.npz")
        np.savez(cache_filename, mu=mu, sigma=sigma)
        self.real_statistics_cache = cache_filename
        
    def on_validation_epoch_start(self, trainer, pl_module):
        self.clear_fake_img_dir()
        pl_module.eval()
        
        with torch.no_grad():
            total_batches = len(self.z_samples)
            for i, z in enumerate(tqdm(self.z_samples,
                desc="Generating test samples")):
                inputs = z.to(pl_module.device)
                samples = pl_module.generator(z.to(pl_module.device)) # get fake images
                if samples.shape[1] == 1:
                    #convert greyscale to RGB
                    samples = torch.cat(3*[samples],dim=1)

                samples = torch.clamp(samples, 0, 1)
                samples = samples.permute(0,2,3,1)
                samples = samples.detach().cpu().numpy()
                samples = (samples*255).astype(int)

                img_indices = [i*self.batch_size + batch_index for batch_index in range(len(z))]
                filenames = [f"{i}.png" for i in img_indices]
                for sample, filename in zip(samples, filenames):
                    imageio.imwrite(f"{self.fake_img_dir}/{filename}", sample)
            
        current_device = pl_module.device
        real_img_path = self.real_statistics_cache\
                if self.real_statistics_cache else self.real_img_dir
        m1, s1, m2, s2 = calculate_inception_statistics_on_paths(
                [real_img_path, self.fake_img_dir],
                batch_size=16,
                device=current_device, dims=2048) 

        if self.real_statistics_cache is None:
            self.cache_statistics(m1,s1)

        fid = calculate_frechet_distance(m1, s1, m2, s2)
        print(f"fid is {fid}")
        pl_module.to(current_device)
        # log FID
        pl_module.log(self.fid_name, fid)

        self.last_global_step = trainer.global_step
