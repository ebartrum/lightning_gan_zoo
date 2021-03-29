import pytorch_lightning as pl
from tqdm import tqdm
import torch
import os
import imageio
import os
from pytorch_fid import fid_score

class FIDCallback(pl.callbacks.base.Callback):
    def __init__(self, real_img_dir, fake_img_dir, fid_name, cfg,
            data_transform=None, n_samples=5000, batch_size=16):
        self.real_img_dir = os.path.join(real_img_dir, "face")
        if not os.path.exists(fake_img_dir):
            os.makedirs(fake_img_dir)
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
        # pl_module.to("cpu")
        # torch.cuda.empty_cache()
        fid = fid_score.calculate_fid_given_paths(
                [self.real_img_dir, self.fake_img_dir],
                batch_size=16,
                device=current_device, dims=2048) 
        print(f"fid is {fid}")
        pl_module.to(current_device)
        # log FID
        pl_module.log(self.fid_name, fid)

        self.last_global_step = trainer.global_step
