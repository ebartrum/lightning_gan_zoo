import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
import os
from torchvision import transforms
import torchvision
import torch
from torch import nn, optim
from torch.nn import functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from core.logger import CustomTensorBoardLogger
from core.networks import Discriminator, Generator
from core.utils import init_weights, VerboseShapeExecution
from core.callback_fid import FIDCallback
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from core.submodules.gan_stability.metrics import FIDEvaluator
import numpy as np
from glob import glob

class GAN(pl.LightningModule):
    def __init__(self, cfg, logging_dir):
        super().__init__()
        self.discriminator = instantiate(cfg.discriminator)
        self.generator = instantiate(cfg.generator)
        self.cfg=cfg
        self.hparams=cfg
        self.logging_dir=logging_dir
        self.transform = transforms.Compose([
            transforms.Resize((cfg.train.img_size,cfg.train.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(cfg.train.channels_img)],
                [0.5 for _ in range(cfg.train.channels_img)])])
        self.criterion = nn.BCELoss()
        self.fixed_noise = torch.randn(32, cfg.train.noise_dim, 1, 1)
        self.discriminator.apply(init_weights)
        self.generator.apply(init_weights)
        if cfg.debug.verbose_shape:
            self.apply(VerboseShapeExecution)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real, _ = batch
        noise = torch.randn(self.cfg.train.batch_size,
                self.cfg.train.noise_dim, 1, 1).to(self.device)
        fake = self.generator(noise)

        # train discriminator
        if optimizer_idx == 0:
            disc_real = self.discriminator(real).reshape(-1)
            loss_disc_real = self.criterion(disc_real,
                    torch.ones_like(disc_real))
            disc_fake = self.discriminator(fake.detach()).reshape(-1)
            loss_disc_fake = self.criterion(disc_fake,
                    torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            self.log('train/d_loss', loss_disc)
            return loss_disc

        # train generator
        if optimizer_idx == 1:
            output = self.discriminator(fake).reshape(-1)
            loss_gen = self.criterion(output, torch.ones_like(output))
            self.log('train/g_loss', loss_gen)
            return loss_gen

    def validation_step(self, batch, batch_idx):
        real, _ = batch
        noise = self.fixed_noise.to(self.device)
        fake = self.generator(noise)
        
        img_grid_real = torchvision.utils.make_grid(real, normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
        self.logger.experiment.add_image('Real',
                img_grid_real, self.current_epoch)
        self.logger.experiment.add_image('Fake',
                img_grid_fake, self.current_epoch)

    def configure_optimizers(self):
        opt_gen = optim.Adam(self.generator.parameters(),
                lr=self.cfg.train.lr,
                betas=(self.cfg.train.beta1, self.cfg.train.beta2))
        opt_disc = optim.Adam(self.discriminator.parameters(),
                lr=self.cfg.train.lr,
                betas=(self.cfg.train.beta1, self.cfg.train.beta2))
        return [opt_disc, opt_gen], []

    def train_dataloader(self):
        dataset = instantiate(self.cfg.dataset.train, transform=self.transform)
        return DataLoader(dataset, num_workers=self.cfg.train.num_workers,
                batch_size=self.cfg.train.batch_size)

    def val_dataloader(self):
        dataset = instantiate(self.cfg.dataset.val, transform=self.transform)
        return DataLoader(dataset, num_workers=self.cfg.train.num_workers,
            batch_size=self.cfg.train.batch_size)

    def test_dataloader(self):
        dataset = instantiate(self.cfg.dataset.test, transform=self.transform)
        return DataLoader(dataset, num_workers=self.cfg.train.num_workers,
            batch_size=self.cfg.train.batch_size)

def find_ckpt(ckpt_dir):
    ckpt_list = [y for x in os.walk(ckpt_dir) for y in glob(os.path.join(x[0], '*.ckpt'))]
    assert len(ckpt_list) <= 1, "Multiple ckpts found!"
    if len(ckpt_list):
        return ckpt_list[0]
    
@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    seed_everything(42)
    tb_logger = CustomTensorBoardLogger('output/',
            name=cfg.name, default_hp_metric=False)
    model = GAN(cfg, logging_dir=tb_logger.log_dir)
    callbacks = [instantiate(fig,
                cfg=cfg.figure_details,
                parent_dir=tb_logger.log_dir)
            for fig in cfg.figures.values()]
                
    callbacks.append(ModelCheckpoint(monitor='validation/fid',
            filename='model-{epoch:02d}-{validation/fid:.2f}'))
    callbacks.append(FIDCallback(db_stats=cfg.val.inception_stats_filepath,
            cfg=cfg, data_transform=model.transform,
            fid_name="validation/fid"))
    ckpt_path = find_ckpt(cfg.train.ckpt_dir) if cfg.train.ckpt_dir else None

    trainer = pl.Trainer(gpus=1, max_epochs=cfg.train.num_epochs,
            logger=tb_logger, deterministic=True,
            fast_dev_run=cfg.debug.fast_dev_run, callbacks=callbacks,
            resume_from_checkpoint=ckpt_path)    
    trainer.fit(model) 

if __name__ == "__main__":
    train()
