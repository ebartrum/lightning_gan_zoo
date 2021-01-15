import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
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
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

class GAN(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.discriminator = instantiate(cfg.discriminator)
        self.generator = instantiate(cfg.generator)
        self.cfg=cfg
        self.hparams=cfg
        self.transform = transforms.Compose([
            transforms.Resize(cfg.train.img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(cfg.train.channels_img)],
                [0.5 for _ in range(cfg.train.channels_img)])])
        self.criterion = nn.BCELoss()
        self.fixed_noise = torch.randn(32, cfg.train.noise_dim, 1, 1)
        self.discriminator.apply(init_weights)
        self.generator.apply(init_weights)
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

    # def test_step(self, batch, batch_nb):
    #     import ipdb;ipdb.set_trace()
    #     x, y = batch
    #     loss = F.cross_entropy(self(x), y)
    #     self.log('test/loss', loss)
    #     return loss

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

        # compute FID and KID
        self.log('validation/fid', 1)
        self.log('validation/kid', 1)

    def configure_optimizers(self):
        opt_gen = optim.Adam(self.generator.parameters(),
                lr=self.cfg.train.lr,
                betas=(self.cfg.train.beta1, self.cfg.train.beta2))
        opt_disc = optim.Adam(self.discriminator.parameters(),
                lr=self.cfg.train.lr,
                betas=(self.cfg.train.beta1, self.cfg.train.beta2))
        return [opt_disc, opt_gen], []

    def train_dataloader(self):
        return DataLoader(MNIST("~/datasets", train=True, download=True,
        transform=self.transform),
            num_workers=self.cfg.train.num_workers,
            batch_size=self.cfg.train.batch_size)

    def val_dataloader(self):
        return DataLoader(MNIST("~/datasets", train=False, download=True,
        transform=self.transform),
            num_workers=self.cfg.train.num_workers,
            batch_size=self.cfg.train.batch_size)

    def test_dataloader(self):
        return DataLoader(MNIST("~/datasets", train=False, download=True,
        transform=self.transform),
            num_workers=self.cfg.train.num_workers,
            batch_size=self.cfg.train.batch_size)

@hydra.main(config_name="config")
def train(cfg: DictConfig) -> None:
    model = GAN(cfg)
    tb_logger = CustomTensorBoardLogger('output/',
            name=cfg.name, default_hp_metric=False)
    checkpoint_callback = ModelCheckpoint(monitor='validation/fid',
            filename='model-{epoch:02d}-{fid:.2f}')
    trainer = pl.Trainer(gpus=1, max_epochs=cfg.train.num_epochs, logger=tb_logger,
            callbacks=[checkpoint_callback])    
    trainer.fit(model) 

if __name__ == "__main__":
    train()
