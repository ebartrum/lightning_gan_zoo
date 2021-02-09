import pytorch_lightning as pl
from hydra.utils import instantiate, call
from torchvision import transforms
from torch import nn, optim
import torch
from core.utils import init_weights, VerboseShapeExecution
from torch.utils.data import DataLoader
import torchvision

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
        return call(self.cfg.train.training_step, self, batch, batch_idx, optimizer_idx)

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
        opt_disc = instantiate(self.cfg.d_optimiser,
                    self.discriminator.parameters())
        opt_gen = instantiate(self.cfg.g_optimiser,
                    self.generator.parameters())
        scheduler_disc = instantiate(self.cfg.scheduler, opt_disc)
        scheduler_gen = instantiate(self.cfg.scheduler, opt_gen)
        return ({'optimizer': opt_disc, 'lr_scheduler': scheduler_disc,
                    'frequency': self.cfg.optimisation.disc_freq},
               {'optimizer': opt_gen, 'lr_scheduler': scheduler_gen,
                   'frequency': self.cfg.optimisation.gen_freq})

    def train_dataloader(self):
        dataset = instantiate(self.cfg.dataset.train, transform=self.transform)
        return DataLoader(dataset, num_workers=self.cfg.train.num_workers,
            batch_size=self.cfg.train.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        dataset = instantiate(self.cfg.dataset.val, transform=self.transform)
        return DataLoader(dataset, num_workers=self.cfg.train.num_workers,
            batch_size=self.cfg.train.batch_size)

    def test_dataloader(self):
        dataset = instantiate(self.cfg.dataset.test, transform=self.transform)
        return DataLoader(dataset, num_workers=self.cfg.train.num_workers,
            batch_size=self.cfg.train.batch_size)
