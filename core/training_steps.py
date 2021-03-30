import pytorch_lightning as pl
import torch
from torch import nn, optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from core.utils import gradient_penalty, compute_grad2
from core.utils import init_weights, VerboseShapeExecution
from hydra.utils import instantiate, call

class BaseGAN(pl.LightningModule):
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
        self.fixed_noise = torch.randn(8, cfg.train.noise_dim, 1, 1)
        self.discriminator.apply(init_weights)
        self.generator.apply(init_weights)
        if cfg.debug.verbose_shape:
            self.apply(VerboseShapeExecution)

    def training_step(self, batch, batch_idx, optimizer_idx):
        return call(self.cfg.train.training_step, self, batch, batch_idx, optimizer_idx)

    def validation_step(self, batch, batch_idx):
        real, _ = batch
        return {'real': real}

    def validation_epoch_end(self, outputs):
        real = outputs[0]['real'][:len(self.fixed_noise)]
        noise = self.fixed_noise.to(self.device)
        fake = self.generator(noise)
        img_grid_real = torchvision.utils.make_grid(real, normalize=True)
        img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
        self.logger.experiment.add_image('Real',
                img_grid_real, self.current_epoch)
        self.logger.experiment.add_image('Fake',
                img_grid_fake, self.current_epoch)

    def configure_optimizers(self):
        opt_disc = instantiate(self.cfg.optimiser,
                    self.discriminator.parameters())
        opt_gen = instantiate(self.cfg.optimiser,
                    self.generator.parameters())
        scheduler_disc = optim.lr_scheduler.StepLR(opt_disc,
                step_size=self.cfg.optimisation.anneal_every,
                gamma=self.cfg.optimisation.lr_anneal)
        scheduler_gen = optim.lr_scheduler.StepLR(opt_gen,
                step_size=self.cfg.optimisation.anneal_every,
                gamma=self.cfg.optimisation.lr_anneal)
        return ({'optimizer': opt_disc, 'lr_scheduler': scheduler_disc,
                    'frequency': self.cfg.optimisation.disc_freq},
               {'optimizer': opt_gen, 'lr_scheduler': scheduler_gen,
                   'frequency': self.cfg.optimisation.gen_freq})

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

def dc(lm, batch, batch_idx, optimizer_idx):
    real, _ = batch
    noise = torch.randn(len(real),
            lm.cfg.train.noise_dim, 1, 1).to(lm.device)
    fake = lm.generator(noise)

    # train discriminator
    if optimizer_idx == 0:
        disc_real = lm.discriminator(real).reshape(-1)
        loss_disc_real = lm.criterion(disc_real,
                torch.ones_like(disc_real))
        disc_fake = lm.discriminator(fake.detach()).reshape(-1)
        loss_disc_fake = lm.criterion(disc_fake,
                torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        lm.log('train/d_loss', loss_disc)
        return loss_disc

    # train generator
    if optimizer_idx == 1:
        output = lm.discriminator(fake).reshape(-1)
        loss_gen = lm.criterion(output, torch.ones_like(output))
        lm.log('train/g_loss', loss_gen)
        return loss_gen

def gan_stability_r1(lm, batch, batch_idx, optimizer_idx):
    real, _ = batch
    noise = torch.randn(len(real),
            lm.cfg.train.noise_dim, 1, 1).to(lm.device)
    fake = lm.generator(noise)

    # train discriminator
    if optimizer_idx == 0:
        real.requires_grad_()
        disc_real = lm.discriminator(real).reshape(-1)
        loss_disc_real = lm.criterion(disc_real,
                torch.ones_like(disc_real))
        disc_fake = lm.discriminator(fake.detach()).reshape(-1)
        loss_disc_fake = lm.criterion(disc_fake,
                torch.zeros_like(disc_fake))
        r1_reg = lm.cfg.loss_weight.reg * compute_grad2(disc_real, real).mean()
        loss_disc = r1_reg + (loss_disc_real + loss_disc_fake) 
        lm.log('train/d_loss', loss_disc)
        return loss_disc

    # train generator
    if optimizer_idx == 1:
        output = lm.discriminator(fake).reshape(-1)
        loss_gen = lm.criterion(output, torch.ones_like(output))
        lm.log('train/g_loss', loss_gen)
        return loss_gen

def wgan(lm, batch, batch_idx, optimizer_idx):
    for p in lm.discriminator.parameters(): #Clip discriminator weights
        p.data.clamp_(-lm.cfg.train.weight_clip, lm.cfg.train.weight_clip)

    real, _ = batch
    noise = torch.randn(len(real),
            lm.cfg.train.noise_dim, 1, 1).to(lm.device)
    fake = lm.generator(noise)

    # train discriminator
    if optimizer_idx == 0:
        disc_real = lm.discriminator(real).reshape(-1)
        disc_fake = lm.discriminator(fake.detach()).reshape(-1)
        loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake))
        lm.log('train/d_loss', loss_disc)
        return loss_disc

    # train generator
    if optimizer_idx == 1:
        gen_fake = lm.discriminator(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        lm.log('train/g_loss', loss_gen)
        return loss_gen

def wgan_gp(lm, batch, batch_idx, optimizer_idx):
    real, _ = batch
    noise = torch.randn(len(real),
            lm.cfg.train.noise_dim, 1, 1).to(lm.device)
    fake = lm.generator(noise)

    # train discriminator
    if optimizer_idx == 0:
        disc_real = lm.discriminator(real).reshape(-1)
        disc_fake = lm.discriminator(fake.detach()).reshape(-1)
        gp = gradient_penalty(lm.discriminator, real, fake, device=lm.device)
        loss_disc = (lm.cfg.loss_weight.lambda_gp*gp)-(torch.mean(disc_real) - torch.mean(disc_fake))
        lm.log('train/d_loss', loss_disc)
        return loss_disc

    # train generator
    if optimizer_idx == 1:
        gen_fake = lm.discriminator(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        lm.log('train/g_loss', loss_gen)
        return loss_gen
