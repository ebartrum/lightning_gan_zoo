import pytorch_lightning as pl
import torch
from torch import nn, optim, distributions
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from core.utils import gradient_penalty, compute_grad2
from core.utils import init_weights, VerboseShapeExecution
from hydra.utils import instantiate, call
from abc import abstractmethod

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
        self.noise_distn = instantiate(cfg.model.noise_distn)
        self.fixed_noise = self.noise_distn.sample((8, cfg.model.noise_dim, 1, 1))
        self.discriminator.apply(init_weights)
        self.generator.apply(init_weights)
        if cfg.debug.verbose_shape:
            self.apply(VerboseShapeExecution)

    @abstractmethod
    def training_step(self, batch, batch_idx, optimizer_idx):
        pass

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
        scheduler_disc = instantiate(self.cfg.optimisation.lr_scheduler,
                opt_disc)
        scheduler_gen = instantiate(self.cfg.optimisation.lr_scheduler,
                opt_gen)
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

class DCGAN(BaseGAN):
    def training_step(self, batch, batch_idx, optimizer_idx):
        real, _ = batch
        noise = self.noise_distn.sample((len(real),
                self.cfg.model.noise_dim, 1, 1)).to(self.device)
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

class GANStabilityR1(BaseGAN):
    def training_step(self, batch, batch_idx, optimizer_idx):
        real, _ = batch
        noise = self.noise_distn.sample((len(real),
                self.cfg.model.noise_dim, 1, 1)).to(self.device)
        fake = self.generator(noise)

        # train discriminator
        if optimizer_idx == 0:
            real.requires_grad_()
            disc_real = self.discriminator(real).reshape(-1)
            loss_disc_real = self.criterion(disc_real,
                    torch.ones_like(disc_real))
            disc_fake = self.discriminator(fake.detach()).reshape(-1)
            loss_disc_fake = self.criterion(disc_fake,
                    torch.zeros_like(disc_fake))
            r1_reg = self.cfg.loss_weight.reg * compute_grad2(disc_real, real).mean()
            loss_disc = r1_reg + (loss_disc_real + loss_disc_fake) 
            self.log('train/d_loss', loss_disc)
            return loss_disc

        # train generator
        if optimizer_idx == 1:
            output = self.discriminator(fake).reshape(-1)
            loss_gen = self.criterion(output, torch.ones_like(output))
            self.log('train/g_loss', loss_gen)
            return loss_gen

class WGAN(BaseGAN):
    def training_step(self, batch, batch_idx, optimizer_idx):
        for p in self.discriminator.parameters(): #Clip discriminator weights
            p.data.clamp_(-self.cfg.train.weight_clip,
                    self.cfg.train.weight_clip)

        real, _ = batch
        noise = self.noise_distn.sample((len(real),
                self.cfg.model.noise_dim, 1, 1)).to(self.device)
        fake = self.generator(noise)

        # train discriminator
        if optimizer_idx == 0:
            disc_real = self.discriminator(real).reshape(-1)
            disc_fake = self.discriminator(fake.detach()).reshape(-1)
            loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake))
            self.log('train/d_loss', loss_disc)
            return loss_disc

        # train generator
        if optimizer_idx == 1:
            gen_fake = self.discriminator(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            self.log('train/g_loss', loss_gen)
            return loss_gen

class WGANGP(BaseGAN):
    def training_step(self, batch, batch_idx, optimizer_idx):
        real, _ = batch
        noise = self.noise_distn.sample((len(real),
                self.cfg.model.noise_dim, 1, 1)).to(self.device)
        fake = self.generator(noise)

        # train discriminator
        if optimizer_idx == 0:
            disc_real = self.discriminator(real).reshape(-1)
            disc_fake = self.discriminator(fake.detach()).reshape(-1)
            gp = gradient_penalty(self.discriminator,
                    real, fake, device=self.device)
            loss_disc = (self.cfg.loss_weight.lambda_gp*gp)\
                    -(torch.mean(disc_real) - torch.mean(disc_fake))
            self.log('train/d_loss', loss_disc)
            return loss_disc

        # train generator
        if optimizer_idx == 1:
            gen_fake = self.discriminator(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            self.log('train/g_loss', loss_gen)
            return loss_gen

class HOLOGAN(BaseGAN):
    def training_step(self, batch, batch_idx, optimizer_idx):
        real, _ = batch
        noise = self.noise_distn.sample((len(real),
                self.cfg.model.noise_dim, 1, 1)).to(self.device)
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