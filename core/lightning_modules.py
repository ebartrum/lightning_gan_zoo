import pytorch_lightning as pl
from hydra.utils import instantiate, call
from torchvision import transforms
from torch import nn, optim
import torch
from core.utils import init_weights, VerboseShapeExecution, compute_grad2
from torch.utils.data import DataLoader
import torchvision
from core.submodules.graf.graf.config import get_hwfr
from core.submodules.graf.graf.transforms import ImgToPatch
from torch.nn import functional as F
from torch import autograd

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

class GRAF(GAN):
    def __init__(self, cfg, logging_dir=None):
        super().__init__(cfg, logging_dir)
        assert(not cfg.data.orthographic), "orthographic not yet supported"
        hwfr = get_hwfr(cfg)
        self.img_to_patch = ImgToPatch(self.generator.ray_sampler,
                hwfr[:3])
        self.reg_param = cfg.loss_weight.reg_param
        self.transform = transforms.Compose([
            transforms.Resize(cfg.data.imsize),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1),
        ])

    def training_step(self, batch, batch_idx, optimizer_idx):
        x_real = batch
        
        self.generator.ray_sampler.iterations = self.global_step//2   # for scale annealing

        # Sample patches for real data
        rgbs = self.img_to_patch(x_real.to(self.device))
        z = torch.randn(self.cfg['train']['batch_size'],
                self.cfg['train']['noise_dim'], device=self.device)

        if optimizer_idx == 0:
            return self.discriminator_trainstep(rgbs,z=z)
        if optimizer_idx == 1:
            return self.generator_trainstep(z=z)

    def generator_trainstep(self, z):
        if self.cfg['nerf']['decrease_noise']:
          self.generator.decrease_nerf_noise(self.global_step//2)
        x_fake = self.generator(z)
        d_fake = self.discriminator(x_fake)
        gloss = self.criterion(d_fake, torch.ones_like(d_fake))
        self.log('generator_loss', gloss)
        return gloss

    def discriminator_trainstep(self, x_real, z):
        x_real.requires_grad_()
        d_real = self.discriminator(x_real)
        dloss_real = self.criterion(d_real, torch.ones_like(d_real))

        with torch.no_grad():
            x_fake = self.generator(z)
        x_fake.requires_grad_()
        d_fake = self.discriminator(x_fake)
        dloss_fake = self.criterion(d_fake, torch.zeros_like(d_fake))
        dloss = (dloss_real + dloss_fake)
        r1_reg = self.reg_param * compute_grad2(d_real, x_real).mean()
        self.log('discriminator_loss', dloss)
        self.log('regularizer_loss', r1_reg)
        return r1_reg + dloss

    def validation_step(self, batch, batch_idx):
        self.log("fid", 1/(self.global_step+1))
