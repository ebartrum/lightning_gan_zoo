import pytorch_lightning as pl
import torch
from torch import nn, optim, distributions
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from core.utils.utils import gradient_penalty, compute_grad2
from core.utils.utils import init_weights, VerboseShapeExecution
from core.utils.anigan import convert_cam_pred
from hydra.utils import instantiate, call
from abc import abstractmethod
import numpy as np
import math
from core.nerf.utils import sample_images_at_xys, sample_full_xys
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    SoftPhongShader,
    TexturesVertex
)
from pytorch3d.structures import Meshes, Pointclouds
from torch.cuda.amp import autocast
from core.submodules.tps_deformation.tps import functions as tps_functions

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
                mean=[cfg.train.data_mean  for _ in range(cfg.train.channels_img)],
                std=[cfg.train.data_std for _ in range(cfg.train.channels_img)])])
        self.criterion = instantiate(cfg.train.criterion)
        self.noise_distn = instantiate(cfg.model.noise_distn)
        self.fixed_noise = self.noise_distn.sample((8, cfg.model.noise_dim))
        # self.discriminator.apply(init_weights)
        # self.generator.apply(init_weights)
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
        opt_disc = instantiate(self.cfg.disc_optimiser,
                    self.discriminator.parameters())
        opt_gen = instantiate(self.cfg.gen_optimiser,
                    self.generator.parameters())
        scheduler_disc = instantiate(self.cfg.optimisation.lr_scheduler,
                optimizer=opt_disc)
        scheduler_gen = instantiate(self.cfg.optimisation.lr_scheduler,
                optimizer=opt_gen)
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
        noise = self.noise_distn.sample((len(real),self.cfg.model.noise_dim)
                ).to(self.device)
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
                self.cfg.model.noise_dim)).to(self.device)
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
                self.cfg.model.noise_dim)).to(self.device)
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
                self.cfg.model.noise_dim)).to(self.device)
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
        z = self.noise_distn.sample((len(real),
                self.cfg.model.noise_dim)).to(self.device)
        fake = self.generator(z)

        # train discriminator
        if optimizer_idx == 0:
            disc_real, _ = self.discriminator(real)
            loss_disc_real = self.criterion(disc_real,
                    torch.ones_like(disc_real))
            disc_fake, d_z_pred = self.discriminator(fake.detach())
            loss_disc_fake = self.criterion(disc_fake,
                    torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            q_loss = torch.mean((d_z_pred - z)**2)
            self.log('train/d_loss', loss_disc)
            self.log('train/q_loss', q_loss)
            return loss_disc + q_loss

        # train generator
        if optimizer_idx == 1:
            output, d_z_pred = self.discriminator(fake)
            loss_gen = self.criterion(output, torch.ones_like(output))
            q_loss = torch.mean((d_z_pred - z)**2)
            self.log('train/g_loss', loss_gen)
            self.log('train/q_loss', q_loss)
            return loss_gen + q_loss

class PIGAN(BaseGAN):
    def __init__(self, cfg, logging_dir):
        super().__init__(cfg, logging_dir)
        self.resolution_list=self.cfg.resolution_annealing.resolutions
        self.training_resolution = self.resolution_list[0]
        self.current_batch_size = self.cfg.variable_batch_size.batch_sizes[0]

    def configure_optimizers(self):
        opt_disc = instantiate(self.cfg.disc_optimiser,
                    self.discriminator.parameters())
        opt_gen = instantiate(self.cfg.gen_optimiser,
                    self.generator.parameters())
        scheduler_disc = instantiate(self.cfg.optimisation.lr_scheduler,
                optimizer=opt_disc)

        lr_decay_span = 10000
        lr_discr = self.cfg.disc_optimiser.lr
        target_lr_discr = lr_discr/4
        lr_gen = self.cfg.gen_optimiser.lr
        target_lr_gen = lr_gen/5
        D_decay_fn = lambda i: max(1 - i / lr_decay_span, 0) +\
                (target_lr_discr / lr_discr) * min(i / lr_decay_span, 1)
        G_decay_fn = lambda i: max(1 - i / lr_decay_span, 0) +\
                (target_lr_gen / lr_gen) * min(i / lr_decay_span, 1)

        scheduler_disc = LambdaLR(opt_disc, D_decay_fn)
        scheduler_gen = LambdaLR(opt_gen, G_decay_fn)

        return ({'optimizer': opt_disc, 'lr_scheduler': scheduler_disc,
                    'frequency': self.cfg.optimisation.disc_freq},
               {'optimizer': opt_gen, 'lr_scheduler': scheduler_gen,
                   'frequency': self.cfg.optimisation.gen_freq})
        
    def train_dataloader(self):
        if self.current_epoch in\
                self.cfg.variable_batch_size.update_epochs:
            batch_size_index =\
                    self.cfg.variable_batch_size.update_epochs.index(
                    self.current_epoch)+1
            self.current_batch_size = self.cfg.variable_batch_size.\
                    batch_sizes[batch_size_index]
        print(f"Batch size for this epoch: {self.current_batch_size}")
        dataset = instantiate(self.cfg.dataset.train, transform=self.transform)
        return DataLoader(dataset, num_workers=self.cfg.train.num_workers,
                batch_size=self.current_batch_size)

    def pigan_disc_loss(self, real_sampled, fake):
        real_sampled.requires_grad_()
        disc_real = self.discriminator(real_sampled).reshape(-1)
        disc_fake = self.discriminator(fake.clone().detach()).reshape(-1)
        divergence = (F.relu(1 + disc_real) + F.relu(1 - disc_fake)).mean()
        r1_reg = self.cfg.loss_weight.reg * compute_grad2(
                disc_real, real_sampled).mean()
        loss_disc = r1_reg + divergence
        self.log('train/d_loss', loss_disc)
        return loss_disc
        
    def pigan_gen_loss(self, fake):
        output = self.discriminator(fake).reshape(-1)
        loss_gen = output.mean()
        self.log('train/g_loss', loss_gen)
        return loss_gen

    def training_step(self, batch, batch_idx, optimizer_idx,
            cameras=None):
        real, _ = batch
        rays_xy = sample_full_xys(batch_size=len(real),
                img_size=self.training_resolution).to(self.device)
        real_sampled = sample_images_at_xys(real.permute(0,2,3,1), rays_xy)
        real_sampled = real_sampled.permute(0,3,1,2)

        z = self.noise_distn.sample((len(real),
                self.cfg.model.noise_dim)).to(self.device)
        fake = self.generator(z, sample_res=self.training_resolution,
                cameras=cameras)

        if optimizer_idx == 0:
            out = self.pigan_disc_loss(real_sampled, fake[:,:3])
        if optimizer_idx == 1:
            out = self.pigan_gen_loss(fake[:,:3])

        #Step the training resolution scheduler
        self.discriminator.update_iter_()
        return out

class ANIGAN(PIGAN):
    def __init__(self, cfg, logging_dir):
        super().__init__(cfg, logging_dir)
        sigma = 1e-4
        self.lights = PointLights(device=self.device, location=[[0.0, 0.0, -3.0]])
        raster_settings_silhouette = RasterizationSettings(
            image_size=self.cfg.train.img_size, 
            blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
            faces_per_pixel=50, 
        )

        self.renderer_silhouette = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings_silhouette
            ),
            shader=SoftSilhouetteShader()
        )

    def training_step(self, batch, batch_idx, optimizer_idx):
        real, _, shape_analysis = batch
        cameras, scale = convert_cam_pred(shape_analysis['cam_pred'],
                device=self.device)
        scale = torch.ones_like(scale)  #TODO: use scale

        template_verts = shape_analysis['mean_shape']
        template_verts = scale.unsqueeze(1).unsqueeze(1)*template_verts

        verts_rgb = torch.ones_like(template_verts)
        textures = TexturesVertex(verts_features=verts_rgb.to(self.device))
        mesh = Meshes(verts=shape_analysis['verts'],
                faces=shape_analysis['faces'],
                textures=textures)

        with autocast(enabled=False):
            silhouette_images = self.renderer_silhouette(
                    mesh, cameras=cameras, lights=self.lights).detach()
            silhouette_images = silhouette_images[:,:,:,3].unsqueeze(-1)

        # silhouette_images = shape_analysis['mask_pred'].unsqueeze(-1)

        # import matplotlib.pyplot as plt
        # silhouette_images = silhouette_images.permute(0,3,1,2)
        # plt.imshow((silhouette_images[0,3]*255).cpu().int())
        # plt.show()
        # plt.imshow((real[0].permute(1,2,0)*255).cpu().int())
        # plt.show()

        rays_xy = sample_full_xys(batch_size=len(real),
                img_size=self.training_resolution).to(self.device)
        real_sampled = sample_images_at_xys(real.permute(0,2,3,1), rays_xy)
        real_sampled = real_sampled.permute(0,3,1,2)

        z = self.noise_distn.sample((len(real),
                self.cfg.model.noise_dim)).to(self.device)

        deformation_field = self.calculate_deformation(shape_analysis)
        fake = self.generator(z, sample_res=self.training_resolution,
                cameras=cameras, ray_scale=scale,
                deformation_field=deformation_field,
                deformed_verts=shape_analysis['verts']\
                        [:,::self.cfg.nerf.template_subdivision])

        if optimizer_idx == 0:
            out = self.pigan_disc_loss(real_sampled, fake[:,:3])
        if optimizer_idx == 1:
            out = self.pigan_gen_loss(fake[:,:3])
            silhouette_sampled = sample_images_at_xys(
                    silhouette_images, rays_xy).squeeze(-1)
            silhouette_loss = ((fake[:,3] - silhouette_sampled)**2).mean()
            out = out + (self.cfg.loss_weight.silhouette * silhouette_loss)

        #Step the training resolution scheduler
        self.discriminator.update_iter_()
        return out

    def calculate_deformation(self, shape_analysis):
        verts = shape_analysis['verts'][:,::self.cfg.nerf.template_subdivision]
        template_verts =\
            shape_analysis['mean_shape'][:,::self.cfg.nerf.template_subdivision]
        tps_lambda = 0
        coefficient = tps_functions.find_coefficients(
            verts, template_verts, tps_lambda).detach()
        return coefficient

    def validation_step(self, batch, batch_idx):
        real, _, shape_analysis = batch
        return {'real': real}
