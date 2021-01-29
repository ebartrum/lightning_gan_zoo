import torch
from core.utils import gradient_penalty, compute_grad2

def dc(lm, batch, batch_idx, optimizer_idx):
    real, _ = batch
    noise = torch.randn(len(real),
            lm.cfg.train.noise_dim).to(lm.device)
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
            lm.cfg.train.noise_dim).to(lm.device)
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
            lm.cfg.train.noise_dim).to(lm.device)
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
            lm.cfg.train.noise_dim).to(lm.device)
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

def graf(lm, batch, batch_idx, optimizer_idx):
    lm.generator.ray_sampler.iterations = lm.global_step   # for scale annealing
    real, _ = batch
    rgbs = lm.img_to_patch(real)          # N_samples x C
    z = torch.randn(len(real),
            lm.cfg.train.noise_dim).to(lm.device)
    y = torch.zeros(len(real)).to(real.device)
    fake = lm.generator(z,y)

    # train discriminator
    if optimizer_idx == 0:
        real.requires_grad_()
        disc_real = lm.discriminator(real, y)
        loss_disc_real = lm.criterion(disc_real,
                torch.ones_like(disc_real))
        disc_fake = lm.discriminator(fake.detach(), y)
        loss_disc_fake = lm.criterion(disc_fake,
                torch.zeros_like(disc_fake))
        r1_reg = lm.cfg.loss_weight.reg * compute_grad2(disc_real, real).mean()
        loss_disc = loss_disc_real + loss_disc_fake
        lm.log('train/loss_disc', loss_disc)
        lm.log('train/r1_reg', r1_reg)
        return loss_disc + r1_reg

    # train generator
    if optimizer_idx == 1:
        # Generators updates
        if lm.cfg.nerf.decrease_noise:
          lm.generator.decrease_nerf_noise(lm.global_step)

        disc_fake = lm.discriminator(fake, y)
        loss_gen = lm.criterion(disc_fake, torch.ones_like(disc_fake))
        lm.log('train/loss_gen', loss_gen)

        #TODO: implement this
        if lm.cfg.train.take_model_average:
            raise NotImplementedError
            update_average(generator_test, generator,
                           beta=config['training']['model_average_beta'])
        return loss_gen
