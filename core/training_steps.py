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
    it = lm.global_step
    x_real, _ = batch
    y = torch.zeros(len(x_real)).to(lm.device)

    lm.generator.ray_sampler.iterations = it

    # Sample patches for real data
    rgbs = lm.img_to_patch(x_real.to(lm.device))          # N_samples x C

    # Discriminator updates
    z = torch.randn(len(x_real),
            lm.cfg.train.noise_dim).to(lm.device)
    dloss, reg = lm.gan_trainer.discriminator_trainstep(rgbs, y=y, z=z)

    # Generators updates
    if lm.cfg['nerf']['decrease_noise']:
      lm.generator.decrease_nerf_noise(it)

    z = torch.randn(len(x_real),
            lm.cfg.train.noise_dim).to(lm.device)
    gloss = lm.gan_trainer.generator_trainstep(y=y, z=z)

    # Update learning rate
    lm.g_scheduler.step()
    lm.d_scheduler.step()

    # (ii) Sample if necessary
    # if ((it % config['training']['sample_every']) == 0) or ((it < 500) and (it % 100 == 0)):
    #     print("Creating samples...")
    #     rgb, depth, acc = evaluator.create_samples(ztest.to(device), poses=ptest)
    #     logger.add_imgs(rgb, 'rgb', it)
    #     logger.add_imgs(depth, 'depth', it)
    #     logger.add_imgs(acc, 'acc', it)
    # # (vi) Create video if necessary
    # if ((it+1) % config['training']['video_every']) == 0:
    #     N_samples = 4
    #     zvid = zdist.sample((N_samples,))

    #     basename = os.path.join(out_dir, '{}_{:06d}_'.format(os.path.basename(config['expname']), it))
    #     evaluator.make_video(basename, zvid, render_poses, as_gif=True)
