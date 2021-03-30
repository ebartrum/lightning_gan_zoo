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
from pytorch_lightning.loggers import TestTubeLogger
from core.networks import Discriminator, Generator
from core.utils import init_weights, VerboseShapeExecution
from core.callback_fid import FIDCallback
import hydra
from hydra.utils import instantiate, call
from omegaconf import DictConfig
from core.submodules.gan_stability.metrics import FIDEvaluator
import numpy as np
from glob import glob
import submitit

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

def find_ckpt(ckpt_dir):
    ckpt_list = [y for x in os.walk(ckpt_dir) for y in glob(os.path.join(x[0], '*.ckpt'))]
    assert len(ckpt_list) <= 1, "Multiple ckpts found!"
    if len(ckpt_list):
        return ckpt_list[0]
    
@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    seed_everything(42)
    version = submitit.JobEnvironment().job_id if cfg.version=="$slurm_job_id"\
            else cfg.version
    tb_logger = TestTubeLogger('output/',
            name=cfg.name, version=int(version))

    logging_dir = tb_logger.experiment.get_data_path(
            tb_logger.experiment.name, tb_logger.experiment.version)
    model = GAN(cfg, logging_dir=logging_dir)
    callbacks = [instantiate(fig,
                cfg=cfg.figure_details,
                parent_dir=logging_dir,
                monitor='fid')
            for fig in cfg.figures.values()]
                
    callbacks.append(ModelCheckpoint(
            monitor='fid', 
            filename='model_best'))
    callbacks.append(FIDCallback(real_img_dir=cfg.dataset.val.root,
            fake_img_dir=os.path.join(logging_dir,"test_samples"), cfg=cfg,
            data_transform=model.transform, fid_name="fid",
            n_samples=cfg.val.fid_n_samples))
    ckpt_path = find_ckpt(cfg.train.ckpt_dir) if cfg.train.ckpt_dir else None

    trainer = pl.Trainer(gpus=1, max_epochs=cfg.train.num_epochs,
            logger=tb_logger,
            deterministic=True,
            fast_dev_run=cfg.debug.fast_dev_run, callbacks=callbacks,
            resume_from_checkpoint=ckpt_path)    
    trainer.fit(model) 

if __name__ == "__main__":
    train()
