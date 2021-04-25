import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import os
import torch
from torch.nn import functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger
from core.models.standard_networks import Discriminator, Generator
from core.callback_inception_metrics import InceptionMetrics
import hydra
from hydra.utils import instantiate, call
from omegaconf import DictConfig
from core.submodules.gan_stability.metrics import FIDEvaluator
import numpy as np
from glob import glob
import submitit

def find_ckpt(ckpt_dir):
    ckpt_list = [y for x in os.walk(ckpt_dir) for y in glob(os.path.join(x[0], '*.ckpt'))]
    assert len(ckpt_list) <= 1, "Multiple ckpts found!"
    if len(ckpt_list):
        return ckpt_list[0]
    
@hydra.main(config_path="conf", config_name="config")
def train(cfg: DictConfig) -> None:
    seed_everything(42)
    try:
        cluster_job_id = int(submitit.JobEnvironment().job_id)
    except RuntimeError:
        cluster_job_id = None
    if cfg.version is None:
        version = cluster_job_id
    else:
        version = cfg.version
    tb_logger = TestTubeLogger('output/',
            name=cfg.name, version=version)

    logging_dir = tb_logger.experiment.get_data_path(
            tb_logger.experiment.name, tb_logger.experiment.version)
    model = instantiate(cfg.model.lm, cfg, logging_dir=logging_dir)
    callbacks = [instantiate(fig,
                cfg=cfg.figure_details,
                parent_dir=logging_dir,
                monitor='fid' if cfg.figure_details.fid_callback else None)
            for fig in cfg.figures.values()]
                
    if cfg.save_ckpts:
        callbacks.append(ModelCheckpoint(
                monitor='fid', filename='model_best-{fid:.2f}'))
    if cfg.calc_fid:
        callbacks.append(InceptionMetrics(model=model, real_img_dir=cfg.dataset.val.root,
                fake_img_dir=os.path.join(logging_dir,"test_samples"),
                data_transform=model.transform, fid_name="fid",
                n_samples=cfg.val.fid_n_samples))
    ckpt_path = find_ckpt(cfg.train.ckpt_dir) if cfg.train.ckpt_dir else None

    trainer = pl.Trainer(gpus=1, max_epochs=cfg.train.num_epochs,
            logger=tb_logger,
            deterministic=False,
            fast_dev_run=cfg.debug.fast_dev_run, callbacks=callbacks,
            resume_from_checkpoint=ckpt_path, precision=cfg.precision)    
    trainer.fit(model) 

if __name__ == "__main__":
    train()
