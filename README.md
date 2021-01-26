# lightning_gan_zoo :zap: :elephant: :penguin: :panda_face:
GAN models implemented with pytorch lightning and hydra configuration (work in progress)

Usage examples:<br/>
```python train.py +expt=dc_gan dataset=celeb_a```<br/>
```python train.py +expt=dc_gan dataset=mnist```<br/>
```python train.py +expt=wgan dataset=celeb_a```<br/>
```python train.py +expt=wgan dataset=mnist```<br/>

By default, the FID score will be monitored on the validation set during the validation step. Model checkpoints are saved when the best FID score is attained.
Generator samples and latent space interpolations are saved to the output directory.

### Models currently supported
- DCGAN
- WGAN
