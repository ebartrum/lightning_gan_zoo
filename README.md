# Lightning GAN Zoo :zap: :elephant: :penguin: :panda_face:
GAN models implemented with pytorch lightning and hydra configuration (work in progress)

Usage examples:<br/>
```python train.py +expt=dc_gan dataset=celeb_a```<br/>
```python train.py +expt=wgan dataset=mnist```<br/>
```python train.py +expt=wgan_gp dataset=celeb_a```<br/>
```python train.py +expt=gan_stability_r1 dataset=celeb_a```<br/>

By default, the **FID** score will be monitored on the validation set during the validation step. Model checkpoints are saved when the best **FID** score is attained.
Generator samples and latent space interpolations are saved to the output directory.

### Models currently supported
- DCGAN
- WGAN
- WGAN-GP
- R1 Regularisation GAN from https://github.com/LMescheder/GAN_stability 

### How to specify filepaths
```cp conf/filepaths/example.yaml conf/filepaths/local.yaml```

Edit conf/filepaths/local.yaml with dataset locations on your system

### Example outputs
#### DCGAN:
![DCGAN output](/examples/dc_gan.png)
![DCGAN interpolation](/examples/dc_gan_interpolation.gif)
