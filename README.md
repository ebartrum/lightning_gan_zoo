# Lightning GAN Zoo :zap: :elephant: :penguin: :panda_face:
GAN models (including 3D controllable models) implemented with [Pytorch Lightning](https://www.pytorchlightning.ai/) and [Hydra](https://hydra.cc/) configuration.
This is an unofficial project and work in progress. **Model correctness is not guaranteed.**

Usage examples:<br/>
```python train.py +expt=dc_gan dataset=celeb_a```<br/>
```python train.py +expt=wgan dataset=mnist```<br/>
```python train.py +expt=wgan_gp dataset=celeb_a```<br/>
```python train.py +expt=gan_stability_r1 dataset=celeb_a```<br/>
```python train.py +expt=hologan dataset=celeb_a```<br/>

By default, the **FID** score will be monitored on the validation set during the validation step. Model checkpoints are saved when the best **FID** score is attained.
Generator samples and latent space interpolations are saved to the output directory. Varying view outputs saved for 3D controllable models.

### Models currently supported
- [DCGAN](https://arxiv.org/abs/1511.06434v2)
- [WGAN](https://arxiv.org/abs/1701.07875v3)
- [WGAN-GP](https://arxiv.org/abs/1704.00028v3)
- [R1 Regularisation GAN](https://github.com/LMescheder/GAN_stability)
- [HoloGAN](https://www.monkeyoverflow.com/hologan-unsupervised-learning-of-3d-representations-from-natural-images/)

### How to specify filepaths
```cp conf/filepaths/example.yaml conf/filepaths/local.yaml```

Edit conf/filepaths/local.yaml with dataset locations on your system

### Example outputs
#### DCGAN:
![DCGAN output](/examples/dc_gan.png)
![DCGAN interpolation](/examples/dc_gan_interpolation.gif)

#### HoloGAN (3D controllable):
![HoloGAN output](/examples/hologan.png)
![HoloGAN azimuth control](/examples/hologan_azimuth.gif)

### Setup Environment
Conda environment yaml file coming soon.<br />
Install pytorch and pytorch lightning.<br />
Install required hydra version using:<br />
``` pip install hydra-core==1.1.0dev3 ```
