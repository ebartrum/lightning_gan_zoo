import pytorch_lightning as pl
from tqdm import tqdm
import torch
import os
import imageio
from pytorch_fid.fid_score import get_activations,\
        calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
import numpy as np
import pathlib
import glob
import sys
from sklearn.metrics.pairwise import polynomial_kernel

def _sqn(arr):
    flat = np.ravel(arr)
    return flat.dot(flat)

def polynomial_mmd_averages(codes_g, codes_r, n_subsets=50, subset_size=1000,
                            ret_var=True, output=sys.stdout, **kernel_args):
    m = min(codes_g.shape[0], codes_r.shape[0])
    mmds = np.zeros(n_subsets)
    if ret_var:
        vars = np.zeros(n_subsets)
    choice = np.random.choice

    with tqdm(range(n_subsets), desc='MMD', file=output) as bar:
        for i in bar:
            g = codes_g[choice(len(codes_g), subset_size, replace=False)]
            r = codes_r[choice(len(codes_r), subset_size, replace=False)]
            o = polynomial_mmd(g, r, **kernel_args, var_at_m=m, ret_var=ret_var)
            if ret_var:
                mmds[i], vars[i] = o
            else:
                mmds[i] = o
            bar.set_postfix({'mean': mmds[:i+1].mean()})
    return (mmds, vars) if ret_var else mmds


def polynomial_mmd(codes_g, codes_r, degree=3, gamma=None, coef0=1,
                   var_at_m=None, ret_var=True):
    # use  k(x, y) = (gamma <x, y> + coef0)^degree
    # default gamma is 1 / dim
    X = codes_g
    Y = codes_r

    K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

    return _mmd2_and_variance(K_XX, K_XY, K_YY,
                              var_at_m=var_at_m, ret_var=ret_var)

def _mmd2_and_variance(K_XX, K_XY, K_YY, unit_diagonal=False,
                       mmd_est='unbiased', block_size=1024,
                       var_at_m=None, ret_var=True):
    # based on
    # https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py
    # but changed to not compute the full kernel matrix at once
    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)
    if var_at_m is None:
        var_at_m = m

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
        sum_diag2_X = sum_diag2_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

        sum_diag2_X = _sqn(diag_X)
        sum_diag2_Y = _sqn(diag_Y)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)
    K_XY_sums_1 = K_XY.sum(axis=1)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == 'biased':
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2 * K_XY_sum / (m * m))
    else:
        assert mmd_est in {'unbiased', 'u-statistic'}
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m-1))
        if mmd_est == 'unbiased':
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m-1))

    if not ret_var:
        return mmd2

    Kt_XX_2_sum = _sqn(K_XX) - sum_diag2_X
    Kt_YY_2_sum = _sqn(K_YY) - sum_diag2_Y
    K_XY_2_sum = _sqn(K_XY)

    dot_XX_XY = Kt_XX_sums.dot(K_XY_sums_1)
    dot_YY_YX = Kt_YY_sums.dot(K_XY_sums_0)

    m1 = m - 1
    m2 = m - 2
    zeta1_est = (
        1 / (m * m1 * m2) * (
            _sqn(Kt_XX_sums) - Kt_XX_2_sum + _sqn(Kt_YY_sums) - Kt_YY_2_sum)
        - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 1 / (m * m * m1) * (
            _sqn(K_XY_sums_1) + _sqn(K_XY_sums_0) - 2 * K_XY_2_sum)
        - 2 / m**4 * K_XY_sum**2
        - 2 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 2 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    zeta2_est = (
        1 / (m * m1) * (Kt_XX_2_sum + Kt_YY_2_sum)
        - 1 / (m * m1)**2 * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 2 / (m * m) * K_XY_2_sum
        - 2 / m**4 * K_XY_sum**2
        - 4 / (m * m * m1) * (dot_XX_XY + dot_YY_YX)
        + 4 / (m**3 * m1) * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
    )
    var_est = (4 * (var_at_m - 2) / (var_at_m * (var_at_m - 1)) * zeta1_est
               + 2 / (var_at_m * (var_at_m - 1)) * zeta2_est)

    return mmd2, var_est

def compute_activations_of_path(path, model, batch_size, dims, device):
    IMAGE_EXTENSIONS = ['bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp']
    files = sorted(list(filter(lambda f: f.endswith(tuple(IMAGE_EXTENSIONS)),
        glob.iglob(path+'**/**', recursive=True))))
    act = get_activations(files, model, batch_size, dims, device)
    return act

class FIDCallback(pl.callbacks.base.Callback):
    def __init__(self, model, real_img_dir, fake_img_dir, fid_name,
            data_transform=None, n_samples=5000, batch_size=16):
        self.real_img_dir = real_img_dir
        if not os.path.exists(fake_img_dir):
            os.makedirs(fake_img_dir)

        cache_files = list(
                filter(lambda s: ".npz" in s, os.listdir(self.real_img_dir)))
        self.real_statistics_cache =\
                os.path.join(self.real_img_dir, cache_files[0])\
                if len(cache_files) else None

        self.fake_img_dir = fake_img_dir
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.fid_name = fid_name
        self.cfg = model.cfg
        self.z_samples = torch.split(
                model.noise_distn.sample((n_samples, self.cfg.model.noise_dim)),
                batch_size)

    def clear_fake_img_dir(self):
        img_filepath = os.path.join(self.fake_img_dir, "*.png")
        cached_filepath = os.path.join(self.fake_img_dir, "*.npz")
        if os.path.exists(img_filepath):
            os.remove(img_filepath)
        if os.path.exists(cached_filepath):
            os.remove(cached_filepath)

    def cache_statistics(self, mu, sigma):
        cache_filename = os.path.join(self.real_img_dir, "inception_cache.npz")
        np.savez(cache_filename, mu=mu, sigma=sigma)
        self.real_statistics_cache = cache_filename
        
    def on_validation_epoch_start(self, trainer, pl_module):
        self.clear_fake_img_dir()
        pl_module.eval()
        
        with torch.no_grad():
            total_batches = len(self.z_samples)
            for i, z in enumerate(tqdm(self.z_samples,
                desc="Generating test samples")):
                inputs = z.to(pl_module.device)
                samples = pl_module.generator(z.to(pl_module.device)) # get fake images
                if samples.shape[1] == 1:
                    #convert greyscale to RGB
                    samples = torch.cat(3*[samples],dim=1)

                samples = torch.clamp(samples, 0, 1)
                samples = samples.permute(0,2,3,1)
                samples = samples.detach().cpu().numpy()
                samples = (samples*255).astype(int)

                img_indices = [i*self.batch_size + batch_index for batch_index in range(len(z))]
                filenames = [f"{i}.png" for i in img_indices]
                for sample, filename in zip(samples, filenames):
                    imageio.imwrite(f"{self.fake_img_dir}/{filename}", sample)
            
        current_device = pl_module.device
        real_img_path = self.real_statistics_cache\
                if self.real_statistics_cache else self.real_img_dir
        
        # if self.real_statistics_cache is None:
        #     self.cache_statistics(m1,s1)

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = InceptionV3([block_idx]).to(current_device)
        real_act = compute_activations_of_path(
                path=self.real_img_dir, model=model,
                batch_size=16, device=current_device, dims=2048)
        real_mu = np.mean(real_act, axis=0)
        real_sigma = np.cov(real_act, rowvar=False)

        fake_act = compute_activations_of_path(
                path=self.fake_img_dir, model=model,
                batch_size=16, device=current_device, dims=2048)
        fake_mu = np.mean(fake_act, axis=0)
        fake_sigma = np.cov(fake_act, rowvar=False)

        fid = calculate_frechet_distance(
                real_mu, real_sigma, fake_mu, fake_sigma)
        print(f"FID: {fid}")
        kid_values = polynomial_mmd_averages(real_act, fake_act, n_subsets=100)
        kid_mean, kid_stddev = kid_values[0].mean(), kid_values[0].std()
        print(f"KID mean: {kid_mean}, KID stddev: {kid_stddev}")
        
        # log FID/KID
        pl_module.log(self.fid_name, fid)
        pl_module.log("kid", kid_mean)

        pl_module.to(current_device)
        self.last_global_step = trainer.global_step
