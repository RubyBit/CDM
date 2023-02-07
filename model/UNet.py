import math
import random
from inspect import isfunction
from functools import partial

# %matplotlib inline
# import matplotlib.pyplot as plt
# from tqdm.auto import tqdm
# from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np


# https://github.com/g2archie/UNet-MRI-Reconstruction
# https://amaarora.github.io/2020/09/13/unet.html#understanding-input-and-output-shapes-in-u-net This implementation
# includes the added padding to prevent change of dimensions to produce image of same dimensions as input to
# reconstruct the image The dimensions were adjusted to be compatible with the dimensions of the time embedding of
# the MNIST dataset
class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))


class Encoder(nn.Module):
    def __init__(self, chs=(1, 32, 64, 128, 256)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return x  # previously ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(256, 128, 64, 32)):  # should the chs be the same (reverse) as the encoder?
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            #enc_ftrs = self.crop(encoder_features[i], x)
            #x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class AutoEncoder(nn.Module):  # autoencoder
    def __init__(self, enc_chs=(1, 32, 64, 128, 256), dec_chs=(256, 128, 64, 32), num_class=1, retain_dim=False,
                 out_sz=(572, 572)):
        super().__init__()
        self.out_sz = out_sz
        self.encoder = Encoder(enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim = retain_dim

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, self.out_sz)
        return out


# Measures the reconstruction loss from the encoding the image to latent space and then decoding it back to the image
def autoencoder_loss(x, x_hat):
    return F.binary_cross_entropy(x_hat, x)  # For MNIST dataset (or log prob if we get distributions)


# Latent loss
def latent_loss(f):
    var_1 = sigma2(gamma(f))
    mean1_sqr = (1.0 - var_1) * np.square(f)
    loss_lat = 0.5 * np.sum(mean1_sqr + var_1 - np.log(var_1) - 1.0)
    return loss_lat


def recon_loss(img, enc_img, decoder: Decoder):
    g_0 = gamma(0)
    # numpy normal distribution
    eps_0 = np.random.normal(size=enc_img.size())
    z_0 = variance_map(enc_img, g_0, eps_0)
    # rescale
    z_0_rescaled = z_0 / alpha(g_0)
    # decode
    decoded_img = decoder(z_0_rescaled)
    return autoencoder_loss(img, decoded_img)


############################################################################################################
# Diffusion process functions
############################################################################################################
# The timestep embedding is for the diffusion model to learn the temporal information of the time series
def get_timestep_embedding(timesteps, embedding_dim):
    timesteps *= 1000
    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim) * -emb)
    emb = np.outer(timesteps, emb)
    emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=1)

    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return torch.from_numpy(emb).float()


# Forward diffusion process functions
def gamma(ts, gamma_min=-6, gamma_max=6):
    return gamma_max + (gamma_min - gamma_max) * ts


def sigma2(gamma_x):
    tensor = torch.tensor(gamma_x)
    return torch.sigmoid(-tensor)  # correct?


def alpha(gamma_x):
    return np.sqrt(1 - sigma2(gamma_x))


def variance_map(x, gamma_x, eps):
    return alpha(gamma_x) * x + np.sqrt(sigma2(gamma_x)) * eps


class ResNet(nn.Module):
    def __init__(self, in_ch, out_ch, num_blocks=4, num_layers=4, num_filters=64, kernel_size=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=True, padding_mode='zeros', activation=nn.ReLU, norm=nn.BatchNorm2d,
                 dropout=nn.Dropout2d, residual=True, **kwargs):
        super().__init__()
        self.residual = residual
        self.activation = activation
        self.norm = norm
        self.dropout = dropout
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.kwargs = kwargs

        self.blocks = nn.ModuleList([self._make_block() for _ in range(self.num_blocks)])
        self.head = nn.Conv2d(self.num_filters, self.out_ch, 1)

    def _make_block(self):
        layers = []
        for _ in range(self.num_layers):
            layers.append(nn.Conv2d(self.in_ch, self.num_filters, self.kernel_size, self.stride, self.padding,
                                    self.dilation, self.groups, self.bias, self.padding_mode))
            if self.norm is not None:
                layers.append(self.norm(self.num_filters))
            if self.activation is not None:
                layers.append(self.activation())
            if self.dropout is not None:
                layers.append(self.dropout())
            self.in_ch = self.num_filters
        return nn.Sequential(*layers)

    def forward(self, x, cond=None):
        for block in self.blocks:
            res = x
            x = block(x)
            if cond is not None:
                x += nn.Linear(cond.shape[1], x.shape[1], bias=False)(cond)
            if self.residual:
                x = x + res
        return self.head(x)


# Score neural network for the diffusion process. Approximates what you should do at each timestep
class ScoreNet(nn.Module):
    def __init__(self, latent_dim, embedding_dim, n_blocks=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.resnet = ResNet(self.latent_dim, self.latent_dim, num_blocks=n_blocks, num_layers=4,
                             num_filters=64, kernel_size=1, stride=1, padding=1, dilation=1, groups=1, bias=True,
                             padding_mode='zeros', activation=nn.ReLU, norm=nn.BatchNorm2d, dropout=nn.Dropout2d,
                             residual=True)

    def forward(self, x, t, conditioning):
        timestep = get_timestep_embedding(t, self.embedding_dim)
        cond = torch.cat([timestep, conditioning], dim=1)
        cond = nn.SiLU()(nn.Linear(self.latent_dim, self.embedding_dim * 4)(cond))
        cond = nn.SiLU()(nn.Linear(self.embedding_dim * 4, self.embedding_dim * 4)(cond))
        cond = nn.Linear(self.embedding_dim * 4, self.embedding_dim)(cond)

        h = nn.Linear(self.latent_dim, self.embedding_dim)(x)
        h = self.resnet(h, cond)
        return x + h


def diffusion_loss(z_0, t, score_net, conditioning):
    # z_0 is the initial latent variable
    # t is the time step (time steps need to be discrete)
    # z_t is the latent variable at time t
    # z_t is a function of z_0 and t

    eps = torch.randn_like(z_0)
    gamma_x = gamma(t)
    z_t = variance_map(z_0, gamma_x, eps)

    # The score function is the derivative of the latent variable with respect to time
    score = score_net(z_t, t, conditioning)
    loss_diff_mse = torch.mean((score - z_t) ** 2)

    # The diffusion process is a stochastic process
    T = len(t)
    s = t - (1. / T)
    g_s = gamma(s)
    loss_diff = .5 * np.expm1(g_s - gamma_x) * loss_diff_mse

    return loss_diff


class VariationalDiffusion(nn.Module):
    timesteps: int = 1000
    layers: int = 32
    gamma_min: float = -3.0
    gamma_max: float = 3.0

    def __init__(self, latent_dim, embedding_dim, n_blocks=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.score_net = ScoreNet(self.latent_dim, self.embedding_dim, n_blocks=n_blocks)
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, img, conditioning=None):  # combined loss for diffusion and reconstruction
        # encoding image
        z_0 = self.encoder(img)
        # encoder loss
        loss_recon = recon_loss(img, z_0, self.decoder)

        loss_latent = latent_loss(z_0)

        # diffusion loss
        # we need to sample time steps
        t = torch.rand((z_0.shape[0], 1))
        # discretize time steps
        t = np.ceil(t * self.timesteps)
        loss_diff = diffusion_loss(z_0, t, self.score_net, conditioning)
        return loss_recon + loss_latent + loss_diff

    def sample(self, z, t, conditioning, num_samples=1):
        eps = torch.randn((num_samples, self.latent_dim))
        gamma_x = gamma(t)
        z_t = variance_map(eps, gamma_x, eps)
        score = self.score_net(z_t, t, conditioning)
        return z_t + score

    def sample_from_prior(self, t, num_samples=1):
        return self.sample(t, conditioning=torch.zeros((num_samples, 0)), num_samples=num_samples)

    def sample_from_posterior(self, t, conditioning, num_samples=1):
        return self.sample(t, conditioning=conditioning, num_samples=num_samples)


if __name__ == "__main__":
    # model
    model = VariationalDiffusion(32, 32)
    # a random image 28x28x1
    img = torch.randn(1, 1, 28, 28)
    model(img)
