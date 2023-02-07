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
from torchvision import transforms



#https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html

class Encoder(nn.Module):

    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.GELU):

        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), 
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), 
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2),
            act_fn(),
            nn.Flatten(), 
            nn.Linear(2*16*c_hid, latent_dim)
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):

    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object = nn.GELU):

        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2*16*c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), 
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=0),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.Tanh() 
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x
encoder = Encoder(num_input_channels=1, base_channel_size=32, latent_dim=256)
# input image
x    = torch.randn(10000,1, 28, 28)
encoder(x).shape
decoder = Decoder(num_input_channels=1, base_channel_size=32, latent_dim=256)
# input image
x    = torch.randn(1000,256)
decoder(x).shape

# Measures the reconstruction loss from the encoding the image to latent space and then decoding it back to the image
def autoencoder_loss(x, x_hat):
    return F.binary_cross_entropy(x_hat, x)  # For MNIST dataset (or log prob if we get distributions)


# Latent loss
def latent_loss(x_hat):
    var_1=sigma2(gamma(x_hat))
    mean1_sqr = (1.0 - var_1) * np.square(x_hat)
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
        #assert conditioning.shape[0]==timestep.shape[0] #as the output of encoder is (1, encoded_dim) this condition must eb satisfied
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
