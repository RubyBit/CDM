import math
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
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs=(256, 128, 64, 32)):  # should the chs be the same (reverse) as the encoder?
        super().__init__()
        self.chs = chs
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
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
    return torch.sigmoid(-gamma_x)


def alpha(gamma_x):
    return np.sqrt(1 - sigma2(gamma_x))


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

    def forward(self, x):
        for block in self.blocks:
            res = x
            x = block(x)
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
        cond = torch.cat([t, conditioning], dim=1)
        cond = nn.SiLU()(nn.Linear(self.latent_dim, self.embedding_dim * 4)(cond))
        cond = nn.SiLU()(nn.Linear(self.embedding_dim * 4, self.embedding_dim * 4)(cond))
        cond = nn.Linear(self.embedding_dim * 4, self.embedding_dim)(cond)

        h = nn.Linear(self.latent_dim, self.embedding_dim)(x)
        h = self.resnet(torch.cat([h, timestep, cond], dim=1))
        return x + h
