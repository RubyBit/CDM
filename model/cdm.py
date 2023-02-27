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


# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html

class Encoder(nn.Module):

    def __init__(self, z_dim=32, hidden_size=256, n_layers=3):
        super().__init__()
        self.z_dim = z_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.resnet = ResNet(embed_dim=hidden_size, middle_size=hidden_size, num_layers=n_layers)
        self.dense1 = torch.nn.Linear(28 * 28, self.hidden_size)
        self.dense2 = torch.nn.Linear(self.hidden_size, self.z_dim)

    def forward(self, x, cond=None):
        img = 2 * x - 1.0
        # reshape img to combine last 3 dimensions
        img = img.reshape(img.shape[0], -1)
        # encode
        img = self.dense1(img)
        img = self.resnet(img, cond)
        weights = self.dense2(img)
        return weights


class Decoder(nn.Module):

    def __init__(self, hidden_size=512, n_layers=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.dense1 = torch.nn.Linear(32, self.hidden_size)
        self.resnet = ResNet(embed_dim=hidden_size, middle_size=hidden_size, num_layers=n_layers)
        self.dense2 = torch.nn.Linear(self.hidden_size, 28 * 28)

    def forward(self, x, cond=None):
        x = x.to(torch.float32)  # because linear layer expects float32
        x = self.dense1(x)
        x = self.resnet(x, cond)

        # return a distribution (make x all positive)
        logits = self.dense2(x)
        # reshape
        logits = logits.reshape(logits.shape[0], 1, 28, 28)
        return torch.distributions.independent.Independent(torch.distributions.Bernoulli(logits=logits), 3)


# Latent loss
def latent_loss(x_hat):
    var_1 = sigma2(gamma(1.0))
    mean_sqr = (1. - var_1) * torch.square(x_hat)
    loss_lat = 0.5 * torch.sum(mean_sqr + var_1 - torch.log(var_1) - 1, dim=1)
    return loss_lat


def recon_loss(img, enc_img, decoder: Decoder, cond=None):
    g_0 = gamma(0.)
    # numpy normal distribution
    eps_0 = torch.normal(0, 1, size=enc_img.shape)
    z_0 = variance_map(enc_img, g_0, eps_0)
    # rescale
    z_0_rescaled = z_0 / alpha(g_0)
    # decode
    decoded_img = decoder(z_0_rescaled, cond)
    # loss
    # convert img from numpy
    img_int = img.round()
    loss = -decoded_img.log_prob(img_int)
    return loss


############################################################################################################
# Diffusion process functions
############################################################################################################
# The timestep embedding is for the diffusion model to learn the temporal information of the time series
def get_timestep_embedding(timesteps, embedding_dim):
    t = timesteps
    t = t * 1000
    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = np.exp(np.arange(half_dim) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=1)
    if embedding_dim % 2 == 1:  # pad
        emb = np.pad(emb, [(0, 0, 0), (0, 1, 0)], mode='constant')

    # combine last 2 dimensions
    emb = emb.reshape(emb.shape[0], -1)
    assert emb.shape == (t.shape[0], embedding_dim)

    return torch.from_numpy(emb).float()


# Forward diffusion process functions
def gamma(ts, gamma_min=-5.0, gamma_max=1.0):
    return gamma_max + (gamma_min - gamma_max) * ts


def sigma2(gamma_x):
    tensor = torch.tensor(gamma_x)
    return torch.sigmoid(-tensor)  # correct?


def alpha(gamma_x):
    return np.sqrt(1 - sigma2(gamma_x))


def variance_map(x, gamma_x, eps):
    return alpha(gamma_x) * x + np.sqrt(sigma2(gamma_x)) * eps


class ResNet(nn.Module):
    # Residual network
    def __init__(self, embed_dim, middle_size=1024, num_layers=10, activation=nn.GELU, norm=nn.LayerNorm):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.activation = activation
        self.norm = norm
        self.middle_size = middle_size

        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            self.blocks.append(self._make_block())

        self.cond_dense = nn.Linear(32, self.middle_size, bias=False)
        self.last_block = nn.ModuleList([self.norm([self.middle_size]), self.activation(),
                                         nn.Linear(self.middle_size, self.embed_dim)])

    def _make_block(self):
        # without convolutional layers
        layers = [self.norm([self.embed_dim]), self.activation(), nn.Linear(self.embed_dim, self.middle_size)]
        return nn.Sequential(*layers)

    def forward(self, x, cond):
        z = x
        for block in self.blocks:
            h = block(z)
            if cond is not None:
                h = h + self.cond_dense(cond)
            h = self.last_block[0](h)

        z = z + h
        return z


# Score neural network for the diffusion process. Approximates what you should do at each timestep
class ScoreNet(nn.Module):
    def __init__(self, latent_dim, embedding_dim, n_blocks=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.dense1 = nn.Linear(64, self.embedding_dim * 2)
        self.dense2 = nn.Linear(self.embedding_dim * 2, self.embedding_dim * 2)
        self.dense3 = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.dense4 = nn.Linear(self.latent_dim, self.embedding_dim)
        self.resnet = ResNet(embed_dim=embedding_dim, middle_size=embedding_dim, num_layers=n_blocks)

    def forward(self, x, t, conditioning):
        timestep = get_timestep_embedding(t, self.embedding_dim)
        # assert conditioning.shape[0]==timestep.shape[0] #as the output of encoder is (1, encoded_dim) this condition must eb satisfied
        cond = timestep
        cond = torch.cat((cond, conditioning), dim=1)
        cond = nn.SiLU()(self.dense1(cond))
        cond = nn.SiLU()(self.dense2(cond))
        cond = self.dense3(cond)

        h = self.dense4(x)  # hardcoded but should be latent_dim
        # h = torch.reshape(h, (1, 32, 1, 1))  # Reshaped for convolutional layers
        h = self.resnet(h, cond)
        return x + h


################################################

# Shortcut for training the autoencoder (encoder and decoder are separate functions here)
###############################################

# EXAMPLE EXECUTION OF short_cut:     short_cut(5,50,100)

# https://nextjournal.com/gkoehler/pytorch-mnist
def train(epoch, train_loader, optimizer, encoder, decoder):
    log_interval = 50
    train_losses = []
    train_counter = []
    loss_f = torch.nn.MSELoss()
    encoder.train()
    decoder.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        encoded_data = encoder(data)
        # Decode data
        decoded_data = decoder(encoded_data)
        loss = loss_f(decoded_data, data)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 1000) + ((epoch - 1) * len(train_loader.dataset)))


def test(test_loader, encoder, decoder):
    loss_f = torch.nn.MSELoss()
    test_losses = []
    encoder.eval()
    decoder.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            encoded_data = encoder(data)
            # Decode data
            output = decoder(encoded_data)
            test_loss += loss_f(output, data).item()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f} \n'.format(
        test_loss))


def short_cut(n_epochs, batch_size_train, batch_size_test):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./', train=True, download=False,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])), batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./', train=False, download=False,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])), batch_size=batch_size_test, shuffle=True)

    encoder = Encoder(num_input_channels=1, base_channel_size=32, latent_dim=256)
    decoder = Decoder(num_input_channels=1, base_channel_size=32, latent_dim=256)
    mean = (0.1307,)
    std = (0.3081,)
    learning_rate = 0.01

    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]
    optimizer = torch.optim.Adam(params_to_optimize, lr=learning_rate)

    for epoch in range(1, n_epochs + 1):
        train(epoch=epoch, train_loader=train_loader, optimizer=optimizer, encoder=encoder, decoder=decoder)
        test(train_loader=train_loader, encoder=encoder, decoder=decoder)


def diffusion_loss(z_0, t, score_net, conditioning, timesteps):
    # z_0 is the initial latent variable
    # t is the time step (time steps need to be discrete)
    # z_t is the latent variable at time t
    # z_t is a function of z_0 and t

    # Eps is a random tensor with the same shape as z_0 drawn from a normal distribution
    eps = torch.randn_like(z_0)
    gamma_x = gamma(t)
    z_t = variance_map(z_0, gamma_x, eps)

    # The score function is the derivative of the latent variable with respect to time
    score = score_net(z_t, t, conditioning)
    loss_diff_mse = torch.sum((torch.square(eps - score)), dim=-1)

    # The diffusion process is a stochastic process
    T = timesteps
    s = t - (1. / T)
    g_s = gamma(s)
    loss_diff = .5 * T * np.expm1(g_s - gamma_x) * loss_diff_mse

    return loss_diff


class VariationalDiffusion(nn.Module):
    timesteps: int = 1000
    layers: int = 32
    gamma_min: float = -3.0
    gamma_max: float = 3.0
    antithetic: bool = True
    classes: int = 10  # 10 digits

    def __init__(self, latent_dim, embedding_dim, n_blocks=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.score_net = ScoreNet(self.latent_dim, self.embedding_dim, n_blocks=n_blocks)
        self.encoder = Encoder(z_dim=embedding_dim)
        self.decoder = Decoder()
        self.embedding_vectors = nn.Embedding(self.classes, self.embedding_dim)

    def forward(self, img, conditioning=None):  # combined loss for diffusion and reconstruction
        cond = self.embedding_vectors(conditioning)

        # encoding image
        z_0 = self.encoder(img, cond)
        # encoder loss
        loss_recon = recon_loss(img, z_0, self.decoder, cond=cond)

        loss_latent = latent_loss(z_0)

        # diffusion loss
        # we need to sample time steps
        if self.antithetic:
            orig_t = torch.rand(1)
            t = np.mod(orig_t + np.arange(0., 1., step=1. / img.shape[0]), 1.0)
            # turn to float32
            t = t.to(torch.float32)
            t = torch.reshape(t, (img.shape[0], 1))
        else:
            t = torch.rand((img.shape[0], 1))

        # discretize time steps
        t = np.ceil(t * self.timesteps) / self.timesteps
        loss_diff = diffusion_loss(z_0, t, self.score_net, cond, self.timesteps)
        return loss_recon, loss_latent, loss_diff

    def sample(self, z_t, step, timesteps, conditioning, guidance_weight=0.):
        eps = torch.randn_like(z_t)
        t = (timesteps - step) / timesteps
        s = (timesteps - step - 1) / timesteps

        g_s = gamma(s)
        g_t = gamma(t)

        cond = conditioning

        eps_hat_cond = self.score_net(z_t, g_t * torch.ones(z_t.shape[0]), cond)

        eps_hat_uncond = self.score_net(z_t, g_t * torch.ones(z_t.shape[0]), torch.zeros_like(z_t))

        eps_hat = (1. + guidance_weight) * eps_hat_cond - guidance_weight * eps_hat_uncond
        a = torch.sigmoid(torch.tensor(g_s))
        b = torch.sigmoid(torch.tensor(g_t))
        c = -np.expm1(g_t - g_s)
        sigma_t = torch.sqrt(sigma2(g_t))
        z_s = torch.sqrt(a / b) * (z_t - sigma_t * c * eps_hat) + np.sqrt((1. - a) * c) * eps
        return z_s

    def sample_from_prior(self, t, num_samples=1):
        return self.sample(t, conditioning=torch.zeros((num_samples, 0)), num_samples=num_samples)

    def sample_from_posterior(self, t, conditioning, num_samples=1):
        return self.sample(t, conditioning=conditioning, num_samples=num_samples)

    def recon(self, img, timesteps, t, conditioning, num_samples=1):
        cond = self.embedding_vectors(conditioning)

        z_0 = self.encoder(img, cond)
        T = timesteps
        t_n = np.ceil(t * T)
        t = t_n / T
        g_t = gamma(t)
        eps = torch.randn(img.shape[0], self.latent_dim)
        # not sure about difference between
        # rng_body = jax.random.fold_in(rng, i)
        # eps = random.normal(rng_body, z_t.shape)
        # and this
        # rng, spl = random.split(rng)
        z_t = variance_map(z_0, g_t, eps)
        diffused = z_t
        for t in range((T - t_n).astype('int'), self.timesteps):
            diffused = self.sample(diffused, t, T, cond)

        g0 = gamma(0.0)
        var0 = sigma2(g0)
        z0_rescaled = diffused / np.sqrt(1.0 - var0)
        reconstructed = self.decoder(z0_rescaled, cond)
        return reconstructed.mean


def TrainVDM(batch_size_train, n_epochs):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('../', train=True, download=False,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor()
                                   ])), batch_size=batch_size_train, shuffle=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = VariationalDiffusion(128, 128).to(device)
    model.train()
    log_interval = 50
    train_losses = []
    train_counter = []
    logs = {}
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.008, weight_decay=1e-4)
    for epoch in range(1, n_epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):

            optimizer.zero_grad()
            loss, values, IMG = model(data)
            loss.backward()
            optimizer.step()
            # plt.plot(train_losses)
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f},{}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(), values))
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx * 1000) + ((epoch - 1) * len(train_loader.dataset)))


if __name__ == "__main__":
    # model
    model = VariationalDiffusion(32, 32, 4)
    # a random image 28x28x1 in range [0,1]
    img = torch.rand((512, 1, 28, 28))
    conditioning = torch.zeros(img.shape[0], dtype=torch.int32)
    losses = model(img, conditioning)
    # rescale losses
    for i in losses:
        print((i * (1. / (np.prod(img.shape[1:]) * np.log(2)))).mean())

    output = model.recon(img, 100, 0.8, conditioning, 1)
    print(output)
