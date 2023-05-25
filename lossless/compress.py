import os
import time

import torch
import numpy as np
import utils as util
import utils as rans
import torch.nn as nn
from lossless_cdm import VDM
import lossless_cdm as loss_cdm
from model.advanced_cdm import UNetVDM
from torchvision import datasets, transforms


def tensor_to_ndarray(tensor):
    if type(tensor) is tuple:
        return tuple(tensor_to_ndarray(t) for t in tensor)
    else:
        return tensor.detach().numpy()


def ndarray_to_tensor(arr):
    if type(arr) is tuple:
        return tuple(ndarray_to_tensor(a) for a in arr)
    elif type(arr) is torch.Tensor:
        return arr
    else:
        return torch.from_numpy(arr)


def torch_fun_to_numpy_fun(fun):
    def numpy_fun(*args, **kwargs):
        torch_args = ndarray_to_tensor(args)
        return tensor_to_ndarray(fun(*torch_args, **kwargs))

    return numpy_fun


def bernoulli_obs_append(precision):
    def obs_append(probs):
        def append(state, data):
            return util.bernoullis_append(probs, precision)(
                state, np.int64(data))

        return append

    return obs_append


def bernoulli_obs_pop(precision):
    def obs_pop(probs):
        def pop(state):
            state, data = util.bernoullis_pop(probs, precision)(state)
            return state, torch.Tensor(data)

        return pop

    return obs_pop


def beta_binomial_obs_append(n, precision):
    def obs_append(params):
        a, b = params

        def append(state, data):
            return util.beta_binomials_append(a, b, n, precision)(
                state, np.int64(data))

        return append

    return obs_append


def beta_binomial_obs_pop(n, precision):
    def obs_pop(params):
        a, b = params

        def pop(state):
            state, data = util.beta_binomials_pop(a, b, n, precision)(state)
            return state, torch.Tensor(data)

        return pop

    return obs_pop


if __name__ == '__main__':

    rng = np.random.RandomState(0)
    np.seterr(over='raise')

    prior_precision = 8
    obs_precision = 14
    q_precision = 14

    num_images = 2

    compress_lengths = []

    latent_dim = 50
    latent_shape = (1, latent_dim)
    config = loss_cdm.VDMConfig(noise_schedule="fixed_linear")
    config_unet = loss_cdm.UnetConfig()
    model = loss_cdm.create_model(config, config_unet, (1, 28, 28))
    # train the model
    loss_cdm.train_model_mnist(model, 10)
    model.eval()

    def model_encode(x):
        # encode x to
        mean, gamma, variance = model.encode_x(x)
        return mean, variance

    def model_decode(z):
        mean, variance = model.reconstruct(z, 500)
        return mean, variance


    rec_net = torch_fun_to_numpy_fun(model_encode)
    gen_net = torch_fun_to_numpy_fun(model_decode)

    obs_append = beta_binomial_obs_append(255, obs_precision)
    obs_pop = beta_binomial_obs_pop(255, obs_precision)

    vae_append = util.vae_append(latent_shape, gen_net, rec_net, obs_append,
                                 prior_precision, q_precision)
    vae_pop = util.vae_pop(latent_shape, gen_net, rec_net, obs_pop,
                           prior_precision, q_precision)

    # load some mnist images
    mnist = datasets.MNIST('data/mnist', train=False, download=True,
                           transform=transforms.Compose([transforms.ToTensor()]))
    images = mnist.test_data[:num_images]

    images = [image.float().view(1, -1) for image in images]

    # randomly generate some 'other' bits
    other_bits = rng.randint(low=1 << 16, high=1 << 31, size=50, dtype=np.uint32)
    state = rans.unflatten(other_bits)

    print_interval = 10
    encode_start_time = time.time()
    for i, image in enumerate(images):
        state = vae_append(state, image)

        if not i % print_interval:
            print('Encoded {}'.format(i))

        compressed_length = 32 * (len(rans.flatten(state)) - len(other_bits)) / (i + 1)
        compress_lengths.append(compressed_length)

    print('\nAll encoded in {:.2f}s'.format(time.time() - encode_start_time))
    compressed_message = rans.flatten(state)

    compressed_bits = 32 * (len(compressed_message) - len(other_bits))
    print("Used " + str(compressed_bits) + " bits.")
    print('This is {:.2f} bits per pixel'.format(compressed_bits
                                                 / (num_images * 784)))

    if not os.path.exists('results'):
        os.mkdir('results')
    np.savetxt('compressed_lengths_cts', np.array(compress_lengths))

    state = rans.unflatten(compressed_message)
    decode_start_time = time.time()

    for n in range(len(images)):
        state, image_ = vae_pop(state)
        original_image = images[len(images) - n - 1].numpy()
        np.testing.assert_allclose(original_image, image_)

        if not n % print_interval:
            print('Decoded {}'.format(n))

    print('\nAll decoded in {:.2f}s'.format(time.time() - decode_start_time))

    recovered_bits = rans.flatten(state)
    assert all(other_bits == recovered_bits)
