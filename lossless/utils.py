from functools import reduce

import numpy as np
from scipy.stats import norm, beta

# Range Asymmetrical Numeral System
head_precision = 64
tail_precision = 32
tail_mask = (1 << tail_precision) - 1
head_min = 1 << head_precision - tail_precision

message_init = head_min, ()


# RANS (Range Asymmetrical Numeral System) functions
def append(msg, start, prob, precision):
    start, prob, precision = map(int, [start, prob, precision])
    head, tail = msg
    if head >= prob << head_precision - precision:
        head, tail = head >> tail_precision, (head & tail_mask, tail)

    return (head // prob << precision) + head % prob + start, tail


def pop(msg, stat_func, precision):
    precision = int(precision)
    head, tail = msg

    cf = head & ((1 << precision) - 1)  # cf in interval [start, start + prob)

    symbol, (start, prob) = stat_func(cf)

    start, prob = map(int, [start, prob])

    head = prob * (head >> precision) + cf - start

    if head < head_min:
        head_new, tail = tail
        head = (head << tail_precision) + head_new

    return (head, tail), symbol


def append_symbol(stat_func, precision):
    def _append_symbol(msg, symbol):
        start, prob = stat_func(symbol)
        return append(msg, start, prob, precision)

    return _append_symbol


def pop_symbol(stat_func, precision):
    def _pop_symbol(msg):
        return pop(msg, stat_func, precision)

    return _pop_symbol


def flatten(msg):  # to 1d np array
    out, msg = [msg[0] >> tail_precision, msg[0]], msg[1]

    while msg:
        x_head, msg = msg
        out.append(x_head)

    return np.array(out, dtype=np.uint32)


def unflatten(msg):  # from 1d np array
    return msg[0] << tail_precision | msg[1], reduce(lambda x, y: (y, x), reversed(msg[2:])), ()  # reverse


# ----------------------------------------------------------------------------
# Bits back append and pop
# ----------------------------------------------------------------------------
def bb_ans_append(post_pop, lik_append, prior_append):
    def append(state, data):
        state, latent = post_pop(data)(state)
        state = lik_append(latent)(state, data)
        state = prior_append(state, latent)
        return state

    return append


def bb_ans_pop(prior_pop, lik_pop, post_append):
    def pop(state):
        state, latent = prior_pop(state)
        state, data = lik_pop(latent)(state)
        state = post_append(data)(state, latent)
        return state, data

    return pop


def vae_append(latent_shape, gen_net, rec_net, obs_append, prior_prec=8,
               latent_prec=12):
    """
    Assume that the vae uses an isotropic Gaussian for its prior and diagonal
    Gaussian for its posterior.
    """

    def post_pop(data):
        post_mean, post_stdd = rec_net(data)
        post_mean, post_stdd = np.ravel(post_mean), np.ravel(post_stdd)
        cdfs = [gaussian_latent_cdf(m, s, prior_prec, latent_prec)
                for m, s in zip(post_mean, post_stdd)]
        ppfs = [gaussian_latent_ppf(m, s, prior_prec, latent_prec)
                for m, s in zip(post_mean, post_stdd)]
        return non_uniforms_pop(latent_prec, ppfs, cdfs)

    def lik_append(latent_idxs):
        y = std_gaussian_centres(prior_prec)[latent_idxs]
        obs_params = gen_net(np.reshape(y, latent_shape))
        return obs_append(obs_params)

    prior_append = uniforms_append(prior_prec)
    return bb_ans_append(post_pop, lik_append, prior_append)


def vae_pop(
        latent_shape, gen_net, rec_net, obs_pop, prior_prec=8, latent_prec=12):
    """
    Assume that the vae uses an isotropic Gaussian for its prior and diagonal
    Gaussian for its posterior.
    """
    prior_pop = uniforms_pop(prior_prec, np.prod(latent_shape))

    def lik_pop(latent_idxs):
        y = std_gaussian_centres(prior_prec)[latent_idxs]
        obs_params = gen_net(np.reshape(y, latent_shape))
        return obs_pop(obs_params)

    def post_append(data):
        post_mean, post_stdd = rec_net(data)
        post_mean, post_stdd = np.ravel(post_mean), np.ravel(post_stdd)
        cdfs = [gaussian_latent_cdf(m, s, prior_prec, latent_prec)
                for m, s in zip(post_mean, post_stdd)]
        return non_uniforms_append(latent_prec, cdfs)

    return bb_ans_pop(prior_pop, lik_pop, post_append)


# ----------------------------------------------------------------------------
# Cumulative distribution functions and inverse cumulative distribution
# functions (ppf) for discretised Gaussian and Beta latent distributions.
#
# Latent cdf inputs are indices of buckets of equal width under the 'prior',
# assumed for the purposes of bits back to be in the same family. They lie in
# the range of ints [0, 1 << prior_prec)
#
# cdf outputs are scaled and rounded to map to integers in the range of ints
# [0, 1 << post_prec) instead of the range [0, 1]
#
# For decodability we must satisfy
#     all(ppf(cf) == s for s in range(1 << prior_prec) for cf in
#         range(cdf(s), cdf(s + 1)))
# ----------------------------------------------------------------------------
def _nearest_int(arr):
    # This will break when vectorized
    return int(np.around(arr))


std_gaussian_bucket_cache = {}  # Stores bucket endpoints
std_gaussian_centres_cache = {}  # Stores bucket centres


def std_gaussian_buckets(precision):
    """
    Return the endpoints of buckets partioning the domain of the prior. Each
    bucket has mass 1 / (1 << precision) under the prior.
    """
    if precision in std_gaussian_bucket_cache:
        return std_gaussian_bucket_cache[precision]
    else:
        buckets = np.float32(
            norm.ppf(np.arange((1 << precision) + 1) / (1 << precision)))
        std_gaussian_bucket_cache[precision] = buckets
        return buckets


def std_gaussian_centres(precision):
    """
    Return the centres of mass of buckets partioning the domain of the prior.
    Each bucket has mass 1 / (1 << precision) under the prior.
    """
    if precision in std_gaussian_centres_cache:
        return std_gaussian_centres_cache[precision]
    else:
        centres = np.float32(
            norm.ppf((np.arange(1 << precision) + 0.5) / (1 << precision)))
        std_gaussian_centres_cache[precision] = centres
        return centres


def gaussian_latent_cdf(mean, stdd, prior_prec, post_prec):
    def cdf(idx):
        x = std_gaussian_buckets(prior_prec)[idx]
        return _nearest_int(norm.cdf(x, mean, stdd) * (1 << post_prec))

    return cdf


def gaussian_latent_ppf(mean, stdd, prior_prec, post_prec):
    def ppf(cf):
        x = norm.ppf((cf + 0.5) / (1 << post_prec), mean, stdd)
        # Binary search is faster than using the actual gaussian cdf for the
        # precisions we typically use, however the cdf is O(1) whereas search
        # is O(precision), so for high precision cdf will be faster.
        return np.searchsorted(
            std_gaussian_buckets(prior_prec), x, 'right') - 1

    return ppf


def beta_latent_cdf(
        a_prior, b_prior, a_post, b_post, prior_prec, post_prec):
    def cdf(idx):
        x = beta.ppf(idx / (1 << prior_prec), a_prior, b_prior)
        return _nearest_int(beta.cdf(x, a_post, b_post) * (1 << post_prec))

    return cdf


def beta_latent_ppf(
        a_prior, b_prior, a_post, b_post, prior_prec, post_prec):
    def ppf(cf):
        x = beta.ppf((cf + 0.5) / (1 << post_prec), a_post, b_post)
        return (beta.cdf(x, a_prior, b_prior) * (1 << prior_prec)).astype(int)

    return ppf


# ----------------------------------------------------------------------------
# Statistics functions for encoding and decoding according to uniform and non-
# uniform distributions over the integer symbols in range(1 << precision).
#
# An encoder statfun performs the mapping
#     symbol |--> (start, freq)
#
# A decoder statfun performs the mapping
#     cumulative_frequency |--> symbol, (start, freq)
# ----------------------------------------------------------------------------
uniform_enc_statfun = lambda s: (s, 1)
uniform_dec_statfun = lambda cf: (cf, (cf, 1))


def uniforms_append(precision):
    append_fun = append_symbol(uniform_enc_statfun, precision)

    def append(state, symbols):
        for symbol in reversed(symbols):
            state = append_fun(state, symbol)
        return state

    return append


def uniforms_pop(precision, n):
    pop_fun = pop_symbol(uniform_dec_statfun, precision)

    def pop(state):
        symbols = []
        for i in range(n):
            state, symbol = pop_fun(state)
            symbols.append(symbol)
        return state, np.asarray(symbols)

    return pop


def non_uniform_enc_statfun(cdf):
    def enc(s):
        start = cdf(s)
        freq = cdf(s + 1) - start
        return start, freq

    return enc


def non_uniform_dec_statfun(ppf, cdf):
    def dec(cf):
        idx = ppf(cf)
        start, freq = non_uniform_enc_statfun(cdf)(idx)
        assert start <= cf < start + freq
        return idx, (start, freq)

    return dec


def non_uniforms_append(precision, cdfs):
    def append(state, symbols):
        for symbol, cdf in reversed(list(zip(symbols, cdfs))):
            statfun = non_uniform_enc_statfun(cdf)
            state = append_symbol(statfun, precision)(state, symbol)
        return state

    return append


def non_uniforms_pop(precision, ppfs, cdfs):
    def pop(state):
        symbols = []
        for ppf, cdf in zip(ppfs, cdfs):
            statfun = non_uniform_dec_statfun(ppf, cdf)
            state, symbol = pop_symbol(statfun, precision)(state)
            symbols.append(symbol)
        return state, np.asarray(symbols)

    return pop
