from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class Hypermodel(hk.Module):
    def __init__(self, output_sizes):
        super().__init__()
        self.linear = hk.nets.MLP(output_sizes=output_sizes)

    def __call__(self, x):
        x = jnp.mean(x, axis=0)
        return self.linear(x) / 100


def init_fn(shape, dtype, alpha):
    return alpha * jnp.ones(shape=shape, dtype=dtype)


def gaussian_log_prob(mu, sigma, sample):
    factor = 1 / (sigma * jnp.sqrt(2 * jnp.pi))
    density = (-(sample - mu) ** 2) / (2 * sigma ** 2)
    log_density = jnp.log(factor) + density
    return jnp.sum(log_density)


class VariationalInference(hk.Module):
    def __init__(self, num_params, init_std, pi, small_prior_std, large_prior_std):
        super().__init__()
        self.num_params = num_params
        self.init_fn = partial(init_fn, alpha=init_std)
        self.pi = pi
        self.small_prior_std = small_prior_std
        self.large_prior_std = large_prior_std

    def __call__(self):
        mu_initializer = hk.initializers.RandomUniform(-1, 1)
        rho_initializer = hk.initializers.RandomUniform(-5, -4)
        mu = hk.get_parameter('mu', shape=(self.num_params,), init=mu_initializer)
        rho = hk.get_parameter('rho', shape=(self.num_params,), init=rho_initializer)
        rng = hk.next_rng_key()
        epsilon = jax.random.normal(rng, shape=(self.num_params,))
        sigma = jnp.log(1 + jnp.exp(rho))
        sample_weights = mu + sigma * epsilon
        prior_mu = jnp.zeros(self.num_params)
        std_1 = jnp.log(1 + jnp.exp(self.small_prior_std * jnp.ones(self.num_params)))
        std_2 = jnp.log(1 + jnp.exp(self.large_prior_std * jnp.ones(self.num_params)))
        prior_log_prob = self.pi * gaussian_log_prob(prior_mu, std_1, sample_weights) \
                         + (1 - self.pi) * gaussian_log_prob(prior_mu, std_2, sample_weights)
        posterior_log_prob = gaussian_log_prob(mu, sigma, sample_weights)
        return sample_weights, posterior_log_prob, prior_log_prob
