import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class Hypermodel(hk.Module):
    def __init__(self, linear_size):
        super().__init__()
        self.linear = hk.nets.MLP(output_sizes=[100, linear_size])

    def __call__(self, x):
        x = jnp.mean(x, axis=0)
        return self.linear(x) / 100


def init_fn(shape, dtype):
    return 0.1 * jnp.ones(shape=shape, dtype=dtype)


def gaussian_log_prob(mu, sigma, sample):
    factor = 1 / (sigma * jnp.sqrt(2 * jnp.pi))
    density = (-(sample - mu) ** 2) / (2*sigma**2)
    log_density = jnp.log(factor) + density
    return jnp.sum(log_density)


class VariationalInference(hk.Module):
    def __init__(self, num_params):
        super().__init__()
        self.num_params = num_params

    def __call__(self):
        mu = hk.get_parameter('mu', shape=(self.num_params,), init=jnp.zeros)
        rho = hk.get_parameter('rho', shape=(self.num_params,), init=init_fn)
        rng = hk.next_rng_key()
        epsilon = jax.random.normal(rng, shape=(self.num_params,))
        sigma = jnp.log(1 + jnp.exp(rho))
        sample_weights = mu + sigma * epsilon
        prior_mu = jnp.zeros(self.num_params)
        prior_rho = 0.1*jnp.ones(shape=(self.num_params,))
        prior_sigma = jnp.log(1 + jnp.exp(prior_rho))
        return sample_weights, gaussian_log_prob(mu, sigma, sample_weights), gaussian_log_prob(prior_mu, prior_sigma, sample_weights)
