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
        return self.linear(x) / 10


def init_fn(shape, dtype):
    return 0.1*jnp.ones(shape=shape, dtype=dtype)


class VariationalInference(hk.Module):
    def __init__(self, num_params):
        super().__init__()
        self.num_params = num_params

    def __call__(self, x):
        mu = hk.get_parameter('mu', shape=(self.num_params,), init=init_fn)
        rho = hk.get_parameter('rho', shape=(self.num_params,), init=jnp.ones)
        rng = hk.next_rng_key()
        epsilon = jax.random.normal(rng, shape=(self.num_params,))
        sample_weights = mu + jnp.log(1 + jnp.exp(rho)) * epsilon
        return sample_weights







