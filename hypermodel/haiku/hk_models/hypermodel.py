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


