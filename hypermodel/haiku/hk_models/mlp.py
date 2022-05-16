import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class MLP(hk.Module):
    def __init__(self, output_sizes):
        super(MLP, self).__init__()
        self.mlp = hk.nets.MLP(output_sizes, name='mlp')

    def __call__(self, x):
        return self.mlp(x)





