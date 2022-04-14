import flax
import flax.linen as nn
import optax
import jax
import jax.numpy as jnp


class MLP(nn.Module):
    features = [10, 1]

    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat, name=f'layers_{i}')(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)
        return x


class VariationalInference(nn.Module):
    weights = 101 * 512 + 512 + 512 * 1 + 1
    std_dev = 101 * 512 + 512 + 512 * 1 + 1

    def setup(self):
        self.means = self.param('means', nn.initializers.zeros, (self.weights,))
        self.rhos = self.param('std_devs', nn.initializers.ones, (self.std_dev,))

    def __call__(self, sample):
        sampled_weights = self.means + jnp.log(1 + jnp.exp(self.rhos)) * sample
        return sampled_weights
