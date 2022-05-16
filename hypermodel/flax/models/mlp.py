import flax
import flax.linen as nn
import optax
import jax
import jax.numpy as jnp


class MLP(nn.Module):
    features = [10, 1]
    unflattener = None

    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat, name=f'layers_{i}')(x)
            if i != len(self.features) - 1:
                x = nn.relu(x)
        return x

    @staticmethod
    def inference(params, x_in):
        x_out = MLP().apply(params, x_in)
        return x_out.flatten()


class LinearHypermodel(nn.Module):
    features: int

    @nn.compact
    def __call__(self, x):
        x = jnp.sum(x, axis=0)
        return nn.Dense(self.features)(x) / 10

    @classmethod
    def set_features(cls, base_model_features):
        pass

