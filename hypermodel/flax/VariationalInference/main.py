import flax
import flax.linen as nn
import jax.flatten_util
import optax
from flax.core import freeze, unfreeze
import torch
import matplotlib.pyplot as plt
import hydra
from omegaconf import OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path
import sys
from data_utils import gaussian_process, encoding_fun, create_train_test_loaders
from models import MLP, VariationalInference
from train import train_and_evaluate
from utils import draw_network
import jax.numpy as jnp

@hydra.main(config_path='configs', config_name='default')
def main(config):
    run(config)


def run(config):
    # initial key
    key = jax.random.PRNGKey(config.random.PRNGSeed)

    # generate data via gaussian process
    data = config.data
    x_train, y_train, x_draw, y_draw = gaussian_process(key, data.num_draw_points,
                                                        data.num_train_points, data.x_lims, data.length_scale)

    # fourier encoding
    fourier = config.data.fourier
    x_train_encoded = encoding_fun(x_train, fourier.max_freq, fourier.num_bands, fourier.base)

    # Get unflattener
    MLP.features = config.model.features
    base_model = MLP()
    key, new_key = jax.random.split(key)
    base_params = base_model.init(new_key, x_train_encoded)
    _, unflattener = jax.flatten_util.ravel_pytree(base_params)

    # data loaders
    trainloader, testloader = create_train_test_loaders(x_train_encoded, y_train, config.train.train_split,
                                                        config.train.batch_size)

    key, new_key = jax.random.split(key)
    final_state = train_and_evaluate(key, config, VariationalInference(), MLP(), trainloader,
                                     testloader, unflattener, 0, 1)

    plt.plot(x_draw, y_draw, label='true function')
    x_draw_encoded = encoding_fun(x_draw, fourier.max_freq, fourier.num_bands, fourier.base)
    draw_network(x_draw, x_draw_encoded, MLP(), final_state.params, label='MLP')
    plt.scatter(x_train, y_train, color='orange')
    plt.savefig('figure.png')


if __name__ == '__main__':
    main()
