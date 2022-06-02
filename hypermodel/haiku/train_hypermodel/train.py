from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from hypermodel.haiku.hk_models.mlp import MLP
from hypermodel.haiku.hk_models.hypermodel import Hypermodel
from hypermodel.haiku.data_gen import fourier_positional_encoding, gaussian_process
from hypermodel.haiku.loaders import VariableLengthDataset, NumpyDataset, collate_fn, variable_collate
import hydra
import matplotlib.pyplot as plt
import optax
from torch.utils.data import DataLoader
import sys


def mlp_forward(x, output_sizes):
    model = MLP(output_sizes=output_sizes, activation=jax.nn.relu)
    return model(x)


def hypermodel_forward(x, linear_size, unraveler, base_model_apply):
    model = Hypermodel([linear_size])
    base_params = unraveler(model(x))
    return base_model_apply(base_params, x)


def train(model, optimizer, opt_state, params, train_loader, num_epochs, print_epoch):
    @jax.jit
    def loss(params, data):
        x_train, y_train = data
        out = model.apply(params, x_train).flatten()
        return jnp.sum((out - y_train) ** 2) / len(y_train)

    @jax.jit
    def train_step(params, opt_state, data):
        loss_value, grads = jax.value_and_grad(loss, argnums=0)(params, data)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return loss_value, new_params, opt_state

    for epoch in range(num_epochs):
        running_loss = 0.0
        for data in train_loader:
            loss_value, params, opt_state = train_step(params, opt_state, data)
            running_loss += loss_value
        if epoch % print_epoch == 0:
            print('Epoch {}: {}'.format(epoch, running_loss / len(train_loader)))
    return params


@hydra.main(config_path='./configs', config_name='default')
def main(config):
    splitter = jax.random.PRNGKey(config.PRNGSeed)
    splitter, key = jax.random.split(splitter)
    kwargs = config.data.generate
    x_train, y_train, x_draw, y_draw = gaussian_process(key, **kwargs)
    args = config.data.fourier
    encoding = partial(fourier_positional_encoding, max_freq=args.max_freq, num_bands=args.num_bands, base=args.base)
    encoding = jax.vmap(encoding, 0, 0)
    x_train_encoded = encoding(x_train)

    print(config.model.output_sizes)
    base_model = hk.without_apply_rng(hk.transform(partial(mlp_forward, output_sizes=config.model.output_sizes)))
    splitter, key = jax.random.split(splitter)
    base_params = base_model.init(key, x=x_train_encoded)
    flattened_params, unraveler = jax.flatten_util.ravel_pytree(base_params)
    hypermodel = hk.without_apply_rng(hk.transform(
        partial(hypermodel_forward, linear_size=len(flattened_params),
                unraveler=unraveler, base_model_apply=base_model.apply)
    ))
    splitter, key = jax.random.split(splitter)
    hypermodel_params = hypermodel.init(key, x=x_train_encoded)
    optimizer = optax.adam(config.train.lr)
    opt_state = optimizer.init(hypermodel_params)
    train_loader = DataLoader(NumpyDataset(x_train_encoded, y_train),
                              batch_size=config.train.batch_size, collate_fn=collate_fn)
    trained_params = train(hypermodel, optimizer, opt_state, hypermodel_params,
                           train_loader, num_epochs=config.train.num_epochs,
                           print_epoch=config.train.print_epoch_loss)

    draw_plot(hypermodel, trained_params, x_draw, y_draw, x_train, y_train, encoding, 'figure')
    draw_plot(hypermodel, trained_params, x_draw, y_draw, x_train[:2], y_train[:2], encoding, 'figure1')
    draw_plot(hypermodel, trained_params, x_draw, y_draw, x_train[2:4], y_train[2:4], encoding, 'figure2')


def draw_plot(model, params, x_draw, y_draw, x_train, y_train, encoding, name):
    plt.clf()
    all_x = np.concatenate((x_draw, x_train))
    all_y = np.concatenate((y_draw, y_train))
    idxs = np.argsort(all_x)
    all_x = all_x[idxs]
    all_y = all_y[idxs]
    all_preds = model.apply(params, encoding(all_x))
    plt.plot(all_x, all_y, label='True function')
    plt.plot(all_x, all_preds, label='Hypermodel')
    plt.scatter(x_train, y_train)
    plt.legend()
    plt.savefig(name + '.png')


if __name__ == '__main__':
    main()
