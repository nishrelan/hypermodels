from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from hypermodel.haiku.hk_models.mlp import MLP
from hypermodel.haiku.hk_models.hypermodel import Hypermodel, VariationalInference, gaussian_log_prob
from hypermodel.haiku.data_gen import fourier_positional_encoding, gaussian_process
from hypermodel.haiku.loaders import VariableLengthDataset, NumpyDataset, collate_fn, variable_collate
import hydra
import matplotlib.pyplot as plt
import optax
from torch.utils.data import DataLoader
import sys


def mlp_forward(x, output_sizes, activation):
    model = MLP(output_sizes, activation)
    return model(x)


def variational_forward(x, model_size, unraveler, model_apply, init_std, data_std, pi, small_prior_std, big_prior_std):
    model = VariationalInference(model_size, init_std, pi, small_prior_std, big_prior_std)
    base_model_params, posterior_log_prob, prior_log_prob = model()
    base_model_params = unraveler(base_model_params)
    return model_apply(base_model_params, x), posterior_log_prob, prior_log_prob, data_std


def train(model, optimizer, opt_state, params, train_loader, num_epochs, print_epoch, splitter):
    @jax.jit
    def loss(params, data, rng_key):
        x_train, y_train = data
        preds, log_q, log_p, data_std = model.apply(params=params, x=x_train, rng=rng_key)
        preds = preds.flatten()
        likelihood = gaussian_log_prob(y_train, data_std * jnp.ones(len(y_train)), preds)
        return log_q - log_p - likelihood, (log_q, log_p, likelihood)

    @jax.jit
    def train_step(params, opt_state, data, rng_key):
        loss_vals, grads = jax.value_and_grad(loss, argnums=0, has_aux=True)(params, data, rng_key)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return loss_vals, new_params, opt_state

    for epoch in range(num_epochs):
        for data in train_loader:
            splitter, key = jax.random.split(splitter)
            loss_vals, params, opt_state = train_step(params, opt_state, data, key)
            loss_value, (log_q, log_p, likelihood) = loss_vals
            print(
                "Epoch {}: Loss: {} Log_q: {} Log_p {}: Likelihood {}".format(
                    epoch, loss_value, log_q, log_p, likelihood
                )
            )
    return params


@hydra.main(config_path='./configs', config_name='default')
def main(config):
    """
        generate and encode data
    """
    splitter = jax.random.PRNGKey(config.PRNGSeed)
    splitter, key = jax.random.split(splitter)
    kwargs = config.data.generate
    x_train, y_train, x_draw, y_draw = gaussian_process(key, **kwargs)
    sorted_idxs = np.argsort(x_train)
    x_train = x_train[sorted_idxs]
    y_train = y_train[sorted_idxs]
    x_train = jnp.concatenate((x_train[:50], x_train[-50:]))
    y_train = jnp.concatenate((y_train[:50], y_train[-50:]))
    args = config.data.fourier
    encoding = partial(fourier_positional_encoding, max_freq=args.max_freq, num_bands=args.num_bands, base=args.base)
    encoding = jax.vmap(encoding, 0, 0)
    x_train_encoded = encoding(x_train)

    """
        get pure function for hypermodel
    """
    activation_func = jax.nn.relu if config.model.activation == 'relu' else jax.nn.leaky_relu
    base_model = hk.without_apply_rng(hk.transform(partial(mlp_forward, output_sizes=config.model.output_sizes,
                                                           activation=activation_func)))
    splitter, key = jax.random.split(splitter)
    base_params = base_model.init(key, x=x_train_encoded, output_sizes=config.model.output_sizes)
    flattened_params, unraveler = jax.flatten_util.ravel_pytree(base_params)
    var_inf = hk.transform(
        partial(variational_forward, model_size=len(flattened_params),
                unraveler=unraveler, model_apply=base_model.apply,
                init_std=config.varinf.init_std, data_std=config.varinf.data_std,
                pi=config.varinf.prior.pi, small_prior_std=config.varinf.prior.small_std,
                big_prior_std=config.varinf.prior.big_std)
    )
    splitter, key = jax.random.split(splitter)
    initial_params = var_inf.init(rng=key, x=x_train_encoded)

    """
        Train variational inference model
    """
    optimizer = optax.adam(config.train.lr)
    opt_state = optimizer.init(initial_params)
    train_loader = DataLoader(NumpyDataset(x_train_encoded, y_train), batch_size=300,
                              collate_fn=collate_fn, shuffle=True)
    trained_params = train(var_inf, optimizer, opt_state, initial_params,
                           train_loader, num_epochs=config.train.num_epochs,
                           print_epoch=config.train.print_epoch_loss, splitter=splitter)
    keys = jax.random.split(splitter, num=10)
    draw_plot(var_inf, trained_params, x_draw, y_draw, x_train, y_train, encoding, keys)
    print(trained_params['variational_inference'].keys())
    plt.clf()
    plt.hist(trained_params['variational_inference']['rho'], bins=jnp.linspace(-1, 1, 10))
    plt.savefig('rho_hist.png')
    plt.clf()
    plt.hist(trained_params['variational_inference']['mu'])
    plt.savefig('mu_hist.png')


def draw_plot(model, params, x_draw, y_draw, x_train, y_train, encoding, keys):
    all_preds_list = []
    all_x = np.concatenate((x_draw, x_train))
    all_y = np.concatenate((y_draw, y_train))
    idxs = np.argsort(all_x)
    all_x = all_x[idxs]
    all_y = all_y[idxs]
    for key in keys:
        plt.clf()
        all_preds, _, _, _ = model.apply(params=params, x=encoding(all_x), rng=key)
        all_preds = all_preds.flatten()
        all_preds_list.append(all_preds)
        plt.plot(all_x, all_y, label='True function')
        plt.plot(all_x, all_preds, label='Hypermodel')
        plt.scatter(x_train, y_train)
        plt.legend()
        plt.savefig(str(key) + '.png')
    plt.clf()
    avg_preds = jnp.mean(jnp.stack(all_preds_list), axis=0)
    plt.plot(all_x, all_y, label='True function')
    plt.plot(all_x, avg_preds, label='Hypermodel')
    plt.scatter(x_train, y_train)
    plt.legend()
    plt.savefig('average.png')


if __name__ == '__main__':
    main()
