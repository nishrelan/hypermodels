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


def mlp_forward(x, output_sizes):
    model = MLP(output_sizes, activation=jax.nn.relu)
    return model(x)


def hypermodel_forward(x, linear_size, unraveler, base_model_apply):
    model = Hypermodel(linear_size)
    base_params = unraveler(model(x))
    return base_model_apply(base_params, x), base_params


def variational_forward(x, hypermodel_size, unraveler, hypermodel_apply, init_std, data_std, pi, small_prior_std,
                        big_prior_std):
    model = VariationalInference(hypermodel_size, init_std, pi, small_prior_std, big_prior_std)
    hypermodel_params, posterior_log_prob, prior_log_prob = model()
    hypermodel_params = unraveler(hypermodel_params)
    base_model_preds, base_params = hypermodel_apply(hypermodel_params, x)
    return base_model_preds, base_params, posterior_log_prob, prior_log_prob, data_std


def train(model, optimizer, opt_state, params, train_loader, num_epochs, print_epoch, splitter):
    @jax.jit
    def loss(params, data, rng_key):
        x_train, y_train = data
        preds, _, log_q, log_p, data_std = model.apply(params=params, x=x_train, rng=rng_key)
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
    args = config.data.fourier
    encoding = partial(fourier_positional_encoding, max_freq=args.max_freq, num_bands=args.num_bands, base=args.base)
    encoding = jax.vmap(encoding, 0, 0)
    x_train_encoded = encoding(x_train)

    """
        get pure function for hypermodel
    """
    base_model = hk.without_apply_rng(hk.transform(partial(mlp_forward, output_sizes=config.model.output_sizes)))
    splitter, key = jax.random.split(splitter)
    base_params = base_model.init(key, x=x_train_encoded, output_sizes=config.model.output_sizes)
    flattened_params, unraveler = jax.flatten_util.ravel_pytree(base_params)
    hypermodel = hk.without_apply_rng(hk.transform(
        partial(hypermodel_forward, linear_size=config.hypermodel.hidden_layers + [len(flattened_params)],
                unraveler=unraveler, base_model_apply=base_model.apply)
    ))

    """
        get pure function for var_inf model
    """
    splitter, key = jax.random.split(splitter)
    hypermodel_params = hypermodel.init(key, x=x_train_encoded)
    flattened_params, unraveler = jax.flatten_util.ravel_pytree(hypermodel_params)
    var_inf = hk.transform(
        partial(variational_forward, hypermodel_size=len(flattened_params),
                unraveler=unraveler, hypermodel_apply=hypermodel.apply,
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
    train_loader = DataLoader(NumpyDataset(x_train_encoded, y_train), batch_size=config.train.batch_size,
                              collate_fn=collate_fn, shuffle=True)
    trained_params = train(var_inf, optimizer, opt_state, initial_params,
                           train_loader, num_epochs=config.train.num_epochs,
                           print_epoch=config.train.print_epoch_loss, splitter=splitter)
    keys = jax.random.split(splitter, num=10)
    splitter = keys[0]
    bs = config.train.batch_size
    draw_plot(var_inf, base_model, trained_params, x_draw, y_draw, x_train[:bs], y_train[:bs], encoding, keys[1:])
    keys = jax.random.split(splitter, num=5)
    print(train_mse(var_inf, trained_params, x_train[:bs], y_train[:bs], encoding, keys))
    plt.clf()
    rhos = trained_params['variational_inference']['rho']
    sigmas = jnp.log(1 + jnp.exp(rhos))
    plt.hist(sigmas, rwidth=0.5, density=True)
    plt.savefig('rho_hist.png')
    plt.clf()
    plt.hist(trained_params['variational_inference']['mu'], rwidth=0.5, density=True)
    plt.savefig('mu_hist.png')


def draw_plot(model, base_model, params, x_draw, y_draw, x_train, y_train, encoding, keys):
    all_preds_list = []
    all_x = np.concatenate((x_draw, x_train))
    all_y = np.concatenate((y_draw, y_train))
    idxs = np.argsort(all_x)
    all_x = all_x[idxs]
    all_y = all_y[idxs]
    for key in keys:
        plt.clf()
        _, base_model_params, _, _, _ = model.apply(params=params, x=encoding(x_train), rng=key)
        all_preds = base_model.apply(params=base_model_params, x=encoding(all_x))
        all_preds = all_preds.flatten()
        all_preds_list.append(all_preds)
        plt.plot(all_x, all_y, label='True function')
        plt.plot(all_x, all_preds, label='Hypermodel')
        plt.scatter(x_train, y_train)
        plt.legend()
        plt.savefig(str(key) + '.png')
    plt.clf()
    stacked_preds = jnp.stack(all_preds_list)
    avg_preds = jnp.mean(stacked_preds, axis=0)
    plt.plot(all_x, all_y, label='True function')
    plt.plot(all_x, avg_preds, label='Hypermodel')
    plt.scatter(x_train, y_train)
    plt.legend()
    plt.savefig('average.png')
    plt.clf()
    variance = jnp.std(stacked_preds, axis=0)
    plt.plot(all_x, all_y, label='True function')
    plt.plot(all_x, avg_preds, label='Hypermodel')
    plt.scatter(x_train, y_train)
    plt.fill_between(all_x, avg_preds - variance, avg_preds + variance, color='orange', alpha=0.3)
    plt.legend()
    plt.savefig('variance.png')


def train_mse(model, params, x_train, y_train, encoding, keys):
    mses = []
    for i, key in enumerate(keys):
        preds, _, _, _, _ = model.apply(params=params, x=encoding(x_train), rng=key)
        preds = preds.flatten()
        mse = jnp.sum((preds - y_train) ** 2) / len(y_train)
        mses.append(mse)
    return jnp.mean(jnp.stack(mses), axis=0)



if __name__ == '__main__':
    main()