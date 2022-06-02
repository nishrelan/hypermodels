from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from hypermodel.haiku.hk_models.mlp import MLP
from hypermodel.haiku.data_gen import fourier_positional_encoding, gaussian_process
import hydra
import matplotlib.pyplot as plt
import optax


def mlp_forward(x, output_sizes):
    model = MLP(output_sizes, activation=jax.nn.relu)
    return model(x)


def train(model, optimizer, opt_state, params, data, num_epochs, print_epoch):
    @jax.jit
    def loss(params, data):
        x_train, y_train = data
        out = model.apply(params, x_train).flatten()
        element_l2 = optax.l2_loss(out, y_train)
        return jnp.sum(element_l2 / len(y_train))

    @jax.jit
    def train_step(params, opt_state, data):
        loss_value, grads = jax.value_and_grad(loss, argnums=0)(params, data)
        updates, opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return loss_value, new_params, opt_state

    for epoch in range(num_epochs):
        loss_value, params, opt_state = train_step(params, opt_state, data)
        if epoch % print_epoch == 0:
            print('Epoch {}: {}'.format(epoch, loss_value))
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
    print(jnp.shape(x_train_encoded))

    model = hk.without_apply_rng(hk.transform(partial(mlp_forward, output_sizes=config.model.output_sizes)))
    splitter, key = jax.random.split(splitter)
    params = model.init(key, x=x_train_encoded)
    optimizer = optax.adam(2e-4)
    opt_state = optimizer.init(params)
    trained_params = train(model, optimizer, opt_state,
                           params, (x_train_encoded[:2], y_train[:2]), num_epochs=10000, print_epoch=10)

    pred_func = model.apply(trained_params, encoding(x_draw)).flatten()
    plt.plot(x_draw, y_draw, label='True function')
    plt.plot(x_draw, pred_func, label='Predicted function')
    plt.scatter(x_train[:2], y_train[:2])
    plt.legend()
    plt.savefig('fig.png')


if __name__ == '__main__':
    main()
