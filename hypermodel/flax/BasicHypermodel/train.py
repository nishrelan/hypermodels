from flax.training import train_state
import flax
import flax.linen as nn
import optax
import jax
import jax.numpy as jnp
import sys
from functools import partial


def create_train_state(config, key, model, trainloader):
    """Creates initial `TrainState`."""
    x, _ = next(iter(trainloader))
    params = model.init(key, x)
    tx = optax.adam(config.train.lr)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)


def evaluate(model, state, xb, yb):
    y_pred = model.apply(state.params, xb).flatten()
    n, = y_pred.shape
    diffs = y_pred - yb
    loss = jnp.inner(diffs, diffs) / n
    return loss


@jax.jit
def l2_loss(x, alpha):
    return alpha * (x ** 2).mean()


@partial(jax.jit, static_argnums=(0,1,))
def train_step(hypermodel, base_model, state, xb, yb):
    """Train for a single step."""

    def loss_fn(params):
        base_params_pred = hypermodel.apply(params, xb).flatten()
        base_params_pred = base_model.unflattener(base_params_pred)
        y_pred = base_model.apply(base_params_pred, xb).flatten()
        n, = y_pred.shape
        diffs = y_pred - yb
        loss = jnp.inner(diffs, diffs) / n
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def train_and_evaluate(key, config, hypermodel, base_model, train_loader, test_loader):
    rng, init_rng = jax.random.split(key)
    state = create_train_state(config, init_rng, hypermodel, train_loader)

    for epoch in range(config.train.epochs):
        rng, input_rng = jax.random.split(rng)
        train_loss = 0.0
        for xb, yb in train_loader:
            state, batch_loss = train_step(hypermodel, base_model, state, xb, yb)
            train_loss += batch_loss

        if epoch % config.train.print_epoch == 0:
            print(
                'epoch:% 3d, train_loss: %.4f'
                % (epoch, train_loss)
            )

    return state
