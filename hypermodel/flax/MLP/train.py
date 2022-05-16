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


@partial(jax.jit, static_argnums=(0,))
def train_step(model, state, xb, yb, alpha):
    """Train for a single step."""

    def loss_fn(params):
        y_pred = model.apply(params, xb).flatten()
        n, = y_pred.shape
        diffs = y_pred - yb
        loss = jnp.inner(diffs, diffs) / n
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def train_and_evaluate(key, config, model, train_loader, test_loader):
    rng, init_rng = jax.random.split(key)
    state = create_train_state(config, init_rng, model, train_loader)

    for epoch in range(config.train.epochs):
        rng, input_rng = jax.random.split(rng)
        train_loss = 0.0
        for xb, yb in train_loader:
            state, batch_loss = train_step(model, state, xb, yb, config.train.alpha)
            train_loss += batch_loss

        test_loss = 0.0
        for xb, yb in test_loader:
            test_loss += evaluate(model, state, xb, yb)

        if epoch % config.train.print_epoch == 0:
            print(
                'epoch:% 3d, train_loss: %.4f, test_loss: %.4f'
                % (epoch, train_loss, test_loss)
            )

    return state
