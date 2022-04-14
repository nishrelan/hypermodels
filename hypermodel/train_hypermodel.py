from typing import Any, Callable
from flax.training import train_state
import flax
import flax.linen as nn
import optax
import jax
import jax.numpy as jnp
import sys
from functools import partial
from models import MLP
from flax import struct, core
import sys

def create_train_state(config, key, model, trainloader):
    """Creates initial `TrainState`."""
    x, _ = next(iter(trainloader))
    model_vars = model.init(key, x)
    tx = optax.adam(config.train.lr)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=model_vars, tx=tx
    )


def evaluate(model, state, xb, yb):
    y_pred = model.apply(state.params, xb).flatten()
    n, = y_pred.shape
    diffs = y_pred - yb
    loss = jnp.inner(diffs, diffs) / n
    return loss


@jax.jit
def l2_loss(x, alpha):
    return alpha * (x ** 2).mean()


@partial(jax.jit, static_argnums=(0, 1,))
def train_step(model, unflattener, state, xb, yb, alpha):
    """Train for a single step."""

    def loss_fn(params):
        base_param_pred = model.apply(params, xb)
        base_param_pred = base_param_pred.flatten()
        base_param_pred = unflattener(base_param_pred)
        y_pred = MLP().apply(base_param_pred, xb).flatten()
        n, = y_pred.shape
        diffs = y_pred - yb
        loss = jnp.inner(diffs, diffs) / n
        loss += sum([l2_loss(w, alpha=alpha) for w in jax.tree_leaves(params)])
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def train_and_evaluate(key, config, model, trainloader, testloader, MLP_unflattener):
    rng, init_rng = jax.random.split(key)
    state = create_train_state(config, init_rng, model, trainloader)

    for epoch in range(config.train.epochs):
        rng, input_rng = jax.random.split(rng)
        train_loss = 0.0
        batch_size = 0
        for xb, yb in trainloader:
            state, batch_loss = train_step(model, MLP_unflattener, state, xb, yb, config.train.alpha)
            train_loss += batch_loss
            batch_size = len(xb)

        test_loss = 0.0
        for xb, yb in testloader:
            test_loss += evaluate(model, state, xb.numpy(), yb.numpy())

        if epoch % config.train.print_epoch == 0:
            print(
                'epoch:% 3d, train_loss: %.4f, test_loss: %.4f, batch_size:% 3d'
                % (epoch, train_loss, test_loss, batch_size)
            )

    return state
