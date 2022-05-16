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
    params = model.init(key, jnp.zeros(52737))
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


@partial(jax.jit, static_argnums=(0,1, 2))
def train_step(model, base_model, unflattener, state, xb, yb, rng, data_mean, data_stddev):
    """Train for a single step."""

    def loss_fn(params):
        sample = jax.random.normal(rng, shape=(52737,))
        weights = model.apply(params, sample).flatten()
        unflattened_weights = unflattener(weights)
        y_preds = base_model.apply(unflattened_weights, xb).flatten()
        n, = y_preds.shape
        diffs = y_preds - yb
        data_likelihood = jnp.inner(diffs, diffs) * -1/(2*data_stddev**2) + -n/2*jnp.log(2*jnp.pi*data_stddev**2)
        posterior_means = params['params']['means']
        posterior_stddevs = jnp.log(1 + jnp.exp(params['params']['std_devs']))
        posterior_likelihood = -1/2*jnp.sum(jnp.log(2*jnp.pi*posterior_stddevs**2)) + jnp.sum(1/(2*posterior_stddevs**2)*(weights - posterior_means)**2)
        prior_likelihood = -52737/2*jnp.log(2*jnp.pi) - 1/2*jnp.sum(weights**2)
        return posterior_likelihood - prior_likelihood + data_likelihood

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


def train_and_evaluate(key, config, model, base_model, trainloader, testloader, unflattener, prior_mean, prior_stddev):
    rng, init_rng = jax.random.split(key)
    state = create_train_state(config, init_rng, model, trainloader)

    for epoch in range(config.train.epochs):
        rng, input_rng = jax.random.split(rng)
        train_loss = 0.0
        for xb, yb in trainloader:
            rng, train_step_rng = jax.random.split(rng)
            state, batch_loss = train_step(model, base_model, unflattener, state, xb.numpy(), yb.numpy(),
                                           train_step_rng, prior_mean, prior_stddev)
            train_loss += batch_loss

        test_loss = 0.0
        for xb, yb in testloader:
            test_loss += evaluate(model, state, xb.numpy(), yb.numpy())

        if epoch % config.train.print_epoch == 0:
            print(
                'epoch:% 3d, train_loss: %.4f, test_loss: %.4f'
                % (epoch, train_loss, test_loss)
            )

    return state
