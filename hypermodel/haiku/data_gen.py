import jax
import jax.numpy as jnp
from sklearn.gaussian_process.kernels import RBF
import numpy as np


def fourier_positional_encoding(x, max_freq, num_bands, base):
    x = jnp.expand_dims(x, -1)
    dtype, orig_x = x.dtype, x

    scales = jnp.logspace(0., jnp.log(max_freq / 2) / jnp.log(base), num_bands, base=base, dtype=dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * jnp.pi
    x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)
    x = jnp.concatenate((x, orig_x), axis=-1)
    return x


def gaussian_process(key, num_draw, num_train, length_scale, xlim=(0, 1)):
    minval, maxval = xlim
    num_total_points = num_draw + num_train
    draw_points = jnp.linspace(minval, maxval, num_draw)
    x_points = jax.random.uniform(key, shape=(num_train,), minval=minval, maxval=maxval)

    all_points = jnp.concatenate((draw_points, x_points)).flatten()
    all_points = jnp.expand_dims(all_points, axis=1)
    all_points = np.array(all_points)
    K = RBF(length_scale=length_scale)(all_points)
    np.random.seed(0)
    y_vals = np.random.multivariate_normal(np.zeros(num_total_points), K)

    y_vals = jnp.array(y_vals)
    all_points = jnp.ravel(jnp.array(all_points))
    x_train = all_points[num_draw:]
    y_train = y_vals[num_draw:]
    x_draw = all_points[:num_draw]
    y_draw = y_vals[:num_draw]
    return x_train, y_train, x_draw, y_draw
