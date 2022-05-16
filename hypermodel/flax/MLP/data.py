import jax.numpy as jnp
import jax
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
from utils import *
from hydra.utils import get_original_cwd, to_absolute_path

import sklearn.gaussian_process.kernels
from sklearn.gaussian_process.kernels import RBF


def polynomial(coeffs):
    def f(x):
        return jnp.polyval(jnp.array(coeffs), x)

    return f


def generate_points(poly_fun, rng_key, xlim=None, num_points=100):
    minval, maxval = xlim
    x_points = jax.random.uniform(rng_key, shape=(num_points,), minval=minval, maxval=maxval)
    y_points = jax.vmap(poly_fun, 0, 0)(x_points)
    return {'x': x_points, 'y': y_points}


# Copied from Robert's Perceiver repo
# Reference:
# https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
def fourier_positional_encoding(x, max_freq, num_bands, base):
    x = jnp.expand_dims(x, -1)
    dtype, orig_x = x.dtype, x

    scales = jnp.logspace(0., jnp.log(max_freq / 2) / jnp.log(base), num_bands, base=base, dtype=dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * jnp.pi
    x = jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)
    x = jnp.concatenate((x, orig_x), axis=-1)
    return x


encoding_fun = jax.vmap(fourier_positional_encoding, (0, None, None, None), 0)


class PolyDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


class VariableLengthDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)

    def __len__(self):
        return 1 if len(self.x_data) > 0 else 0

    def __getitem__(self, idx):
        probs = torch.ones(len(self.x_data)) * 0.1
        selections = torch.bernoulli(probs)
        selections = torch.flatten(torch.nonzero(selections))
        x_points = self.x_data[selections]
        y_points = self.y_data[selections]

        if len(x_points) == 0:
            return np.expand_dims(self.x_data[0], axis=0), np.expand_dims(self.y_data[0], axis=0)
        else:
            return x_points, y_points


def create_train_test_loaders(x, y, train_split=0.8, batch_size=64):
    num_train = int(train_split * len(x))

    train_x = x[:num_train]
    train_y = y[:num_train]
    train_ds = VariableLengthDataset(train_x, train_y)

    test_x = x[num_train:]
    test_y = y[num_train:]
    test_ds = VariableLengthDataset(test_x, test_y)

    # disable automatic batching, return numpy arrays
    return DataLoader(train_ds, batch_size=None, collate_fn=lambda x: x), DataLoader(test_ds, batch_size=None,
                                                                                     collate_fn=lambda x: x)


def get_poly_data(config):
    coeff = omegaconf_list_to_array(config.data.polynomial.coeff)
    coefficients = jnp.array(coeff) * config.data.polynomial.scale
    f = polynomial(coefficients)
    key = jax.random.PRNGKey(0)
    points = generate_points(f, key, xlim=config.data.xlim, num_points=config.data.num_points)
    encode = jax.vmap(fourier_positional_encoding, in_axes=(0, None, None, None), out_axes=0)

    params = config.data.fourier
    encoded_points = encode(points['x'], params.max_freq, params.num_bands, params.base)

    return create_train_test_loaders(encoded_points, points['y'],
                                     train_split=config.data.train_split, batch_size=config.train.batch_size)


def gaussian_process(key, num_draw_points, num_train_points, xlims, l):
    minval, maxval = xlims
    num_total_points = num_draw_points + num_train_points
    draw_points = jnp.linspace(minval, maxval, num_draw_points)
    x_points = jax.random.uniform(key, shape=(num_train_points,), minval=minval, maxval=maxval)

    all_points = jnp.concatenate((draw_points, x_points)).flatten()
    all_points = jnp.expand_dims(all_points, axis=1)
    all_points = np.array(all_points)
    K = RBF(length_scale=l)(all_points)
    y_vals = np.random.multivariate_normal(np.zeros(num_total_points), K)

    y_vals = jnp.array(y_vals)
    all_points = jnp.ravel(jnp.array(all_points))
    x_train = all_points[num_draw_points:]
    y_train = y_vals[num_draw_points:]
    x_draw = all_points[:num_draw_points]
    y_draw = y_vals[:num_draw_points]
    return x_train, y_train, x_draw, y_draw


def get_data(config, key):
    if config.data.generate:
        key, new_key = jax.random.split(key)
        x_train, y_train, x_draw, y_draw = gaussian_process(new_key,
                                                            config.data.num_draw_points, config.data.num_train_points,
                                                            config.data.xlims, sftf(config.data.length_scale))
        if config.data.save_file:
            save_data('datasets', config.data.save_file, x_train, y_train, x_draw, y_draw)

    else:
        x_train, y_train, x_draw, y_draw = get_function_data(config.data.data_dir, config.data.file)

    return x_train, y_train, x_draw, y_draw


def save_data(save_dir, save_filename, x_train, y_train, x_draw, y_draw):
    save_dir = get_original_cwd() + '/{}'.format(save_dir)
    file_path = save_dir + '/{}'.format(save_filename)

    with open(file_path, 'wb') as f:
        jnp.save(f, x_train)
        jnp.save(f, y_train)
        jnp.save(f, x_draw)
        jnp.save(f, y_draw)


def get_function_data(save_dir, save_filename):
    save_dir = get_original_cwd() + '/{}'.format(save_dir)
    file_path = save_dir + '/{}'.format(save_filename)

    with open(file_path, 'rb') as f:
        x_train = jnp.load(f)
        y_train = jnp.load(f)
        x_draw = jnp.load(f)
        y_draw = jnp.load(f)

    return x_train, y_train, x_draw, y_draw
