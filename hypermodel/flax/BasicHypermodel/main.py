import jax.flatten_util
from data import *
from hypermodel.models.mlp import MLP, LinearHypermodel
from train import *
import flax.linen as nn

import matplotlib.pyplot as plt
import hydra

class A(nn.Module):
    features: int




def test():
    a = A(features=5)
    b = A(features=6)

    print(a.features)
    print(b.features)
    print(A.features)

def run(config):
    key = jax.random.PRNGKey(config.PRNGSeed)
    x_train, y_train, x_draw, y_draw = get_data(config, key)
    fourier = config.data.fourier
    x_train_encoded = encoding_fun(x_train, fourier.max_freq, fourier.num_bands, fourier.base)
    x_draw_encoded = encoding_fun(x_draw, fourier.max_freq, fourier.num_bands, fourier.base)

    train_loader, test_loader = create_train_test_loaders(x_train_encoded, y_train,
                                                          train_split=config.train.train_split,
                                                          batch_size=config.train.batch_size)
    MLP.features = config.model.features
    key, new_key = jax.random.split(key)
    final_train_state = train_and_evaluate(new_key, config, MLP(), train_loader, test_loader)
    preds = MLP.inference(final_train_state.params, x_draw_encoded)

    # draw function
    plt.plot(x_draw, y_draw, label="True function")
    plt.plot(x_draw, preds, label="MLP")
    plt.scatter(x_train, y_train, label='Training points')
    plt.legend()
    plt.savefig('figure.png')


@hydra.main(config_path='./configs', config_name='default')
def main(config):
    test()


if __name__ == '__main__':
    main()
