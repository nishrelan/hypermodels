import matplotlib.pyplot as plt


def draw_network(x_in, x_encoded, model, params, label):
    y_values = model.apply(params, x_encoded).flatten()
    plt.plot(x_in, y_values, label=label)

