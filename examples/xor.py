"""Solve XOR with a small feed forward network."""

from pynn import trainer
from pynn import modules
from pynn import losses
from pynn import optimizers
from pynn import Tensor
import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(model: modules.Module):
    """Plots the decision boundary of the XOR model."""
    x_vals = []
    y_vals = []
    z_vals = []

    for x in np.linspace(0, 1, 20):
        for y in np.linspace(0, 1, 20):
            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(model.forward(np.array([[x], [y]]))[0][0])

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(x_vals, y_vals, z_vals)
    plt.show()


def main():
    x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).reshape((4, 2, 1))
    y_train = np.array([[0], [1], [1], [0]]).reshape((4, 1, 1))
    x_train = [Tensor.array(x) for x in x_train]
    y_train = [Tensor.array(y) for y in y_train]

    model = modules.Sequential([
        modules.Linear(2, 3),
        modules.Tanh(),
        modules.Linear(3, 1),
        modules.Tanh(),
    ])

    trainer.train(
        model,
        x_train,
        y_train,
        losses.MSE(),
        optimizers.SGD(model, learning_rate=0.1),
        epochs=1000,
    )

    for x in x_train:
        print(x.data, model.forward(x).data)

    plot_decision_boundary(model)


if __name__ == "__main__":
    main()
