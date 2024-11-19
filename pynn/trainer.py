"""Helper to train a neural network and make evaluations."""

import numpy as np
from pynn import losses
from pynn import modules


def train(
    module: modules.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    loss: losses.Loss,
    epochs: int,
    learning_rate: float,
):
    """Performs SGD training on the given data and loss."""
    for i in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            output = module.forward(x)
            error += loss.loss(y, output)
            module.zero_gradients()
            module.backward(loss.loss_prime(y, output))
            for param, grad in zip(module.parameters(), module.gradients()):
                param -= learning_rate * grad

        error /= len(x_train)
        print(f"{i+1}/{epochs} error={error:.5f}")


def evaluate(
    module: modules.Module,
    x_test: np.ndarray,
    y_test: np.ndarray,
    loss: losses.Loss,
) -> float:
    """Evaluates the network on the given data and loss."""
    error = sum(loss.loss(y, module.forward(x))
                for x, y in zip(x_test, y_test))
    return error / len(x_test)
