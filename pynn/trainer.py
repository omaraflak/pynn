"""Helper to train a neural network and make evaluations."""

from pynn import losses
from pynn import modules
from pynn import optimizers
from pynn import Tensor


def train(
    module: modules.Module,
    x_train: Tensor,
    y_train: Tensor,
    loss: losses.Loss,
    optimizer: optimizers.Optimizer,
    epochs: int,
) -> list[float]:
    """Performs training on the given data, loss, and optimizer."""
    errors = []
    for i in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            output = module.forward(x)
            error += loss.loss(y, output)
            optimizer.zero_gradients()
            module.backward(loss.loss_prime(y, output))
            optimizer.step()

        error /= len(x_train)
        errors.append(error)
        print(f"{i+1}/{epochs} error={error:.5f}")
    return errors


def evaluate(
    module: modules.Module,
    x_test: Tensor,
    y_test: Tensor,
    loss: losses.Loss,
) -> float:
    """Evaluates the network on the given data and loss."""
    error = sum(loss.loss(y, module.forward(x))
                for x, y in zip(x_test, y_test))
    return error / len(x_test)
