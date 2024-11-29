"""Solve MNIST with a simple feed forward network."""

from pynn import trainer
from pynn import modules
from pynn import losses
from pynn import optimizers
from pynn import Tensor
from examples import mnist_canvas
import numpy as np
import keras


def preprocess_data(
    x: np.ndarray, y: np.ndarray
) -> tuple[Tensor, Tensor]:
    # reshape and normalize input data
    x = x.reshape(-1, 28 * 28, 1)
    x = x.astype(np.float32) / 255
    # encode output which is a number in range [0, 9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = keras.utils.to_categorical(y)
    y = y.reshape(-1, 10, 1)
    return Tensor(x.flatten(), x.shape), Tensor(y.flatten(), y.shape)


def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    model = modules.Sequential([
        modules.Linear(28 * 28, 50),
        modules.Tanh(),
        modules.Linear(50, 10),
        modules.Softmax(),
    ])

    trainer.train(
        model,
        x_train,
        y_train,
        losses.CrossEntropy(),
        optimizers.SGD(model, learning_rate=0.01),
        epochs=50,
    )

    for x, y in zip(x_test[:20], y_test[:20]):
        output = model.forward(x)
        print("pred:", np.argmax(output), "true:", np.argmax(y))

    print(
        "Loss over test set:",
        trainer.evaluate(model, x_test, y_test, losses.CrossEntropy()),
    )

    canvas = mnist_canvas.MnistCanvas(model)
    canvas.show()


if __name__ == "__main__":
    main()
