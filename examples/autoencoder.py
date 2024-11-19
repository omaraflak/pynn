"""Solve MNIST with a simple feed forward network."""

from pynn import trainer
from pynn import modules
from pynn import losses
from pynn import optimizers
import numpy as np
import matplotlib.pyplot as plt
import keras


def preprocess_data(x: np.ndarray) -> np.ndarray:
    x = x.reshape(-1, 28 * 28, 1)
    x = x.astype(np.float32) / 255
    return x


def plot_grid(
    images: list[np.ndarray],
    figsize: tuple[int, int] = (20, 10),
):
    cols = 3
    rows = len(images) // cols
    _, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize)
    for idx, image in enumerate(images):
        axs[idx // cols][idx % cols].imshow(image, cmap="gray")
    plt.show()


def main():
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    x_train = preprocess_data(x_train[:1000])
    x_test = preprocess_data(x_test)

    encoder = modules.Sequential([
        modules.Linear(784, 30, xavier=True),
        modules.Tanh(),
        modules.Linear(30, 16, xavier=True),
        modules.Tanh(),
    ])
    decoder = modules.Sequential([
        modules.Linear(16, 30, xavier=True),
        modules.Tanh(),
        modules.Linear(30, 784, xavier=True),
        modules.Tanh(),
    ])
    model = modules.Sequential([encoder, decoder])

    trainer.train(
        model,
        x_train,
        x_train,
        losses.MSE(),
        optimizers.SGD(model, learning_rate=0.1),
        epochs=300,
    )

    print("Test loss:", trainer.evaluate(model, x_test, x_test, losses.MSE()))

    images = []
    for x in x_test[:5]:
        noise = encoder.forward(x)
        decoded = decoder.forward(noise)
        images.append(np.reshape(x, (28, 28)))
        images.append(np.reshape(noise, (4, 4)))
        images.append(np.reshape(decoded, (28, 28)))
    plot_grid(images)


if __name__ == "__main__":
    main()
