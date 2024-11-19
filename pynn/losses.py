"""Loss functions for training neural networks."""

import numpy as np


class Loss:
    """Abstract class for a loss function."""

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError()

    def loss_prime(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError()


class MSE(Loss):
    """Mean squared error loss."""

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.power(y_true - y_pred, 2))

    def loss_prime(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2.0 * (y_pred - y_true) / np.size(y_true)


class CrossEntropy(Loss):
    """Cross entropy loss."""

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return -np.sum(y_true * np.log(y_pred))

    def loss_prime(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return -y_true / y_pred
