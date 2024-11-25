"""Loss functions for training neural networks."""

from pynn import Tensor


class Loss:
    """Abstract class for a loss function."""

    def loss(self, y_true: Tensor, y_pred: Tensor) -> float:
        raise NotImplementedError()

    def loss_prime(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        raise NotImplementedError()


class MSE(Loss):
    """Mean squared error loss."""

    def loss(self, y_true: Tensor, y_pred: Tensor) -> float:
        return (y_true - y_pred).power(2).mean()

    def loss_prime(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return 2.0 * (y_pred - y_true) / y_true.size


class CrossEntropy(Loss):
    """Cross entropy loss."""

    def loss(self, y_true: Tensor, y_pred: Tensor) -> float:
        return -(y_true * y_pred.log()).sum()

    def loss_prime(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        return -y_true / y_pred
