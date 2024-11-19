"""Optimizers for parameters."""


import abc
from pynn import modules


class Optimizer(abc.ABC):
    def __init__(self, module: modules.Module):
        self.module = module

    def step(self):
        raise NotImplementedError()

    def zero_gradients(self):
        for grad in self.module.gradients():
            grad.fill(0)


class SGD(Optimizer):
    def __init__(self, module: modules.Module, learning_rate: float = 0.1):
        super().__init__(module)
        self.learning_rate = learning_rate

    def step(self):
        for param, grad in zip(self.module.parameters(), self.module.gradients()):
            param -= self.learning_rate * grad
