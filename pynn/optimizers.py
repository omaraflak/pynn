"""Optimizers for parameters."""


import abc
import numpy as np
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
    def __init__(self, module: modules.Module, learning_rate: float = 0.01, momentum: float = 0.9):
        super().__init__(module)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = [np.zeros_like(param) for param in module.parameters()]

    def step(self):
        parameters = self.module.parameters()
        gradients = self.module.gradients()
        for i in range(len(parameters)):
            self.v[i] = self.v[i] * self.momentum + self.learning_rate * gradients[i]
            parameters[i] -= self.v[i]

