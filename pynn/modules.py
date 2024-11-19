"""Layers for neural networks."""

import numpy as np


class Module:
    """Abstract class for a module in a neural network."""

    def __init__(self):
        self.inputs: np.ndarray = None
        self.outputs: np.ndarray = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def parameters(self) -> list[np.ndarray]:
        return []

    def gradients(self) -> list[np.ndarray]:
        return []


class Linear(Module):
    """Linear transformation applied to input column vector."""

    def __init__(self, input_size: int, output_size: int, xavier: bool = False):
        super().__init__()
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.weights_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.bias)
        # TODO: extract this logic into initializers...
        if xavier:
            self.weights = self.weights / np.sqrt(input_size)
            self.bias = self.bias / np.sqrt(output_size)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.outputs = np.dot(self.weights, inputs) + self.bias
        return self.outputs

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        self.weights_grad += np.dot(output_grad, self.inputs.T)
        self.bias_grad += output_grad
        return np.dot(self.weights.T, output_grad)

    def parameters(self) -> list[np.ndarray]:
        return [self.weights, self.bias]

    def gradients(self) -> list[np.ndarray]:
        return [self.weights_grad, self.bias_grad]


class Softmax(Module):
    """Softmax activation layer."""

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.outputs = np.exp(inputs)
        self.outputs /= np.sum(self.outputs)
        return self.outputs

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        n = np.size(self.outputs)
        return np.dot((np.identity(n) - self.outputs.T) * self.outputs, output_grad)


class Activation(Module):
    """Applies an activation function to the input, element wise."""

    def activation(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def activation_prime(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return self.activation(inputs)

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad * self.activation_prime(self.inputs)


class Tanh(Activation):
    """Hyperbolic tangent activation layer."""

    def activation(self, inputs: np.ndarray) -> np.ndarray:
        return np.tanh(inputs)

    def activation_prime(self, inputs: np.ndarray) -> np.ndarray:
        return 1.0 - np.power(np.tanh(inputs), 2)


class Sequential(Module):
    """Sequential neural network."""

    def __init__(self, modules: list[Module]):
        super().__init__()
        self.modules = modules

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        outputs = inputs
        for module in self.modules:
            outputs = module.forward(outputs)
        return outputs

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        input_grad = output_grad
        for module in reversed(self.modules):
            input_grad = module.backward(input_grad)
        return input_grad

    def parameters(self) -> list[np.ndarray]:
        return [param for module in self.modules for param in module.parameters()]

    def gradients(self) -> list[np.ndarray]:
        return [grad for module in self.modules for grad in module.gradients()]
