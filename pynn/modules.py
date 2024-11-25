"""Layers for neural networks."""

from pynn import Tensor


class Module:
    """Abstract class for a module in a neural network."""

    def __init__(self):
        self.inputs: Tensor = None
        self.outputs: Tensor = None

    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError()

    def backward(self, output_grad: Tensor) -> Tensor:
        raise NotImplementedError()

    def parameters(self) -> list[Tensor]:
        return []

    def gradients(self) -> list[Tensor]:
        return []


class Linear(Module):
    """Linear transformation applied to input column vector."""

    def __init__(self, input_size: int, output_size: int, xavier: bool = False):
        super().__init__()
        self.weights = Tensor.random_normal((output_size, input_size))
        self.bias = Tensor.random_normal((output_size, 1))
        self.weights_grad = Tensor.zeros(self.weights.shape)
        self.bias_grad = Tensor.zeros(self.bias.shape)
        # TODO: extract this logic into initializers...
        if xavier:
            self.weights = self.weights / (input_size ** 0.5)
            self.bias = self.bias / (output_size ** 0.5)

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        self.outputs = self.weights @ inputs + self.bias
        return self.outputs

    def backward(self, output_grad: Tensor) -> Tensor:
        self.weights_grad += output_grad @ self.inputs.T
        self.bias_grad += output_grad
        return self.weights.T @ output_grad

    def parameters(self) -> list[Tensor]:
        return [self.weights, self.bias]

    def gradients(self) -> list[Tensor]:
        return [self.weights_grad, self.bias_grad]


class Softmax(Module):
    """Softmax activation layer."""

    def forward(self, inputs: Tensor) -> Tensor:
        self.outputs = inputs.exp()
        self.outputs /= self.outputs.sum()
        return self.outputs

    def backward(self, output_grad: Tensor) -> Tensor:
        n = self.outputs.size
        return ((Tensor.identity(n) - self.outputs.T) * self.outputs) @ output_grad


class Activation(Module):
    """Applies an activation function to the input, element wise."""

    def activation(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError()

    def activation_prime(self, inputs: Tensor) -> Tensor:
        raise NotImplementedError()

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.activation(inputs)

    def backward(self, output_grad: Tensor) -> Tensor:
        return output_grad * self.activation_prime(self.inputs)


class Tanh(Activation):
    """Hyperbolic tangent activation layer."""

    def activation(self, inputs: Tensor) -> Tensor:
        return inputs.tanh()

    def activation_prime(self, inputs: Tensor) -> Tensor:
        return 1.0 - inputs.tanh().power(2)


class Sequential(Module):
    """Sequential neural network."""

    def __init__(self, modules: list[Module]):
        super().__init__()
        self.modules = modules

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = inputs
        for module in self.modules:
            outputs = module.forward(outputs)
        return outputs

    def backward(self, output_grad: Tensor) -> Tensor:
        input_grad = output_grad
        for module in reversed(self.modules):
            input_grad = module.backward(input_grad)
        return input_grad

    def parameters(self) -> list[Tensor]:
        return [param for module in self.modules for param in module.parameters()]

    def gradients(self) -> list[Tensor]:
        return [grad for module in self.modules for grad in module.gradients()]
