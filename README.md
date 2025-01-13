# PyNN

Python neural networks library with GPU support.

Under development...

# Development

```
pip install -e .
python3 examples/xor.py
```

# Usage

```python
from pynn import modules
from pynn import losses
from pynn import optimizers
from pynn import Tensor
from pynn import trainer


def to_gpu(tensors: list[Tensor]):
    for tensor in tensors:
        tensor.to_gpu()

def to_cpu(tensors: list[Tensor]):
    for tensor in tensors:
        tensor.to_cpu()

# Tensor slicing is not yet supported for GPU tensors
# So Tensors cannot be iterated over when they're on GPU.
# In the meantime, I'm using a list of Tensors instead.

x_train = [
    Tensor([0, 0], (2, 1)),
    Tensor([0, 1], (2, 1)),
    Tensor([1, 0], (2, 1)),
    Tensor([1, 1], (2, 1)),
]
y_train = [
    Tensor([0], (1, 1)),
    Tensor([1], (1, 1)),
    Tensor([1], (1, 1)),
    Tensor([0], (1, 1)),
]

model = modules.Sequential([
    modules.Linear(2, 3),
    modules.Tanh(),
    modules.Linear(3, 1),
    modules.Tanh(),
])

to_gpu(x_train)
to_gpu(y_train)
model.to_gpu()

sgd = optimizers.SGD(model, learning_rate=0.1)
sgd.to_gpu()

trainer.train(model, x_train, y_train, losses.MSE(), sgd, epochs=1000)

to_cpu(x_train)
model.to_cpu()
sgd.to_cpu()

for x in x_train:
    print(x.data, model.forward(x).data)
```