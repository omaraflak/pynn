# PyNN

Python neural networks library with GPU support.

Under development...

# Development

```
pip install -e .
make -C ctensor
python examples/xor.py
```

# Usage

```python
from pynn import modules
from pynn import losses
from pynn import optimizers
from pynn import Tensor
from pynn import trainer

x_train = Tensor.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = Tensor.array([[0], [1], [1], [0]])

model = modules.Sequential([
    modules.Linear(2, 3),
    modules.Tanh(),
    modules.Linear(3, 1),
    modules.Tanh(),
])

x_train.to_gpu()
y_train.to_gpu()
model.to_gpu()

sgd = optimizers.SGD(model, learning_rate=0.1)
sgd.to_gpu()

trainer.train(model, x_train, y_train, losses.MSE(), sgd, epochs=1000)

x_train.to_cpu()
model.to_cpu()

for x in x_train:
    print(x.data, model.forward(x).data)
```
