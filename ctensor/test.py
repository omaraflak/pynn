from tensor import Tensor

a = Tensor([1, 2, 3, 4, 5, 6], (6, ))
b = Tensor([1, 2, 3, 4, 5, 6], (6, ))
a.to_gpu()
b.to_gpu()
c = a * b
c.to_cpu()

for i in range(c.size):
    print(c.get(i))
