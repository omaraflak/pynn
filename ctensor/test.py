from tensor import Tensor

a = Tensor([1, 2, 3, 4, 5, 6], (1, 3))
b = Tensor([1, 2, 3, 4, 5, 6], (3, 1))
a.to_gpu()
b.to_gpu()
c = a.dot(b)
c.to_cpu()

print(c.get(0, 0))
