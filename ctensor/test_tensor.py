import math
import unittest
from tensor import Tensor


class TestTensor(unittest.TestCase):
    def test_create_tensor(self):
        x = Tensor([1, 2, 3, 4, 5, 6], (3, 2))

        self.assertEqual(x.size, 6)
        self.assertEqual(x.dims, 2)
        self.assertEqual(x.shape, (3, 2))
        self.assertEqual(x.device, 0)
        self.assertEqual(x.data, [1, 2, 3, 4, 5, 6])

    def test_to_gpu(self):
        x = Tensor([1, 2, 3, 4, 5, 6], (3, 2))

        x.to_gpu()

        self.assertEqual(x.size, 6)
        self.assertEqual(x.dims, 2)
        self.assertEqual(x.shape, (3, 2))
        self.assertEqual(x.stride, (2, 1))
        self.assertEqual(x.device, 1)
        self.assertEqual(x.offset, 0)
        self.assertEqual(x.indices, [0, 1, 2, 3, 4, 5])

        x.to_cpu()
        self.assertEqual(x.data, [1, 2, 3, 4, 5, 6])

    def test_to_cpu(self):
        x = Tensor([1, 2, 3, 4, 5, 6], (3, 2))
        x.to_gpu()

        x.to_cpu()

        self.assertEqual(x.size, 6)
        self.assertEqual(x.dims, 2)
        self.assertEqual(x.shape, (3, 2))
        self.assertEqual(x.stride, (2, 1))
        self.assertEqual(x.device, 0)
        self.assertEqual(x.offset, 0)
        self.assertEqual(x.data, [1, 2, 3, 4, 5, 6])
        self.assertEqual(x.indices, [0, 1, 2, 3, 4, 5])

    def test_copy_cpu(self):
        x = Tensor([1, 2, 3, 4, 5, 6], (3, 2))

        y = x.copy()

        self.assertEqual(y.size, 6)
        self.assertEqual(y.dims, 2)
        self.assertEqual(y.shape, (3, 2))
        self.assertEqual(y.stride, (2, 1))
        self.assertEqual(y.device, 0)
        self.assertEqual(y.data, [1, 2, 3, 4, 5, 6])
        self.assertNotEqual(y, x)

    def test_copy_gpu(self):
        x = Tensor([1, 2, 3, 4, 5, 6], (3, 2))
        x.to_gpu()

        y = x.copy()
        x.to_cpu()
        y.to_cpu()

        self.assertEqual(y.size, 6)
        self.assertEqual(y.dims, 2)
        self.assertEqual(y.shape, (3, 2))
        self.assertEqual(y.stride, (2, 1))
        self.assertEqual(y.device, 0)
        self.assertEqual(y.data, [1, 2, 3, 4, 5, 6])
        self.assertNotEqual(y, x)

    def test_copy_sliced_cpu(self):
        x = Tensor.array([[1, 2], [3, 4], [5, 6]])

        y = x[::2].copy()

        self.assertEqual(y.size, 4)
        self.assertEqual(y.dims, 2)
        self.assertEqual(y.shape, (2, 2))
        self.assertEqual(y.stride, (2, 1))
        self.assertEqual(y.device, 0)
        self.assertEqual(y.data, [1, 2, 5, 6])
        self.assertEqual(y.indices, [0, 1, 2, 3])
        self.assertEqual(y.base, None)
        self.assertNotEqual(y, x)

    def test_copy_sliced_gpu(self):
        x = Tensor.array([[1, 2], [3, 4], [5, 6]])
        x.to_gpu()

        y = x[::2].copy()

        self.assertEqual(y.size, 4)
        self.assertEqual(y.dims, 2)
        self.assertEqual(y.shape, (2, 2))
        self.assertEqual(y.stride, (2, 1))
        self.assertEqual(y.device, 1)
        self.assertEqual(y.indices, [0, 1, 2, 3])
        self.assertEqual(y.base, None)
        self.assertNotEqual(y, x)

        y.to_cpu()
        self.assertEqual(y.data, [1, 2, 5, 6])


    def test_fill_cpu(self):
        x = Tensor([1, 2, 3, 4, 5, 6], (3, 2))

        x.fill(1)

        self.assertEqual(x.data, [1, 1, 1, 1, 1, 1])

    def test_fill_gpu(self):
        x = Tensor([1, 2, 3, 4, 5, 6], (3, 2))
        x.to_gpu()

        x.fill(1)
        x.to_cpu()

        self.assertEqual(x.data, [1, 1, 1, 1, 1, 1])

    def test_reshape(self):
        x = Tensor([1, 2, 3, 4, 5, 6], (6, ))

        x.reshape(3, 2)

        self.assertEqual(x.size, 6)
        self.assertEqual(x.dims, 2)
        self.assertEqual(x.shape, (3, 2))
        self.assertEqual(x.device, 0)
        self.assertEqual(x.data, [1, 2, 3, 4, 5, 6])

    def test_unary_minus_cpu(self):
        x = Tensor([1, 2, 3, 4, 5, 6], (3, 2))

        y = -x

        self.assertEqual(y.size, 6)
        self.assertEqual(y.dims, 2)
        self.assertEqual(y.shape, (3, 2))
        self.assertEqual(y.device, 0)
        self.assertEqual(y.data, [-1, -2, -3, -4, -5, -6])

    def test_unary_minus_gpu(self):
        x = Tensor([1, 2, 3, 4, 5, 6], (3, 2))
        x.to_gpu()

        y = -x
        y.to_cpu()

        self.assertEqual(y.size, 6)
        self.assertEqual(y.dims, 2)
        self.assertEqual(y.shape, (3, 2))
        self.assertEqual(y.device, 0)
        self.assertEqual(y.data, [-1, -2, -3, -4, -5, -6])

    def test_add_cpu(self):
        a = Tensor([1, 2, 3, 4, 5, 6], (3, 2))
        b = Tensor([1, 2, 3, 4, 5, 6], (3, 2))

        c = a + b

        self.assertEqual(c.size, 6)
        self.assertEqual(c.dims, 2)
        self.assertEqual(c.shape, (3, 2))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [2, 4, 6, 8, 10, 12])

    def test_add_gpu(self):
        a = Tensor([1, 2, 3, 4, 5, 6], (3, 2))
        b = Tensor([1, 2, 3, 4, 5, 6], (3, 2))
        a.to_gpu()
        b.to_gpu()

        c = a + b
        c.to_cpu()

        self.assertEqual(c.size, 6)
        self.assertEqual(c.dims, 2)
        self.assertEqual(c.shape, (3, 2))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [2, 4, 6, 8, 10, 12])

    def test_broadcast_add_cpu(self):
        a = Tensor([1, 2, 3, 4, 5, 6], (3, 2))

        c = a + 1

        self.assertEqual(c.size, 6)
        self.assertEqual(c.dims, 2)
        self.assertEqual(c.shape, (3, 2))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [2, 3, 4, 5, 6, 7])

    def test_broadcast_add_gpu(self):
        a = Tensor([1, 2, 3, 4, 5, 6], (3, 2))
        a.to_gpu()

        c = a + 1
        c.to_cpu()

        self.assertEqual(c.size, 6)
        self.assertEqual(c.dims, 2)
        self.assertEqual(c.shape, (3, 2))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [2, 3, 4, 5, 6, 7])

    def test_broadcast_right_add_cpu(self):
        a = Tensor([1, 2, 3, 4, 5, 6], (3, 2))

        c = 1 + a

        self.assertEqual(c.size, 6)
        self.assertEqual(c.dims, 2)
        self.assertEqual(c.shape, (3, 2))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [2, 3, 4, 5, 6, 7])

    def test_broadcast_right_add_gpu(self):
        a = Tensor([1, 2, 3, 4, 5, 6], (3, 2))
        a.to_gpu()

        c = 1 + a
        c.to_cpu()

        self.assertEqual(c.size, 6)
        self.assertEqual(c.dims, 2)
        self.assertEqual(c.shape, (3, 2))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [2, 3, 4, 5, 6, 7])

    def test_subtract_cpu(self):
        a = Tensor([1, 2, 3, 4, 5, 6], (3, 2))
        b = Tensor([6, 4, 3, 2, 7, 6], (3, 2))

        c = a - b

        self.assertEqual(c.size, 6)
        self.assertEqual(c.dims, 2)
        self.assertEqual(c.shape, (3, 2))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [-5, -2, 0, 2, -2, 0])

    def test_subtract_gpu(self):
        a = Tensor([1, 2, 3, 4, 5, 6], (3, 2))
        b = Tensor([6, 4, 3, 2, 7, 6], (3, 2))
        a.to_gpu()
        b.to_gpu()

        c = a - b
        c.to_cpu()

        self.assertEqual(c.size, 6)
        self.assertEqual(c.dims, 2)
        self.assertEqual(c.shape, (3, 2))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [-5, -2, 0, 2, -2, 0])

    def test_broadcast_subtract_cpu(self):
        a = Tensor([1, 2, 3, 4, 5, 6], (3, 2))

        c = a - 1

        self.assertEqual(c.size, 6)
        self.assertEqual(c.dims, 2)
        self.assertEqual(c.shape, (3, 2))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [0, 1, 2, 3, 4, 5])

    def test_broadcast_subtract_gpu(self):
        a = Tensor([1, 2, 3, 4, 5, 6], (3, 2))
        a.to_gpu()

        c = a - 1
        c.to_cpu()

        self.assertEqual(c.size, 6)
        self.assertEqual(c.dims, 2)
        self.assertEqual(c.shape, (3, 2))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [0, 1, 2, 3, 4, 5])

    def test_broadcast_right_subtract_cpu(self):
        a = Tensor([1, 2, 3, 4, 5, 6], (3, 2))

        c = 1 - a

        self.assertEqual(c.size, 6)
        self.assertEqual(c.dims, 2)
        self.assertEqual(c.shape, (3, 2))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [0, -1, -2, -3, -4, -5])

    def test_broadcast_right_subtract_gpu(self):
        a = Tensor([1, 2, 3, 4, 5, 6], (3, 2))
        a.to_gpu()

        c = 1 - a
        c.to_cpu()

        self.assertEqual(c.size, 6)
        self.assertEqual(c.dims, 2)
        self.assertEqual(c.shape, (3, 2))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [0, -1, -2, -3, -4, -5])

    def test_multiply_cpu(self):
        a = Tensor([1, 2, 3, 4, 5, 6], (3, 2))
        b = Tensor([6, 4, 3, 2, 7, 6], (3, 2))

        c = a * b

        self.assertEqual(c.size, 6)
        self.assertEqual(c.dims, 2)
        self.assertEqual(c.shape, (3, 2))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [6, 8, 9, 8, 35, 36])

    def test_multiply_gpu(self):
        a = Tensor([1, 2, 3, 4, 5, 6], (3, 2))
        b = Tensor([6, 4, 3, 2, 7, 6], (3, 2))
        a.to_gpu()
        b.to_gpu()

        c = a * b
        c.to_cpu()

        self.assertEqual(c.size, 6)
        self.assertEqual(c.dims, 2)
        self.assertEqual(c.shape, (3, 2))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [6, 8, 9, 8, 35, 36])

    def test_broadcast_multiply_cpu(self):
        a = Tensor([1, 2, 3, 4, 5, 6], (3, 2))

        c = a * 2

        self.assertEqual(c.size, 6)
        self.assertEqual(c.dims, 2)
        self.assertEqual(c.shape, (3, 2))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [2, 4, 6, 8, 10, 12])

    def test_broadcast_multiply_gpu(self):
        a = Tensor([1, 2, 3, 4, 5, 6], (3, 2))
        a.to_gpu()

        c = a * 2
        c.to_cpu()

        self.assertEqual(c.size, 6)
        self.assertEqual(c.dims, 2)
        self.assertEqual(c.shape, (3, 2))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [2, 4, 6, 8, 10, 12])

    def test_broadcast_right_multiply_cpu(self):
        a = Tensor([1, 2, 3, 4, 5, 6], (3, 2))

        c = 2 * a

        self.assertEqual(c.size, 6)
        self.assertEqual(c.dims, 2)
        self.assertEqual(c.shape, (3, 2))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [2, 4, 6, 8, 10, 12])

    def test_broadcast_right_multiply_gpu(self):
        a = Tensor([1, 2, 3, 4, 5, 6], (3, 2))
        a.to_gpu()

        c = 2 * a
        c.to_cpu()

        self.assertEqual(c.size, 6)
        self.assertEqual(c.dims, 2)
        self.assertEqual(c.shape, (3, 2))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [2, 4, 6, 8, 10, 12])

    def test_divide_cpu(self):
        a = Tensor([6, 4, 3], (3,))
        b = Tensor([3, 1, 2], (3,))

        c = a / b

        self.assertEqual(c.size, 3)
        self.assertEqual(c.dims, 1)
        self.assertEqual(c.shape, (3,))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [2, 4, 1.5])

    def test_divide_gpu(self):
        a = Tensor([6, 4, 3], (3,))
        b = Tensor([3, 1, 2], (3,))
        a.to_gpu()
        b.to_gpu()

        c = a / b
        c.to_cpu()

        self.assertEqual(c.size, 3)
        self.assertEqual(c.dims, 1)
        self.assertEqual(c.shape, (3,))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [2, 4, 1.5])

    def test_broadcast_divide_cpu(self):
        a = Tensor([6, 4, 3], (3,))

        c = a / 2

        self.assertEqual(c.size, 3)
        self.assertEqual(c.dims, 1)
        self.assertEqual(c.shape, (3,))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [3, 2, 1.5])

    def test_broadcast_divide_gpu(self):
        a = Tensor([6, 4, 3], (3,))
        a.to_gpu()

        c = a / 2
        c.to_cpu()

        self.assertEqual(c.size, 3)
        self.assertEqual(c.dims, 1)
        self.assertEqual(c.shape, (3,))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [3, 2, 1.5])

    def test_broadcast_right_divide_cpu(self):
        a = Tensor([6, 4, 3], (3,))

        c = 12 / a

        self.assertEqual(c.size, 3)
        self.assertEqual(c.dims, 1)
        self.assertEqual(c.shape, (3,))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [2, 3, 4])

    def test_broadcast_right_divide_gpu(self):
        a = Tensor([6, 4, 3], (3,))
        a.to_gpu()

        c = 12 / a
        c.to_cpu()

        self.assertEqual(c.size, 3)
        self.assertEqual(c.dims, 1)
        self.assertEqual(c.shape, (3,))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [2, 3, 4])

    def test_matmul_cpu(self):
        a = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
        b = Tensor([6, 4, 3, 2, 7, 6], (3, 2))

        c = a @ b

        self.assertEqual(c.size, 4)
        self.assertEqual(c.dims, 2)
        self.assertEqual(c.shape, (2, 2))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [33, 26, 81, 62])

    def test_matmul_gpu(self):
        a = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
        b = Tensor([6, 4, 3, 2, 7, 6], (3, 2))
        a.to_gpu()
        b.to_gpu()

        c = a @ b
        c.to_cpu()

        self.assertEqual(c.size, 4)
        self.assertEqual(c.dims, 2)
        self.assertEqual(c.shape, (2, 2))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [33, 26, 81, 62])

    def test_matmul_batch_cpu(self):
        a = Tensor([1, 2, 3, 4, 5, 6, 1, 0, 2, 3, 4, 1], (2, 2, 3))
        b = Tensor([6, 4, 3, 2, 7, 6, 0, 5, 2, 1, 6, 0], (2, 3, 2))

        c = a @ b

        self.assertEqual(c.size, 8)
        self.assertEqual(c.dims, 3)
        self.assertEqual(c.shape, (2, 2, 2))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [33, 26, 81, 62, 12, 5, 14, 19])

    def test_matmul_batch_gpu(self):
        a = Tensor([1, 2, 3, 4, 5, 6, 1, 0, 2, 3, 4, 1], (2, 2, 3))
        b = Tensor([6, 4, 3, 2, 7, 6, 0, 5, 2, 1, 6, 0], (2, 3, 2))
        a.to_gpu()
        b.to_gpu()

        c = a @ b
        c.to_cpu()

        self.assertEqual(c.size, 8)
        self.assertEqual(c.dims, 3)
        self.assertEqual(c.shape, (2, 2, 2))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [33, 26, 81, 62, 12, 5, 14, 19])

    def test_matmul_ultra_batch_cpu(self):
        import numpy as np
        na = np.random.randint(0, 100, (6, 5, 4, 3, 2))
        nb = np.random.randint(0, 100, (6, 5, 4, 2, 7))
        a = Tensor(na.flatten().tolist(), (6, 5, 4, 3, 2))
        b = Tensor(nb.flatten().tolist(), (6, 5, 4, 2, 7))

        c = a @ b

        self.assertEqual(c.size, 6 * 5 * 4 * 3 * 7)
        self.assertEqual(c.dims, 5)
        self.assertEqual(c.shape, (6, 5, 4, 3, 7))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, np.einsum(
            'abcij,abcjk->abcik', na, nb).flatten().tolist())

    def test_matmul_ultra_batch_gpu(self):
        import numpy as np
        na = np.random.randint(0, 100, (6, 5, 4, 3, 2))
        nb = np.random.randint(0, 100, (6, 5, 4, 2, 7))
        a = Tensor(na.flatten().tolist(), (6, 5, 4, 3, 2))
        b = Tensor(nb.flatten().tolist(), (6, 5, 4, 2, 7))
        a.to_gpu()
        b.to_gpu()

        c = a @ b
        c.to_cpu()

        self.assertEqual(c.size, 6 * 5 * 4 * 3 * 7)
        self.assertEqual(c.dims, 5)
        self.assertEqual(c.shape, (6, 5, 4, 3, 7))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, np.einsum(
            'abcij,abcjk->abcik', na, nb).flatten().tolist())

    def test_fill_random_uniform_cpu(self):
        x = Tensor.zeros(30)

        x.fill_random_uniform(lower=37, upper=38)

        for i in x.data:
            self.assertTrue(37 <= i <= 38)

    def test_fill_random_uniform_gpu(self):
        x = Tensor.zeros(30)
        x.to_gpu()

        x.fill_random_uniform(lower=37, upper=38)
        x.to_cpu()

        for i in x.data:
            self.assertTrue(37 <= i <= 38)

    def test_fill_random_normal_cpu(self):
        n = 10000
        x = Tensor.zeros(n)

        x.fill_random_normal(mean=5, std=2)

        mean = sum(x.data) / n
        std = (sum((i - mean) ** 2 for i in x.data) / n) ** 0.5
        self.assertAlmostEqual(mean, 5, delta=0.3)
        self.assertAlmostEqual(std, 2, delta=0.3)

    def test_fill_random_normal_gpu(self):
        n = 100000
        x = Tensor.zeros(n)
        x.to_gpu()

        x.fill_random_normal(mean=5, std=2)
        x.to_cpu()

        mean = sum(x.data) / n
        std = (sum((i - mean) ** 2 for i in x.data) / n) ** 0.5
        self.assertAlmostEqual(mean, 5, delta=0.3)
        self.assertAlmostEqual(std, 2, delta=0.3)

    def test_zeros(self):
        x = Tensor.zeros(3, 3)

        self.assertEqual(x.data, [0] * 9)

    def test_ones(self):
        x = Tensor.ones(3, 3)

        self.assertEqual(x.data, [1] * 9)

    def test_sum_cpu(self):
        x = Tensor([1, 2, 3, 4, 5, 6], (2, 3))

        s = x.sum()

        self.assertEqual(s, 21)

    def test_sum_gpu(self):
        x = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
        x.to_gpu()

        s = x.sum()

        self.assertEqual(s, 21)

    def test_mean_cpu(self):
        x = Tensor([1, 2, 3], (3,))

        m = x.mean()

        self.assertEqual(m, 2)

    def test_mean_cgpu(self):
        x = Tensor([1, 2, 3], (3,))
        x.to_gpu()

        m = x.mean()

        self.assertEqual(m, 2)

    def test_min_cpu(self):
        x = Tensor([8, 2, 1, -7, 3], (5,))

        m = x.min()

        self.assertEqual(m, -7)

    def test_min_gpu(self):
        x = Tensor([8, 2, 1, -7, 3], (5,))
        x.to_gpu()

        m = x.min()

        self.assertEqual(m, -7)

    def test_max_cpu(self):
        x = Tensor([8, 2, 1, 11, 3], (5,))

        m = x.max()

        self.assertEqual(m, 11)

    def test_max_gpu(self):
        x = Tensor([8, 2, 1, 11, 3], (5,))
        x.to_gpu()

        m = x.max()

        self.assertEqual(m, 11)

    def test_power_cpu(self):
        x = Tensor([1, 2, 3], (3,))

        y = x.power(2)

        self.assertEqual(y.size, 3)
        self.assertEqual(y.dims, 1)
        self.assertEqual(y.shape, (3,))
        self.assertEqual(y.device, 0)
        self.assertEqual(y.data, [1, 4, 9])

    def test_power_gpu(self):
        x = Tensor([1, 2, 3], (3,))
        x.to_gpu()

        y = x.power(2)
        y.to_cpu()

        self.assertEqual(y.size, 3)
        self.assertEqual(y.dims, 1)
        self.assertEqual(y.shape, (3,))
        self.assertEqual(y.device, 0)
        self.assertEqual(y.data, [1, 4, 9])

    def test_cos_cpu(self):
        x = Tensor([0, math.pi / 2, math.pi, 3 * math.pi / 2], (4,))

        y = x.cos()

        self.assertEqual(y.size, 4)
        self.assertEqual(y.dims, 1)
        self.assertEqual(y.shape, (4,))
        self.assertEqual(y.device, 0)
        self.assertAlmostEqual(y.data[0], 1, delta=1e-3)
        self.assertAlmostEqual(y.data[1], 0, delta=1e-3)
        self.assertAlmostEqual(y.data[2], -1, delta=1e-3)
        self.assertAlmostEqual(y.data[3], 0, delta=1e-3)

    def test_cos_gpu(self):
        x = Tensor([0, math.pi / 2, math.pi, 3 * math.pi / 2], (4,))
        x.to_gpu()

        y = x.cos()
        y.to_cpu()

        self.assertEqual(y.size, 4)
        self.assertEqual(y.dims, 1)
        self.assertEqual(y.shape, (4,))
        self.assertEqual(y.device, 0)
        self.assertAlmostEqual(y.data[0], 1, delta=1e-3)
        self.assertAlmostEqual(y.data[1], 0, delta=1e-3)
        self.assertAlmostEqual(y.data[2], -1, delta=1e-3)
        self.assertAlmostEqual(y.data[3], 0, delta=1e-3)

    def test_sin_cpu(self):
        x = Tensor([0, math.pi / 2, math.pi, 3 * math.pi / 2], (4,))

        y = x.sin()

        self.assertEqual(y.size, 4)
        self.assertEqual(y.dims, 1)
        self.assertEqual(y.shape, (4,))
        self.assertEqual(y.device, 0)
        self.assertAlmostEqual(y.data[0], 0, delta=1e-3)
        self.assertAlmostEqual(y.data[1], 1, delta=1e-3)
        self.assertAlmostEqual(y.data[2], 0, delta=1e-3)
        self.assertAlmostEqual(y.data[3], -1, delta=1e-3)

    def test_sin_gpu(self):
        x = Tensor([0, math.pi / 2, math.pi, 3 * math.pi / 2], (4,))
        x.to_gpu()

        y = x.sin()
        y.to_cpu()

        self.assertEqual(y.size, 4)
        self.assertEqual(y.dims, 1)
        self.assertEqual(y.shape, (4,))
        self.assertEqual(y.device, 0)
        self.assertAlmostEqual(y.data[0], 0, delta=1e-3)
        self.assertAlmostEqual(y.data[1], 1, delta=1e-3)
        self.assertAlmostEqual(y.data[2], 0, delta=1e-3)
        self.assertAlmostEqual(y.data[3], -1, delta=1e-3)

    def test_exp_cpu(self):
        x = Tensor.random_uniform((10,))

        y = x.exp()

        self.assertEqual(y.size, 10)
        self.assertEqual(y.dims, 1)
        self.assertEqual(y.shape, (10,))
        self.assertEqual(y.device, 0)
        for i, j in zip(x.data, y.data):
            self.assertAlmostEqual(j, math.exp(i), delta=1e-3)

    def test_exp_gpu(self):
        x = Tensor.random_uniform((10,))
        x.to_gpu()

        y = x.exp()
        x.to_cpu()
        y.to_cpu()

        self.assertEqual(y.size, 10)
        self.assertEqual(y.dims, 1)
        self.assertEqual(y.shape, (10,))
        self.assertEqual(y.device, 0)
        for i, j in zip(x.data, y.data):
            self.assertAlmostEqual(j, math.exp(i), delta=1e-3)

    def test_log_cpu(self):
        x = Tensor.random_uniform((10,), lower=1, upper=10)

        y = x.log()

        self.assertEqual(y.size, 10)
        self.assertEqual(y.dims, 1)
        self.assertEqual(y.shape, (10,))
        self.assertEqual(y.device, 0)
        for i, j in zip(x.data, y.data):
            self.assertAlmostEqual(j, math.log(i), delta=1e-3)

    def test_log_gpu(self):
        x = Tensor.random_uniform((10,), lower=1, upper=10)
        x.to_gpu()

        y = x.log()
        x.to_cpu()
        y.to_cpu()

        self.assertEqual(y.size, 10)
        self.assertEqual(y.dims, 1)
        self.assertEqual(y.shape, (10,))
        self.assertEqual(y.device, 0)
        for i, j in zip(x.data, y.data):
            self.assertAlmostEqual(j, math.log(i), delta=1e-3)

    def test_log10_cpu(self):
        x = Tensor.random_uniform((10,), lower=1, upper=10)

        y = x.log10()

        self.assertEqual(y.size, 10)
        self.assertEqual(y.dims, 1)
        self.assertEqual(y.shape, (10,))
        self.assertEqual(y.device, 0)
        for i, j in zip(x.data, y.data):
            self.assertAlmostEqual(j, math.log10(i), delta=1e-3)

    def test_log10_gpu(self):
        x = Tensor.random_uniform((10,), lower=1, upper=10)
        x.to_gpu()

        y = x.log10()
        x.to_cpu()
        y.to_cpu()

        self.assertEqual(y.size, 10)
        self.assertEqual(y.dims, 1)
        self.assertEqual(y.shape, (10,))
        self.assertEqual(y.device, 0)
        for i, j in zip(x.data, y.data):
            self.assertAlmostEqual(j, math.log10(i), delta=1e-3)

    def test_logb_cpu(self):
        x = Tensor.random_uniform((10,), lower=1, upper=10)

        y = x.logb(2)

        self.assertEqual(y.size, 10)
        self.assertEqual(y.dims, 1)
        self.assertEqual(y.shape, (10,))
        self.assertEqual(y.device, 0)
        for i, j in zip(x.data, y.data):
            self.assertAlmostEqual(j, math.log2(i), delta=1e-3)

    def test_logb_gpu(self):
        x = Tensor.random_uniform((10,), lower=1, upper=10)
        x.to_gpu()

        y = x.logb(2)
        x.to_cpu()
        y.to_cpu()

        self.assertEqual(y.size, 10)
        self.assertEqual(y.dims, 1)
        self.assertEqual(y.shape, (10,))
        self.assertEqual(y.device, 0)
        for i, j in zip(x.data, y.data):
            self.assertAlmostEqual(j, math.log2(i), delta=1e-3)

    def test_add_into_cpu(self):
        x = Tensor([1, 2, 3, 4], (2, 2))
        y = Tensor([1, 0, 1, 1], (2, 2))

        x += y

        self.assertEqual(x.size, 4)
        self.assertEqual(x.dims, 2)
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(x.device, 0)
        self.assertEqual(x.data, [2, 2, 4, 5])

    def test_add_into_gpu(self):
        x = Tensor([1, 2, 3, 4], (2, 2))
        y = Tensor([1, 0, 1, 1], (2, 2))
        x.to_gpu()
        y.to_gpu()

        x += y
        x.to_cpu()

        self.assertEqual(x.size, 4)
        self.assertEqual(x.dims, 2)
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(x.device, 0)
        self.assertEqual(x.data, [2, 2, 4, 5])

    def test_subtract_into_cpu(self):
        x = Tensor([1, 2, 3, 4], (2, 2))
        y = Tensor([1, 0, 1, 1], (2, 2))

        x -= y

        self.assertEqual(x.size, 4)
        self.assertEqual(x.dims, 2)
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(x.device, 0)
        self.assertEqual(x.data, [0, 2, 2, 3])

    def test_subtract_into_gpu(self):
        x = Tensor([1, 2, 3, 4], (2, 2))
        y = Tensor([1, 0, 1, 1], (2, 2))
        x.to_gpu()
        y.to_gpu()

        x -= y
        x.to_cpu()

        self.assertEqual(x.size, 4)
        self.assertEqual(x.dims, 2)
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(x.device, 0)
        self.assertEqual(x.data, [0, 2, 2, 3])

    def test_multiply_into_cpu(self):
        x = Tensor([1, 2, 3, 4], (2, 2))
        y = Tensor([1, 0, 1, 1], (2, 2))

        x *= y

        self.assertEqual(x.size, 4)
        self.assertEqual(x.dims, 2)
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(x.device, 0)
        self.assertEqual(x.data, [1, 0, 3, 4])

    def test_multiply_into_gpu(self):
        x = Tensor([1, 2, 3, 4], (2, 2))
        y = Tensor([1, 0, 1, 1], (2, 2))
        x.to_gpu()
        y.to_gpu()

        x *= y
        x.to_cpu()

        self.assertEqual(x.size, 4)
        self.assertEqual(x.dims, 2)
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(x.device, 0)
        self.assertEqual(x.data, [1, 0, 3, 4])

    def test_divide_into_cpu(self):
        x = Tensor([1, 2, 3, 4], (2, 2))
        y = Tensor([2, 1, 2, 2], (2, 2))

        x /= y

        self.assertEqual(x.size, 4)
        self.assertEqual(x.dims, 2)
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(x.device, 0)
        self.assertEqual(x.data, [0.5, 2, 1.5, 2])

    def test_divide_into_gpu(self):
        x = Tensor([1, 2, 3, 4], (2, 2))
        y = Tensor([2, 1, 2, 2], (2, 2))
        x.to_gpu()
        y.to_gpu()

        x /= y
        x.to_cpu()

        self.assertEqual(x.size, 4)
        self.assertEqual(x.dims, 2)
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(x.device, 0)
        self.assertEqual(x.data, [0.5, 2, 1.5, 2])

    def test_tanh_cpu(self):
        x = Tensor([1, 2, 3, 4], (2, 2))

        y = x.tanh()

        self.assertEqual(x.size, 4)
        self.assertEqual(x.dims, 2)
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(x.device, 0)
        for i, j in zip(x.data, y.data):
            self.assertAlmostEqual(j, math.tanh(i))

    def test_tanh_gpu(self):
        x = Tensor([1, 2, 3, 4], (2, 2))
        x.to_gpu()

        y = x.tanh()
        x.to_cpu()
        y.to_cpu()

        self.assertEqual(x.size, 4)
        self.assertEqual(x.dims, 2)
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(x.device, 0)
        for i, j in zip(x.data, y.data):
            self.assertAlmostEqual(j, math.tanh(i))

    def test_get_item(self):
        x = Tensor([1, 2, 3, 4], (2, 2))

        n = x[1, 0]

        self.assertEqual(n, 3)

    def test_set_item(self):
        x = Tensor([1, 2, 3, 4], (2, 2))

        x[1, 0] = 20

        self.assertEqual(x[1, 0], 20)

    def test_fill_identity_cpu(self):
        x = Tensor.zeros(3, 3)

        x.fill_identity()

        self.assertEqual(x.data, [1, 0, 0, 0, 1, 0, 0, 0, 1])

    def test_fill_identity_gpu(self):
        x = Tensor.zeros(3, 3)
        x.to_gpu()

        x.fill_identity()
        x.to_cpu()

        self.assertEqual(x.data, [1, 0, 0, 0, 1, 0, 0, 0, 1])

    def test_fill_identity_3d_cpu(self):
        x = Tensor.zeros(2, 2, 2)

        x.fill_identity()

        self.assertEqual(x.data, [1, 0, 0, 0, 0, 0, 0, 1])

    def test_fill_identity_3d_gpu(self):
        x = Tensor.zeros(2, 2, 2)
        x.to_gpu()

        x.fill_identity()
        x.to_cpu()

        self.assertEqual(x.data, [1, 0, 0, 0, 0, 0, 0, 1])

    def test_transpose(self):
        x = Tensor([1, 2, 3, 4, 5, 6], (1, 2, 3))

        y = x.T

        self.assertEqual(y.size, 6)
        self.assertEqual(y.dims, 3)
        self.assertEqual(y.shape, (1, 3, 2))
        self.assertEqual(y.device, 0)
        self.assertEqual(y.data, [1, 2, 3, 4, 5, 6])
        self.assertEqual(y[0, 0, 0], 1)
        self.assertEqual(y[0, 0, 1], 4)
        self.assertEqual(y[0, 1, 0], 2)
        self.assertEqual(y[0, 1, 1], 5)
        self.assertEqual(y[0, 2, 0], 3)
        self.assertEqual(y[0, 2, 1], 6)

    def test_transpose_square(self):
        x = Tensor([1, 2, 3, 4], (2, 2))

        y = x.T

        self.assertEqual(y.size, 4)
        self.assertEqual(y.dims, 2)
        self.assertEqual(y.shape, (2, 2))
        self.assertEqual(y.device, 0)
        self.assertEqual(y.data, [1, 2, 3, 4])
        self.assertEqual(y[0, 0], 1)
        self.assertEqual(y[0, 1], 3)
        self.assertEqual(y[1, 0], 2)
        self.assertEqual(y[1, 1], 4)

    def test_create_from_array(self):
        x = Tensor.array([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
            [[13, 14, 15], [16, 17, 18]],
        ])

        self.assertEqual(x.size, 18)
        self.assertEqual(x.dims, 3)
        self.assertEqual(x.shape, (3, 2, 3))
        self.assertEqual(x.device, 0)
        self.assertEqual(x.data, list(range(1, 19)))

    def test_slice(self):
        x = Tensor.array([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
            [[13, 14, 15], [16, 17, 18]],
        ])

        y = x[::2, :, 1::]

        self.assertEqual(y.size, 8)
        self.assertEqual(y.dims, 3)
        self.assertEqual(y.shape, (2, 2, 2))
        self.assertEqual(y.stride, (12, 3, 1))
        self.assertEqual(y.offset, 1)
        self.assertEqual(y.device, 0)
        self.assertEqual(y.data, [2, 3, 5, 6, 14, 15, 17, 18])
        self.assertEqual(y.indices, [1, 2, 4, 5, 13, 14, 16, 17])
        self.assertEqual(y.base, x)

    def test_slice_all(self):
        x = Tensor.array([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
            [[13, 14, 15], [16, 17, 18]],
        ])

        y = x[:]

        self.assertEqual(y.size, x.size)
        self.assertEqual(y.dims, x.dims)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.stride, x.stride)
        self.assertEqual(y.offset, x.offset)
        self.assertEqual(y.device, x.device)
        self.assertEqual(y.data, x.data)
        self.assertEqual(y.indices, x.indices)
        self.assertEqual(y.base, x)

    def test_slice_negative_start_stop(self):
        x = Tensor.array([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
            [[13, 14, 15], [16, 17, 18]],
        ])

        y = x[-1:, :-1]

        self.assertEqual(y.size, 3)
        self.assertEqual(y.dims, 3)
        self.assertEqual(y.shape, (1, 1, 3))
        self.assertEqual(y.stride, x.stride)
        self.assertEqual(y.offset, 12)
        self.assertEqual(y.device, 0)
        self.assertEqual(y.data, [13, 14, 15])
        self.assertEqual(y.indices, [12, 13, 14])
        self.assertEqual(y.base, x)

    def test_slice_with_index(self):
        x = Tensor.array([
            [[1, 2], [4, 5], [6, 7]],
            [[8, 9], [10, 11], [12, 13]],
            [[14, 15], [16, 17], [18, 19]],
            [[20, 21], [22, 23], [24, 25]],
        ])

        y = x[0]

        self.assertEqual(y.size, 6)
        self.assertEqual(y.dims, 2)
        self.assertEqual(y.shape, (3, 2))
        self.assertEqual(y.stride, (2, 1))
        self.assertEqual(y.offset, 0)
        self.assertEqual(y.device, 0)
        self.assertEqual(y.data, [1, 2, 4, 5, 6, 7])
        self.assertEqual(y.indices, [0, 1, 2, 3, 4, 5])
        self.assertEqual(y.base, x)

    def test_slice_multiple(self):
        x = Tensor(list(range(24)), (3, 2, 4))

        y1 = x[0:3:2, 0:2, 1:4]
        y2 = y1[1:2, :2, :2]
        y3 = y2[:1, :1, 1:2]

        self.assertEqual(x.offset, 0)
        self.assertEqual(y1.offset, 1)
        self.assertEqual(y1.stride, (16, 4, 1))
        self.assertEqual(y2.offset, 17)
        self.assertEqual(y2.stride, (16, 4, 1))
        self.assertEqual(y3.size, 1)
        self.assertEqual(y3.dims, 3)
        self.assertEqual(y3.shape, (1, 1, 1))
        self.assertEqual(y3.stride, (16, 4, 1))
        self.assertEqual(y3.offset, 18)
        self.assertEqual(y3.device, 0)
        self.assertEqual(y3.data, [18])
        self.assertEqual(y3.indices, [18])
        self.assertEqual(y3.base, x)
        self.assertEqual(y2.base, x)
        self.assertEqual(y1.base, x)

    def test_slice_and_add(self):
        x = Tensor.array([
            [[1, 2], [4, 5], [6, 7]],
            [[8, 9], [10, 11], [12, 13]],
            [[1, 2], [2, 1], [1, 0]],
            [[20, 21], [22, 23], [24, 25]],
        ])

        y = x[0] + x[2]

        self.assertEqual(y.size, 6)
        self.assertEqual(y.dims, 2)
        self.assertEqual(y.shape, (3, 2))
        self.assertEqual(y.stride, (2, 1))
        self.assertEqual(y.offset, 0)
        self.assertEqual(y.device, 0)
        self.assertEqual(y.data, [2, 4, 6, 6, 7, 7])
        self.assertEqual(y.indices, [0, 1, 2, 3, 4, 5])
        self.assertEqual(y.base, None)

    def test_slice_and_matmul(self):
        a = Tensor([1, 2, 3, 4, 5, 6, 1, 0, 2, 3, 4, 1, -2, -2, -2, -2, -2, -2], (3, 2, 3))
        b = Tensor([6, 4, 3, 2, 7, 6, 0, 5, 2, 1, 6, 0, -4, -4, -4, -4, -4, -4], (3, 3, 2))

        c = a[:-1] @ b[:-1]

        self.assertEqual(c.size, 8)
        self.assertEqual(c.dims, 3)
        self.assertEqual(c.shape, (2, 2, 2))
        self.assertEqual(c.stride, (4, 2, 1))
        self.assertEqual(c.offset, 0)
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [33, 26, 81, 62, 12, 5, 14, 19])
        self.assertEqual(c.indices, [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(c.base, None)

    def test_slice_and_reshape_will_copy(self):
        x = Tensor.array([
            [[1, 2], [4, 5], [6, 7]],
            [[8, 9], [10, 11], [12, 13]],
            [[14, 15], [16, 17], [18, 19]],
            [[20, 21], [22, 23], [24, 25]],
        ])

        y = x[::2, -2:] # shape=(2, 2, 2), stride=(12, 2, 1)
        z = y.reshape(4, 2)

        self.assertEqual(y.shape, (2, 2, 2))
        self.assertEqual(y.stride, (12, 2, 1))
        self.assertEqual(y.base, x)

        self.assertEqual(z.size, 8)
        self.assertEqual(z.dims, 2)
        self.assertEqual(z.shape, (4, 2))
        self.assertEqual(z.stride, (2, 1))
        self.assertEqual(z.offset, 0)
        self.assertEqual(z.device, 0)
        self.assertEqual(z.data, [4, 5, 6, 7, 16, 17, 18, 19])
        self.assertEqual(z.indices, [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(z.base, None)
        self.assertNotEqual(z, y)

    def test_slice_and_reshape_will_not_copy(self):
        x = Tensor.array([
            [[1, 2], [4, 5], [6, 7]],
            [[8, 9], [10, 11], [12, 13]],
            [[14, 15], [16, 17], [18, 19]],
            [[20, 21], [22, 23], [24, 25]],
        ])

        y = x[::2, -2:] # shape=(2, 2, 2), stride=(12, 2, 1)
        z = y.reshape(2, 1, 2, 2, 1)

        self.assertEqual(z.size, 8)
        self.assertEqual(z.dims, 5)
        self.assertEqual(z.shape, (2, 1, 2, 2, 1))
        self.assertEqual(z.stride, (12, 1, 2, 1, 1))
        self.assertEqual(z.offset, 2)
        self.assertEqual(z.device, 0)
        self.assertEqual(z.data, [4, 5, 6, 7, 16, 17, 18, 19])
        self.assertEqual(z.indices, [2, 3, 4, 5, 14, 15, 16, 17])
        self.assertEqual(z.base, x)
        self.assertEqual(z, y)

    def test_squeeze(self):
        x = Tensor([1, 2, 3, 4], (1, 1, 2, 1, 2, 1))

        x.squeeze()

        self.assertEqual(x.size, 4)
        self.assertEqual(x.dims, 2)
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(x.device, 0)
        self.assertEqual(x.data, [1, 2, 3, 4])

    def test_squeeze_keep_at_least_one_dim(self):
        x = Tensor([5], (1, 1, 1))

        x.squeeze()

        self.assertEqual(x.size, 1)
        self.assertEqual(x.dims, 1)
        self.assertEqual(x.shape, (1,))
        self.assertEqual(x.device, 0)
        self.assertEqual(x.data, [5])

    def test_iter(self):
        x = Tensor([1, 2, 3, 4], (2, 2))

        y, z = list(iter(x))

        self.assertEqual(y.size, 2)
        self.assertEqual(y.dims, 1)
        self.assertEqual(y.shape, (2,))
        self.assertEqual(y.device, 0)
        self.assertEqual(y.data, [1, 2])

        self.assertEqual(z.size, 2)
        self.assertEqual(z.dims, 1)
        self.assertEqual(z.shape, (2,))
        self.assertEqual(z.device, 0)
        self.assertEqual(z.data, [3, 4])

    def test_len(self):
        x = Tensor.zeros(6, 4, 2)

        n = len(x)

        self.assertEqual(n, 6)

    def test_broadcase_add_into_cpu(self):
        x = Tensor([1, 2, 3, 4], (2, 2))

        x += 1

        self.assertEqual(x.size, 4)
        self.assertEqual(x.dims, 2)
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(x.device, 0)
        self.assertEqual(x.data, [2, 3, 4, 5])

    def test_broadcase_add_into_gpu(self):
        x = Tensor([1, 2, 3, 4], (2, 2))
        x.to_gpu()

        x += 1
        x.to_cpu()

        self.assertEqual(x.size, 4)
        self.assertEqual(x.dims, 2)
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(x.device, 0)
        self.assertEqual(x.data, [2, 3, 4, 5])

    def test_broadcase_subtract_into_cpu(self):
        x = Tensor([1, 2, 3, 4], (2, 2))

        x -= 1

        self.assertEqual(x.size, 4)
        self.assertEqual(x.dims, 2)
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(x.device, 0)
        self.assertEqual(x.data, [0, 1, 2, 3])

    def test_broadcase_subtract_into_gpu(self):
        x = Tensor([1, 2, 3, 4], (2, 2))
        x.to_gpu()

        x -= 1
        x.to_cpu()

        self.assertEqual(x.size, 4)
        self.assertEqual(x.dims, 2)
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(x.device, 0)
        self.assertEqual(x.data, [0, 1, 2, 3])

    def test_broadcase_multiply_into_cpu(self):
        x = Tensor([1, 2, 3, 4], (2, 2))

        x *= 2

        self.assertEqual(x.size, 4)
        self.assertEqual(x.dims, 2)
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(x.device, 0)
        self.assertEqual(x.data, [2, 4, 6, 8])

    def test_broadcase_multiply_into_gpu(self):
        x = Tensor([1, 2, 3, 4], (2, 2))
        x.to_gpu()

        x *= 2
        x.to_cpu()

        self.assertEqual(x.size, 4)
        self.assertEqual(x.dims, 2)
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(x.device, 0)
        self.assertEqual(x.data, [2, 4, 6, 8])

    def test_broadcase_divide_into_cpu(self):
        x = Tensor([1, 2, 3, 4], (2, 2))

        x /= 2

        self.assertEqual(x.size, 4)
        self.assertEqual(x.dims, 2)
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(x.device, 0)
        self.assertEqual(x.data, [0.5, 1, 1.5, 2])

    def test_broadcase_divide_into_gpu(self):
        x = Tensor([1, 2, 3, 4], (2, 2))
        x.to_gpu()

        x /= 2
        x.to_cpu()

        self.assertEqual(x.size, 4)
        self.assertEqual(x.dims, 2)
        self.assertEqual(x.shape, (2, 2))
        self.assertEqual(x.device, 0)
        self.assertEqual(x.data, [0.5, 1, 1.5, 2])


if __name__ == "__main__":
    unittest.main()
