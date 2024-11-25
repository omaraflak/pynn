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

    def test_to_gpu_changes_device(self):
        x = Tensor([1, 2, 3], (3,))

        x.to_gpu()

        self.assertEqual(x.device, 1)

    def test_to_cpu_changes_device(self):
        x = Tensor([1, 2, 3], (3,))
        x.to_gpu()

        x.to_cpu()

        self.assertEqual(x.device, 0)

    def test_copy_cpu(self):
        x = Tensor([1, 2, 3, 4, 5, 6], (3, 2))

        y = x.copy()

        self.assertEqual(y.size, 6)
        self.assertEqual(y.dims, 2)
        self.assertEqual(y.shape, (3, 2))
        self.assertEqual(y.device, 0)
        self.assertEqual(y.data, [1, 2, 3, 4, 5, 6])

    def test_copy_gpu(self):
        x = Tensor([1, 2, 3, 4, 5, 6], (3, 2))
        x.to_gpu()

        y = x.copy()
        y.to_cpu()

        self.assertEqual(y.size, 6)
        self.assertEqual(y.dims, 2)
        self.assertEqual(y.shape, (3, 2))
        self.assertEqual(y.device, 0)
        self.assertEqual(y.data, [1, 2, 3, 4, 5, 6])

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

        x.reshape((3, 2))

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

    def test_dot_cpu(self):
        a = Tensor([1, 2, 3, 4, 5, 6], (2, 3))
        b = Tensor([6, 4, 3, 2, 7, 6], (3, 2))

        c = a @ b

        self.assertEqual(c.size, 4)
        self.assertEqual(c.dims, 2)
        self.assertEqual(c.shape, (2, 2))
        self.assertEqual(c.device, 0)
        self.assertEqual(c.data, [33, 26, 81, 62])

    def test_dot_gpu(self):
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

    def test_random_uniform(self):
        x = Tensor.random_uniform((3, 3), lower=37, upper=38)

        for i in x.data:
            self.assertTrue(37 <= i <= 38)

    def test_random_normal(self):
        n = 10000

        x = Tensor.random_normal((n,), mean=5, std=2)

        mean = sum(x.data) / n
        std = (sum((i - mean) ** 2 for i in x.data) / n) ** 0.5
        self.assertAlmostEqual(mean, 5, delta=0.3)
        self.assertAlmostEqual(std, 2, delta=0.3)

    def test_ones(self):
        x = Tensor.ones((3, 3))

        self.assertListEqual(x.data, [0] * 9)

    def test_ones(self):
        x = Tensor.ones((3, 3))

        self.assertListEqual(x.data, [1] * 9)

    def test_sum(self):
        x = Tensor([1, 2, 3, 4, 5, 6], (2, 3))

        s = x.sum()

        self.assertEqual(s, 21)

    def test_mean(self):
        x = Tensor([1, 2, 3], (3,))

        m = x.mean()

        self.assertEqual(m, 2)

    def test_min(self):
        x = Tensor([8, 2, 1, -7, 3], (5,))

        m = x.min()

        self.assertEqual(m, -7)

    def test_max(self):
        x = Tensor([8, 2, 1, 11, 3], (5,))

        m = x.max()

        self.assertEqual(m, 11)

    def test_power(self):
        x = Tensor([1, 2, 3], (3,))

        y = x.power(2)

        self.assertEqual(y.size, 3)
        self.assertEqual(y.dims, 1)
        self.assertEqual(y.shape, (3,))
        self.assertEqual(y.device, 0)
        self.assertEqual(y.data, [1, 4, 9])

    def test_cos(self):
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

    def test_sin(self):
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


if __name__ == "__main__":
    unittest.main()
