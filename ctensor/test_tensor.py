import unittest
from tensor import Tensor


class TestTensor(unittest.TestCase):

    def test_create_tensor(self):
        x = Tensor([1, 2, 3, 4, 5, 6], (3, 2))

        self.assertEqual(x.size, 6)
        self.assertEqual(x.dims, 2)
        self.assertEqual(x.shape, (3, 2))
        self.assertEqual(x.device, 0)

    def test_to_gpu_changes_device(self):
        x = Tensor([1, 2, 3], (3,))

        x.to_gpu()

        self.assertEqual(x.device, 1)

    def test_to_cpu_changes_device(self):
        x = Tensor([1, 2, 3], (3,))
        x.to_gpu()

        x.to_cpu()

        self.assertEqual(x.device, 0)

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

    # def test_broadcast_right_subtract_cpu(self):
    #     a = Tensor([1, 2, 3, 4, 5, 6], (3, 2))

    #     c = 1 - a

    #     self.assertEqual(c.size, 6)
    #     self.assertEqual(c.dims, 2)
    #     self.assertEqual(c.shape, (3, 2))
    #     self.assertEqual(c.device, 0)
    #     self.assertEqual(c.data, [0, -1, -2, -3, -4, -5])

    # def test_broadcast_right_subtract_gpu(self):
    #     a = Tensor([1, 2, 3, 4, 5, 6], (3, 2))
    #     a.to_gpu()

    #     c = 1 - a
    #     c.to_cpu()

    #     self.assertEqual(c.size, 6)
    #     self.assertEqual(c.dims, 2)
    #     self.assertEqual(c.shape, (3, 2))
    #     self.assertEqual(c.device, 0)
    #     self.assertEqual(c.data, [0, -1, -2, -3, -4, -5])

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

    # def test_broadcast_right_divide_cpu(self):
    #     a = Tensor([6, 4, 3], (3,))

    #     c = 12 / a

    #     self.assertEqual(c.size, 3)
    #     self.assertEqual(c.dims, 1)
    #     self.assertEqual(c.shape, (3,))
    #     self.assertEqual(c.device, 0)
    #     self.assertEqual(c.data, [2, 3, 4])

    # def test_broadcast_right_divide_gpu(self):
    #     a = Tensor([1, 2, 3, 4, 5, 6], (3, 2))
    #     a.to_gpu()

    #     c = 2 / a
    #     c.to_cpu()

    #     self.assertEqual(c.size, 3)
    #     self.assertEqual(c.dims, 1)
    #     self.assertEqual(c.shape, (3,))
    #     self.assertEqual(c.device, 0)
    #     self.assertEqual(c.data, [2, 3, 4])

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


if __name__ == "__main__":
    unittest.main()
