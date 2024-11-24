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
