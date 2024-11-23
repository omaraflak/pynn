import ctypes


class CTensor(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.POINTER(ctypes.c_float)),
        ('shape', ctypes.POINTER(ctypes.c_uint32)),
        ('stride', ctypes.POINTER(ctypes.c_uint32)),
        ('dims', ctypes.c_uint32),
        ('size', ctypes.c_uint32),
        ('device', ctypes.c_uint32),
    ]


def init_tensor_c_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL('/content/libtensor.so')

    lib.create_tensor.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
    lib.create_tensor.restype = ctypes.POINTER(CTensor)
    lib.copy_tensor.argtypes = [ctypes.POINTER(CTensor)]
    lib.copy_tensor.restype = ctypes.POINTER(CTensor)
    lib.delete_tensor.argtypes = [ctypes.POINTER(CTensor)]
    lib.delete_tensor.restype = None

    lib.tensor_cpu_to_gpu.argtypes = [ctypes.POINTER(CTensor)]
    lib.tensor_cpu_to_gpu.restype = None
    lib.tensor_gpu_to_cpu.argtypes = [ctypes.POINTER(CTensor)]
    lib.tensor_gpu_to_cpu.restype = None

    lib.fill_tensor_data.argtypes = [ctypes.POINTER(CTensor), ctypes.c_float]
    lib.fill_tensor_data.restype = None
    lib.reshape_tensor.argtypes = [ctypes.POINTER(
        CTensor), ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
    lib.reshape_tensor.restype = None
    lib.get_tensor_item.argtypes = [ctypes.POINTER(
        CTensor), ctypes.POINTER(ctypes.c_uint32)]
    lib.get_tensor_item.restype = ctypes.c_float

    lib.add_tensors.argtypes = [
        ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    lib.add_tensors.restype = ctypes.POINTER(CTensor)
    lib.matmul_tensors.argtypes = [
        ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    lib.matmul_tensors.restype = ctypes.POINTER(CTensor)

    return lib


class Tensor:
    _C = init_tensor_c_lib()

    def __init__(self, data: list[float], shape: tuple[int, ...], c_tensor=None):
        if c_tensor:
            self.c_tensor = c_tensor
            return

        self.c_data = (ctypes.c_float * len(data))(*data)
        self.c_shape = (ctypes.c_uint32 * len(data))(*shape)
        self.c_tensor = Tensor._C.create_tensor(
            self.c_data,
            self.c_shape,
            ctypes.c_uint32(len(shape)),
        )

    @property
    def dims(self) -> int:
        return self.c_tensor.contents.dims

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.c_tensor.contents.shape[i] for i in range(self.dims))

    @property
    def size(self) -> int:
        return self.c_tensor.contents.size

    @property
    def stride(self) -> tuple[int, ...]:
        return tuple(self.c_tensor.contents.stride[i] for i in range(self.dims))

    @property
    def device(self) -> int:
        return self.c_tensor.contents.device

    def copy(self) -> 'Tensor':
        c_tensor = Tensor._C.tensor_copy(self.c_tensor)
        return Tensor(None, None, c_tensor)

    def to_gpu(self):
        Tensor._C.tensor_cpu_to_gpu(self.c_tensor)

    def to_cpu(self):
        Tensor._C.tensor_gpu_to_cpu(self.c_tensor)

    def reshape(self, shape: tuple[int, ...]):
        Tensor._C.reshape_tensor(
            self.c_tensor,
            (ctypes.c_uint32 * len(shape))(*shape),
            ctypes.c_uint32(len(shape))
        )

    def add(self, other: 'Tensor') -> 'Tensor':
        c_tensor = Tensor._C.add_tensors(self.c_tensor, other.c_tensor)
        return Tensor(None, None, c_tensor)

    def dot(self, other: 'Tensor') -> 'Tensor':
        c_tensor = Tensor._C.matmul_tensors(self.c_tensor, other.c_tensor)
        return Tensor(None, None, c_tensor)

    def get(self, *key: tuple[int, ...]) -> float:
        return Tensor._C.get_tensor_item(self.c_tensor, (ctypes.c_uint32 * len(key))(*key))

    def __del__(self):
        Tensor._C.delete_tensor(self.c_tensor)


a = Tensor([1, 2, 3, 4, 5, 6], (1, 3))
b = Tensor([1, 2, 3, 4, 5, 6], (3, 1))
c = a.dot(b)

print(a.shape, a.stride, a.size)
print(b.shape, b.stride, b.size)
print(c.shape, c.stride, c.size)

for i in range(c.shape[0]):
    for j in range(c.shape[1]):
        print(c.get(i, j), end=' ')
    print()
