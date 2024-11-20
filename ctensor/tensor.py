import ctypes
from functools import reduce


class CTensor(ctypes.Structure):
    _fields_ = [
        ('data', ctypes.POINTER(ctypes.c_float)),
        ('shape', ctypes.POINTER(ctypes.c_uint32)),
        ('stride', ctypes.POINTER(ctypes.c_uint32)),
        ('dims', ctypes.c_uint32),
        ('size', ctypes.c_uint32),
    ]


def init_tensor_c_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL('libtensor.so')
    lib.create_tensor.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32
    ]
    lib.create_tensor.restype = ctypes.POINTER(CTensor)
    lib.free_tensor.argtypes = [ctypes.POINTER(CTensor)]
    lib.free_tensor.restype = None
    lib.add_tensors.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(CTensor)]
    lib.add_tensors.restype = ctypes.POINTER(CTensor)
    lib.reshape_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.c_uint32), ctypes.c_uint32]
    lib.reshape_tensor.restype = ctypes.POINTER(CTensor)
    lib.get_tensor_item.argtypes = [ctypes.POINTER(CTensor), ctypes.POINTER(ctypes.c_uint32)]
    lib.get_tensor_item.restype = ctypes.c_float
    return lib


class Tensor:
    _C = init_tensor_c_lib()


    def __init__(self, data: list[float], shape: tuple[int,...], c_tensor = None):
        if c_tensor:
            self.c_tensor = c_tensor
            self.dims = self.c_tensor.contents.dims
            self.shape = tuple(self.c_tensor.contents.shape[i] for i in range(self.dims))
            self.size = self.c_tensor.contents.size
            return

        self.shape = shape
        self.dims = len(shape)
        self.size = reduce(lambda a, b: a * b, shape, 1)
        self.c_data = (ctypes.c_float * len(data))(*data)
        self.c_shape = (ctypes.c_uint32 * len(data))(*shape)
        self.c_tensor = Tensor._C.create_tensor(
            self.c_data,
            self.c_shape,
            ctypes.c_uint32(len(shape)),
        )


    def reshape(self, shape: tuple[int,...]) -> 'Tensor':
        c_tensor = Tensor._C.reshape_tensor(
            self.c_tensor,
            (ctypes.c_uint32 * len(shape))(*shape),
            ctypes.c_uint32(len(shape))
        )
        return Tensor(None, None, c_tensor)


    def add(self, other: 'Tensor') -> 'Tensor':
        c_tensor = Tensor._C.add_tensors(self.c_tensor, other.c_tensor)
        return Tensor(None, None, c_tensor)


    def get(self, *key: tuple[int,...]) -> float:
        return Tensor._C.get_tensor_item(self.c_tensor, (ctypes.c_uint32 * len(key))(*key))


    def __del__(self):
        Tensor._C.free_tensor(self.c_tensor)


a = Tensor([1, 2, 3], (3,))
b = a.reshape((3, 1))

