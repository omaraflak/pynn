from __future__ import annotations
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


def _init_tensor_c_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL('/content/libtensor.so')

    lib.tensor_create.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32
    ]
    lib.tensor_create.restype = ctypes.POINTER(CTensor)
    lib.tensor_create_empty.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32
    ]
    lib.tensor_create_empty.restype = ctypes.POINTER(CTensor)
    lib.tensor_copy.argtypes = [ctypes.POINTER(CTensor)]
    lib.tensor_copy.restype = ctypes.POINTER(CTensor)
    lib.tensor_delete.argtypes = [ctypes.POINTER(CTensor)]
    lib.tensor_delete.restype = None

    lib.tensor_cpu_to_gpu.argtypes = [ctypes.POINTER(CTensor)]
    lib.tensor_cpu_to_gpu.restype = None
    lib.tensor_gpu_to_cpu.argtypes = [ctypes.POINTER(CTensor)]
    lib.tensor_gpu_to_cpu.restype = None

    lib.tensor_fill.argtypes = [ctypes.POINTER(CTensor), ctypes.c_float]
    lib.tensor_fill.restype = None
    lib.tensor_fill_random_uniform.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.tensor_fill_random_uniform.restype = None
    lib.tensor_fill_random_normal.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float,
        ctypes.c_float,
    ]
    lib.tensor_fill_random_normal.restype = None
    lib.tensor_reshape.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32
    ]
    lib.tensor_reshape.restype = None
    lib.tensor_get_item.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.c_uint32)
    ]
    lib.tensor_get_item.restype = ctypes.c_float
    lib.tensor_sum.argtypes = [ctypes.POINTER(CTensor)]
    lib.tensor_sum.restype = ctypes.c_float
    lib.tensor_mean.argtypes = [ctypes.POINTER(CTensor)]
    lib.tensor_mean.restype = ctypes.c_float

    lib.tensor_unary_minus.argtypes = [ctypes.POINTER(CTensor)]
    lib.tensor_unary_minus.restype = ctypes.POINTER(CTensor)
    lib.tensor_add.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor)
    ]
    lib.tensor_add.restype = ctypes.POINTER(CTensor)
    lib.tensor_subtract.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor)
    ]
    lib.tensor_subtract.restype = ctypes.POINTER(CTensor)
    lib.tensor_multiply.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor)
    ]
    lib.tensor_multiply.restype = ctypes.POINTER(CTensor)
    lib.tensor_divide.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor)
    ]
    lib.tensor_divide.restype = ctypes.POINTER(CTensor)
    lib.tensor_matmul.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor)
    ]
    lib.tensor_matmul.restype = ctypes.POINTER(CTensor)

    lib.tensor_broadcast_add.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float
    ]
    lib.tensor_broadcast_add.restype = ctypes.POINTER(CTensor)
    lib.tensor_broadcast_subtract.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float
    ]
    lib.tensor_broadcast_subtract.restype = ctypes.POINTER(CTensor)
    lib.tensor_broadcast_multiply.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float
    ]
    lib.tensor_broadcast_multiply.restype = ctypes.POINTER(CTensor)
    lib.tensor_broadcast_divide.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float
    ]
    lib.tensor_broadcast_divide.restype = ctypes.POINTER(CTensor)
    lib.tensor_broadcast_right_divide.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float
    ]
    lib.tensor_broadcast_right_divide.restype = ctypes.POINTER(CTensor)
    return lib


class Tensor:
    _C = _init_tensor_c_lib()

    def __init__(
        self,
        data: list[float] | None,
        shape: tuple[int, ...] | None,
        c_tensor=None
    ):
        if c_tensor:
            self.c_tensor = c_tensor
            return

        if not data or not shape:
            raise ValueError("Must provide data and shape.")

        self.c_tensor = Tensor._C.tensor_create(
            (ctypes.c_float * len(data))(*data),
            (ctypes.c_uint32 * len(shape))(*shape),
            ctypes.c_uint32(len(shape)),
        )

    @classmethod
    def _empty(cls, shape: tuple[int, ...]) -> Tensor:
        c_tensor = Tensor._C.tensor_create_empty(
            (ctypes.c_uint32 * len(shape))(*shape),
            ctypes.c_uint32(len(shape)),
        )
        return Tensor(None, None, c_tensor)

    @classmethod
    def random_uniform(cls, shape: tuple[int, ...], lower: float = 0, upper: float = 1) -> Tensor:
        tensor = Tensor._empty(shape)
        tensor.fill_random_uniform(lower, upper)
        return tensor

    @classmethod
    def random_normal(cls, shape: tuple[int, ...], mean: float = 0, std: float = 1) -> Tensor:
        tensor = Tensor._empty(shape)
        tensor.fill_random_normal(mean, std)
        return tensor

    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> Tensor:
        tensor = Tensor._empty(shape)
        tensor.fill(0)
        return tensor

    @classmethod
    def ones(cls, shape: tuple[int, ...]) -> Tensor:
        tensor = Tensor._empty(shape)
        tensor.fill(1)
        return tensor

    @property
    def dims(self) -> int:
        return self.c_tensor.contents.dims

    @property
    def size(self) -> int:
        return self.c_tensor.contents.size

    @property
    def data(self) -> list[float]:
        return [self.c_tensor.contents.data[i] for i in range(self.size)]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.c_tensor.contents.shape[i] for i in range(self.dims))

    @property
    def stride(self) -> tuple[int, ...]:
        return tuple(self.c_tensor.contents.stride[i] for i in range(self.dims))

    @property
    def device(self) -> int:
        return self.c_tensor.contents.device

    def copy(self) -> Tensor:
        c_tensor = Tensor._C.tensor_copy(self.c_tensor)
        return Tensor(None, None, c_tensor)

    def to_gpu(self):
        Tensor._C.tensor_cpu_to_gpu(self.c_tensor)

    def to_cpu(self):
        Tensor._C.tensor_gpu_to_cpu(self.c_tensor)

    def reshape(self, shape: tuple[int, ...]):
        Tensor._C.tensor_reshape(
            self.c_tensor,
            (ctypes.c_uint32 * len(shape))(*shape),
            ctypes.c_uint32(len(shape))
        )

    def fill(self, value: float):
        Tensor._C.tensor_fill(self.c_tensor, ctypes.c_float(value))

    def fill_random_uniform(self, lower: float = 0, upper: float = 1):
        Tensor._C.tensor_fill_random_uniform(
            self.c_tensor,
            ctypes.c_float(lower),
            ctypes.c_float(upper)
        )

    def fill_random_normal(self, mean: float = 0, std: float = 1):
        Tensor._C.tensor_fill_random_normal(
            self.c_tensor,
            ctypes.c_float(mean),
            ctypes.c_float(std)
        )

    def unary_minus(self) -> Tensor:
        c_tensor = Tensor._C.tensor_unary_minus(self.c_tensor)
        return Tensor(None, None, c_tensor)

    def add(self, other: Tensor | float) -> Tensor:
        if isinstance(other, Tensor):
            c_tensor = Tensor._C.tensor_add(self.c_tensor, other.c_tensor)
        else:
            c_tensor = Tensor._C.tensor_broadcast_add(
                self.c_tensor, ctypes.c_float(other))
        return Tensor(None, None, c_tensor)

    def subtract(self, other: Tensor | float) -> Tensor:
        if isinstance(other, Tensor):
            c_tensor = Tensor._C.tensor_subtract(
                self.c_tensor, other.c_tensor)
        else:
            c_tensor = Tensor._C.tensor_broadcast_subtract(
                self.c_tensor, ctypes.c_float(other))
        return Tensor(None, None, c_tensor)

    def multiply(self, other: Tensor | float) -> Tensor:
        if isinstance(other, Tensor):
            c_tensor = Tensor._C.tensor_multiply(
                self.c_tensor, other.c_tensor)
        else:
            c_tensor = Tensor._C.tensor_broadcast_multiply(
                self.c_tensor, ctypes.c_float(other))
        return Tensor(None, None, c_tensor)

    def divide(self, other: Tensor | float) -> Tensor:
        if isinstance(other, Tensor):
            c_tensor = Tensor._C.tensor_divide(self.c_tensor, other.c_tensor)
        else:
            c_tensor = Tensor._C.tensor_broadcast_divide(
                self.c_tensor, ctypes.c_float(other))
        return Tensor(None, None, c_tensor)

    def right_divide(self, other: float) -> Tensor:
        c_tensor = Tensor._C.tensor_broadcast_right_divide(
            self.c_tensor, ctypes.c_float(other))
        return Tensor(None, None, c_tensor)

    def matmul(self, other: Tensor) -> Tensor:
        c_tensor = Tensor._C.tensor_matmul(self.c_tensor, other.c_tensor)
        return Tensor(None, None, c_tensor)

    def get(self, *key: tuple[int, ...]) -> float:
        return Tensor._C.tensor_get_item(self.c_tensor, (ctypes.c_uint32 * len(key))(*key))

    def sum(self) -> float:
        return Tensor._C.tensor_sum(self.c_tensor)

    def mean(self) -> float:
        return Tensor._C.tensor_mean(self.c_tensor)

    def __del__(self):
        Tensor._C.tensor_delete(self.c_tensor)

    def __add__(self, other: Tensor | float) -> Tensor:
        return self.add(other)

    def __radd__(self, other: Tensor | float) -> Tensor:
        return self.add(other)

    def __sub__(self, other: Tensor | float) -> Tensor:
        return self.subtract(other)

    def __rsub__(self, other: Tensor | float) -> Tensor:
        return self.unary_minus().add(other)

    def __mul__(self, other: Tensor | float) -> Tensor:
        return self.multiply(other)

    def __rmul__(self, other: Tensor | float) -> Tensor:
        return self.multiply(other)

    def __truediv__(self, other: Tensor | float) -> Tensor:
        return self.divide(other)

    def __rtruediv__(self, other: float) -> Tensor:
        return self.right_divide(other)

    def __matmul__(self, other: Tensor) -> Tensor:
        return self.matmul(other)

    def __neg__(self) -> Tensor:
        return self.unary_minus()
