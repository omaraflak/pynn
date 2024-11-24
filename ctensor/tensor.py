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

    lib.print_tensor_info.argtypes = [ctypes.POINTER(CTensor)]
    lib.print_tensor_info.restype = None
    lib.create_tensor.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32
    ]
    lib.create_tensor.restype = ctypes.POINTER(CTensor)
    lib.copy_tensor.argtypes = [ctypes.POINTER(CTensor)]
    lib.copy_tensor.restype = ctypes.POINTER(CTensor)
    lib.delete_tensor.argtypes = [ctypes.POINTER(CTensor)]
    lib.delete_tensor.restype = None

    lib.tensor_cpu_to_gpu.argtypes = [ctypes.POINTER(CTensor)]
    lib.tensor_cpu_to_gpu.restype = None
    lib.tensor_gpu_to_cpu.argtypes = [ctypes.POINTER(CTensor)]
    lib.tensor_gpu_to_cpu.restype = None

    lib.fill_tensor.argtypes = [ctypes.POINTER(CTensor), ctypes.c_float]
    lib.fill_tensor.restype = None
    lib.reshape_tensor.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint32
    ]
    lib.reshape_tensor.restype = None
    lib.get_tensor_item.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.c_uint32)
    ]
    lib.get_tensor_item.restype = ctypes.c_float

    lib.unary_minus_tensor.argtypes = [ctypes.POINTER(CTensor)]
    lib.unary_minus_tensor.restype = ctypes.POINTER(CTensor)
    lib.add_tensors.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor)
    ]
    lib.add_tensors.restype = ctypes.POINTER(CTensor)
    lib.subtract_tensors.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor)
    ]
    lib.subtract_tensors.restype = ctypes.POINTER(CTensor)
    lib.multiply_tensors.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor)
    ]
    lib.multiply_tensors.restype = ctypes.POINTER(CTensor)
    lib.divide_tensors.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor)
    ]
    lib.divide_tensors.restype = ctypes.POINTER(CTensor)
    lib.matmul_tensors.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor)
    ]
    lib.matmul_tensors.restype = ctypes.POINTER(CTensor)

    lib.broadcast_add_tensor.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float
    ]
    lib.broadcast_add_tensor.restype = ctypes.POINTER(CTensor)
    lib.broadcast_subtract_tensor.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float
    ]
    lib.broadcast_subtract_tensor.restype = ctypes.POINTER(CTensor)
    lib.broadcast_multiply_tensor.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float
    ]
    lib.broadcast_multiply_tensor.restype = ctypes.POINTER(CTensor)
    lib.broadcast_divide_tensor.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float
    ]
    lib.broadcast_divide_tensor.restype = ctypes.POINTER(CTensor)
    lib.broadcast_right_divide_tensor.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float
    ]
    lib.broadcast_right_divide_tensor.restype = ctypes.POINTER(CTensor)
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

        self.c_tensor = Tensor._C.create_tensor(
            (ctypes.c_float * len(data))(*data),
            (ctypes.c_uint32 * len(data))(*shape),
            ctypes.c_uint32(len(shape)),
        )

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
        c_tensor = Tensor._C.copy_tensor(self.c_tensor)
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

    def fill(self, value: float):
        Tensor._C.fill_tensor(self.c_tensor, ctypes.c_float(value))

    def unary_minus(self) -> Tensor:
        c_tensor = Tensor._C.unary_minus_tensor(self.c_tensor)
        return Tensor(None, None, c_tensor)

    def add(self, other: Tensor | float) -> Tensor:
        if isinstance(other, Tensor):
            c_tensor = Tensor._C.add_tensors(self.c_tensor, other.c_tensor)
        else:
            c_tensor = Tensor._C.broadcast_add_tensor(
                self.c_tensor, ctypes.c_float(other))
        return Tensor(None, None, c_tensor)

    def subtract(self, other: Tensor | float) -> Tensor:
        if isinstance(other, Tensor):
            c_tensor = Tensor._C.subtract_tensors(
                self.c_tensor, other.c_tensor)
        else:
            c_tensor = Tensor._C.broadcast_subtract_tensor(
                self.c_tensor, ctypes.c_float(other))
        return Tensor(None, None, c_tensor)

    def multiply(self, other: Tensor | float) -> Tensor:
        if isinstance(other, Tensor):
            c_tensor = Tensor._C.multiply_tensors(
                self.c_tensor, other.c_tensor)
        else:
            c_tensor = Tensor._C.broadcast_multiply_tensor(
                self.c_tensor, ctypes.c_float(other))
        return Tensor(None, None, c_tensor)

    def divide(self, other: Tensor | float) -> Tensor:
        if isinstance(other, Tensor):
            c_tensor = Tensor._C.divide_tensors(self.c_tensor, other.c_tensor)
        else:
            c_tensor = Tensor._C.broadcast_divide_tensor(
                self.c_tensor, ctypes.c_float(other))
        return Tensor(None, None, c_tensor)

    def right_divide(self, other: float) -> Tensor:
        c_tensor = Tensor._C.broadcast_right_divide_tensor(
            self.c_tensor, ctypes.c_float(other))
        return Tensor(None, None, c_tensor)

    def matmul(self, other: Tensor) -> Tensor:
        c_tensor = Tensor._C.matmul_tensors(self.c_tensor, other.c_tensor)
        return Tensor(None, None, c_tensor)

    def get(self, *key: tuple[int, ...]) -> float:
        return Tensor._C.get_tensor_item(self.c_tensor, (ctypes.c_uint32 * len(key))(*key))

    def print_info(self):
        Tensor._C.print_tensor_info(self.c_tensor)

    def __del__(self):
        Tensor._C.delete_tensor(self.c_tensor)

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
