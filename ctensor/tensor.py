from __future__ import annotations
from typing import Iterator
from collections import defaultdict
import ctypes
import os


class CTensor(ctypes.Structure):
    pass

# must be ordered as the C struct!
CTensor._fields_ = [
    ("data", ctypes.POINTER(ctypes.c_float)),
    ("shape", ctypes.POINTER(ctypes.c_int32)),
    ("stride", ctypes.POINTER(ctypes.c_int32)),
    ("dims", ctypes.c_int32),
    ("size", ctypes.c_int32),
    ("device", ctypes.c_int32),
    ("offset", ctypes.c_int32),
    ("base", ctypes.POINTER(CTensor))
]


class CSlice(ctypes.Structure):
    _fields_ = [
        ("start", ctypes.c_int32),
        ("stop", ctypes.c_int32),
        ("step", ctypes.c_int32),
    ]


def _get_tensorlib_path(libname: str = "libtensor.so") -> str:
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), libname)


def _init_tensor_c_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(_get_tensorlib_path())

    lib.tensor_create.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32
    ]
    lib.tensor_create.restype = ctypes.POINTER(CTensor)
    lib.tensor_create_empty.argtypes = [
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32
    ]
    lib.tensor_create_empty.restype = ctypes.POINTER(CTensor)
    lib.tensor_copy.argtypes = [ctypes.POINTER(CTensor)]
    lib.tensor_copy.restype = ctypes.POINTER(CTensor)
    lib.tensor_delete.argtypes = [ctypes.POINTER(CTensor)]
    lib.tensor_delete.restype = None
    lib.tensor_print_info.argtypes = [ctypes.POINTER(CTensor)]
    lib.tensor_print_info.restype = None

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
    lib.tensor_fill_identity.argtypes = [ctypes.POINTER(CTensor)]
    lib.tensor_fill_identity.restype = None
    lib.tensor_reshape.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32
    ]
    lib.tensor_reshape.restype = ctypes.POINTER(CTensor)
    lib.tensor_get_item.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.c_int32)
    ]
    lib.tensor_get_item.restype = ctypes.c_float
    lib.tensor_get_item_at.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_int32
    ]
    lib.tensor_get_item_at.restype = ctypes.c_float
    lib.tensor_set_item.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_float,
    ]
    lib.tensor_set_item.restype = None
    lib.tensor_get_data_index.restype = ctypes.c_int32
    lib.tensor_get_data_index.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_int32
    ]
    lib.tensor_slice.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CSlice),
    ]
    lib.tensor_slice.restype = ctypes.POINTER(CTensor)
    lib.tensor_sum.argtypes = [ctypes.POINTER(CTensor)]
    lib.tensor_sum.restype = ctypes.c_float
    lib.tensor_mean.argtypes = [ctypes.POINTER(CTensor)]
    lib.tensor_mean.restype = ctypes.c_float
    lib.tensor_min.argtypes = [ctypes.POINTER(CTensor)]
    lib.tensor_min.restype = ctypes.c_float
    lib.tensor_max.argtypes = [ctypes.POINTER(CTensor)]
    lib.tensor_max.restype = ctypes.c_float

    lib.tensor_unary_minus.argtypes = [ctypes.POINTER(CTensor)]
    lib.tensor_unary_minus.restype = ctypes.POINTER(CTensor)
    lib.tensor_transpose.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    lib.tensor_transpose.restype = ctypes.POINTER(CTensor)
    lib.tensor_add_into.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor)
    ]
    lib.tensor_add_into.restype = None
    lib.tensor_subtract_into.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor)
    ]
    lib.tensor_subtract_into.restype = None
    lib.tensor_multiply_into.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor)
    ]
    lib.tensor_multiply_into.restype = None
    lib.tensor_divide_into.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.POINTER(CTensor)
    ]
    lib.tensor_divide_into.restype = None
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

    lib.tensor_broadcast_add_into.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float
    ]
    lib.tensor_broadcast_add_into.restype = None
    lib.tensor_broadcast_subtract_into.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float
    ]
    lib.tensor_broadcast_subtract_into.restype = None
    lib.tensor_broadcast_multiply_into.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float
    ]
    lib.tensor_broadcast_multiply_into.restype = None
    lib.tensor_broadcast_divide_into.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float
    ]
    lib.tensor_broadcast_divide_into.restype = None
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

    lib.tensor_power.argtypes = [
        ctypes.POINTER(CTensor),
        ctypes.c_float
    ]
    lib.tensor_power.restype = ctypes.POINTER(CTensor)
    lib.tensor_exp.argtypes = [ctypes.POINTER(CTensor)]
    lib.tensor_exp.restype = ctypes.POINTER(CTensor)
    lib.tensor_log.argtypes = [ctypes.POINTER(CTensor)]
    lib.tensor_log.restype = ctypes.POINTER(CTensor)
    lib.tensor_log10.argtypes = [ctypes.POINTER(CTensor)]
    lib.tensor_log10.restype = ctypes.POINTER(CTensor)
    lib.tensor_logb.argtypes = [ctypes.POINTER(CTensor), ctypes.c_float]
    lib.tensor_logb.restype = ctypes.POINTER(CTensor)
    lib.tensor_sin.argtypes = [ctypes.POINTER(CTensor)]
    lib.tensor_sin.restype = ctypes.POINTER(CTensor)
    lib.tensor_cos.argtypes = [ctypes.POINTER(CTensor)]
    lib.tensor_cos.restype = ctypes.POINTER(CTensor)
    lib.tensor_tanh.argtypes = [ctypes.POINTER(CTensor)]
    lib.tensor_tanh.restype = ctypes.POINTER(CTensor)
    return lib


def _get_array_and_shape(array: list[float | list]) -> tuple[list[float], tuple[int, ...]]:
    def _flatten_array(array: list[float | list], shape: list[int]) -> list[float]:
        shape.append(len(array))
        if isinstance(array[0], (int, float)):
            return array
        return _flatten_array([x for arr in array for x in arr], shape)

    shape = []
    flattened = _flatten_array(array, shape)
    shape = [
        shape[i] // (1 if i == 0 else shape[i - 1])
        for i in range(len(shape))
    ]
    return flattened, shape


class Tensor:
    _C = _init_tensor_c_lib()

    # keeps mapping of address(tensor) -> count to avoid early dereferencing
    _REGISTRY: dict[int, int] = defaultdict(int)


    def __init__(
        self,
        data: list[float] | None,
        shape: tuple[int, ...] | None,
        c_tensor=None
    ):
        if c_tensor:
            self.c_tensor = c_tensor
            self._increase_ref_count()
            return

        if data is None or shape is None:
            raise ValueError("Must provide data and shape.")

        self.c_tensor = Tensor._C.tensor_create(
            (ctypes.c_float * len(data))(*data),
            (ctypes.c_int32 * len(shape))(*shape),
            ctypes.c_int32(len(shape)),
        )
        self._increase_ref_count()

    @classmethod
    def _empty(cls, shape: tuple[int, ...]) -> Tensor:
        c_tensor = Tensor._C.tensor_create_empty(
            (ctypes.c_int32 * len(shape))(*shape),
            ctypes.c_int32(len(shape)),
        )
        return Tensor(None, None, c_tensor)

    @classmethod
    def array(cls, array: list[float | list]) -> Tensor:
        data, shape = _get_array_and_shape(array)
        return Tensor(data, shape)

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
    def zeros(cls, *shape: int) -> Tensor:
        tensor = Tensor._empty(shape)
        tensor.fill(0)
        return tensor

    @classmethod
    def ones(cls, *shape: int) -> Tensor:
        tensor = Tensor._empty(shape)
        tensor.fill(1)
        return tensor

    @classmethod
    def identity(cls, size: int) -> Tensor:
        tensor = Tensor._empty((size, size))
        tensor.fill_identity()
        return tensor

    @property
    def dims(self) -> int:
        return self.c_tensor.contents.dims

    @property
    def size(self) -> int:
        return self.c_tensor.contents.size

    @property
    def data(self) -> list[float]:
        return [self.at(i) for i in range(self.size)]

    @property
    def indices(self) -> list[int]:
        return [
            Tensor._C.tensor_get_data_index(self.c_tensor, ctypes.c_int32(i))
            for i in range(self.size)
        ]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.c_tensor.contents.shape[i] for i in range(self.dims))

    @property
    def stride(self) -> tuple[int, ...]:
        return tuple(self.c_tensor.contents.stride[i] for i in range(self.dims))

    @property
    def base(self) -> Tensor | None:
        if self.c_tensor.contents.base:
            return Tensor(None, None, self.c_tensor.contents.base)
        return None

    @property
    def offset(self) -> int:
        return self.c_tensor.contents.offset

    @property
    def device(self) -> int:
        return self.c_tensor.contents.device

    @property
    def T(self) -> Tensor:
        return self.transpose()

    @property
    def address(self) -> int:
        return ctypes.addressof(self.c_tensor.contents)

    def copy(self) -> Tensor:
        c_tensor = Tensor._C.tensor_copy(self.c_tensor)
        return Tensor(None, None, c_tensor)

    def to_gpu(self):
        Tensor._C.tensor_cpu_to_gpu(self.c_tensor)

    def to_cpu(self):
        Tensor._C.tensor_gpu_to_cpu(self.c_tensor)

    def reshape(self, *shape: int) -> Tensor:
        result = Tensor._C.tensor_reshape(
            self.c_tensor,
            (ctypes.c_int32 * len(shape))(*shape),
            ctypes.c_int32(len(shape))
        )
        return Tensor(None, None, result)


    def squeeze(self, *dims: int) -> Tensor:
        if any(self.shape[i] != 1 for i in dims):
            raise ValueError("Trying to squeeze dimension that is not 1")

        if len(dims) == 0:
            new_shape = [i for i in self.shape if i != 1]
        else:
            new_shape = [self.shape[i]
                         for i in range(self.dims) if i not in dims]

        if len(new_shape) == 0:
            new_shape = [1]

        return self.reshape(*new_shape)

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

    def fill_identity(self):
        Tensor._C.tensor_fill_identity(self.c_tensor)

    def unary_minus(self) -> Tensor:
        c_tensor = Tensor._C.tensor_unary_minus(self.c_tensor)
        return Tensor(None, None, c_tensor)

    def transpose(self, axis1: int | None = None, axis2: int | None = None) -> Tensor:
        if axis1 is None or axis2 is None:
            axis1 = self.dims - 1
            axis2 = self.dims - 2
        c_tensor = Tensor._C.tensor_transpose(
            self.c_tensor, ctypes.c_int32(axis1), ctypes.c_int32(axis2))
        return Tensor(None, None, c_tensor)

    def add_into(self, other: Tensor | float):
        if isinstance(other, Tensor):
            Tensor._C.tensor_add_into(self.c_tensor, other.c_tensor)
        else:
            Tensor._C.tensor_broadcast_add_into(
                self.c_tensor, ctypes.c_float(other))

    def subtract_into(self, other: Tensor | float):
        if isinstance(other, Tensor):
            Tensor._C.tensor_subtract_into(self.c_tensor, other.c_tensor)
        else:
            Tensor._C.tensor_broadcast_subtract_into(
                self.c_tensor, ctypes.c_float(other))

    def multiply_into(self, other: Tensor | float):
        if isinstance(other, Tensor):
            Tensor._C.tensor_multiply_into(self.c_tensor, other.c_tensor)
        else:
            Tensor._C.tensor_broadcast_multiply_into(
                self.c_tensor, ctypes.c_float(other))

    def divide_into(self, other: Tensor | float):
        if isinstance(other, Tensor):
            Tensor._C.tensor_divide_into(self.c_tensor, other.c_tensor)
        else:
            Tensor._C.tensor_broadcast_divide_into(
                self.c_tensor, ctypes.c_float(other))

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

    def power(self, pow: float) -> Tensor:
        c_tensor = Tensor._C.tensor_power(self.c_tensor, ctypes.c_float(pow))
        return Tensor(None, None, c_tensor)

    def exp(self) -> Tensor:
        c_tensor = Tensor._C.tensor_exp(self.c_tensor)
        return Tensor(None, None, c_tensor)

    def log(self) -> Tensor:
        c_tensor = Tensor._C.tensor_log(self.c_tensor)
        return Tensor(None, None, c_tensor)

    def log10(self) -> Tensor:
        c_tensor = Tensor._C.tensor_log10(self.c_tensor)
        return Tensor(None, None, c_tensor)

    def logb(self, base: float) -> Tensor:
        c_tensor = Tensor._C.tensor_logb(self.c_tensor, ctypes.c_float(base))
        return Tensor(None, None, c_tensor)

    def sin(self) -> Tensor:
        c_tensor = Tensor._C.tensor_sin(self.c_tensor)
        return Tensor(None, None, c_tensor)

    def cos(self) -> Tensor:
        c_tensor = Tensor._C.tensor_cos(self.c_tensor)
        return Tensor(None, None, c_tensor)

    def tanh(self) -> Tensor:
        c_tensor = Tensor._C.tensor_tanh(self.c_tensor)
        return Tensor(None, None, c_tensor)

    def get(self, *key: int) -> float:
        return Tensor._C.tensor_get_item(self.c_tensor, (ctypes.c_int32 * len(key))(*key))

    def at(self, index: int) -> float:
        return Tensor._C.tensor_get_item_at(self.c_tensor, ctypes.c_int32(index))

    def set(self, key: tuple[int, ...], value: float):
        Tensor._C.tensor_set_item(
            self.c_tensor,
            (ctypes.c_int32 * len(key))(*key),
            ctypes.c_float(value)
        )

    def slice(self, *slices: tuple[slice]) -> Tensor:
        c_slices = []
        for i in range(len(self.shape)):
            dim = self.shape[i]
            if i < len(slices):
                s = slices[i]
                r = CSlice(s.start or 0, s.stop or dim, s.step or 1)
            else:
                r = CSlice(0, dim, 1)
            c_slices.append(r)

        c_tensor = Tensor._C.tensor_slice(
            self.c_tensor,
            (CSlice * len(self.shape))(*c_slices)
        )
        return Tensor(None, None, c_tensor)

    def sum(self) -> float:
        return Tensor._C.tensor_sum(self.c_tensor)

    def mean(self) -> float:
        return Tensor._C.tensor_mean(self.c_tensor)

    def min(self) -> float:
        return Tensor._C.tensor_min(self.c_tensor)

    def max(self) -> float:
        return Tensor._C.tensor_max(self.c_tensor)

    def print_info(self):
        Tensor._C.tensor_print_info(self.c_tensor)

    def __del__(self):
        self._decrease_ref_count()
        if self._has_no_ref():
            Tensor._C.tensor_delete(self.c_tensor)

    def __iadd__(self, other: Tensor) -> Tensor:
        self.add_into(other)
        return self

    def __isub__(self, other: Tensor) -> Tensor:
        self.subtract_into(other)
        return self

    def __imul__(self, other: Tensor) -> Tensor:
        self.multiply_into(other)
        return self

    def __itruediv__(self, other: Tensor) -> Tensor:
        self.divide_into(other)
        return self

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

    def __len__(self) -> int:
        return self.shape[0]

    def __iter__(self) -> Iterator[Tensor]:
        for i in range(self.shape[0]):
            yield self[i]

    def __getitem__(self, key: tuple[int | slice, ...] | int | slice) -> float | Tensor:
        if not isinstance(key, tuple):
            key = (key,)

        if isinstance(key[0], int):
            if len(key) == self.dims:
                return self.get(*key)
            elif len(key) == 1:
                idx = key[0]
                tmp = self.slice(slice(idx, idx + 1, 1))
                tmp = tmp.squeeze(0)
                return tmp
            else:
                raise ValueError(
                    "Key must be a tuple of ints the size of dims, or slices")
        elif isinstance(key[0], slice):
            return self.slice(*key)
        else:
            raise ValueError(
                "Key must be a tuple of ints the size of dims, or slices")

    def __setitem__(self, key: tuple[int, ...], value: float):
        self.set(key, value)

    def __str__(self) -> str:
        return str(self.data)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Tensor):
            return self.address == other.address
        return False

    def _increase_ref_count(self):
        self._REGISTRY[self.address] += 1

    def _decrease_ref_count(self):
        self._REGISTRY[self.address] -= 1

    def _has_no_ref(self) -> bool:
        return self._REGISTRY[self.address] == 0
