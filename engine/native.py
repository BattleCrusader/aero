"""ctypes binding for the native Aero kernels.

Why a plain C ABI and ctypes rather than a CPython extension: this module is a
thin shim over one hot loop, and ctypes adds no build step, no ABI coupling to
the interpreter version, and no refcounting bugs to get wrong. The measured
overhead is one Python call per MATMUL (97 per token), not per block.

The kernels take the raw quantized bytes, so the fp32 weight is never built --
which is the entire point (a 9216x2560 Q4_K matrix is 13.3 MB on disk and 94 MB
as fp32).
"""

from __future__ import annotations

import ctypes
import os
from ctypes import POINTER, c_float, c_int, c_int64, c_uint8

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# tensor type ids, mirroring gguf.T_* and the C++ DType enum
DTYPE_F32 = 0
DTYPE_F16 = 1
DTYPE_Q4_K = 12
DTYPE_Q6_K = 14
DTYPE_IQ4_XS = 23
DTYPE_BF16 = 30

# types the native path can compute; anything else must fall back
NATIVE_TYPES = frozenset({DTYPE_Q4_K, DTYPE_Q6_K, DTYPE_IQ4_XS})

_LIB_CANDIDATES = (
    os.path.join(_ROOT, "build", "libaero_kernels.dylib"),
    os.path.join(_ROOT, "build", "libaero_kernels.so"),
)


class KernelsUnavailable(RuntimeError):
    """The native library is not built. Callers must fall back, not guess."""


_lib = None
_load_error: str | None = None


def _load():
    """Load once; remember the failure so callers get a clear reason."""
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    for path in _LIB_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            lib = ctypes.CDLL(path)
            lib.aero_matvec.argtypes = [
                c_int, POINTER(c_uint8), POINTER(c_float), POINTER(c_float),
                c_int64, c_int64, c_int,
            ]
            lib.aero_matvec.restype = c_int
            lib.aero_matmul.argtypes = [
                c_int, POINTER(c_uint8), POINTER(c_float), POINTER(c_float),
                c_int64, c_int64, c_int64, c_int,
            ]
            lib.aero_matmul.restype = c_int
            lib.aero_dtype_block_bytes.argtypes = [c_int]
            lib.aero_dtype_block_bytes.restype = c_int64
            lib.aero_dtype_block_values.argtypes = [c_int]
            lib.aero_dtype_block_values.restype = c_int64
            lib.aero_hardware_threads.argtypes = []
            lib.aero_hardware_threads.restype = c_int
            _lib = lib
            return _lib
        except OSError as e:                      # pragma: no cover
            _load_error = f"{path}: {e}"
    if _load_error is None:
        _load_error = ("libaero_kernels not found; build it with "
                       "`make kernels` (looked in " +
                       ", ".join(_LIB_CANDIDATES) + ")")
    return None


def available() -> bool:
    return _load() is not None


def unavailable_reason() -> str:
    _load()
    return _load_error or ""


def block_bytes(dtype: int) -> int:
    lib = _load()
    if lib is None:
        raise KernelsUnavailable(unavailable_reason())
    return int(lib.aero_dtype_block_bytes(int(dtype)))


def block_values(dtype: int) -> int:
    lib = _load()
    if lib is None:
        raise KernelsUnavailable(unavailable_reason())
    return int(lib.aero_dtype_block_values(int(dtype)))


def hardware_threads() -> int:
    lib = _load()
    if lib is None:
        return 1
    return int(lib.aero_hardware_threads())


def _u8(arr):
    """uint8 view of a bytes-like object, with no copy."""
    if isinstance(arr, (bytes, bytearray, memoryview)):
        buf = (c_uint8 * len(arr)).from_buffer_copy(arr) if isinstance(arr, bytes) \
            else (c_uint8 * len(arr)).from_buffer(arr)
        return buf
    raise TypeError(f"expected bytes-like, got {type(arr).__name__}")


def dequant(dtype: int, raw, shape):
    """Dequantise a whole tensor with the native kernel.

    Returns a numpy array of `shape`. Raises KernelsUnavailable when the library
    is missing so the caller can fall back to the Python dequantiser rather than
    silently getting a wrong buffer.
    """
    import numpy as np

    lib = _load()
    if lib is None:
        raise KernelsUnavailable(unavailable_reason())
    if dtype not in NATIVE_TYPES:
        raise ValueError(f"dtype {dtype} has no native dequantiser")

    n = 1
    for d in shape:
        n *= int(d)
    out = np.empty(n, dtype=np.float32)
    rawbuf = _u8(raw)
    ok = lib.aero_dequant_ref(int(dtype), rawbuf,
                              out.ctypes.data_as(POINTER(c_float)), c_int64(n))
    if not ok:
        raise ValueError(f"native dequant refused dtype={dtype} n={n}")
    return out.reshape(shape)


def matvec(dtype: int, w_bytes, x, threads: int = 1):
    """y = W @ x. `w_bytes` is the raw quantized tensor, `x` a float32 array."""
    import numpy as np

    lib = _load()
    if lib is None:
        raise KernelsUnavailable(unavailable_reason())
    if dtype not in NATIVE_TYPES:
        raise ValueError(f"dtype {dtype} has no native kernel")

    wb = _u8(w_bytes)
    xv = np.ascontiguousarray(x, dtype=np.float32)
    n_cols = xv.size
    bv = block_values(dtype)
    n_rows = (len(w_bytes) // block_bytes(dtype)) * bv // n_cols

    y = np.zeros(n_rows, dtype=np.float32)
    ok = lib.aero_matvec(int(dtype), wb, xv.ctypes.data_as(POINTER(c_float)),
                    y.ctypes.data_as(POINTER(c_float)),
                    c_int64(n_rows), c_int64(n_cols), c_int(threads))
    if not ok:
        raise ValueError(
            f"native matvec refused dtype={dtype} n_rows={n_rows} n_cols={n_cols}")
    return y


def matmul(dtype: int, w_bytes, X, threads: int = 1):
    """Y = X @ W^T. X is (T, n_cols) -> Y is (T, n_rows)."""
    import numpy as np

    lib = _load()
    if lib is None:
        raise KernelsUnavailable(unavailable_reason())
    if dtype not in NATIVE_TYPES:
        raise ValueError(f"dtype {dtype} has no native kernel")

    wb = _u8(w_bytes)
    Xv = np.ascontiguousarray(X, dtype=np.float32)
    if Xv.ndim == 1:
        Xv = Xv[None, :]
    T, n_cols = Xv.shape
    bv = block_values(dtype)
    n_rows = (len(w_bytes) // block_bytes(dtype)) * bv // n_cols

    Y = np.zeros((T, n_rows), dtype=np.float32)
    ok = lib.aero_matmul(int(dtype), wb, Xv.ctypes.data_as(POINTER(c_float)),
                    Y.ctypes.data_as(POINTER(c_float)),
                    c_int64(T), c_int64(n_rows), c_int64(n_cols), c_int(threads))
    if not ok:
        raise ValueError(
            f"native matmul refused dtype={dtype} T={T} n_rows={n_rows} "
            f"n_cols={n_cols}")
    return Y


# --------------------------------------------------------------- memory guard
# Binding for aero::MemoryGuard, the C++ process-wide ceiling. Enforcement is a
# monitor thread inside the library that hard-exits on breach, so the ceiling
# cannot be bypassed from Python.
_GLIB_CANDIDATES = (
    os.path.join(_ROOT, "build", "libaero_guard.dylib"),
    os.path.join(_ROOT, "build", "libaero_guard.so"),
)


class MemoryGuard:
    """Process-wide memory ceiling, backed by the C++ guard."""

    _inst = None

    def __init__(self, lib) -> None:
        self._lib = lib

    @classmethod
    def instance(cls) -> "MemoryGuard":
        if cls._inst is not None:
            return cls._inst
        for path in _GLIB_CANDIDATES:
            if not os.path.exists(path):
                continue
            lib = ctypes.CDLL(path)
            lib.aero_guard_set_budget.argtypes = [ctypes.c_size_t]
            lib.aero_guard_budget.restype = ctypes.c_size_t
            lib.aero_guard_start_monitor.argtypes = [ctypes.c_uint]
            lib.aero_guard_exceeded.restype = c_int
            lib.aero_guard_own_footprint.restype = ctypes.c_size_t
            lib.aero_guard_peak_footprint.restype = ctypes.c_size_t
            lib.aero_guard_system_available.restype = ctypes.c_size_t
            lib.aero_guard_preflight.argtypes = [ctypes.c_size_t]
            lib.aero_guard_preflight.restype = c_int
            cls._inst = cls(lib)
            return cls._inst
        raise KernelsUnavailable(
            "libaero_guard not found; build it with `make guard` (looked in " +
            ", ".join(_GLIB_CANDIDATES) + ")")

    def set_budget(self, bytes_: int) -> None:
        self._lib.aero_guard_set_budget(ctypes.c_size_t(int(bytes_)))

    def budget(self) -> int:
        return int(self._lib.aero_guard_budget())

    def start_monitor(self, interval_ms: int = 50) -> None:
        self._lib.aero_guard_start_monitor(ctypes.c_uint(interval_ms))

    def stop_monitor(self) -> None:
        self._lib.aero_guard_stop_monitor()

    def exceeded(self) -> bool:
        return bool(self._lib.aero_guard_exceeded())

    def own_footprint(self) -> int:
        return int(self._lib.aero_guard_own_footprint())

    def peak_footprint(self) -> int:
        return int(self._lib.aero_guard_peak_footprint())

    def system_available(self) -> int:
        return int(self._lib.aero_guard_system_available())

    def preflight(self, headroom: int = 0) -> bool:
        return bool(self._lib.aero_guard_preflight(ctypes.c_size_t(headroom)))
