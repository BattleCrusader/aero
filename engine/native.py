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
            lib.aero_dequant_threaded.argtypes = [
                c_int, POINTER(c_uint8), c_int64, POINTER(c_float), c_int64, c_int,
            ]
            lib.aero_dequant_threaded.restype = c_int
            lib.aero_dequant_ref.argtypes = [
                c_int, POINTER(c_uint8), c_int64, POINTER(c_float), c_int64,
            ]
            lib.aero_dequant_ref.restype = c_int
            lib.aero_dtype_block_bytes.argtypes = [c_int]
            lib.aero_dtype_block_bytes.restype = c_int64
            lib.aero_dtype_block_values.argtypes = [c_int]
            lib.aero_dtype_block_values.restype = c_int64
            lib.aero_hardware_threads.argtypes = []
            lib.aero_hardware_threads.restype = c_int
            _lib = lib
            return _lib
        except (OSError, AttributeError) as e:
            # AttributeError, not just OSError: ctypes raises AttributeError when a
            # dlsym lookup misses, so a library that loads but is missing one symbol
            # used to escape this handler entirely and crash the caller instead of
            # falling back. That turned "stale .dylib" into "engine is broken".
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


# Values per block, mirrored from the kernel so a thread split lands on block
# boundaries without a round trip. Kept as a literal dict so this module imports
# no engine state (it is a leaf binding).
_BLOCK_VALUES = {DTYPE_F32: 1, DTYPE_F16: 1, DTYPE_Q4_K: 256, DTYPE_Q6_K: 256,
                 DTYPE_IQ4_XS: 256, DTYPE_BF16: 1}

_BLOCK_BYTES = {DTYPE_F32: 4, DTYPE_F16: 2, DTYPE_Q4_K: 144, DTYPE_Q6_K: 210,
                DTYPE_IQ4_XS: 136, DTYPE_BF16: 2}

_threads_hint = 0


def hw_threads() -> int:
    """Hardware thread count, cached. 0 when the library is absent."""
    global _threads_hint
    if _threads_hint:
        return _threads_hint
    lib = _load()
    if lib is None:
        return 0
    _threads_hint = int(lib.aero_hardware_threads())
    return _threads_hint


def _raw_nbytes(raw, dtype: int, n_values: int) -> int:
    """Bytes the caller actually passed, so the kernel can bounds-check.

    A memoryview/bytes carries its own length; anything else has to be sized from
    the dtype. Never returns more than the caller has, because over-reporting
    would re-enable the out-of-bounds read this exists to prevent.
    """
    try:
        return len(raw)
    except TypeError:
        pass
    bv = _BLOCK_VALUES.get(dtype, 0)
    if not bv:
        return 0
    bb = _BLOCK_BYTES.get(dtype, 0)
    return (n_values // bv) * bb


def dequant(dtype: int, raw, shape, threads: int = 0):
    """Dequantise a whole tensor with the native kernel.

    Returns a numpy array of `shape`. Raises KernelsUnavailable when the library
    is missing so the caller can fall back to the Python dequantiser rather than
    silently getting a wrong buffer.

    `threads` defaults to 0 meaning "pick one": the thread count is clamped to the
    tensor's block count, because spreading a 512-block tensor over 8 threads and
    one of 2 blocks over 8 threads want different answers, and below ~8 blocks the
    fork/join costs more than the work. Measured on the 280 tensors a gemma-4
    decode loads, this is the difference between ~11.7 s and ~2 s of dequantise.
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

    if threads == 0:
        bv = _BLOCK_VALUES.get(dtype, 0)
        nblocks = (n // bv) if bv else 0
        hw = hw_threads()
        threads = hw if nblocks >= 2 * max(hw, 1) else 1

    # raw_len is passed so the kernel can refuse a short buffer instead of
    # reading past it -- passing a length it never received segfaulted the suite.
    rl = c_int64(_raw_nbytes(raw, dtype, n))
    if threads > 1:
        fn = lib.aero_dequant_threaded
        ok = fn(int(dtype), rawbuf, rl, out.ctypes.data_as(POINTER(c_float)),
                c_int64(n), c_int(int(threads)))
    else:
        ok = lib.aero_dequant_ref(int(dtype), rawbuf, rl,
                                  out.ctypes.data_as(POINTER(c_float)), c_int64(n))
    if not ok:
        raise ValueError(
            f"native dequant refused dtype={dtype} n={n} raw_len={int(rl.value)} "
            f"(buffer too short, or the type/dimension combination is unsupported)")
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
