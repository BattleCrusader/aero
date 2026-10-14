"""ctypes binding for the native fp32 tensor primitives (libaero_tensor).

WHY THIS FILE EXISTS: src/tensor.cpp was written, tested and correct long before
this binding existed -- and completely unreachable from the engine, because the
module exported no C symbols and was compiled only into its own test binary. The
engine therefore kept running these primitives in numpy while the C++ versions sat
unused. A primitive with no ABI is not ported; it is a private detail. This closes
that gap.

Everything here is a thin, checked wrapper: flat row-major fp32 buffers, explicit
lengths, and a 0/1 return that is surfaced as an exception rather than ignored.
The refusing-on-length convention is deliberate -- a wrong buffer length is the
exact bug class that segfaulted this repo once already.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import POINTER, c_float, c_int, c_int64

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

_LIB_CANDIDATES = (
    os.path.join(_ROOT, "build", "libaero_tensor.dylib"),
    os.path.join(_ROOT, "build", "libaero_tensor.so"),
)


class TensorLibUnavailable(RuntimeError):
    """The native tensor library is not built. Callers must fall back, not guess."""


_lib = None
_load_error: str | None = None


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    for path in _LIB_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            lib = ctypes.CDLL(path)
            cp = POINTER(c_float)
            lib.aero_tensor_matvec.argtypes = [cp, cp, cp, c_int64, c_int64, c_int]
            lib.aero_tensor_matvec.restype = c_int
            lib.aero_tensor_matmul.argtypes = [cp, cp, cp, c_int64, c_int64, c_int64,
                                               c_int]
            lib.aero_tensor_matmul.restype = c_int
            for name in ("silu", "gelu", "softmax"):
                f = getattr(lib, "aero_tensor_" + name)
                f.argtypes = [cp, c_int64]
                f.restype = c_int
            lib.aero_tensor_mul.argtypes = [cp, cp, c_int64]
            lib.aero_tensor_mul.restype = c_int
            lib.aero_tensor_swiglu.argtypes = [cp, cp, c_int64]
            lib.aero_tensor_swiglu.restype = c_int
            lib.aero_tensor_rmsnorm.argtypes = [cp, cp, cp, c_int64, c_float]
            lib.aero_tensor_rmsnorm.restype = c_int
            lib.aero_tensor_l2norm.argtypes = [cp, cp, cp, c_int64, c_float, c_int]
            lib.aero_tensor_l2norm.restype = c_int
            lib.aero_tensor_rmsnorm_rows.argtypes = [cp, c_int64, c_int64, c_float]
            lib.aero_tensor_rmsnorm_rows.restype = c_int
            lib.aero_tensor_argmax.argtypes = [cp, c_int64]
            lib.aero_tensor_argmax.restype = c_int
            lib.aero_tensor_threads.argtypes = []
            lib.aero_tensor_threads.restype = c_int
            _lib = lib
            return _lib
        except (OSError, AttributeError) as e:
            # AttributeError, not OSError: ctypes raises it for a missing symbol, so
            # a stale library that loads but lacks a symbol must fall through here
            # rather than escape and crash the caller.
            _load_error = f"{path}: {e}"
    if _load_error is None:
        _load_error = ("libaero_tensor not found; build it with `make tensor-lib` "
                       "(looked in " + ", ".join(_LIB_CANDIDATES) + ")")
    return None


def available() -> bool:
    return _load() is not None


def unavailable_reason() -> str:
    _load()
    return _load_error or ""


def _p(arr):
    """float32 ndarray -> a checked float pointer. Rejects anything else loudly."""
    import numpy as np
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"expected a numpy array, got {type(arr).__name__}")
    if arr.dtype != np.float32:
        raise TypeError(f"expected float32, got {arr.dtype}")
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr.ctypes.data_as(POINTER(c_float)), arr


def _lib_or_raise():
    lib = _load()
    if lib is None:
        raise TensorLibUnavailable(unavailable_reason())
    return lib


def threads() -> int:
    """Default thread count the native layer would pick. 0 when unavailable."""
    lib = _load()
    return int(lib.aero_tensor_threads()) if lib else 0


# ------------------------------------------------------------------ primitives
def matvec(w, x, threads: int = 0):
    """y = W @ x. w is (n_rows, n_cols), x is (n_cols,) -> (n_rows,)."""
    import numpy as np
    lib = _lib_or_raise()
    w = np.ascontiguousarray(w, dtype=np.float32)
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    n_rows, n_cols = w.shape
    if x.size != n_cols:
        raise ValueError(f"x has {x.size} != W cols {n_cols}")
    y = np.empty(n_rows, dtype=np.float32)
    pw, _ = _p(w); px, _ = _p(x); py, _ = _p(y)
    if not lib.aero_tensor_matvec(pw, px, py, n_rows, n_cols, int(threads)):
        raise ValueError("native matvec refused (null or non-positive dims)")
    return y


def matmul(w, X, threads: int = 0):
    """Y = X @ W.T. w is (n_rows, n_cols), X is (n_tokens, n_cols) -> (n_tokens, n_rows)."""
    import numpy as np
    lib = _lib_or_raise()
    w = np.ascontiguousarray(w, dtype=np.float32)
    X = np.atleast_2d(np.ascontiguousarray(X, dtype=np.float32))
    n_rows, n_cols = w.shape
    n_tokens = X.shape[0]
    if X.shape[1] != n_cols:
        raise ValueError(f"X cols {X.shape[1]} != W cols {n_cols}")
    Y = np.empty((n_tokens, n_rows), dtype=np.float32)
    pw, _ = _p(w); pX, _ = _p(X); pY, _ = _p(Y)
    if not lib.aero_tensor_matmul(pw, pX, pY, n_tokens, n_rows, n_cols, int(threads)):
        raise ValueError("native matmul refused (null or non-positive dims)")
    return Y


def _inplace(name, x):
    import numpy as np
    lib = _lib_or_raise()
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    p, _ = _p(x)
    if not getattr(lib, "aero_tensor_" + name)(p, x.size):
        raise ValueError(f"native {name} refused (null or empty)")
    return x


def silu(x):
    """x * sigmoid(x), elementwise, returned as a flat float32 array."""
    return _inplace("silu", x)


def gelu(x):
    """Exact (erf-form) GELU, elementwise. gemma4 uses this, not SiLU."""
    return _inplace("gelu", x)


def softmax(x):
    """Numerically-stable softmax over a flat vector."""
    return _inplace("softmax", x)


def mul(a, b):
    """a[i] *= b[i] (returns the modified a)."""
    import numpy as np
    lib = _lib_or_raise()
    a = np.ascontiguousarray(a, dtype=np.float32).reshape(-1)
    b = np.ascontiguousarray(b, dtype=np.float32).reshape(-1)
    if a.size != b.size:
        raise ValueError(f"mul: {a.size} != {b.size}")
    pa, _ = _p(a); pb, _ = _p(b)
    if not lib.aero_tensor_mul(pa, pb, a.size):
        raise ValueError("native mul refused")
    return a


def swiglu(gate, up):
    """gate = silu(gate) * up, in place on gate (returned)."""
    import numpy as np
    lib = _lib_or_raise()
    gate = np.ascontiguousarray(gate, dtype=np.float32).reshape(-1)
    up = np.ascontiguousarray(up, dtype=np.float32).reshape(-1)
    if gate.size != up.size:
        raise ValueError(f"swiglu: {gate.size} != {up.size}")
    pg, _ = _p(gate); pu, _ = _p(up)
    if not lib.aero_tensor_swiglu(pg, pu, gate.size):
        raise ValueError("native swiglu refused")
    return gate


def rmsnorm(x, weight, eps: float):
    """out = x / sqrt(mean(x^2) + eps) * weight."""
    import numpy as np
    lib = _lib_or_raise()
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    weight = np.ascontiguousarray(weight, dtype=np.float32).reshape(-1)
    if x.size != weight.size:
        raise ValueError(f"rmsnorm: x {x.size} != weight {weight.size}")
    out = np.empty_like(x)
    px, _ = _p(x); pw, _ = _p(weight); po, _ = _p(out)
    if not lib.aero_tensor_rmsnorm(px, pw, po, x.size, float(eps)):
        raise ValueError("native rmsnorm refused")
    return out


def l2norm(x, weight=None, eps: float = 1e-6):
    """out = x / sqrt(sum(x^2) + eps) * weight, or weightless when weight is None."""
    import numpy as np
    lib = _lib_or_raise()
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    out = np.empty_like(x)
    px, _ = _p(x); po, _ = _p(out)
    if weight is None:
        pw = POINTER(c_float)()               # NULL; kernel must not touch it
        weighted = 0
    else:
        weight = np.ascontiguousarray(weight, dtype=np.float32).reshape(-1)
        if weight.size != x.size:
            raise ValueError(f"l2norm: x {x.size} != weight {weight.size}")
        pw, _ = _p(weight)
        weighted = 1
    if not lib.aero_tensor_l2norm(px, pw, po, x.size, float(eps), weighted):
        raise ValueError("native l2norm refused")
    return out


def rmsnorm_rows(t, eps: float):
    """Per-row weightless RMS over a 2-D block, in place (gemma4's V norm)."""
    import numpy as np
    lib = _lib_or_raise()
    t = np.ascontiguousarray(t, dtype=np.float32)
    if t.ndim != 2:
        raise ValueError("rmsnorm_rows expects a 2-D array")
    rows, cols = t.shape
    p, _ = _p(t)
    if not lib.aero_tensor_rmsnorm_rows(p, rows, cols, float(eps)):
        raise ValueError("native rmsnorm_rows refused")
    return t


def argmax(v) -> int:
    import numpy as np
    lib = _lib_or_raise()
    v = np.ascontiguousarray(v, dtype=np.float32).reshape(-1)
    p, _ = _p(v)
    r = lib.aero_tensor_argmax(p, v.size)
    if r < 0:
        raise ValueError("native argmax refused (null or empty)")
    return int(r)
