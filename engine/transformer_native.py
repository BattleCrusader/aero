"""ctypes binding for the native transformer primitives (libaero_transformer).

The C++ port of engine/transformer.py's fp32 core: RoPE tables/application,
grouped-query attention, and the norm/activation wrappers. Same pattern as
engine/native.py -- declare argtypes/restype in _load(), and treat a missing
SYMBOL (ctypes raises AttributeError on a missed dlsym) the same as a missing
library, so a stale .dylib makes callers fall back instead of crashing.

Nothing here owns KV state: k/v are passed as plain numpy buffers and the cache
stays in Python (transformer.KVCache), exactly as before.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import POINTER, c_float, c_int, c_int32, c_int64

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# RoPE pair conventions, mirroring transformer.ROPE_MODE and aero::RopeMode.
ROPE_NEOX = 0
ROPE_INTERLEAVED = 1

_LIB_CANDIDATES = (
    os.path.join(_ROOT, "build", "libaero_transformer.dylib"),
    os.path.join(_ROOT, "build", "libaero_transformer.so"),
)


class TransformerUnavailable(RuntimeError):
    """The native transformer library is not built. Callers must fall back."""


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
            lib.aero_rope_tables.argtypes = [
                c_int64, POINTER(c_int32), c_int64, c_float, c_int64,
                POINTER(c_float), c_int64, POINTER(c_float), POINTER(c_float),
            ]
            lib.aero_rope_tables.restype = c_int64
            lib.aero_apply_rope.argtypes = [
                POINTER(c_float), c_int64, c_int64, c_int64,
                POINTER(c_float), POINTER(c_float), POINTER(c_int32), c_int,
            ]
            lib.aero_apply_rope.restype = None
            lib.aero_rmsnorm_rows.argtypes = [
                POINTER(c_float), POINTER(c_float), POINTER(c_float),
                c_int64, c_int64, c_float,
            ]
            lib.aero_rmsnorm_rows.restype = None
            lib.aero_l2norm_rows.argtypes = lib.aero_rmsnorm_rows.argtypes
            lib.aero_l2norm_rows.restype = None
            lib.aero_gelu.argtypes = [POINTER(c_float), c_int64]
            lib.aero_gelu.restype = None
            lib.aero_silu.argtypes = [POINTER(c_float), c_int64]
            lib.aero_silu.restype = None
            lib.aero_swiglu.argtypes = [POINTER(c_float), POINTER(c_float), c_int64]
            lib.aero_swiglu.restype = None
            lib.aero_softmax.argtypes = [POINTER(c_float), c_int64]
            lib.aero_softmax.restype = None
            lib.aero_softmax_rows.argtypes = [POINTER(c_float), c_int64, c_int64]
            lib.aero_softmax_rows.restype = None
            lib.aero_attention_tiled.argtypes = [
                POINTER(c_float), POINTER(c_float), POINTER(c_float),
                POINTER(c_float), c_int64, c_int64, c_int64, c_int64, c_int64,
                c_float, POINTER(c_int32), POINTER(c_int32), c_int64, c_int,
                c_int64,
            ]
            lib.aero_attention_tiled.restype = c_int
            lib.aero_attention.argtypes = [
                POINTER(c_float), POINTER(c_float), POINTER(c_float),
                POINTER(c_float), c_int64, c_int64, c_int64, c_int64, c_float,
                c_int64, c_int,
            ]
            lib.aero_attention.restype = c_int
            _lib = lib
            return _lib
        except (OSError, AttributeError) as e:
            # AttributeError, not just OSError: ctypes raises it when a dlsym
            # lookup misses, so a library that loads but is missing one symbol
            # must fall back rather than crash the caller (the lesson from
            # engine/native.py: a stale .dylib turned into "engine is broken").
            _load_error = f"{path}: {e}"
    if _load_error is None:
        _load_error = ("libaero_transformer not found; build it with "
                       "`make transformer` (looked in " +
                       ", ".join(_LIB_CANDIDATES) + ")")
    return None


def available() -> bool:
    return _load() is not None


def unavailable_reason() -> str:
    _load()
    return _load_error or ""


def _f32p(a):
    """Pointer to a contiguous float32 copy, or NULL for None."""
    if a is None:
        return ctypes.cast(None, POINTER(c_float))
    return np.ascontiguousarray(a, dtype=np.float32).ctypes.data_as(POINTER(c_float))


def _i32p(a):
    if a is None:
        return ctypes.cast(None, POINTER(c_int32))
    return np.ascontiguousarray(a, dtype=np.int32).ctypes.data_as(POINTER(c_int32))


def _require():
    lib = _load()
    if lib is None:
        raise TransformerUnavailable(unavailable_reason())
    return lib


# ------------------------------------------------------------------------ RoPE
def rope_tables(head_dim: int, positions, base: float, n_dims: int | None = None,
                freq_factors=None):
    """cos/sin tables, shape (len(positions), n_rot//2). Mirrors
    transformer.rope_tables. `n_dims` selects partial rotary; `freq_factors`
    is proportional RoPE (the angle is divided by the factor)."""
    lib = _require()
    pos = np.ascontiguousarray(positions, dtype=np.int32).reshape(-1)
    n_pos = int(pos.size)
    nd = -1 if n_dims is None else int(n_dims)
    n = head_dim if n_dims is None else min(int(n_dims), head_dim)
    n -= n % 2
    half = n // 2
    if freq_factors is None:
        ffa, n_ff = None, 0
    else:
        ffa = np.ascontiguousarray(freq_factors, dtype=np.float32).reshape(-1)
        n_ff = int(ffa.size)
    cos = np.empty(n_pos * half, dtype=np.float32)
    sin = np.empty(n_pos * half, dtype=np.float32)
    n_rot = int(lib.aero_rope_tables(
        c_int64(head_dim), _i32p(pos), c_int64(n_pos), c_float(base), c_int64(nd),
        _f32p(ffa), c_int64(n_ff),
        cos.ctypes.data_as(POINTER(c_float)), sin.ctypes.data_as(POINTER(c_float))))
    if n_rot < 0:
        raise ValueError(
            f"native rope_tables refused head_dim={head_dim} n_dims={nd} "
            f"n_ff={n_ff} (freq_factors too short?)")
    h = int(n_rot) // 2
    return cos[: n_pos * h].reshape(n_pos, h), sin[: n_pos * h].reshape(n_pos, h)


def apply_rope(x, cos, sin, positions=None, mode: int = ROPE_NEOX):
    """Rotate the first 2*cos.shape[1] channels of x's last axis. x has shape
    (..., head_dim); returns a rotated copy (the C side rotates in place)."""
    lib = _require()
    xa = np.ascontiguousarray(x, dtype=np.float32)
    head_dim = int(xa.shape[-1])
    n_rows = int(xa.size // head_dim)
    flat = np.array(xa.reshape(-1), dtype=np.float32, copy=True)
    n_rot = 2 * int(cos.shape[1])
    lib.aero_apply_rope(
        flat.ctypes.data_as(POINTER(c_float)), c_int64(n_rows), c_int64(head_dim),
        c_int64(n_rot), _f32p(cos), _f32p(sin), _i32p(positions), c_int(int(mode)))
    return flat.reshape(xa.shape)


# --------------------------------------------------------------- norm/activations
def rmsnorm_rows(x, weight=None, eps: float = 1e-6):
    """out[r] = x[r]/sqrt(mean(x[r]^2)+eps)*weight, per row over the last axis."""
    lib = _require()
    xa = np.ascontiguousarray(x, dtype=np.float32)
    n_rows = int(xa.shape[0]) if xa.ndim > 1 else 1
    n_cols = int(xa.shape[-1])
    out = np.empty_like(xa)
    lib.aero_rmsnorm_rows(xa.ctypes.data_as(POINTER(c_float)), _f32p(weight),
                          out.ctypes.data_as(POINTER(c_float)), c_int64(n_rows),
                          c_int64(n_cols), c_float(eps))
    return out


def l2norm_rows(x, weight=None, eps: float = 1e-6):
    """As rmsnorm_rows but the divisor is sqrt(sum(x^2)+eps) (L2, not RMS)."""
    lib = _require()
    xa = np.ascontiguousarray(x, dtype=np.float32)
    n_rows = int(xa.shape[0]) if xa.ndim > 1 else 1
    n_cols = int(xa.shape[-1])
    out = np.empty_like(xa)
    lib.aero_l2norm_rows(xa.ctypes.data_as(POINTER(c_float)), _f32p(weight),
                         out.ctypes.data_as(POINTER(c_float)), c_int64(n_rows),
                         c_int64(n_cols), c_float(eps))
    return out


def _inplace1(x, fn):
    xa = np.ascontiguousarray(x, dtype=np.float32)
    out = np.array(xa, dtype=np.float32, copy=True)
    fn(out.ctypes.data_as(POINTER(c_float)), c_int64(out.size))
    return out


def gelu(x):
    return _inplace1(x, _require().aero_gelu)


def silu(x):
    return _inplace1(x, _require().aero_silu)


def swiglu(gate, up):
    lib = _require()
    g = np.array(np.ascontiguousarray(gate, dtype=np.float32), dtype=np.float32, copy=True)
    lib.aero_swiglu(g.ctypes.data_as(POINTER(c_float)), _f32p(up), c_int64(g.size))
    return g


def softmax(x):
    """softmax over the last axis (per row for a 2-D input)."""
    lib = _require()
    xa = np.ascontiguousarray(x, dtype=np.float32)
    out = np.array(xa, dtype=np.float32, copy=True)
    if out.ndim <= 1:
        lib.aero_softmax(out.ctypes.data_as(POINTER(c_float)), c_int64(out.size))
    else:
        n_rows = int(out.size // out.shape[-1])
        lib.aero_softmax_rows(out.ctypes.data_as(POINTER(c_float)), c_int64(n_rows),
                              c_int64(out.shape[-1]))
    return out


# ------------------------------------------------------------------- attention
def attention_tiled(q, k, v, scale, qpos=None, kpos=None, block=1024, window=None):
    """GQA attention with an online softmax. q (T,H,D), k (S,HK,D), v (S,HK,DV)
    -> (T,H,DV). Mirrors transformer.attention_tiled."""
    lib = _require()
    qa = np.ascontiguousarray(q, dtype=np.float32)
    ka = np.ascontiguousarray(k, dtype=np.float32)
    va = np.ascontiguousarray(v, dtype=np.float32)
    T, H, D = qa.shape
    S, HK, DV = ka.shape[0], ka.shape[1], va.shape[2]
    out = np.zeros((T, H, DV), dtype=np.float32)
    win = 0 if window is None else int(window)
    qp = None if qpos is None else np.ascontiguousarray(qpos, dtype=np.int32).reshape(-1)
    kp = None if kpos is None else np.ascontiguousarray(kpos, dtype=np.int32).reshape(-1)
    ok = lib.aero_attention_tiled(
        qa.ctypes.data_as(POINTER(c_float)), ka.ctypes.data_as(POINTER(c_float)),
        va.ctypes.data_as(POINTER(c_float)), out.ctypes.data_as(POINTER(c_float)),
        c_int64(T), c_int64(H), c_int64(HK), c_int64(D), c_int64(DV),
        c_float(scale), _i32p(qp), _i32p(kp), c_int64(S), c_int(win),
        c_int64(int(block)))
    if not ok:
        raise ValueError(
            f"native attention refused H={H} HK={HK} D={D} DV={DV} S={S}")
    return out


def attention(q, k, v, scale, window=None):
    """Single-token decode: one query at position S-1 vs the whole cache."""
    lib = _require()
    qa = np.ascontiguousarray(q, dtype=np.float32)
    ka = np.ascontiguousarray(k, dtype=np.float32)
    va = np.ascontiguousarray(v, dtype=np.float32)
    H, D = qa.shape
    S, HK, DV = ka.shape[0], ka.shape[1], va.shape[2]
    out = np.zeros((H, DV), dtype=np.float32)
    win = 0 if window is None else int(window)
    ok = lib.aero_attention(
        qa.ctypes.data_as(POINTER(c_float)), ka.ctypes.data_as(POINTER(c_float)),
        va.ctypes.data_as(POINTER(c_float)), out.ctypes.data_as(POINTER(c_float)),
        c_int64(H), c_int64(HK), c_int64(D), c_int64(DV), c_float(scale),
        c_int64(S), c_int(win))
    if not ok:
        raise ValueError(f"native attention refused H={H} HK={HK} D={D} DV={DV} S={S}")
    return out
