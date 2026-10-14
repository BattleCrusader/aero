"""ctypes binding for the native SSM (gated delta net) and MoE kernels.

The numerically-heavy inner loops of engine/ssm.py and engine/moe.py live in
build/libaero_ssm.dylib (src/ssm.cpp, src/moe.cpp). This module is the thin shim
that hands numpy buffers to them. Weight loading stays in Python — that is I/O,
not compute.

Same shape as engine/native.py: a plain C ABI, no build step, no interpreter ABI
coupling. `_load()` catches (OSError, AttributeError) because ctypes raises
AttributeError on a missing symbol, so a stale library that loads but lacks one
exported function falls back instead of crashing the caller.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import (CFUNCTYPE, POINTER, c_float, c_int, c_int64, c_void_p)

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

_LIB_CANDIDATES = (
    os.path.join(_ROOT, "build", "libaero_ssm.dylib"),
    os.path.join(_ROOT, "build", "libaero_ssm.so"),
)

# the expert-row reader the MoE block calls back into Python for
AERO_MOE_FETCH = CFUNCTYPE(c_int, c_void_p, c_int, c_int, POINTER(c_float))

_lib = None
_load_error: str | None = None


class SsmUnavailable(RuntimeError):
    """The native library is not built. Callers must fall back, not guess."""


def _f(buf):
    return buf.ctypes.data_as(POINTER(c_float))


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    for path in _LIB_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            lib = ctypes.CDLL(path)
            p = POINTER(c_float)
            lib.aero_ssm_softplus.argtypes = [c_float]
            lib.aero_ssm_softplus.restype = c_float

            lib.aero_ssm_delta_step.argtypes = [
                p, p, p, p, p, p, p, p, c_int, c_int, c_float, c_float, p,
            ]
            lib.aero_ssm_delta_step.restype = None

            lib.aero_ssm_scan.argtypes = [
                p, p, p, p, p, p, p, p, c_int, c_int, c_int, c_int,
                c_float, c_float, p,
            ]
            lib.aero_ssm_scan.restype = None

            lib.aero_ssm_conv.argtypes = [
                p, p, p, POINTER(c_int), p, c_int, c_int, c_int,
            ]
            lib.aero_ssm_conv.restype = None

            lib.aero_ssm_forward.argtypes = [
                p, p, p, p, p, p, p, p, p, p,          # x + 9 weight buffers
                c_int, c_int, c_int, c_int, c_int, c_int, c_int,   # 7 dims
                c_float, c_float,                      # eps, norm_eps
                p, POINTER(c_int), p, p,               # tail, tail_len, state, out
            ]
            lib.aero_ssm_forward.restype = c_int

            lib.aero_moe_route.argtypes = [
                p, c_int, c_int, c_int, POINTER(c_int64), p,
            ]
            lib.aero_moe_route.restype = c_int

            lib.aero_moe_expert_accum.argtypes = [
                p, c_int, p, p, p, c_int, c_float, p,
            ]
            lib.aero_moe_expert_accum.restype = None

            lib.aero_moe_shared_accum.argtypes = [
                p, c_int, c_int, p, p, p, p, c_int, p,
            ]
            lib.aero_moe_shared_accum.restype = None

            lib.aero_moe_ffn_block.argtypes = [
                p, p, c_int, c_int, c_int, c_int, c_int, c_int, c_int,
                p, p, p, p, AERO_MOE_FETCH, c_void_p, p,
            ]
            lib.aero_moe_ffn_block.restype = c_int
            _lib = lib
            return _lib
        except (OSError, AttributeError) as e:
            _load_error = f"{path}: {e}"
    if _load_error is None:
        _load_error = ("libaero_ssm not found; build it with `make ssm` or "
                       "`make all` (looked in " + ", ".join(_LIB_CANDIDATES) + ")")
    return None


def available() -> bool:
    return _load() is not None


def unavailable_reason() -> str:
    _load()
    return _load_error or ""


def _f32(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float32)


# --------------------------------------------------------------------- SSM

def softplus(x) -> np.ndarray:
    lib = _load()
    if lib is None:
        raise SsmUnavailable(unavailable_reason())
    xf = np.asarray(x, dtype=np.float32)
    out = np.empty_like(xf)
    flat = xf.reshape(-1)
    of = out.reshape(-1)
    fn = lib.aero_ssm_softplus
    for i in range(flat.size):
        of[i] = fn(float(flat[i]))
    return out


def delta_step(q, k, v, alpha, beta, a_log, dt_bias, state, scale, eps=1e-6):
    """One step of the gated delta rule. Returns (out, new_state).

    Mirrors engine/ssm.delta_rule_step: q/k are (n_heads, d) ALREADY tiled, and
    the passed `state` is NOT mutated (a copy is stepped and returned).
    """
    lib = _load()
    if lib is None:
        raise SsmUnavailable(unavailable_reason())
    q, k, v = _f32(q), _f32(k), _f32(v)
    alpha, beta = _f32(alpha), _f32(beta)
    a_log, dt_bias = _f32(a_log), _f32(dt_bias)
    n_heads, d = int(state.shape[0]), int(state.shape[1])
    st = _f32(state).copy()
    out = np.empty((n_heads, d), dtype=np.float32)
    lib.aero_ssm_delta_step(_f(q), _f(k), _f(v), _f(alpha), _f(beta),
                            _f(a_log), _f(dt_bias), _f(st), n_heads, d,
                            float(scale), float(eps), _f(out))
    return out, st


def scan(qh, kh, vh, alpha, beta, a_log, dt_bias, state, scale, eps=1e-6):
    """The per-head scan over T tokens. Returns (outs, new_state).

    qh, kh: (T, n_k, d); vh: (T, n_v, d); alpha, beta: (T, n_v).
    v-head h reads k-head h % n_k. `state` (n_v, d, d) is not mutated.
    """
    lib = _load()
    if lib is None:
        raise SsmUnavailable(unavailable_reason())
    qh, kh, vh = _f32(qh), _f32(kh), _f32(vh)
    alpha, beta = _f32(alpha), _f32(beta)
    a_log, dt_bias = _f32(a_log), _f32(dt_bias)
    T = int(qh.shape[0])
    n_k, d = int(qh.shape[1]), int(qh.shape[2])
    n_v = int(vh.shape[1])
    st = _f32(state).copy()
    outs = np.empty((T, n_v, d), dtype=np.float32)
    lib.aero_ssm_scan(_f(qh), _f(kh), _f(vh), _f(alpha), _f(beta), _f(a_log),
                      _f(dt_bias), _f(st), T, n_k, n_v, d, float(scale),
                      float(eps), _f(outs))
    return outs, st


def conv(x, w, tail, tail_len):
    """Causal depthwise conv step. Returns (out, new_tail, new_tail_len).

    `w` is (channels, kernel); `tail` is the (<=kernel-1, channels) history.
    """
    lib = _load()
    if lib is None:
        raise SsmUnavailable(unavailable_reason())
    x, w = _f32(x), _f32(w)
    kernel = int(w.shape[1])
    channels = int(w.shape[0])
    T = int(x.shape[0])
    keep = max(kernel - 1, 0)
    tailbuf = np.zeros((keep, channels), dtype=np.float32)
    tl = int(tail_len)
    if tl > 0:
        tailbuf[:tl] = _f32(tail)[:tl]
    out = np.empty((T, channels), dtype=np.float32)
    c_tl = c_int(tl)
    lib.aero_ssm_conv(_f(x), _f(w), _f(tailbuf), ctypes.byref(c_tl), _f(out),
                      T, channels, kernel)
    return out, tailbuf, int(c_tl.value)


def forward(x, w_qkv, w_gate, w_alpha, w_beta, a_log, dt_bias, conv1d, norm,
            w_out, n_embd, inner_size, state_size, n_group, conv_kernel,
            time_step_rank, eps, norm_eps, conv_tail, tail_len, state):
    """Full SSM layer. `state` and `conv_tail` are UPDATED IN PLACE.

    Returns (out, new_tail_len) on success, or None when the dimension
    combination is refused (the caller then falls back to numpy).
    """
    lib = _load()
    if lib is None:
        raise SsmUnavailable(unavailable_reason())
    x = _f32(x)
    T = int(x.shape[0])
    out = np.empty((T, n_embd), dtype=np.float32)
    c_tl = c_int(int(tail_len))
    rc = lib.aero_ssm_forward(
        _f(x), _f(w_qkv), _f(w_gate), _f(w_alpha), _f(w_beta), _f(a_log),
        _f(dt_bias), _f(conv1d), _f(norm), _f(w_out),
        T, int(n_embd), int(inner_size), int(state_size), int(n_group),
        int(conv_kernel), int(time_step_rank), float(eps), float(norm_eps),
        _f(conv_tail), ctypes.byref(c_tl), _f(state), _f(out))
    if rc != 0:
        return None
    return out, int(c_tl.value)


# --------------------------------------------------------------------- MoE

def route(logits, n_expert, k, norm_topk_prob=True):
    """Top-k routing. Returns (idx int64 (k,), w float32 (k,))."""
    lib = _load()
    if lib is None:
        raise SsmUnavailable(unavailable_reason())
    lg = _f32(logits).reshape(-1)
    idx = np.empty(k, dtype=np.int64)
    w = np.empty(k, dtype=np.float32)
    rc = lib.aero_moe_route(_f(lg), int(n_expert), int(k),
                            1 if norm_topk_prob else 0,
                            idx.ctypes.data_as(POINTER(c_int64)), _f(w))
    if rc != 0:
        raise ValueError(f"route refused k={k} for n_expert={n_expert}")
    return idx, w


def expert_accum(h, gate, up, down, wt, acc):
    """acc += wt * down @ (silu(gate@h) * (up@h)). Mutates `acc` in place."""
    lib = _load()
    if lib is None:
        raise SsmUnavailable(unavailable_reason())
    h = _f32(h).reshape(-1)
    n_embd = h.size
    ffn_len = int(gate.shape[0])
    lib.aero_moe_expert_accum(_f(h), n_embd, _f(gate), _f(up), _f(down),
                             ffn_len, float(wt), _f(acc))


def ffn_block(h, router_logits, n_expert, n_expert_used, ffn_len,
              norm_topk_prob, shexp, expert_rows, block_rows=64):
    """Full MoE FFN over a (T, n_embd) block.

    `expert_rows(name, expert, block_rows)` returns the full expert matrix, the
    same reader engine/moe.py moe_ffn takes (see engine/model.py
    _expert_rows_fn). `shexp` is a dict with gate_inp/gate/up/down or None.
    """
    lib = _load()
    if lib is None:
        raise SsmUnavailable(unavailable_reason())
    h = _f32(h)
    router_logits = _f32(router_logits)
    T, n_embd = int(h.shape[0]), int(h.shape[1])

    err: list[BaseException] = []
    names = {0: "ffn_gate_exps.weight", 1: "ffn_up_exps.weight",
             2: "ffn_down_exps.weight"}

    def _cb(ctx, which, expert, out):
        try:
            mat = _f32(expert_rows(names[which], int(expert), block_rows)).reshape(-1)
            np.ctypeslib.as_array(out, shape=(mat.size,))[:] = mat
            return 0
        except BaseException as e:  # noqa: BLE001 - handed back to the caller
            err.append(e)
            return 1

    cb = AERO_MOE_FETCH(_cb)

    if shexp:
        gi = _f32(shexp["gate_inp"])
        g = _f32(shexp["gate"])
        u = _f32(shexp["up"])
        d = _f32(shexp["down"])
        has_shared = 1
    else:
        gi = g = u = d = np.zeros(1, dtype=np.float32)
        has_shared = 0

    out = np.zeros((T, n_embd), dtype=np.float32)
    rc = lib.aero_moe_ffn_block(
        _f(h), _f(router_logits), T, n_embd, int(n_expert), int(n_expert_used),
        int(ffn_len), 1 if norm_topk_prob else 0, has_shared,
        _f(gi), _f(g), _f(u), _f(d), cb, None, _f(out))
    if err:
        raise err[0]
    if rc != 0:
        raise ValueError(f"moe ffn block refused (rc={rc})")
    return out
