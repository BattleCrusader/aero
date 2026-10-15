"""ctypes binding for the native BLOCK LOOP (libaero_model).

WHY THIS EXISTS. The block loop is correct in C++ (tests/test_model_cpp.cpp, 20
checks) but nothing called it: engine/model.py's forward still ran the whole
per-layer loop in numpy. This module is the missing seam -- it binds aero_model_open
and aero_model_forward, and supplies the per-layer attention/SSM/MoE work through the
C++ model's hook table, backed by the same Python primitives the engine uses today
(so behaviour is identical to the verified Python path, and each hook can be
swapped for its native equivalent later without touching this layer).

SAFETY MODEL:
  * the caller passes a budget from AERO_MAX_RAM / max_ram_bytes -- this module
    NEVER loads without one (same contract as engine/model.py);
  * hooks receive plain fp32 buffers and return 1/0; refusal of a hook is an error,
    not a silent skip, so a missing primitive can never produce wrong output;
  * the load lock (one model at a time) is TAKEN HERE, before anything is read --
    two concurrent loads OOMed this machine and the lock is the enforcement.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import (POINTER, CFUNCTYPE, c_char_p, c_float, c_int, c_int32,
                    c_int64, c_void_p, byref)

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

_LIB_CANDIDATES = (
    os.path.join(_ROOT, "build", "libaero_model.dylib"),
    os.path.join(_ROOT, "build", "libaero_model.so"),
)


# ---------------------------------------------------------------- signatures
# int (*)(void* ctx, int layer, const float* h, const int64_t* pos, int T,
#         int n_embd, float* out)
ATTN_FN = CFUNCTYPE(c_int, c_void_p, c_int, POINTER(c_float), POINTER(c_int64),
                    c_int, c_int, POINTER(c_float))
# int (*)(void* ctx, int layer, const float* h, int T, int n_embd, float* out)
SSM_FN = CFUNCTYPE(c_int, c_void_p, c_int, POINTER(c_float), c_int, c_int,
                   POINTER(c_float))
MOE_FN = SSM_FN



def _guard(name: str, fn):
    """Wrap a ctypes callback so a Python exception cannot be SWALLOWED.

    ctypes prints "Exception ignored in ..." and discards the exception: the C side
    reads the callback's default return (0 for an int) and the model produces wrong
    numbers with no traceback anywhere. That has cost three debugging rounds in this
    project. Returning 0 is the correct signal -- aero_model_forward treats a hook's
    0 as refusal and the whole forward fails loudly in Python, where it can be seen.
    """
    def wrapper(*args):
        try:
            return fn(*args)
        except BaseException as e:            # noqa: BLE001 -- must not escape
            import traceback
            _HOOK_ERRORS.append((name, repr(e)))
            print(f"[native_model] {name} hook FAILED: {type(e).__name__}: {e}",
                  file=sys.stderr)
            traceback.print_exc()
            return 0
    return wrapper


_HOOK_ERRORS: list = []

class ModelLibUnavailable(RuntimeError):
    """libaero_model is not built. Callers must fall back, not guess."""


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
            lib.aero_model_open.argtypes = [c_char_p, c_void_p, c_void_p,
                                            c_char_p, c_int]
            lib.aero_model_open.restype = c_void_p
            lib.aero_model_free.argtypes = [c_void_p]
            lib.aero_model_forward.argtypes = [c_void_p, POINTER(c_int32),
                                               c_int, POINTER(c_int64),
                                               POINTER(c_float)]
            lib.aero_model_forward.restype = c_int
            for nm in ("set_attn", "set_ssm", "set_moe"):
                f = getattr(lib, "aero_model_" + nm)
                f.argtypes = [c_void_p, c_void_p, c_void_p]
                f.restype = None
            _lib = lib
            return _lib
        except (OSError, AttributeError) as e:
            # AttributeError: ctypes raises it for a missing symbol. A library that
            # loads but lacks one must fall through here rather than crash callers.
            _load_error = f"{path}: {e}"
    if _load_error is None:
        _load_error = ("libaero_model not found; build it with `make all` "
                       "(looked in " + ", ".join(_LIB_CANDIDATES) + ")")
    return None


def available() -> bool:
    return _load() is not None


def unavailable_reason() -> str:
    _load()
    return _load_error or ""


# ------------------------------------------------------------------ geometry
class AeroGeom(ctypes.Structure):
    _fields_ = [
        ("arch", ctypes.c_int32),
        ("n_layer", ctypes.c_int32),
        ("n_embd", ctypes.c_int32),
        ("vocab", ctypes.c_int32),
        ("eps", ctypes.c_float),
        ("tied_output", ctypes.c_int32),
        ("softcap", ctypes.c_float),
        ("embed_scale", ctypes.c_float),
        ("block_on_output", ctypes.c_int32),
        ("gemma_out_scale", ctypes.c_int32),
        ("threads", ctypes.c_int32),
        ("block_rows", ctypes.c_int32),
    ]


class AeroLayerGeom(ctypes.Structure):
    _fields_ = [("kind", ctypes.c_int32), ("is_moe", ctypes.c_int32),
                ("out_scale", ctypes.c_float)]


# ---------------------------------------------------------------- the wrapper
class NativeBlockLoop:
    """The per-layer loop, computed by libaero_model, with hook-backed layers."""

    def __init__(self, path: str, *, arch: str, n_layer: int, n_embd: int,
                 vocab: int, eps: float, tied_output: bool, softcap: float,
                 embed_scale: float, block_on_output: bool,
                 gemma_out_scale: int, block_rows: int, threads: int,
                 layer_kinds: list[int], out_scales: list[float],
                 attn_hook, ssm_hook, moe_hook) -> None:
        lib = _load()
        if lib is None:
            raise ModelLibUnavailable(unavailable_reason())

        g = AeroGeom(
            arch=0 if arch == "gemma4" else 1,
            n_layer=n_layer, n_embd=n_embd, vocab=vocab, eps=float(eps),
            tied_output=int(bool(tied_output)), softcap=float(softcap or 0.0),
            embed_scale=float(embed_scale or 0.0),
            block_on_output=int(bool(block_on_output)),
            gemma_out_scale=int(gemma_out_scale), threads=int(threads or 1),
            block_rows=int(block_rows),
        )
        lg = (AeroLayerGeom * n_layer)()
        for i in range(n_layer):
            lg[i].kind = int(layer_kinds[i])
            lg[i].is_moe = int(layer_is_moe(i) if False else 0)
            lg[i].out_scale = float(out_scales[i])

        # Hold references for the lifetime of the object: ctypes does not keep
        # CFUNCTYPE objects alive across calls, and a collected callback = crash.
        self._attn_cb = ATTN_FN(_guard("attn", attn_hook))
        if ssm_hook is not None:
            self._ssm_cb = SSM_FN(_guard("ssm", ssm_hook))
        else:
            self._ssm_cb = None
        if moe_hook is not None:
            self._moe_cb = MOE_FN(_guard("moe", moe_hook))
        else:
            self._moe_cb = None

        err = ctypes.create_string_buffer(256)
        self._h = lib.aero_model_open(path.encode(), byref(g), lg, err, 256)
        if not self._h:
            raise RuntimeError(f"aero_model_open failed: {err.value.decode()!r}")
        lib.aero_model_set_attn(self._h, self._attn_cb, None)
        if self._ssm_cb is not None:
            lib.aero_model_set_ssm(self._h, self._ssm_cb, None)
        if self._moe_cb is not None:
            lib.aero_model_set_moe(self._h, self._moe_cb, None)
        self._lib = lib
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.vocab = vocab

    def forward(self, tokens: list[int], positions, logits_out) -> None:
        """One pass. tokens: list[int]; positions: int64 ndarray; logits_out: f32
        ndarray of vocab, written in place."""
        import numpy as np
        arr = (ctypes.c_int32 * len(tokens))(*[int(t) for t in tokens])
        pos = np.ascontiguousarray(positions, dtype=np.int64)
        ok = self._lib.aero_model_forward(
            self._h, arr, len(tokens),
            pos.ctypes.data_as(POINTER(c_int64)),
            logits_out.ctypes.data_as(POINTER(c_float)))
        if ok != 1:
            raise RuntimeError("aero_model_forward refused (hook or tensor missing)")

    def close(self) -> None:
        if getattr(self, "_h", None):
            self._lib.aero_model_free(self._h)
            self._h = None


def layer_is_moe(i: int) -> bool:
    return False