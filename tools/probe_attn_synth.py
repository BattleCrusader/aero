#!/usr/bin/env python3
"""aero_attention_qwen on tiny synthetic weights -- no model, seconds per run.

WHY: the crash investigation kept costing a 2.78 GB model load per attempt, and lldb
under that load was too slow to be usable. Everything the layer does (projections,
per-head q/k norm, partial rope, GQA, causal+window, gate, output projection) works on
random small weights, so the crash can be bisected here in seconds.

Dims are deliberately awkward in a way real files are NOT: rope_dims < head_dim, a
single kv head with several query heads, head_dim not a power of two times anything.

Run:  AERO_MAX_RAM=5G .venv-cml/bin/python tools/probe_attn_synth.py [T]
"""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import POINTER, c_float, c_int, c_int32, c_int64, c_void_p

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("AERO_MAX_RAM", "5G")

import numpy as np

lib = ctypes.CDLL(os.path.join(ROOT, "build", "libaero_attn.dylib"))
kl = ctypes.CDLL(os.path.join(ROOT, "build", "libaero_kv.dylib"))
lib.aero_attention_qwen.argtypes = [
    POINTER(c_float), c_int64, c_int64, POINTER(c_int32),
    c_int, c_void_p, c_int64, c_int,
    c_int, c_void_p, c_int64,
    c_int, c_void_p, c_int64,
    c_int, c_void_p, c_int64,
    POINTER(c_float), POINTER(c_float),
    c_int, c_float, c_float, c_int64, POINTER(c_float), c_int64,
    c_int64, c_int64, c_int64, c_int64, c_float, c_int, c_int, c_int,
    c_void_p, c_void_p, c_int64, POINTER(c_float)]
lib.aero_attention_qwen.restype = c_int
kl.aero_kv_slot_new.argtypes = [c_int64, c_int64, c_int64, c_int, c_int]
kl.aero_kv_slot_new.restype = c_void_p

# FP32 (dt=0) weights: no quantiser involved, so a crash is structural, not a read.
H, HK, D = 4, 2, 8          # 4 query heads, 2 kv heads (GQA ratio 2), head_dim 8
NE = 16
NROT = 4                    # PARTIAL rope: rotate 4 of 8 dims
ROWS_Q = 2 * H * D          # fused q + gate, per-head interleaved
ROWS_K = HK * D
ROWS_O = H * D              # attention width -> the o projection maps HD
rng = np.random.default_rng(7)
W = {
    "q": (rng.standard_normal((ROWS_Q, NE)) * 0.2).astype(np.float32),
    "k": (rng.standard_normal((ROWS_K, NE)) * 0.2).astype(np.float32),
    "v": (rng.standard_normal((ROWS_K, NE)) * 0.2).astype(np.float32),
    "o": (rng.standard_normal((NE, ROWS_O)) * 0.2).astype(np.float32),
}
qn = (rng.random(D) + 0.5).astype(np.float32)
kn = (rng.random(D) + 0.5).astype(np.float32)


def run(T, use_slots):
    x = (rng.standard_normal((T, NE)) * 0.5).astype(np.float32)
    pos = np.arange(T, dtype=np.int32)
    out = np.zeros((T, NE), np.float32)
    ks = vs = None
    if use_slots:
        ks = kl.aero_kv_slot_new(64, HK, D, 0, 0)
        vs = kl.aero_kv_slot_new(64, HK, D, 0, 0)
    rc = lib.aero_attention_qwen(
        x.ctypes.data_as(POINTER(c_float)), T, NE,
        pos.ctypes.data_as(POINTER(c_int32)),
        0, W["q"].ctypes.data_as(c_void_p), ROWS_Q, 1,
        0, W["k"].ctypes.data_as(c_void_p), ROWS_K,
        0, W["v"].ctypes.data_as(c_void_p), ROWS_K,
        0, W["o"].ctypes.data_as(c_void_p), ROWS_O,
        qn.ctypes.data_as(POINTER(c_float)), kn.ctypes.data_as(POINTER(c_float)),
        0, 1e-6, 1e7, NROT, None, 0,
        H, HK, D, D, float(1.0 / np.sqrt(D)), 0, 1, 1,
        ks, vs, 0,
        out.ctypes.data_as(POINTER(c_float)))
    return rc, out


print(f"  dims: n_embd={NE} H={H} HK={HK} head_dim={D} rope_dims={NROT} "
      f"q_rows={ROWS_Q} k_rows={ROWS_K} o_rows={ROWS_O}")
for T in (1, 2, 4):
    for use_slots in (False, True):
        tag = "slots" if use_slots else "no slots"
        try:
            rc, out = run(T, use_slots)
            fin = bool(np.isfinite(out).all())
            print(f"  T={T} {tag:8} rc={rc} finite={fin} out[0][0..2]={out[0][:3]}")
        except Exception as e:                        # noqa: BLE001
            print(f"  T={T} {tag:8} EXCEPTION {type(e).__name__}: {e}")
        sys.stdout.flush()
print("  (a SIGSEGV would abort the process above; every line printing = no crash)")
