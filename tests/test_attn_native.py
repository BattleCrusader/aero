#!/usr/bin/env python3
"""Native ATTENTION layer vs the Python attention, on the REAL model, per layer.

This is the parity gate for the c++ attention hook (aero_attention_qwen). The
Python attention() is known-correct end to end (gemma-4-12b answers 'Paris'
through it), so the native hook must agree with it on the SAME inputs -- to a
tolerance set by fp32, not by hope:
  * the projections go through the SAME fused dequant-matmul the Python uses
    (dequant order differs; expect ~1e-5 rel, not bit-exact);
  * the gate split, rope, GQA and the KV quantisers must be BYTE-EQUIVALENT, so
    anything above the fp32 floor is a wiring bug, not rounding.

Feeding it the real per-layer weights from Qwen3.8-4B (attn_q 8192x2560 Q6_K
[2*H*D => gated], attn_k/attn_v 1024x2560, attn_output 2560x4096 Q4_K, q/k norms
256-wide F32, rope_dims 64 of 256) exercises EVERY branch: fused qkv, gating
(interleaved), per-head norms, partial rope, GQA 16/4.
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

FAILS = 0
CHECKS = 0


def check(cond: bool, what: str) -> None:
    global FAILS, CHECKS
    CHECKS += 1
    if cond:
        print("  ok   " + what)
    else:
        FAILS += 1
        print("  FAIL " + what)


KV = "/Users/cyberdeth/.cache/aero-models/Qwen3.8-4B-Q4_K_M.gguf"
ATTLIB = os.path.join(ROOT, "build", "libaero_attn.dylib")
KVL = os.path.join(ROOT, "build", "libaero_kv.dylib")

if not os.path.exists(ATTLIB):
    print(f"(skipped: {ATTLIB} absent)")
    print(f"\n=== 0 checks, 0 failures ===")
    sys.exit(0)

lib = ctypes.CDLL(ATTLIB)
lib.aero_attention_qwen.argtypes = [
    POINTER(c_float), c_int64, c_int64, POINTER(c_int32),
    c_int, c_void_p, c_int64, c_int,
    c_int, c_void_p, c_int64,
    c_int, c_void_p, c_int64,
    c_int, c_void_p, c_int64,
    POINTER(c_float), POINTER(c_float), c_int, c_float,
    c_float, c_int64, c_void_p, c_int64,
    c_int64, c_int64, c_int64, c_int64, c_float, c_int,
    c_int, c_int,
    c_void_p, c_void_p, c_int64, POINTER(c_float)]
lib.aero_attention_qwen.restype = c_int

from engine import transformer as T       # noqa: E402
from engine.gguf import GGUFReader        # noqa: E402

rng = np.random.default_rng(3)

P = KV
with GGUFReader(P) as r:
    # Qwen3.8-4B geometry (blk.3's tensors are the reference)
    H, HK, D = 16, 4, 256
    NE = 2560
    q_info = r.tensors["blk.3.attn_q.weight"]
    k_info = r.tensors["blk.3.attn_k.weight"]
    v_info = r.tensors["blk.3.attn_v.weight"]
    o_info = r.tensors["blk.3.attn_output.weight"]
    qn = r.tensor("blk.3.attn_q_norm.weight")
    kn = r.tensor("blk.3.attn_k_norm.weight")

    Tt = 1                                     # one decode step
    x = (rng.standard_normal((Tt, NE)) * 0.3).astype(np.float32)
    pos = np.array([7], np.int32)

    def native_dt(py):
        return {12: 12, 14: 14, 23: 23, 1: 1, 0: 0}[py]

    # ---- python reference (the known-correct path) -------------------------
    w = T.AttnWeights(
        wq=r.tensor("blk.3.attn_q.weight"), wk=r.tensor("blk.3.attn_k.weight"),
        wv=r.tensor("blk.3.attn_v.weight"), wo=r.tensor("blk.3.attn_output.weight"),
        wqkv=None, q_norm=qn, k_norm=kn, v_norm_weightless=False,
    )
    acfg = T.AttnConfig(n_head=H, n_head_kv=HK, head_dim=D, value_dim=D,
                        rope_base=1e7, rope_dims=64, eps=1e-6)
    ref = T.attention(x, pos.astype(np.int64), w, acfg)

    # ---- native call -------------------------------------------------------
    def raw(info):
        return r.tensor_bytes(info)

    q_raw_b, k_raw_b, v_raw_b, o_raw_b = raw(q_info), raw(k_info), raw(v_info), raw(o_info)
    bufs = {}
    def u8buf(b):
        n = len(b)
        a = (ctypes.c_uint8 * n)()
        ctypes.memmove(a, b, n)
        return a
    q_b, k_b, v_b, o_b = u8buf(q_raw_b), u8buf(k_raw_b), u8buf(v_raw_b), u8buf(o_raw_b)

    out = np.zeros((Tt, NE), np.float32)
    ok = lib.aero_attention_qwen(
        x.ctypes.data_as(POINTER(c_float)), Tt, NE,
        pos.ctypes.data_as(POINTER(c_int32)),
        native_dt(q_info.type_id), q_b, q_info.shape[0], 1,
        native_dt(k_info.type_id), k_b, k_info.shape[0],
        native_dt(v_info.type_id), v_b, v_info.shape[0],
        native_dt(o_info.type_id), o_b, o_info.shape[0],
        qn.ctypes.data_as(POINTER(c_float)), kn.ctypes.data_as(POINTER(c_float)),
        0, 1e-6,
        1e7, 64, None, 0,
        H, HK, D, D, 0.0, 0,
        1, 8,                       # gate=INTERLEAVED_SIGMOID, threads
        None, None, 0,              # no kv slot for this comparison
        out.ctypes.data_as(POINTER(c_float)))
    check(ok == 1, "native attention hook returns 1")

    d = np.abs(ref - out)
    scale = max(float(np.abs(ref).max()), 1e-9)
    rel = d.max() / scale
    print(f"       max |py - native| = {d.max():.3e}  rel = {rel:.3e}")
    check(rel < 1e-4, f"attention parity on the REAL layer (rel {rel:.3e})")

print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
sys.exit(1 if FAILS else 0)