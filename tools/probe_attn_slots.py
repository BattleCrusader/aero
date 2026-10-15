#!/usr/bin/env python3
"""Minimal repro: aero_attention_qwen with None slots (works) vs REAL slots (crashes).

tests/test_attn_native.py passes with `None, None, 0` for k_slot/v_slot/kv_phys, so the
projections, rope, gate, GQA and the output projection are proven. The decode path
needs real slots, and that call crashed the process (SIGSEGV/BUS). This isolates it:
same weights, same geometry, same positions -- only the slots differ.

Run:  AERO_MAX_RAM=5G .venv-cml/bin/python tools/probe_attn_slots.py
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

from engine import transformer as T
from engine.gguf import GGUFReader

P = "/Users/cyberdeth/.cache/aero-models/Qwen3.8-4B-Q4_K_M.gguf"
LAYER = 3                      # an attention layer in Qwen3.8-4B
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
kl.aero_kv_slot_bytes.argtypes = [c_void_p]
kl.aero_kv_slot_bytes.restype = c_int64

NATIVE = {12, 14, 23}


def dt(t):
    return int(t) if int(t) in NATIVE else 0


with GGUFReader(P) as r:
    cfg = T.AttnConfig(n_head=16, n_head_kv=4, head_dim=256, value_dim=256,
                       rope_base=1e7, rope_dims=64, eps=1e-6)
    H, HK, D = cfg.n_head, cfg.n_head_kv, cfg.head_dim
    NE = 2560
    p = f"blk.{LAYER}."

    def u8(blob):
        a = (ctypes.c_uint8 * len(blob))()
        ctypes.memmove(a, blob, len(blob))
        return a

    infos = {n: r.tensors[p + n] for n in
             ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight")}
    bufs = {k: u8(r.tensor_bytes(v)) for k, v in infos.items()}
    qn = np.ascontiguousarray(r.tensor(p + "attn_q_norm.weight"), np.float32)
    kn = np.ascontiguousarray(r.tensor(p + "attn_k_norm.weight"), np.float32)

    rng = np.random.default_rng(0)
    x = (rng.standard_normal((1, NE)) * 0.5).astype(np.float32)
    pos = np.zeros(1, np.int32)

    def call(k_slot, v_slot, kv_phys):
        out = np.zeros((1, NE), np.float32)
        return lib.aero_attention_qwen(
            x.ctypes.data_as(POINTER(c_float)), 1, NE,
            pos.ctypes.data_as(POINTER(c_int32)),
            dt(infos["attn_q.weight"].type_id), bufs["attn_q.weight"],
            infos["attn_q.weight"].shape[0], 1,
            dt(infos["attn_k.weight"].type_id), bufs["attn_k.weight"],
            infos["attn_k.weight"].shape[0],
            dt(infos["attn_v.weight"].type_id), bufs["attn_v.weight"],
            infos["attn_v.weight"].shape[0],
            dt(infos["attn_output.weight"].type_id), bufs["attn_output.weight"],
            infos["attn_output.weight"].shape[0],
            qn.ctypes.data_as(POINTER(c_float)), kn.ctypes.data_as(POINTER(c_float)),
            0, 1e-6, 1e7, 64, None, 0,
            H, HK, D, D, float(1.0 / np.sqrt(D)), 0, 1, 1,
            k_slot, v_slot, kv_phys,
            out.ctypes.data_as(POINTER(c_float))), out

    print("  [A] None slots (the path tests/test_attn_native.py proves)")
    rc_a, out_a = call(None, None, 0)
    print(f"      rc={rc_a}  out[0][0..2]={out_a[0][:3]}")
    sys.stdout.flush()

    print("  [B] real fp32 slots, kv_phys=0")
    ks = kl.aero_kv_slot_new(64, HK, D, 0, 0)
    vs = kl.aero_kv_slot_new(64, HK, D, 0, 0)
    print(f"      slots: k={hex(ks or 0)} ({kl.aero_kv_slot_bytes(ks)} B)  "
          f"v={hex(vs or 0)} ({kl.aero_kv_slot_bytes(vs)} B)")
    sys.stdout.flush()
    rc_b, out_b = call(ks, vs, 0)
    print(f"      rc={rc_b}  out[0][0..2]={out_b[0][:3]}")
    sys.stdout.flush()

    d = np.abs(out_a - out_b).max()
    print(f"\n  slots vs no-slots: max |d| = {d:.3e}")
    print(f"  (should be ~0: at position 0 the cache holds exactly this token, so the "
          f"two paths are the same computation)")
