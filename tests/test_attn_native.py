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

# ---------------------------------------------------------------------------
# [2] THE THREE BUGS THAT MADE THIS SUITE DIE, pinned so they cannot come back.
# The suite used to flap between "2 checks pass" and SIGSEGV on unchanged code, which
# is the signature of heap corruption rather than logic -- and all three causes were
# real. Each check below fails loudly on the old code.
kvlib = ctypes.CDLL(KVL) if os.path.exists(KVL) else None
if kvlib is None:
    print("  (skipped: libaero_kv absent)")
else:
    print("\n[2] kv head index, ring wrap, and per-row rope positions")
    kvlib.aero_kv_slot_new.argtypes = [c_int64, c_int64, c_int64, c_int, c_int]
    kvlib.aero_kv_slot_new.restype = c_void_p
    kvlib.aero_kv_slot_write.argtypes = [c_void_p, c_int64, c_int64, POINTER(c_float)]
    kvlib.aero_kv_slot_write.restype = c_int
    kvlib.aero_kv_slot_read.argtypes = [c_void_p, c_int64, c_int64, POINTER(c_float)]
    kvlib.aero_kv_slot_read.restype = c_int
    kvlib.aero_kv_slot_free.argtypes = [c_void_p]

    # (a) THE HEAD INDEX. kv_slot_read used to compute `phys * heads * vec_bytes()`,
    # i.e. head 0's slot, for every head -- so reading head 1 returned head 0's vector.
    cap, heads, dim = 8, 4, 16
    sl = kvlib.aero_kv_slot_new(cap, heads, dim, 0, 0)       # fp32, no ring
    a_vec = np.full(dim, 1.5, np.float32)
    b_vec = np.full(dim, -2.5, np.float32)
    kvlib.aero_kv_slot_write(sl, 2, 0, a_vec.ctypes.data_as(POINTER(c_float)))
    kvlib.aero_kv_slot_write(sl, 2, 1, b_vec.ctypes.data_as(POINTER(c_float)))
    got = np.zeros(dim, np.float32)
    kvlib.aero_kv_slot_read(sl, 2, 0, got.ctypes.data_as(POINTER(c_float)))
    check(np.array_equal(got, a_vec), "head 0 reads back its own vector")
    got2 = np.zeros(dim, np.float32)
    kvlib.aero_kv_slot_read(sl, 2, 1, got2.ctypes.data_as(POINTER(c_float)))
    check(np.array_equal(got2, b_vec),
          "head 1 reads back ITS vector, not head 0's (the old bug)")
    check(not np.array_equal(got2, a_vec), "and it is not merely a copy of head 0")
    kvlib.aero_kv_slot_free(sl)

    # (b) ONE POSITION PER ROW. apply_rope reads positions[r] for every row and was
    # called with T*H rows against a T-entry array, so rows >= T read past the end --
    # EXC_BAD_ACCESS inside apply_rope. At position 0 a step WITH a cache must equal the
    # same step WITHOUT one (both attend over exactly one position).
    k_sl = kvlib.aero_kv_slot_new(8, HK, D, 0, 0)
    v_sl = kvlib.aero_kv_slot_new(8, HK, D, 0, 0)
    pos0 = np.array([0], np.int32)
    out_s = np.zeros((1, NE), np.float32)
    ok_s = lib.aero_attention_qwen(
        x.ctypes.data_as(POINTER(c_float)), 1, NE,
        pos0.ctypes.data_as(POINTER(c_int32)),
        native_dt(q_info.type_id), q_b, q_info.shape[0], 1,
        native_dt(k_info.type_id), k_b, k_info.shape[0],
        native_dt(v_info.type_id), v_b, v_info.shape[0],
        native_dt(o_info.type_id), o_b, o_info.shape[0],
        qn.ctypes.data_as(POINTER(c_float)), kn.ctypes.data_as(POINTER(c_float)),
        0, 1e-6, 1e7, 64, None, 0,
        H, HK, D, D, 0.0, 0, 1, 8,
        ctypes.c_void_p(k_sl), ctypes.c_void_p(v_sl), 0,
        out_s.ctypes.data_as(POINTER(c_float)))
    check(ok_s == 1, "attention with a KV slot returns 1 (no wild rope read)")
    ref0 = T.attention(x, pos0.astype(np.int64), w, acfg)
    rel0 = np.abs(ref0 - out_s).max() / max(float(np.abs(ref0).max()), 1e-9)
    check(rel0 < 1e-4,
          f"position 0 with a cache == without one (rel {rel0:.3e}) -- rope rows correct")

    # (c) RING WRAP. Writing at kv_phys + t ran past a wrapped slot's cap: an
    # out-of-bounds WRITE into the cache vectors (heap corruption). With cap=4 and
    # kv_phys >= cap the physical index must wrap, not overrun.
    rk = kvlib.aero_kv_slot_new(4, HK, D, 0, 1)      # ring=1 -- the wrap cases need it
    rv = kvlib.aero_kv_slot_new(4, HK, D, 0, 1)
    for kvp in (4, 5, 9):
        posn = np.array([kvp], np.int32)
        outn = np.zeros((1, NE), np.float32)
        r_ok = lib.aero_attention_qwen(
            x.ctypes.data_as(POINTER(c_float)), 1, NE,
            posn.ctypes.data_as(POINTER(c_int32)),
            native_dt(q_info.type_id), q_b, q_info.shape[0], 1,
            native_dt(k_info.type_id), k_b, k_info.shape[0],
            native_dt(v_info.type_id), v_b, v_info.shape[0],
            native_dt(o_info.type_id), o_b, o_info.shape[0],
            qn.ctypes.data_as(POINTER(c_float)), kn.ctypes.data_as(POINTER(c_float)),
            0, 1e-6, 1e7, 64, None, 0,
            H, HK, D, D, 0.0, 0, 1, 8,
            ctypes.c_void_p(rk), ctypes.c_void_p(rv), kvp,
            outn.ctypes.data_as(POINTER(c_float)))
        check(r_ok == 1 and np.isfinite(outn).all(),
              f"kv_phys={kvp} past cap=4 (ring) wraps instead of overrunning")
    kvlib.aero_kv_slot_free(rk)
    kvlib.aero_kv_slot_free(rv)
    # a non-ring slot that is too small must be REFUSED, not overrun
    small = kvlib.aero_kv_slot_new(2, HK, D, 0, 0)
    out_x = np.zeros((1, NE), np.float32)
    r_refused = lib.aero_attention_qwen(
        x.ctypes.data_as(POINTER(c_float)), 1, NE,
        np.array([7], np.int32).ctypes.data_as(POINTER(c_int32)),
        native_dt(q_info.type_id), q_b, q_info.shape[0], 1,
        native_dt(k_info.type_id), k_b, k_info.shape[0],
        native_dt(v_info.type_id), v_b, v_info.shape[0],
        native_dt(o_info.type_id), o_b, o_info.shape[0],
        qn.ctypes.data_as(POINTER(c_float)), kn.ctypes.data_as(POINTER(c_float)),
        0, 1e-6, 1e7, 64, None, 0,
        H, HK, D, D, 0.0, 0, 1, 8,
        ctypes.c_void_p(small), ctypes.c_void_p(small), 7,
        out_x.ctypes.data_as(POINTER(c_float)))
    check(r_refused == 0, "a non-ring slot too small for the span is refused (rc=0, not corruption)")
    kvlib.aero_kv_slot_free(small)
    kvlib.aero_kv_slot_free(k_sl)
    kvlib.aero_kv_slot_free(v_sl)


# ---------------------------------------------------------------------------
# [3] THE INVARIANT THE NATIVE PATH FAILS. Prefill n tokens into a slot, then decode one
# more at position n, must equal feeding all n+1 in a single call. The C++ model path misses
# this by 23.7 logits (the Python path holds it to 2e-05), so the layer itself is the place
# to localise it: same tensors, same weights, only the call pattern differs.
print("\n[3] prefill + decode == one-shot prefill (the state path)")
k5 = kvlib.aero_kv_slot_new(16, HK, D, 0, 0)
v5 = kvlib.aero_kv_slot_new(16, HK, D, 0, 0)


def run_slots(Tt, pos_list, kvp, kslot, vslot):
    out = np.zeros((Tt, NE), np.float32)
    rc = lib.aero_attention_qwen(
        x5.ctypes.data_as(POINTER(c_float)), Tt, NE,
        np.array(pos_list, np.int32).ctypes.data_as(POINTER(c_int32)),
        native_dt(q_info.type_id), q_b, q_info.shape[0], 1,
        native_dt(k_info.type_id), k_b, k_info.shape[0],
        native_dt(v_info.type_id), v_b, v_info.shape[0],
        native_dt(o_info.type_id), o_b, o_info.shape[0],
        qn.ctypes.data_as(POINTER(c_float)), kn.ctypes.data_as(POINTER(c_float)),
        0, 1e-6, 1e7, 64, None, 0,
        H, HK, D, D, 0.0, 0, 1, 8,
        ctypes.c_void_p(kslot), ctypes.c_void_p(vslot), kvp,
        out.ctypes.data_as(POINTER(c_float)))
    return rc, out


# five tokens of real input, one shot into a fresh slot
x5 = np.concatenate([x] * 5, axis=0).astype(np.float32)
rcA, outA = run_slots(5, [0, 1, 2, 3, 4], 0, k5, v5)
check(rcA == 1, "one-shot prefill of 5 tokens succeeded")

# four tokens, then the fifth decoded at position 4, into another fresh slot
k6 = kvlib.aero_kv_slot_new(16, HK, D, 0, 0)
v6 = kvlib.aero_kv_slot_new(16, HK, D, 0, 0)
x4 = np.concatenate([x] * 4, axis=0).astype(np.float32)
rcB, outB4 = run_slots(4, [0, 1, 2, 3], 0, k6, v6)
rcC, outC = run_slots(1, [4], 4, k6, v6)
check(rcB == 1 and rcC == 1, "prefill of 4 then a decode at position 4 both succeeded")

d5 = float(np.abs(outA[4] - outC[0]).max())
rel5 = d5 / max(float(np.abs(outA[4]).max()), 1e-9)
print("       one-shot last token vs decoded last token: max|d| = %.4e  rel = %.3e" % (d5, rel5))
check(rel5 < 1e-4,
      "prefill+decode == one-shot prefill at the LAYER level (rel %.3e)" % rel5)
kvlib.aero_kv_slot_free(k5); kvlib.aero_kv_slot_free(v5)
kvlib.aero_kv_slot_free(k6); kvlib.aero_kv_slot_free(v6)

print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
sys.exit(1 if FAILS else 0)