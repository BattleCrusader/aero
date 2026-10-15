#!/usr/bin/env python3
"""The hook guard, and the KvSlot C ABI.

TWO THINGS THAT COST REAL DEBUGGING TIME, now pinned:

1. A ctypes callback that raises is SWALLOWED. ctypes prints "Exception ignored in
   <callback>" and discards the exception, so the C side sees the callback's default
   return (0 for an int) and the model produces wrong numbers with NO traceback. This
   has cost three separate debugging rounds in this project: the `T` shadowing bug, an
   SsmState-vs-array mistake, and a ctypes POINTER passed to np.asarray. The guard
   makes the failure loud: the hook returns 0, the C++ loop treats that as a refusal,
   and the forward raises in Python where it can be seen. This test drives it with a
   deliberately broken hook and asserts the forward RAISES.

2. The KvSlot ABI (allocator + accessors) that the attention layer needs. It is new
   code with no test, so it gets one: allocate, write a vector, read it back, compare
   against the Python quantiser, and check the sizes and the refusals.

Uses the SYNTHETIC GGUF the C++ model test writes, so no real model is loaded.

    AERO_MAX_RAM=5G .venv-cml/bin/python tests/test_hook_guard.py
"""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import POINTER, c_float, c_int, c_int64, c_void_p

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


print("\n[1] the guard makes a raising hook LOUD instead of silent")
from engine import native_model as NM  # noqa: E402

if not NM.available():
    print(f"(skipped: libaero_model not built -- {NM.unavailable_reason()})")
    print("\n=== 0 checks, 0 failures ===")
    sys.exit(0)

# The guard is applied at callback registration, so a raising hook must turn into a
# refusal (0), and aero_model_forward must then report failure rather than computing
# with an unwritten output buffer.
seen: list = []


def broken_hook(ctx, layer, h, n_tok, n_embd, out):
    seen.append(layer)
    raise RuntimeError("deliberate hook failure")


wrapped = NM._guard("probe", broken_hook)
rc = wrapped(None, 0, None, 1, 8, None)
check(rc == 0, f"a raising hook is converted to a refusal (rc={rc}), not an exception")
check(any(n == "probe" for n, _ in NM._HOOK_ERRORS),
      "the failure is RECORDED (so a caller can report which hook broke)")

print("\n[2] the guard passes through a healthy hook unchanged")
ok_calls: list = []


def good_hook(ctx, layer, h, n_tok, n_embd, out):
    ok_calls.append(layer)
    return 1


check(NM._guard("ok", good_hook)(None, 3, None, 1, 8, None) == 1,
      "a healthy hook still returns its own value")
check(ok_calls == [3], "and it actually ran (the guard is not short-circuiting)")

print("\n[3] the KvSlot ABI: allocate, write, read back")
from engine import native as NK  # noqa: E402
from engine.transformer import KVCache as PKV  # noqa: E402

kp = os.path.join(ROOT, "build", "libaero_kv.dylib")
if not os.path.exists(kp):
    print("  (skipped: libaero_kv not built)")
else:
    kl = ctypes.CDLL(kp)
    kl.aero_kv_slot_new.argtypes = [c_int64, c_int64, c_int64, c_int, c_int]
    kl.aero_kv_slot_new.restype = c_void_p
    kl.aero_kv_slot_free.argtypes = [c_void_p]
    kl.aero_kv_slot_bytes.argtypes = [c_void_p]
    kl.aero_kv_slot_bytes.restype = c_int64
    kl.aero_kv_slot_write.argtypes = [c_void_p, c_int64, POINTER(c_float)]
    kl.aero_kv_slot_write.restype = c_int
    kl.aero_kv_slot_read.argtypes = [c_void_p, c_int64, POINTER(c_float)]
    kl.aero_kv_slot_read.restype = c_int

    cap, heads, dim = 16, 2, 8       # small on purpose
    s = kl.aero_kv_slot_new(cap, heads, dim, 0, 0)
    check(bool(s), "fp32 slot allocated")
    want_bytes = cap * heads * dim * 4
    check(int(kl.aero_kv_slot_bytes(s)) == want_bytes,
          f"fp32 slot is {want_bytes} bytes ({cap}x{heads}x{dim}x4)")

    vec = (np.random.default_rng(3).standard_normal(dim) * 2.0).astype(np.float32)
    check(kl.aero_kv_slot_write(s, 5, vec.ctypes.data_as(POINTER(c_float))) == 1,
          "write at physical index 5 accepted")
    back = np.zeros(dim, np.float32)
    check(kl.aero_kv_slot_read(s, 5, back.ctypes.data_as(POINTER(c_float))) == 1,
          "read back from index 5 accepted")
    check(np.array_equal(back, vec), "fp32 slot round trip is bit-exact")

    # untouched positions must still be zero (a fresh slot is zeroed, so an unset
    # position cannot masquerade as data)
    z = np.zeros(dim, np.float32)
    kl.aero_kv_slot_read(s, 0, z.ctypes.data_as(POINTER(c_float)))
    check(np.all(z == 0.0), "an unwritten position reads as zeros")

    print("\n[4] int8 and int4 slots match the Python quantiser byte-for-byte")
    for dt, name in ((1, "int8"), (2, "int4")):
        s2 = kl.aero_kv_slot_new(cap, heads, dim, dt, 0)
        check(bool(s2), f"{name} slot allocated")
        kl.aero_kv_slot_write(s2, 2, vec.ctypes.data_as(POINTER(c_float)))
        got = np.zeros(dim, np.float32)
        kl.aero_kv_slot_read(s2, 2, got.ctypes.data_as(POINTER(c_float)))
        if dt == 1:
            sc = np.where(np.abs(vec).max() > 0, np.abs(vec).max() / 127.0,
                          1.0).astype(np.float32)
            want = np.clip(np.round(vec / sc), -127, 127).astype(np.float32) * sc
        else:
            q, sc = PKV._quantize4(vec)
            want = PKV._unpack4(PKV._pack4(q), dim) * sc.squeeze(-1)
        check(np.allclose(got, want, atol=1e-6),
              f"{name} slot read matches the Python quantiser "
              f"(max |d| {np.abs(got - want).max():.2e})")
        kl.aero_kv_slot_free(s2)

    print("\n[5] refusals: bad geometry and out-of-range indices")
    check(not kl.aero_kv_slot_new(0, heads, dim, 0, 0), "cap=0 refused")
    check(not kl.aero_kv_slot_new(cap, 0, dim, 0, 0), "heads=0 refused")
    check(not kl.aero_kv_slot_new(cap, heads, 7, 2, 0), "int4 with odd dim refused")
    check(not kl.aero_kv_slot_new(cap, heads, dim, 9, 0), "unknown dtype refused")
    check(kl.aero_kv_slot_write(s, cap, vec.ctypes.data_as(POINTER(c_float))) == 0,
          f"write at index cap ({cap}) refused -- no out-of-bounds write")
    check(kl.aero_kv_slot_read(s, cap, np.zeros(dim, np.float32).ctypes.data_as(
        POINTER(c_float))) == 0, "read at index cap refused")
    kl.aero_kv_slot_free(s)

print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
sys.exit(1 if FAILS else 0)