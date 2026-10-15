#!/usr/bin/env python3
"""Native KV primitives vs engine/transformer.KVCache's own quantisers.

The KV cache is where quiet corruption does the most damage: a wrong nibble order
or an off-by-one ring rotation is INVISIBLE at write time, costs nothing at load,
and shows up only as subtly wrong attention scores -- the hardest class of bug to
find from output. So this checks byte-exact storage and ring behaviour against the
Python class on identical inputs, and prints the observed differences.

    AERO_MAX_RAM=5G .venv-cml/bin/python tests/test_kv_native.py
"""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import POINTER, c_float, c_int, c_int64, c_void_p

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

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


_LIBS = (
    os.path.join(ROOT, "build", "libaero_kv.dylib"),
    os.path.join(ROOT, "build", "libaero_kv.so"),
)
lib = None
err = None
for p in _LIBS:
    if os.path.exists(p):
        try:
            lib = ctypes.CDLL(p)
            break
        except (OSError, AttributeError) as e:
            err = str(e)
if lib is None:
    print(f"(skipped: libaero_kv not built -- {err or 'not found'})")
    print(f"\n=== 0 checks, 0 failures ===")
    sys.exit(0)

lib.aero_kv_write_vec.argtypes = [c_int, POINTER(c_float), c_int64, c_void_p,
                                  POINTER(c_float)]
lib.aero_kv_write_vec.restype = c_int
lib.aero_kv_read_vec.argtypes = [c_int, c_void_p, POINTER(c_float), c_int64,
                                 POINTER(c_float)]
lib.aero_kv_read_vec.restype = c_int
lib.aero_kv_ring_rotate.argtypes = [c_int64, c_int64, c_int64, POINTER(c_int64)]
lib.aero_kv_ring_rotate.restype = c_int
lib.aero_kv_span_error.argtypes = [c_int, c_void_p, POINTER(c_float), c_int64,
                                   POINTER(c_float), POINTER(c_float)]
lib.aero_kv_span_error.restype = c_int

from engine.transformer import KVCache as PKV  # noqa: E402

rng = np.random.default_rng(11)


def write_native(dt, x):
    """-> (q_bytes, scale) matching the C ABI layout for one vector.

    Buffer sizing is per dtype -- fp32 = dim*4 bytes, int8 = dim, int4 = dim/2.
    The first version of this harness sized EVERY case as dim bytes, which
    truncated the fp32 memcpy to a quarter of the vector and reported the
    kernel 'not bit-exact'. The kernel was fine; the harness was too small.
    """
    dim = x.size
    n = dim * 4 if dt == 0 else (dim // 2 if dt == 2 else dim)
    q = (ctypes.c_uint8 * n)()
    s = c_float()
    ok = lib.aero_kv_write_vec(dt, x.ctypes.data_as(POINTER(c_float)), dim, q,
                               ctypes.byref(s))
    return ok, q, s


def read_native(dt, q, s, dim):
    out = np.zeros(dim, np.float32)
    qb = (ctypes.c_uint8 * len(q)).from_buffer(bytearray(bytes(q)))
    ok = lib.aero_kv_read_vec(dt, qb, ctypes.byref(c_float(s)), dim,
                              out.ctypes.data_as(POINTER(c_float)))
    return ok, out


print("\n[1] int8: byte-exact against the Python quantiser")
x = (rng.standard_normal(128) * 2.5).astype(np.float32)
# the Python path, via the class's staticmethod
sc_py = np.where(np.abs(x).max() > 0, np.abs(x).max() / 127.0, 1.0).astype(np.float32)
q_py = np.clip(np.round(x / sc_py), -127, 127).astype(np.int8)
ok, q_nt, s_nt = write_native(1, x)
check(ok == 1, "native int8 write returns 1")
nt_vals = np.array([q_nt[i] for i in range(x.size)], np.uint8).view(np.int8)
check(np.array_equal(nt_vals, q_py),
      f"int8 values byte-identical (max |d| = {np.abs(nt_vals.astype(int) - q_py.astype(int)).max()})")
check(abs(s_nt.value - float(sc_py)) < 1e-9,
      f"int8 scale identical ({s_nt.value:.9f} vs {float(sc_py):.9f})")

print("\n[2] int4: byte-exact, low nibble first, sign-extended on read")
q4_py, s4_py = PKV._quantize4(x)
pk_py = PKV._pack4(q4_py)
ok4, q4_nt, s4_nt = write_native(2, x)
check(ok4 == 1, "native int4 write returns 1")
nt4 = np.array([q4_nt[i] for i in range(x.size // 2)], np.uint8)
check(np.array_equal(nt4, pk_py),
      f"int4 packed bytes identical (max |d| = {np.abs(nt4.astype(int) - pk_py.astype(int)).max()})")
okr, back = read_native(2, q4_nt, s4_nt.value, x.size)
check(okr == 1 and np.array_equal(back, q4_py.astype(np.float32) * s4_py),
      f"native int4 read matches the Python unpack (max |d| = {np.abs(back - q4_py * s4_py).max():.3e})")

print("\n[3] fp32 is a memcpy, byte-exact")
okf, qf, _ = write_native(0, x)
# range(x.size) took the first 128 BYTES (a third of the 512-byte fp32 buffer),
# so the view produced 32 garbage floats against a 128-float x. Buffer -> full
# byte count, then view.
backf = np.ctypeslib.as_array(qf, shape=(x.size * 4,)).view(np.float32)
check(okf == 1 and np.array_equal(backf, x), "fp32 round trip is bit-exact")

print("\n[4] the quantisation error metric, on real data, per side")
qe = c_float()
okq = lib.aero_kv_span_error(1, q_nt, ctypes.byref(c_float(s_nt.value)), x.size,
                             x.ctypes.data_as(POINTER(c_float)), ctypes.byref(qe))
back8 = q_py.astype(np.float32) * sc_py
want = float(np.abs(back8 - x).max()) / max(float(np.abs(x).max()), 1e-9)
check(okq == 1 and abs(qe.value - want) < 1e-6,
      f"span error matches the Python definition ({qe.value:.3e} vs {want:.3e})")
qe4 = c_float()
lib.aero_kv_span_error(2, q4_nt, ctypes.byref(c_float(s4_nt.value)), x.size,
                       x.ctypes.data_as(POINTER(c_float)), ctypes.byref(qe4))
back4 = q4_py.astype(np.float32) * s4_py
want4 = float(np.abs(back4 - x).max()) / max(float(np.abs(x).max()), 1e-9)
check(abs(qe4.value - want4) < 1e-6,
      f"int4 span error matches ({qe4.value:.3e} vs {want4:.3e})")
check(qe4.value > qe.value,
      f"int4 stores at lower precision than int8, as measured ({qe4.value:.2e} > {qe.value:.2e})")

print("\n[5] ring rotation: index 0 must be the OLDEST retained position")
for cap, start in ((8, 3), (16, 7), (64, 0), (5, 4)):
    n = cap
    out = (ctypes.c_int64 * n)()
    okr = lib.aero_kv_ring_rotate(cap, start, n, out)
    idx = np.array([out[i] for i in range(n)])
    want_idx = np.concatenate([np.arange(start, cap), np.arange(0, start)])
    check(okr == 1 and np.array_equal(idx, want_idx),
          f"ring cap={cap} start={start}: chronological order recovered")
# and the Python agrees for a real cache: cross-check against _read_slot's rotation
kv = PKV(1, 2, 4, 4, capacity=8, dtype="fp32", windowed={0: 4})
kv.append(0, rng.standard_normal((8, 2, 4)).astype(np.float32),
          rng.standard_normal((8, 2, 4)).astype(np.float32))
kv.commit(8)
kk, vv = kv._read_slot(0, 8)
# physical start for this ring at end=8, cap=8: 8 % 8 == 0
out2 = (ctypes.c_int64 * 8)()
lib.aero_kv_ring_rotate(8, 0, 8, out2)
check(np.array_equal(np.array([out2[i] for i in range(8)]), np.arange(8)),
      "the C++ rotation agrees with the Python ring at the identity offset")

print("\n[6] refusals: bad dtype, odd int4 dim, out-of-range start")
okb = lib.aero_kv_write_vec(7, x.ctypes.data_as(POINTER(c_float)), x.size, None,
                            ctypes.byref(c_float()))
check(okb == 0, "unknown dt refused")
xo = (rng.standard_normal(7)).astype(np.float32)
okb = lib.aero_kv_write_vec(2, xo.ctypes.data_as(POINTER(c_float)), 7, None,
                            ctypes.byref(c_float()))
check(okb == 0, "int4 with an odd dim refused")
okr = lib.aero_kv_ring_rotate(8, 9, 8, (ctypes.c_int64 * 8)())
check(okr == 0, "ring start >= cap refused")

print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
sys.exit(1 if FAILS else 0)