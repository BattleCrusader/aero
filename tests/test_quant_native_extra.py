#!/usr/bin/env python3
"""Q8_0 and Q5_K: bit-exact against the Python dequantiser, and fused-matvec parity.

WHY THESE TWO. They are not a completion exercise: they are the types actually present
in the models on disk. Ornith-1.5-9B-MTP-IQ4_XS uses 8 tensors of Q8_0 (type 8) and 32
of Q5_K (type 13) beside its IQ4_XS, and the C++ supported neither -- so Ornith could not
run through the engine at all. The ternary types (PQ2_0=142, PTQ1_0=143) belong to the
27B Bonsai files, which are deliberately never run on this 16 GB box, so they are NOT
implemented and their absence is not a gap in coverage of anything runnable.

Every check is on SYNTHETIC blocks (random bytes): no model load, no lock, seconds.

    AERO_MAX_RAM=5G .venv-cml/bin/python tests/test_quant_native_extra.py
"""
from __future__ import annotations

import os
import struct
import sys

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


from engine import native as NK  # noqa: E402
from engine import quant as Q  # noqa: E402

if not NK.available():
    print("(skipped: libaero_kernels not built)")
    print("\n=== 0 checks, 0 failures ===")
    sys.exit(0)

rng = np.random.default_rng(4)


def f16(v: float) -> bytes:
    return struct.pack("<e", v)


for label, type_id, block_bytes, block_vals in (
        ("Q8_0 (type 8)", 8, 34, 32),
        ("Q5_K (type 13)", 13, 176, 256)):
    print(f"\n[{label}]")
    check(type_id in NK.NATIVE_TYPES,
          f"declared native (NATIVE_TYPES includes {type_id})")
    check(NK._BLOCK_BYTES[type_id] == block_bytes and
          NK._BLOCK_VALUES[type_id] == block_vals,
          f"block geometry {block_bytes} bytes / {block_vals} values")

    # several blocks, with real-looking scale halves (a small d, a modest min)
    NB = 4
    rnd = rng.integers(0, 256, size=NB * block_bytes, dtype=np.uint8).tobytes()
    # overwrite the fp16 fields with sane values so the dequant is not denormal soup
    buf = bytearray(rnd)
    for b in range(NB):
        off = b * block_bytes
        buf[off:off + 2] = f16(0.02 * (b + 1))
        if type_id == 13:
            buf[off + 2:off + 4] = f16(0.01 * (b + 1))
    raw = bytes(buf)

    got = NK.dequant(type_id, raw, (NB * block_vals,))
    want = Q.dequantize(raw, type_id, (NB * block_vals,))
    want = np.asarray(want, np.float32).reshape(-1)
    d = np.abs(got.reshape(-1) - want).max()
    check(np.array_equal(got.reshape(-1), want),
          f"dequant is BIT-EXACT vs the Python reference (max |d| = {d})")
    check(np.isfinite(got).all(), "output is finite (no denormal/NaN)")

    # the fused path must agree with dequant-then-matmul: a mismatch here means the
    # fused inner loop and the reference decode the same bytes differently
    n_rows, n_cols = NB, block_vals
    x = (rng.standard_normal(n_cols) * 0.5).astype(np.float32)
    y_fused = NK.matvec(type_id, raw, x)
    y_ref = want.reshape(n_rows, n_cols) @ x.astype(np.float64)
    rel = float(np.abs(y_fused.astype(np.float64) - y_ref).max() /
                max(float(np.abs(y_ref).max()), 1e-9))
    check(rel < 1e-5,
          f"fused matvec agrees with dequant-then-matmul (rel {rel:.3e})")

    # and a wrong byte length must be refused rather than read past the buffer
    try:
        NK.dequant(type_id, raw[:block_bytes], (NB * block_vals,))
        check(False, "a short buffer is refused")
    except (ValueError, OSError):
        check(True, "a short buffer is refused (no unbounded read)")

print("\n[c] the ternary types stay unimplemented, deliberately")
for t, name in ((142, "PQ2_0"), (143, "PTQ1_0")):
    check(t not in NK.NATIVE_TYPES,
          f"{name} (type {t}) is NOT native -- its files are the 27B Bonsai models, "
          f"which are never run on this 16 GB host")

print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
sys.exit(1 if FAILS else 0)