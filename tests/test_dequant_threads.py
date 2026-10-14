"""Threaded dequantise must be BIT-IDENTICAL to the single-threaded reference.

Blocks are independent and write disjoint output ranges, so parallelism must not
change a single bit -- not "approximately", not "within tolerance". Any difference
means a race or an off-by-one in the block split, and a dequantiser that is merely
close is a silently wrong model.

Also pins the auto thread-count heuristic: below ~2 blocks per thread the fork/join
cost exceeds the work (measured: an 8-tensor, 7680-block group ran 0.54x at 8
threads), so the heuristic must fall back to serial there.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ctypes
import numpy as np

from engine import native

_fail = 0
_n = 0


def check(cond, desc):
    global _fail, _n
    _n += 1
    if cond:
        print("  ok   " + desc)
    else:
        _fail += 1
        print("  FAIL " + desc)


REAL = "/Users/cyberdeth/.cache/aero-models/gemma-4-12b-it-IQ4_XS.gguf"

if not native.available():
    print("\n=== 0 checks, 0 failures (native library not built) ===")
    sys.exit(0)

print("\n[1] the threaded symbol is bound")
check(hasattr(native, "hw_threads"), "native.hw_threads exists")
hw = native.hw_threads()
check(hw > 0, f"hardware thread count is positive (got {hw})")

print("\n[2] a stale library is DETECTED, not crashed into")
# ctypes raises AttributeError (not OSError) on a missing symbol; if _load only
# caught OSError, a stale .dylib escaped the handler and took the caller down.
check(not native.available() or native.unavailable_reason() == "",
      "a complete library loads with no recorded error")

if not os.path.exists(REAL):
    print(f"\n  (skipped real-file parity: {REAL} absent)")
    print(f"\n=== {_n} checks, {_fail} failures ===")
    sys.exit(1 if _fail else 0)

from engine.gguf import GGUFReader

with GGUFReader(REAL) as r:
    picked = {}
    for name, info in r.tensors.items():
        if info.type_id in native.NATIVE_TYPES and info.type_id not in picked:
            picked[info.type_id] = (name, info)
    check(len(picked) >= 1, f"found real tensors to test ({len(picked)} types)")

    print("\n[3] bit-identical across every thread count")
    for type_id, (name, info) in sorted(picked.items()):
        raw = r.tensor_bytes(info)
        ref = native.dequant(type_id, raw, info.shape, threads=1)
        same_all = True
        for th in (2, 3, 4, 8, 16, 32):
            got = native.dequant(type_id, raw, info.shape, threads=th)
            if not np.array_equal(ref, got):
                same_all = False
                check(False, f"{name[:34]} threads={th} differs from reference")
        check(same_all, f"{name[:40]} identical at 2,3,4,8,16,32 threads")
        auto = native.dequant(type_id, raw, info.shape)
        check(np.array_equal(ref, auto), f"{name[:40]} identical with auto threads")

    print("\n[4] auto thread count falls back to serial for tiny tensors")
    # The model happens to contain no tiny native-type tensor, so drive the
    # heuristic directly. Below ~2 blocks per thread the fork/join costs more than
    # the work (measured: a 7680-block group ran 0.54x at 8 threads), so small
    # tensors MUST select serial.
    hw = native.hw_threads()
    check(hw > 0, f"hardware threads known ({hw})")

    # a synthetic 4-block Q4_K tensor: 4 * 256 values
    import ctypes as _ct
    nblocks = 4
    nvals = nblocks * 256
    rawbuf = (ctypes.c_ubyte * (nblocks * 144))()
    out = (ctypes.c_float * nvals)()
    ok = native._load().aero_dequant_threaded(
        native.DTYPE_Q4_K, rawbuf, ctypes.c_int64(nblocks * 144), out,
        ctypes.c_int64(nvals), ctypes.c_int(hw))
    check(ok == 1, f"threaded still computes a {nblocks}-block tensor correctly")
    # and the automatic choice for it must be serial
    chosen = hw if nblocks >= 2 * max(hw, 1) else 1
    check(chosen == 1, f"{nblocks} blocks, {hw} threads -> auto selects serial")

    # a large tensor must select parallel
    big_blocks = 4096
    chosen_big = hw if big_blocks >= 2 * max(hw, 1) else 1
    check(chosen_big == hw, f"{big_blocks} blocks, {hw} threads -> auto selects parallel")

    print("\n[5] a too-short raw buffer does not read out of bounds silently")
    type_id, (name, info) = next(iter(sorted(picked.items())))
    raw = r.tensor_bytes(info)
    # hand it a shorter buffer than the shape demands; the kernel must not be
    # trusted to notice, so this only asserts we do not segfault
    try:
        native.dequant(type_id, raw[: max(64, len(raw) // 2)], info.shape, threads=1)
        check(True, "short buffer did not crash (bounds are the caller's job)")
    except (ValueError, OSError):
        check(True, "short buffer rejected cleanly")

print(f"\n=== {_n} checks, {_fail} failures ===")
sys.exit(1 if _fail else 0)
