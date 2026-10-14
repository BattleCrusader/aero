#!/usr/bin/env python3
"""Native kernel vs the Python reference, on REAL model weights.

WHY THIS TEST EXISTS. The C++ self-test compares the fused kernel against the
C++ reference dequantiser. Both were written from the same reading of the format,
so when that reading was wrong for Q6_K they AGREED WITH EACH OTHER and the test
passed -- while the model produced garbage. A test that shares its subject's
misreading cannot detect it.

So this one cross-checks the C++ against the PYTHON dequantiser, which is
independently verified against real files by tests/test_quant.py. Two
implementations written separately, compared on real bytes, is the only version
of this test with teeth.

Run: .venv-cml/bin/python tests/test_native.py
"""

from __future__ import annotations

import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine import gguf as G          # noqa: E402
from engine import native             # noqa: E402
from engine import quant as Q         # noqa: E402

CHECKS = FAILS = 0

QWEN = ("/Volumes/Backup/lmstudio/models/empero-ai/"
        "Qwen3.8-4B-Distill-GGUF/Qwen3.8-4B-Q4_K_M.gguf")
_CACHED = glob.glob(os.path.expanduser("~/.cache/aero-models/*.gguf"))


def check(cond, what):
    global CHECKS, FAILS
    CHECKS += 1
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS += 1


def banner(t):
    print(f"\n{t}")


def test_library():
    banner("[1] the native library is built and reports the formats correctly")
    if not native.available():
        check(False, f"library present -- {native.unavailable_reason()}")
        return False
    check(True, "libaero_kernels loaded")
    # block geometry must match the Python side exactly, or every offset is wrong
    for dtype, name in ((12, "Q4_K"), (14, "Q6_K")):
        pv = {12: 256, 14: 256}[dtype]
        pb = {12: 144, 14: 210}[dtype]
        check(native.block_bytes(dtype) == pb and native.block_values(dtype) == pv,
              f"{name}: {native.block_bytes(dtype)} B / {native.block_values(dtype)} "
              f"values, matching the Python side")
        check(Q.ggml_type_size(dtype, pv) == native.block_bytes(dtype),
              f"{name}: total size agrees between C++ and Python")
    check(native.hardware_threads() >= 1,
          f"hardware_threads() = {native.hardware_threads()}")
    return True


def test_real_weights(path):
    banner(f"[2] C++ kernel == Python reference on REAL weights\n    {path}")
    with G.GGUFReader(path) as r:
        # one tensor of each type actually present in this file
        by_type = {}
        for nm, info in r.tensors.items():
            if info.type_id in native.NATIVE_TYPES and len(info.shape) == 2:
                by_type.setdefault(info.type_id, nm)
        if not by_type:
            check(False, "the file has no tensor with a native kernel")
            return
        all_ok = True
        for dtype, nm in sorted(by_type.items()):
            info = r.tensors[nm]
            n_rows, n_cols = info.shape
            raw = r.tensor_bytes(info)
            rb = r.row_bytes(info)
            k = min(64, n_rows)
            x = np.random.default_rng(1).standard_normal(n_cols).astype(np.float32)

            py = Q.dequantize(raw[:k * rb], info.type_id, (k, n_cols))
            want = py @ x
            got = native.matvec(info.type_id, raw[:k * rb], x, 1)
            rel = float(np.abs(got - want).max()) / max(float(np.abs(want).max()), 1e-9)
            ok = rel < 1e-4
            all_ok &= ok
            print(f"       {nm} {info.type_name} {n_rows}x{n_cols} "
                  f"({n_cols // 256} blocks/row)")
            check(ok, f"{info.type_name}: relative error {rel:.2e} vs the Python "
                      f"dequantiser (must be fp32 accumulation, < 1e-4)")

        # a LARGE tensor, to exercise many blocks per row
        big = max((nm for nm, i in r.tensors.items()
                   if i.type_id in native.NATIVE_TYPES and len(i.shape) == 2),
                  key=lambda n: r.tensors[n].n_elements, default=None)
        if big:
            info = r.tensors[big]
            n_rows, n_cols = info.shape
            raw = r.tensor_bytes(info)
            rb = r.row_bytes(info)
            k = min(64, n_rows)
            x = np.random.default_rng(2).standard_normal(n_cols).astype(np.float32)
            want = Q.dequantize(raw[:k * rb], info.type_id, (k, n_cols)) @ x
            got = native.matvec(info.type_id, raw[:k * rb], x, 1)
            rel = float(np.abs(got - want).max()) / max(float(np.abs(want).max()), 1e-9)
            check(rel < 1e-4,
                  f"the largest tensor ({big}, {n_cols // 256} blocks/row) agrees "
                  f"({rel:.2e}) -- this is the case a 1-block fixture misses")
        check(all_ok, "every tensor type in this file matches the Python reference")


def test_threads_are_safe():
    banner("[3] threads are a speed knob, never a correctness one")
    path = _CACHED[0] if _CACHED else (QWEN if os.path.exists(QWEN) else None)
    if path is None:
        print("       (skipped: no model)")
        return
    with G.GGUFReader(path) as r:
        nm = next((n for n, i in r.tensors.items()
                   if i.type_id == native.DTYPE_Q4_K and len(i.shape) == 2), None)
        if nm is None:
            print("       (skipped: no Q4_K tensor)")
            return
        info = r.tensors[nm]
        raw = r.tensor_bytes(info)
        n_rows, n_cols = info.shape
        x = np.random.default_rng(3).standard_normal(n_cols).astype(np.float32)
        base = native.matvec(native.DTYPE_Q4_K, raw, x, 1)
        for th in (2, 4, 8):
            got = native.matvec(native.DTYPE_Q4_K, raw, x, th)
            check(np.allclose(got, base, rtol=1e-5, atol=1e-4),
                  f"threads={th} matches threads=1 ({nm})")


def test_refusals():
    banner("[4] it refuses what it cannot do")
    if not native.available():
        return
    try:
        native.matvec(999, b"\x00" * 288, np.zeros(256, dtype=np.float32))
        check(False, "an unknown dtype must raise")
    except ValueError:
        check(True, "an unknown dtype raises instead of computing garbage")
    # an F32 tensor has no fused kernel -- the caller must fall back
    check(3 not in native.NATIVE_TYPES,
          "F16/Q8_0/other types are NOT in NATIVE_TYPES, so gguf falls back")


def test_fallback_is_honest():
    banner("[5] the engine falls back rather than silently mis-computing")
    import inspect
    src = inspect.getsource(G.GGUFReader.matmul)
    check("native.available()" in src and "NATIVE_TYPES" in src,
          "matmul checks availability and the type before using the kernel")
    check("if read_once:" in src,
          "...and the pure-Python path is still there underneath")


def main():
    print("=== native kernel vs the Python reference ===")
    have = test_library()
    path = _CACHED[0] if _CACHED else (QWEN if os.path.exists(QWEN) else None)
    if have and path:
        test_real_weights(path)
        test_threads_are_safe()
    elif have:
        print("\n[2] (skipped: model not present)")
    test_refusals()
    test_fallback_is_honest()
    print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
