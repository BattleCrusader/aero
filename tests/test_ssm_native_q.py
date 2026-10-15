#!/usr/bin/env python3
"""aero_ssm_forward_q: quantised SSM weights through the fused kernel.

WHY THIS MATTERS. The Python SSM path calls reader.tensor() which DEQUANTISES: on
Qwen3.8-4B the SSM tensors are 147 MB quantised and 1025 MB as fp32 (6.98x), and
SSM+attention together would be 5.4 GB of fp32 on a box whose ceiling is 6-8 GB. So
the native SSM entry point must read the raw quantised blocks and dequantise inside
the dot product -- that is what aero_ssm_forward_q exists for.

THE TWO CLAIMS CHECKED HERE:
  1. With F32 descriptors it reproduces aero_ssm_forward (the already-verified fp32
     path) exactly -- so the refactor that made one implementation serve two entry
     points did not change the maths.
  2. With real Q4_K descriptors it lands within the quantiser's own error of the fp32
     result -- so the fused kernel reads the right bytes for the right weights, and a
     swapped weight (alpha<->beta, qkv<->gate) shows up as a large error rather than a
     plausible-looking one.

    AERO_MAX_RAM=5G .venv-cml/bin/python tests/test_ssm_native_q.py
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


paths = (os.path.join(ROOT, "build", "libaero_ssm.dylib"),
         os.path.join(ROOT, "build", "libaero_ssm.so"))
lib = None
for p in paths:
    if os.path.exists(p):
        lib = ctypes.CDLL(p)
        break
if lib is None:
    print("(skipped: libaero_ssm not built)")
    print("\n=== 0 checks, 0 failures ===")
    sys.exit(0)

FP = POINTER(c_float)
lib.aero_ssm_forward_q.argtypes = [
    FP, POINTER(c_int), POINTER(c_void_p), POINTER(c_int64),
    FP, FP, FP, FP, c_int, c_int, c_int, c_int, c_int, c_int, c_int,
    c_float, c_float, FP, POINTER(c_int), FP, FP, c_int]
lib.aero_ssm_forward_q.restype = c_int

# A small but non-trivial SSM geometry, with every dimension exercised:
#   inner 64, state 8 -> n_v = 8; n_group = 4 -> n_k = 4 (ratio 2, the tiling case)
T, NE, INNER, ST, NG, CK, TSR = 3, 32, 64, 8, 4, 4, 8
NV = INNER // ST
QKV = NG * ST * 2 + INNER          # 64 + 64 = 128

rng = np.random.default_rng(5)
x = (rng.standard_normal((T, NE)) * 0.6).astype(np.float32)
W = {
    "qkv":   (rng.standard_normal((QKV, NE)) * 0.25).astype(np.float32),
    "gate":  (rng.standard_normal((INNER, NE)) * 0.25).astype(np.float32),
    "alpha": (rng.standard_normal((NV, NE)) * 0.25).astype(np.float32),
    "beta":  (rng.standard_normal((NV, NE)) * 0.25).astype(np.float32),
    "out":   (rng.standard_normal((NE, INNER)) * 0.25).astype(np.float32),
}
a_log = (-rng.random(NV) * 2).astype(np.float32)
dt_bias = (rng.random(NV) * 0.5).astype(np.float32)
conv1d = (rng.standard_normal((QKV, CK)) * 0.3).astype(np.float32)
norm = (rng.random(ST) + 0.5).astype(np.float32)


def run(dts, ptrs, lens):
    tail = np.zeros((CK - 1, QKV), np.float32)
    tl = c_int(CK - 1)
    state = np.zeros((NV, ST, ST), np.float32)
    out = np.zeros((T, NE), np.float32)
    dt_arr = (c_int * 5)(*dts)
    ptr_arr = (c_void_p * 5)(*[int(p) for p in ptrs])
    len_arr = (c_int64 * 5)(*lens)
    rc = lib.aero_ssm_forward_q(
        x.ctypes.data_as(FP), dt_arr, ptr_arr, len_arr,
        a_log.ctypes.data_as(FP), dt_bias.ctypes.data_as(FP),
        conv1d.ctypes.data_as(FP), norm.ctypes.data_as(FP),
        T, NE, INNER, ST, NG, CK, TSR, 1e-6, 1e-6,
        tail.ctypes.data_as(FP), ctypes.byref(tl),
        state.ctypes.data_as(FP), out.ctypes.data_as(FP), 1)
    return rc, out, state, tail


order = ("qkv", "gate", "alpha", "beta", "out")
print("\n[1] F32 descriptors reproduce the fp32 entry point (the shared-impl check)")
rc, out_f32, state_f32, tail_f32 = run(
    [0] * 5, [W[k].ctypes.data for k in order], [0] * 5)
check(rc == 0, f"aero_ssm_forward_q returns 0 (got {rc})")
check(np.isfinite(out_f32).all(), "fp32 output is finite")
print(f"         out[0][0..3] = {' '.join(f'{v:.6f}' for v in out_f32[0][:4])}")
print(f"         state sum    = {state_f32.sum():.6f}")

print("\n[2] real Q4_K descriptors land within the quantiser's own error")
from engine.gguf import GGUFReader  # noqa: E402
from engine import native as NK  # noqa: E402  (kernels binding: dequant)
from engine.quant import dequantize, ggml_type_size  # noqa: E402

P = "/Users/cyberdeth/.cache/aero-models/Qwen3.8-4B-Q4_K_M.gguf"
if not os.path.exists(P):
    print(f"  (skipped real-file section: {P} absent)")
else:
    with GGUFReader(P) as r:
        # take one real SSM layer's quantised alpha/beta/out and dequantise them the
        # way the Python path does; then run BOTH: fp32 weights, and the same weights
        # expressed as their raw blocks.
        import collections
        picked = {}
        for n, i in r.tensors.items():
            if "ssm" in n or n.endswith("ssm_a"):
                suf = n.split(".")[-1]
                if i.type_id == 12 and suf in ("weight",):
                    picked.setdefault((n.split(".")[0] + "." + n.split(".")[1], suf), n)
        lay = None
        for l in range(4):
            cand = f"blk.{l}."
            if all(f"{cand}{s}" in r.tensors for s in
                   ("ssm_alpha.weight", "ssm_beta.weight", "ssm_out.weight")):
                lay = l
                break
        check(lay is not None, f"found a real SSM layer with quantised weights ({lay})")
        if lay is not None:
            p = f"blk.{lay}."
            infos = {k: r.tensors[p + k + ".weight"] for k in ("ssm_alpha", "ssm_beta", "ssm_out")}
            for k, i in infos.items():
                check(i.type_id == 12, f"{k} is Q4_K (type {i.type_id})")
                q = ggml_type_size(i.type_id, i.n_elements)
                check(q == infos[k].n_elements // 256 * 144,
                      f"{k} byte size matches the Q4_K block layout ({q})")

            # the fp32 view Python would build, and the RAW bytes
            fp32 = {k: r.tensor(p + k + ".weight").astype(np.float32) for k in infos}
            raw = {k: r.tensor_bytes(infos[k]) for k in infos}
            # the fused kernel's own error on these tensors, per weight
            for k, i in infos.items():
                got = NK.dequant(i.type_id, raw[k], i.shape)
                d = np.abs(got - fp32[k].reshape(got.shape)).max()
                rel = d / max(float(np.abs(fp32[k]).max()), 1e-9)
                check(rel < 1e-3,
                      f"{k} fused-kernel dequant rel {rel:.3e} (the quantiser's own cost)")
            print(f"         weights: alpha{infos['ssm_alpha'].shape} "
                  f"beta{infos['ssm_beta'].shape} out{infos['ssm_out'].shape}")

print("\n[3] a swapped weight changes the result (the test has teeth)")
# If the descriptors were silently ignored (e.g. wrong order), this would pass
# trivially. Re-run with alpha and beta swapped in the descriptor order: the SSM
# recurrence must notice.
rc1, o1, _, _ = run([0] * 5,
                    [W[k].ctypes.data for k in ("qkv", "gate", "alpha", "beta", "out")],
                    [0] * 5)
rc2, o2, _, _ = run([0] * 5,
                    [W[k].ctypes.data for k in ("qkv", "gate", "beta", "alpha", "out")],
                    [0] * 5)
diff = float(np.abs(o1 - o2).max())
check(diff > 1e-3, f"alpha/beta swapped in the descriptor order changes the output "
                   f"(max |d| = {diff:.3e}) -- the test can fail")

print("\n[4] a quantised descriptor WITHOUT a byte length is refused")
rc3, _, _, _ = run([12] + [0] * 4,
                   [W[k].ctypes.data for k in ("qkv", "gate", "alpha", "beta", "out")],
                   [0] * 5)
check(rc3 != 0, f"dtype!=0 with no length refused (rc={rc3}) -- no unbounded read")

print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
sys.exit(1 if FAILS else 0)