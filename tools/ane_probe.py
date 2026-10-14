#!/usr/bin/env python3
"""ANE bandwidth probe.

Measures what Apple's Neural Engine actually delivers for the op shape that
matters here: a large streaming matvec (y = W x), i.e. every weight read once.

Same access shape as tools/aero-bwprobe.mm, so the numbers are comparable to
the Metal/CPU measurements (GPU 54.20 GB/s, CPU 15.23 GB/s on this box).

CoreML is the ONLY public door to the ANE -- Apple exposes no way to submit a
custom kernel to it. So this measures the best case available: let CoreML place
the op where it likes, and see what we get.

SAFETY: weights are ~134 MB. Small, bounded, safe on a 16 GB machine.
"""

import sys
import threading
import time
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types as miltypes

# ---- problem size: a weight matrix big enough to be bandwidth-dominated ----
N = 8192          # output dim
K = 8192          # input dim
W_BYTES = N * K * 2           # fp16
X_BYTES = K * 2
BYTES_PER_CALL = W_BYTES + X_BYTES   # every weight read once per call

print(f"weight matrix {N}x{K} fp16 = {W_BYTES/1e6:.1f} MB")
print(f"bytes streamed per inference = {BYTES_PER_CALL/1e6:.1f} MB\n")

rng = np.random.default_rng(0)
W = (rng.standard_normal((N, K)) * 0.02).astype(np.float16)
x = (rng.standard_normal((K,)) * 0.5).astype(np.float16)


@mb.program(input_specs=[mb.TensorSpec(shape=(K,), dtype=miltypes.fp16)])
def prog(x_in):
    w = mb.const(val=W, name="W")
    return mb.linear(x=x_in, weight=w, name="y")


def build(compute_unit):
    t0 = time.perf_counter()
    ml = ct.convert(
        prog,
        convert_to="mlprogram",
        compute_units=compute_unit,
        minimum_deployment_target=ct.target.macOS15,
    )
    return ml, time.perf_counter() - t0


def bench(ml, iters=20):
    # warm up (first predict includes lazy init / ANE compile)
    for _ in range(3):
        ml.predict({"x_in": x})
    best = 0.0
    for _ in range(iters):
        t0 = time.perf_counter()
        ml.predict({"x_in": x})
        dt = time.perf_counter() - t0
        if dt > 0:
            best = max(best, BYTES_PER_CALL / dt / 1e9)
    return best


UNITS = [
    ("CPU_ONLY", ct.ComputeUnit.CPU_ONLY),
    ("CPU_AND_GPU", ct.ComputeUnit.CPU_AND_GPU),
    ("CPU_AND_NE  (ANE)", ct.ComputeUnit.CPU_AND_NE),
    ("ALL", ct.ComputeUnit.ALL),
]

print("=== TEST A: one large op (a single device must do it all) ===")
results = {}
for name, unit in UNITS:
    try:
        ml, ctime = build(unit)
        bw = bench(ml)
        results[name] = bw
        print(f"  {name:22s} {bw:7.2f} GB/s   (compiled in {ctime:.1f}s)")
    except Exception as e:  # noqa: BLE001
        print(f"  {name:22s} FAILED: {type(e).__name__}: {str(e)[:120]}")

# ---------------------------------------------------------------------------
# TEST B: many INDEPENDENT ops. This is the real concurrency test -- a single op
# can only be placed on one device, so it can never show what parallelism buys.
# With independent branches the scheduler is free to spread them across units.
# ---------------------------------------------------------------------------
BRANCHES = 8
WB = (N // BRANCHES, K)   # per-branch weight
Wb = [np.ascontiguousarray(W[i * WB[0]:(i + 1) * WB[0]]) for i in range(BRANCHES)]


@mb.program(input_specs=[mb.TensorSpec(shape=(K,), dtype=miltypes.fp16)])
def prog_multi(x_in):
    outs = []
    for i, w in enumerate(Wb):
        c = mb.const(val=w, name=f"W{i}")
        outs.append(mb.linear(x=x_in, weight=c, name=f"y{i}"))
    return mb.concat(values=outs, axis=0, name="y")


print(f"\n=== TEST B: {BRANCHES} independent ops (scheduler may spread them) ===")
results_multi = {}
for name, unit in UNITS:
    try:
        ml, ctime = build(unit)
        bw = bench(ml)
        results_multi[name] = bw
        print(f"  {name:22s} {bw:7.2f} GB/s   (compiled in {ctime:.1f}s)")
    except Exception as e:  # noqa: BLE001
        print(f"  {name:22s} FAILED: {type(e).__name__}: {str(e)[:120]}")

# ---------------------------------------------------------------------------
# TEST C: GPU + ANE running CONCURRENTLY.
# CoreML exposes NO "GPU+ANE" compute-unit mode (the enum is only ALL,
# CPU_AND_GPU, CPU_AND_NE, CPU_ONLY), and ALL drags in the slow CPU (~8 GB/s),
# so ALL cannot isolate GPU+ANE. Instead: run two models side by side, one
# pinned to the GPU and one to the ANE, and measure AGGREGATE throughput.
# If the two units add, aggregate > either alone. If they contend, it won't be.
# ---------------------------------------------------------------------------
print("\n=== TEST C: GPU and ANE running CONCURRENTLY (side-by-side models) ===")
try:
    ml_gpu_ = build(ct.ComputeUnit.CPU_AND_GPU)[0]
    ml_ne_ = build(ct.ComputeUnit.CPU_AND_NE)[0]

    for _ in range(3):  # warm up both
        ml_gpu_.predict({"x_in": x})
        ml_ne_.predict({"x_in": x})

    def solo(ml, iters=15):
        best = 0.0
        for _ in range(iters):
            t0 = time.perf_counter()
            ml.predict({"x_in": x})
            dt = time.perf_counter() - t0
            if dt > 0:
                best = max(best, BYTES_PER_CALL / dt / 1e9)
        return best

    s_gpu = solo(ml_gpu_)
    s_ne = solo(ml_ne_)

    best_agg = 0.0
    for _ in range(15):
        t0 = time.perf_counter()
        th = threading.Thread(target=ml_ne_.predict, args=({"x_in": x},))
        th.start()
        ml_gpu_.predict({"x_in": x})   # this thread drives the GPU
        th.join()                      # that thread drove the ANE
        dt = time.perf_counter() - t0
        if dt > 0:
            best_agg = max(best_agg, 2 * BYTES_PER_CALL / dt / 1e9)

    base = max(s_gpu, s_ne)
    print(f"  GPU alone          : {s_gpu:7.2f} GB/s")
    print(f"  ANE alone          : {s_ne:7.2f} GB/s")
    print(f"  GPU + ANE together : {best_agg:7.2f} GB/s  -> {best_agg/base:.2f}x of the best single unit")
    if best_agg / base < 1.15:
        print("  -> NO aggregate gain. GPU and ANE contend for the same memory bus.")
        print("     (Apple not offering a GPU+ANE mode is consistent with this.)")
    else:
        print("  -> THEY ADD. GPU+ANE IS a real lever at this op shape.")
except Exception as e:  # noqa: BLE001
    print(f"  FAILED: {type(e).__name__}: {str(e)[:160]}")

print("\n=== verdict ===")
if results:
    best = max(results.items(), key=lambda kv: kv[1])
    print(f"  single-op best unit : {best[0]} at {best[1]:.2f} GB/s")
    print(f"  reference: Metal 54.20 GB/s, CPU 15.23 GB/s (measured separately)")
    ne = results.get("CPU_AND_NE  (ANE)")
    if ne is not None:
        print(f"  ANE-targeted        : {ne:.2f} GB/s")
if results_multi:
    bm_ = max(results_multi.items(), key=lambda kv: kv[1])
    print(f"  multi-op  best      : {bm_[0]} at {bm_[1]:.2f} GB/s")
    gpu_m = results_multi.get("CPU_AND_GPU")
    all_m = results_multi.get("ALL")
    ne_m = results_multi.get("CPU_AND_NE  (ANE)")
    if gpu_m and all_m:
        print(f"\n  CONCURRENCY TEST: GPU-only {gpu_m:.2f} vs ALL(any unit) {all_m:.2f}"
              f"  -> {all_m/gpu_m:.2f}x")
        if all_m / gpu_m < 1.15:
            print("  -> enabling every unit buys <15%. The units share one memory bus,")
            print("     so they contend rather than add. Multi-unit is NOT a lever.")
        else:
            print("  -> units DO add throughput here. Investigate further.")
    if ne_m:
        print(f"  ANE-only on multi-op: {ne_m:.2f} GB/s")


