#!/usr/bin/env python3
"""Aero engine -- self-test.

Proves the device-count flags work in ANY AND ALL COMBINATIONS:
  * every non-empty subset of {cpu, gpu, ane} parses
  * counts are range-checked against each device's real limit
  * all-zero is an error, not a silent fallback
  * CPU counts allocate PERFORMANCE-FIRST, then efficiency
  * every combination actually RUNS and produces the same numbers as a plain
    NumPy reference
  * every device with a non-zero count actually EXECUTED (call counters), so
    "must be used" is measured rather than assumed

SAFETY: weights are tiny (64x32). No model loading, no large allocation.

Run:  .venv-cml/bin/python tests/test_engine.py
"""

from __future__ import annotations

import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import (Device, DeviceCountOutOfRange, NoDevicesSelected,  # noqa: E402
                    all_combinations, build_engine, cpu_topology, format_limits,
                    parse_device_alloc)
from engine.limits import all_limits  # noqa: E402

CHECKS = 0
FAILS = 0


def check(cond: bool, what: str) -> None:
    global CHECKS, FAILS
    CHECKS += 1
    if cond:
        print(f"  ok   {what}")
    else:
        FAILS += 1
        print(f"  FAIL {what}")


# ---- tiny fixture: fast CoreML compiles, exact reference ----------------
N_OUT, N_IN = 64, 32
rng = np.random.default_rng(7)
W = (rng.standard_normal((N_OUT, N_IN)) * 0.1).astype(np.float32)
X = (rng.standard_normal((N_IN,)) * 0.5).astype(np.float32)
REFERENCE = W @ X


def main() -> int:
    print("=== aero engine self-test ===\n")

    LIMITS = all_limits()
    print("detected limits:")
    print(format_limits(LIMITS))
    print()

    # ---------------------------------------------------------------- limits
    print("[1] limits are real, not invented")
    check(LIMITS["cpu"].limit >= 1, f"cpu limit = {LIMITS['cpu'].limit}")
    check(LIMITS["gpu"].limit >= 1, f"gpu limit = {LIMITS['gpu'].limit}")
    check(LIMITS["ane"].limit >= 1, f"ane limit = {LIMITS['ane'].limit}")
    check(LIMITS["cpu"].detected, "cpu limit is DETECTED on this machine")
    check(LIMITS["gpu"].detected, "gpu limit is DETECTED on this machine")
    check(not LIMITS["ane"].detected,
          "ane limit is marked DOCUMENTED (no public API reports it) -- honest labelling")

    topo = cpu_topology()
    print(f"       cpu topology: {topo.performance}P + {topo.efficiency}E = {topo.total}")

    # ------------------------------------------------------- CPU P-then-E
    print("\n[2] CPU counts allocate PERFORMANCE-first, then efficiency")
    check(topo.split(0) == (0, 0), "cpu=0 -> none")
    check(topo.split(1) == (1, 0), "cpu=1 -> 1 performance, 0 efficiency")
    check(topo.split(topo.performance) == (topo.performance, 0),
          f"cpu={topo.performance} -> all performance")
    if topo.efficiency > 0:
        check(topo.split(topo.performance + 1) == (topo.performance, 1),
              f"cpu={topo.performance + 1} -> performance full, then 1 efficiency")
        check(topo.split(topo.total) == (topo.performance, topo.efficiency),
              f"cpu={topo.total} -> all cores")
        check(topo.split(topo.total + 99) == (topo.performance, topo.efficiency),
              "over-allocation clamps to the total")
    print(f"       describe(6) = {topo.describe(min(6, topo.total))}")

    # ------------------------------------------------------------- defaults
    print("\n[3] defaults")
    d = parse_device_alloc(limits=LIMITS)
    check(str(d) == "gpu=1", f"no flags -> {d} (default is --use-gpu)")
    d = parse_device_alloc(use_cpu=None, use_gpu=None, use_ane=None, limits=LIMITS)
    check(str(d) == "gpu=1", "explicit Nones == no flags")

    # ------------------------------------------------------------ validation
    print("\n[4] range validation")
    for name, lim in LIMITS.items():
        try:
            parse_device_alloc(**{f"use_{name}": lim.limit + 1}, limits=LIMITS)
            check(False, f"--use-{name}={lim.limit + 1} must be rejected")
        except DeviceCountOutOfRange:
            check(True, f"--use-{name}={lim.limit + 1} rejected (limit {lim.limit})")
        try:
            parse_device_alloc(**{f"use_{name}": -1}, limits=LIMITS)
            check(False, f"--use-{name}=-1 must be rejected")
        except DeviceCountOutOfRange:
            check(True, f"--use-{name}=-1 rejected")
    try:
        parse_device_alloc(use_cpu=0, use_gpu=0, use_ane=0, limits=LIMITS)
        check(False, "all-zero must raise NoDevicesSelected")
    except NoDevicesSelected:
        check(True, "all-zero raises NoDevicesSelected (not a silent fallback)")

    # ------------------------------------------------- all combinations parse
    print("\n[5] every non-empty combination parses")
    combos = all_combinations(LIMITS)
    check(len(combos) == 7, f"7 non-empty device subsets found (got {len(combos)})")
    for c in combos:
        print(f"       {str(c):<28s} {c.total_streams} stream(s)")
    check(all(c.total_streams > 0 for c in combos), "every combination has >0 streams")

    # ---------------------------------------- all combinations RUN + agree
    print("\n[6] every combination RUNS and matches the NumPy reference")
    ref_norm = float(np.linalg.norm(REFERENCE))
    for c in combos:
        label = str(c)
        try:
            eng = build_engine(use_cpu=c.count(Device.CPU) or None,
                               use_gpu=c.count(Device.GPU) or None,
                               use_ane=c.count(Device.ANE) or None,
                               limits=LIMITS)
            eng.build(W)
            y, stats = eng.matvec(X)
            err = float(np.linalg.norm(y - REFERENCE)) / ref_norm
            ok_shape = y.shape == REFERENCE.shape
            ok_num = err < 5e-2          # fp16 CoreML vs fp32 numpy
            check(ok_shape and ok_num,
                  f"{label:<28s} runs, shape ok, rel.err={err:.2e}")

            # every allocated device must have ACTUALLY executed
            used = eng.devices_used()
            expected = tuple(d.value for d, _ in c)
            check(set(used) == set(expected),
                  f"{label:<28s} all allocated devices executed (used={used})")
        except Exception as e:  # noqa: BLE001
            check(False, f"{label} raised {type(e).__name__}: {e}")
            traceback.print_exc()

    # ------------------------------------------------ partitioning integrity
    print("\n[7] streams partition the rows exactly (no gap, no overlap)")
    eng = build_engine(use_cpu=2, use_gpu=2, use_ane=2, limits=LIMITS)
    if LIMITS["cpu"].limit >= 2 and LIMITS["gpu"].limit >= 2 and LIMITS["ane"].limit >= 2:
        eng.build(W)
        spans = [(s.row_start, s.row_end) for s in eng.streams]
        check(spans[0][0] == 0, "first stream starts at row 0")
        check(spans[-1][1] == N_OUT, "last stream ends at the final row")
        check(all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1)),
              "spans are contiguous with no gap or overlap")
        check(sum(s.n_rows for s in eng.streams) == N_OUT,
              f"row counts sum to {N_OUT}")
        print(f"       {len(eng.streams)} streams: {spans}")
    else:
        print("       (skipped: machine lacks enough cores for a 2/2/2 split)")

    # --------------------------------------------------------- CPU QoS tags
    print("\n[8] CPU streams are tagged performance-first")
    n_cpu = min(6, LIMITS["cpu"].limit)
    eng = build_engine(use_cpu=n_cpu, use_gpu=0, limits=LIMITS)
    if n_cpu >= 1:
        eng.build(W)
        classes = [s.core_class for s in eng.device_streams(Device.CPU)]
        n_perf = sum(1 for c in classes if c == "performance")
        n_eff = sum(1 for c in classes if c == "efficiency")
        check(n_perf == min(n_cpu, topo.performance),
              f"cpu={n_cpu} -> {n_perf} performance streams (expected {min(n_cpu, topo.performance)})")
        check(n_eff == max(0, n_cpu - topo.performance),
              f"cpu={n_cpu} -> {n_eff} efficiency streams (expected {max(0, n_cpu - topo.performance)})")
        print(f"       {classes}")
    else:
        print("       (skipped)")

    print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
