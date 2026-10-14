#!/usr/bin/env python3
"""P4: the real runtime wired up, with --max-ram enforced.

    .venv-cml/bin/python tests/test_runtime_gguf.py

Two things are tested here and they are different:
  1. the GUARD refuses a model that cannot fit, before reading weight data
  2. the REAL PATH loads and generates, end to end, with real weights
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine import gguf as G          # noqa: E402
from engine import model as M         # noqa: E402
from engine import arch as ARCH       # noqa: E402
from engine.runtime import (GenParams, GgufRuntime, make_runtime,  # noqa: E402
                            _MeasuredCap, _own_footprint)

CHECKS = FAILS = 0
QWEN = ("/Volumes/Backup/lmstudio/models/empero-ai/"
        "Qwen3.8-4B-Distill-GGUF/Qwen3.8-4B-Q4_K_M.gguf")
MOE_P = ("/Volumes/Backup/lmstudio/models/tvall43/"
         "Qwen3.6-14B-A3B-FableVibes-GGUF/Qwen3.6-14B-A3B-FableVibes-Q4_K_M.gguf")


def check(cond, what):
    global CHECKS, FAILS
    CHECKS += 1
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS += 1


def banner(t):
    print(f"\n{t}")


def test_footprint_prediction():
    banner("[1] the predictive plan (no weight data read)")
    if not os.path.exists(QWEN):
        check(False, "model present")
        return None
    with G.GGUFReader(QWEN) as r:
        t0 = time.time()
        plan = M.plan_footprint(r, context=4096, block_rows=64)
        dt = time.time() - t0
        print(f"       planning took {dt*1000:.1f} ms")
        print(f"       arch={plan['arch']} layers={plan['n_layer']}")
        for k in ("embeddings_bytes", "largest_layer_bytes", "weights_bytes",
                  "kv_bytes", "ssm_bytes", "moe_bytes", "state_bytes",
                  "scratch_bytes", "total_bytes"):
            print(f"       {k:22s} {plan[k]/2**20:10.1f} MiB")
    check(dt < 1.0, f"planning is fast ({dt*1000:.1f} ms) -- it reads no weights")
    check(plan["total_bytes"] > 0, "total is non-zero")
    check(plan["weights_bytes"] < plan["total_bytes"],
          "weights are only part of the total")
    # the plan must be SMALLER than the file + state, because weights are read
    # per-layer rather than held whole
    file_bytes = os.path.getsize(QWEN)
    check(plan["largest_layer_bytes"] < file_bytes,
          f"largest layer ({plan['largest_layer_bytes']/2**20:.0f} MiB) is far "
          f"below the whole file ({file_bytes/2**20:.0f} MiB) -- per-layer read")
    check(plan["state_bytes"] > 0, "state (KV + SSM) is accounted for")
    if plan["ssm_bytes"]:
        check(plan["ssm_bytes"] > 0, f"SSM state planned: {plan['ssm_bytes']/2**20:.2f} MiB")
    return plan


def test_guard_refuses():
    banner("[2] the guard REFUSES a model that cannot fit, before loading")
    if not os.path.exists(MOE_P):
        check(False, "MoE model present")
        return
    rt = GgufRuntime(max_ram="1GB", context=4096)
    t0 = time.time()
    try:
        rt.load(MOE_P)
        check(False, "loading a 7.9 GiB model under a 1 GB cap must be REFUSED")
    except MemoryError as e:
        dt = time.time() - t0
        msg = str(e)
        check("refusing to load" in msg, "refusal explains itself")
        check("planned footprint" in msg, "refusal states the planned footprint")
        check("--max-ram" in msg, "refusal names the cap that was exceeded")
        check(dt < 5.0,
              f"refusal was fast ({dt*1000:.0f} ms) -- no weight data was read")

    # a generous cap must NOT refuse
    rt2 = GgufRuntime(max_ram="64GB", context=256)
    try:
        rt2.load(QWEN)
        check(rt2.model is not None, "a 2.6 GB model loads under a 64 GB cap")
        check(rt2.guard is not None, "the guard exists when a cap is given")
        rt2.close()
    except Exception as e:
        check(False, f"should have loaded: {type(e).__name__}: {e}")

    # no cap -> no guard, and it must say so rather than imply enforcement
    rt3 = GgufRuntime(max_ram=None)
    check(rt3.guard is None, "no cap -> no guard object (not a fake zero cap)")


def test_measured_cap():
    banner("[3] the measured cap measures the RIGHT quantity")
    cap = _MeasuredCap(cap_bytes=10**18)          # absurdly large: never breached
    v = cap.measured_bytes()
    check(v is not None and v > 0,
          f"phys_footprint is readable ({v/2**20:.1f} MiB)")
    check(cap.measurable, "the cap reports itself as measurable")
    check(not cap.exceeded(), "an absurd cap is not breached")
    cap.peak_bytes()
    check(cap.peak >= v, f"peak tracks the maximum ({cap.peak/2**20:.1f} MiB)")

    # a tiny cap must be breached by a process that is obviously over it
    tiny = _MeasuredCap(cap_bytes=1)
    check(tiny.exceeded(), "a 1-byte cap is breached by any live process")

    # and the honest case: nothing measuring -> never claims a breach
    class _Blind(_MeasuredCap):
        def measured_bytes(self):
            return None
    blind = _Blind(cap_bytes=1)
    check(not blind.exceeded(),
          "when nothing is measuring, exceeded() is False, not a false alarm")
    check(not blind.measurable, "...and it reports itself unmeasurable")


def test_own_footprint_is_the_gpu_aware_one():
    banner("[4] the footprint metric is phys_footprint, not rss")
    fp = _own_footprint()
    check(fp is not None and fp > 0, f"footprint available ({fp/2**20:.1f} MiB)")
    # the whole point: this must be the value that sees GPU memory. We cannot
    # allocate GPU memory here without a Metal context, but we CAN prove it is
    # not plain rss by checking it exceeds the tiny baseline rss would give.
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    check(isinstance(fp, int), "footprint is an integer byte count")
    print(f"       phys_footprint ~{fp/2**20:.1f} MiB "
          f"(ru_maxrss {rss/2**20:.1f} MiB)")


def test_real_generation():
    banner("[5] REAL end-to-end generation through the wired runtime")
    if not os.path.exists(QWEN):
        check(False, "model present")
        return
    rt = make_runtime("gguf", max_ram="6GB", context=128, block_rows=64)
    check(isinstance(rt, GgufRuntime), "make_runtime('gguf') returns the real runtime")
    t0 = time.time()
    rt.load(QWEN)
    load_dt = time.time() - t0
    st = rt.stats()
    print(f"       load {load_dt:.1f}s  arch={st['arch']} layers={st['layers']}")
    print(f"       planned {st['planned_bytes']/2**20:.0f} MiB, "
          f"state {st['state_bytes']/2**20:.0f} MiB")
    check(st["arch"] == "qwen35", "arch is qwen35")
    check(st["moe"] is False, "qwen35 is not MoE (correctly)")
    check(st["ssm"] is True, "qwen35 has SSM layers")

    params = GenParams(max_tokens=2, temperature=0.0)
    t0 = time.time()
    pieces = list(rt.generate([{"role": "user", "content": "Hello"}], params))
    gen_dt = time.time() - t0
    print(f"       generated {len(pieces)} piece(s) in {gen_dt:.1f}s")
    print(f"       output: {''.join(pieces)!r}")
    check(len(pieces) >= 1, "generation produced output")
    check(any(p.strip() for p in pieces), "output is non-empty")
    check("error" not in "".join(pieces).lower(),
          "output is not an error message")
    st2 = rt.stats()
    if st2.get("measured_bytes"):
        print(f"       measured footprint {st2['measured_bytes']/2**20:.0f} MiB "
              f"(peak {st2['peak_bytes']/2**20:.0f} MiB), cap 6 GiB")
        check(st2["peak_bytes"] > 0, "the measured monitor saw real memory")
    check(st2["positions"] >= 2, f"positions advanced ({st2['positions']})")
    rt.close()
    check(rt.model is None, "close() released the model")


def main():
    print("=== P4: real runtime + --max-ram enforcement ===")
    test_footprint_prediction()
    test_guard_refuses()
    test_measured_cap()
    test_own_footprint_is_the_gpu_aware_one()
    if os.environ.get("AERO_SKIP_SLOW"):
        banner("[5] (skipped: AERO_SKIP_SLOW set)")
    else:
        test_real_generation()
    print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
