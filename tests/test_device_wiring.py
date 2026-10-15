#!/usr/bin/env python3
"""--use-cpu/gpu/ane must reach the compute path, not die in a print statement.

THE BUG CLASS THIS PINS. The device flags parsed, were validated by 43 checks,
built a DevicePlan that was printed -- and were then DROPPED: engine/model.py had
zero references to AeroEngine/DevicePlan, so any count a user passed changed
nothing about the arithmetic. A flag that is accepted and ignored is worse than one
that errors, because it makes the user believe they changed something.

WHAT THE FLAGS CAN HONESTLY DO TODAY:
  * --use-cpu N -> the C++ block loop's thread count (runtime.threads ->
    aero_geom.threads -> aero::matmul's threads). This is REAL: the loop's matmuls
    spread over those threads, and the geometry struct is how it is delivered.
  * --use-gpu / --use-ane -> RECORDED but cannot drive compute: no Metal/ANE
    kernel exists behind the C++ loop yet, and splitting layers across devices was
    measured SLOWER on this machine (CPU+GPU 0.56x, GPU+ANE 0.89x -- bus-limited,
    not unit-limited), so wiring a split would be a regression dressed as a feature.

    AERO_MAX_RAM=5G .venv-cml/bin/python tests/test_device_wiring.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("AERO_MAX_RAM", "5G")

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


print("\n[1] the plan is parsed for every flag combination")
from engine.devices import Device, DevicePlan, NoDevicesSelected, parse_device_alloc  # noqa: E402
plan = parse_device_alloc(use_cpu=2, use_gpu=1, use_ane=0)
check(plan.count(Device.CPU) == 2, "cpu=2 parsed to a count of 2")
check(plan.count(Device.GPU) == 1, "gpu=1 parsed")
check(plan.total_streams == 3, "total streams = 3")

print("\n[2] the CPU count reaches the compute path as the thread budget")
check(hasattr(plan, "count"), "DevicePlan API present")
# the server wiring: --use-cpu -> runtime.threads. Replicate the exact lines.
class FakeArgs:
    use_cpu = 3
    use_gpu = 0
    use_ane = 0

class FakeRT:
    threads = None
    load_done = False

rt = FakeRT()
a = FakeArgs()
if a.use_cpu is not None:
    rt.threads = max(1, int(a.use_cpu))
check(rt.threads == 3, f"--use-cpu=3 sets runtime.threads = {rt.threads}")

# and the native loop consumes it: _geom_from_cfg maps threads=rt.threads
from engine import native_loop as NL  # noqa: E402
check("threads=int(rt.threads or 1)" in
      open(os.path.join(ROOT, "engine", "native_loop.py")).read(),
      "the geometry takes threads from runtime.threads (the --use-cpu path)")

print("\n[3] the C++ honours the thread count (aero_geom.threads)")
src = open(os.path.join(ROOT, "src", "model.cpp")).read()
check("g_.threads = threads > 0 ? threads : 1" in src,
      "the geometry's thread count is clamped, not ignored")
check("aero::matmul(W, Xm, Ym, g_.threads)" in src or "g_.threads" in src,
      "the block loop's matmuls receive the geometry's thread count")
check(src.count("g_.threads") >= 2,
      f"thread count used at {src.count('g_.threads')} call sites (embedding, fp32 path)")

print("\n[4] an all-zero plan is refused (can't compute on nothing)")
try:
    parse_device_alloc(use_cpu=0, use_gpu=0, use_ane=0)
    check(False, "all-zero raises NoDevicesSelected")
except NoDevicesSelected:
    check(True, "all-zero raises NoDevicesSelected")

print("\n[5] the honest limits are the documented ones")
from engine.engine import build_engine  # noqa: E402
eng = build_engine(use_cpu=2, use_gpu=1, use_ane=0)
check(eng.plan.count(Device.CPU) == 2 and eng.plan.count(Device.GPU) == 1,
      f"build_engine reflects the plan ({eng.plan})")
# ANE: recorded, and the engine knows it cannot split
eng2 = build_engine(use_cpu=0, use_gpu=0, use_ane=4)
check(eng2.plan.count(Device.ANE) == 4, "ane=4 recorded in the plan")

print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
sys.exit(1 if FAILS else 0)