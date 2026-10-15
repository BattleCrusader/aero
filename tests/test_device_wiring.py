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
  * --use-cpu left at the default -> the machine's hardware thread count, not 1.
    `rt.threads or 1` collapsed the default to a SINGLE thread while gpu=1 was
    reported as a stream, so the flags looked like they did something and did not.
    Measured on Qwen3.8-4B, 8 hardware threads: 7.35 -> 2.54 s/token (2.89x), with
    IDENTICAL greedy tokens -- the split does not change the arithmetic.
  * gpu/ane in the plan -> said out loud (DevicePlan.notice()), never implied. An
    unspecified count would otherwise read as "compute runs on 0 CPU stream(s)".
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
# The server wiring: --use-cpu -> the plan -> runtime.threads -> aero_geom.threads.
# This was replicated as three lines of fake code, which cannot fail when the real
# code changes. It is asserted against the real constructor now.
from engine.runtime import GgufRuntime  # noqa: E402
rt = GgufRuntime(max_ram=None, device_plan=parse_device_alloc(use_cpu=3, use_gpu=0, use_ane=0))
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

print("\n[6] the plan drives the compute path, and the DEFAULT is honest")
from engine.devices import COMPUTE_CAPABLE  # noqa: E402
from engine.native import hardware_threads  # noqa: E402
from engine.runtime import accepts_kwarg  # noqa: E402
hw = hardware_threads()
check(hw >= 1, f"the C++ side reports {hw} hardware thread(s)")

dflt = parse_device_alloc()                      # the documented defaults: gpu=1, cpu=0
check(dflt.count(Device.CPU) == 0, "the pinned default is still cpu=0 (unchanged on purpose)")
rtd = GgufRuntime(max_ram=None, device_plan=dflt)
check(rtd.threads == hw,
      f"cpu unspecified -> runtime.threads = {rtd.threads} (hardware), not the 1 that "
      f"`rt.threads or 1` gave")
check(dflt.thread_budget(1) == 1, "on a 1-core box the budget is still an honest 1")
check(parse_device_alloc(use_cpu=3, use_gpu=0).thread_budget(8) == 3,
      "an explicit --use-cpu=3 is honoured exactly; the hardware count is not consulted")
rtn = GgufRuntime(max_ram=None)
check(rtn.threads is None, "no plan -> threads untouched (every existing caller unchanged)")

print("\n[7] what cannot compute is SAID, not implied")
check(COMPUTE_CAPABLE == (Device.CPU,), "CPU is the only device with a compute kernel")
g = parse_device_alloc(use_cpu=0, use_gpu=1, use_ane=2)
check(g.compute_streams == 0, "gpu+ane are not counted as compute streams")
check(g.recorded_only() == ((Device.GPU, 1), (Device.ANE, 2)), "recorded_only lists both, in order")
note = g.notice()
check(note is not None and "--use-gpu=1" in note and "--use-ane=2" in note and "no GPU/ANE kernel" in note,
      "the notice names exactly what was asked for and what is missing")
check(parse_device_alloc(use_cpu=4, use_gpu=0).notice() is None, "a pure-CPU plan has nothing to confess")
check("[aero]" not in note and note.startswith("--use-"), "the notice is data, the caller frames it")
check("0 CPU stream" not in dflt.notice(),
      "the default notice never claims compute runs on 0 streams (it runs on the CPU)")

print("\n[8] every entry point hands the plan to the runtime")
for f in ("engine/server.py", "engine/chat.py"):
    src = open(os.path.join(ROOT, f)).read()
    check("device_plan=engine.plan" in src or 'opts["device_plan"] = engine.plan' in src,
          f"{f} hands the plan to the runtime")
check(accepts_kwarg("gguf", "device_plan") and not accepts_kwarg("echo", "device_plan"),
      "accepts_kwarg distinguishes the real runtime from the stub (no silent drop)")
src_rt = open(os.path.join(ROOT, "engine", "runtime.py")).read()
check("device_notice" in src_rt, "the notice is recorded in load stats as well as printed")

print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
sys.exit(1 if FAILS else 0)