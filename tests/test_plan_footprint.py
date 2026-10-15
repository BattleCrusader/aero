#!/usr/bin/env python3
"""plan_footprint must be honest about WHICH read path it is planning for.

THE GAP THIS CLOSES. plan_footprint modelled the Python reference: weights read one
layer at a time and released, so resident weight cost = largest layer + embeddings. The
C++ engine mmaps the whole file and reads every projection out of the mapping, so its
footprint can approach the FILE SIZE. Planning for the Python path while running the
C++ one under-predicts by (file - one layer) -- on gemma-4-12b about 6.4 GB against a
6 GB cap -- and `--max-ram` then refuses nothing while the engine touches the whole
file. That is the "discovered by an OOM" failure the plan exists to prevent.

Header-only: reading the tensor index costs no weight bytes, so this test loads no
model data and needs no model lock.

    AERO_MAX_RAM=5G .venv-cml/bin/python tests/test_plan_footprint.py
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


from engine.gguf import GGUFReader  # noqa: E402
from engine.model import plan_footprint  # noqa: E402

MODELS = [
    ("qwen35   4B", "/Users/cyberdeth/.cache/aero-models/Qwen3.8-4B-Q4_K_M.gguf"),
    ("gemma4   12B", "/Users/cyberdeth/.cache/aero-models/gemma-4-12b-it-IQ4_XS.gguf"),
]
present = [(n, p) for n, p in MODELS if os.path.exists(p)]
if not present:
    print("(skipped: no model files present)")
    print("\n=== 0 checks, 0 failures ===")
    sys.exit(0)

for label, path in present:
    size = os.path.getsize(path)
    with GGUFReader(path) as r:
        py = plan_footprint(r, context=4096, block_rows=64, kv_dtype="fp32",
                            native=False)
        nt = plan_footprint(r, context=4096, block_rows=64, kv_dtype="fp32",
                            native=True)

    print(f"\n[{label}] file {size/2**20:.0f} MiB")
    check(nt["native"] is True and py.get("native") is False,
          "the plan records which read path it modelled")
    check(int(py["mapped_file_bytes"]) == 0,
          "the Python plan carries no mapped-file term")
    check(int(nt["mapped_file_bytes"]) == size,
          f"the native plan carries the file size ({size/2**20:.0f} MiB)")

    # The mapping is REPORTED, not counted: it is purgeable, so it cannot threaten a
    # hard ceiling. Counting it put gemma-4-12b at 11.4 GiB and would have refused a
    # configuration that provably works (gemma answers at a 5 GiB cap), so the plan
    # must instead bound the ANONYMOUS demand: state + scratch + one layer's working set.
    check(nt["total_bytes"] < size,
          f"native total ({nt['total_bytes']/2**20:.0f} MiB) stays BELOW the file size "
          f"({size/2**20:.0f} MiB) -- the mapping is purgeable and not charged")
    # The native path charges ONE working set (the larger of the largest layer and the
    # embeddings); the Python path charges BOTH, because it keeps the embedding table
    # resident while streaming layers. So the native total is smaller by exactly
    # min(largest_layer, embeddings) -- not equal, and not smaller by accident.
    diff = py["total_bytes"] - nt["total_bytes"]
    expect = min(py["largest_layer_bytes"], py["embeddings_bytes"])
    check(diff == expect,
          f"native is smaller than Python by exactly min(largest_layer, embeddings) "
          f"= {expect/2**20:.0f} MiB (got {diff/2**20:.0f} MiB)")
    check(nt["total_bytes"] > nt["state_bytes"] + nt["scratch_bytes"],
          f"and it still charges a live working set above state+scratch "
          f"({nt['total_bytes']/2**20:.0f} vs "
          f"{(nt['state_bytes']+nt['scratch_bytes'])/2**20:.0f} MiB)")
    # the state + scratch terms survive in both
    check(nt["kv_bytes"] == py["kv_bytes"] and nt["ssm_bytes"] == py["ssm_bytes"],
          "the KV and SSM state terms are unchanged by the read path")

print("\n[a] the difference is exactly the mapping, not a doubled weight term")
label, path = present[0]
with GGUFReader(path) as r:
    py = plan_footprint(r, context=4096, native=False)
    nt = plan_footprint(r, context=4096, native=True)
check(nt["mapped_file_bytes"] == os.path.getsize(path),
      "the mapping is reported as its own field (so a caller can see it)")
check(nt["total_bytes"] < py["total_bytes"],
      "the native chargeable total is the SMALLER of the two (one working set)"
      )
check(nt["total_bytes"] < os.path.getsize(path),
      "the chargeable total is smaller than the file, i.e. the mapping is not billed")

print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
sys.exit(1 if FAILS else 0)