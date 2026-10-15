#!/usr/bin/env python3
"""Cross-check our dequantizers against the ggml authors' own implementation.

WHY THIS FILE EXISTS. Every other quant test compares the C++ to OUR Python -- two
implementations written from the same (mis)understanding. That is why a wrong Q5_K
5th-bit mask survived a "bit-exact vs the Python reference" claim and shipped: both
sides were wrong the same way, and the test built its input blocks from the same wrong
formula. The only thing that catches this class of bug is an INDEPENDENT implementation,
which gguf-py (shipped with llama.cpp, written by the people who defined the formats)
is. It found the Q5_K bug at rel 0.624 on a real tensor.

It SKIPS (exit 0, loudly) when gguf-py or the model files are absent, so the gate stays
green on a machine without them -- but on this machine it checks real weights.

    AERO_MAX_RAM=5G .venv-cml/bin/python tests/test_quant_oracle.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("AERO_MAX_RAM", "5G")

FAILS = 0
CHECKS = 0
SKIPS = 0


def check(cond, what):
    global FAILS, CHECKS
    CHECKS += 1
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS += 1


CANDIDATES = [
    "/Volumes/Backup/Development/PrismML-llama.cpp/gguf-py",
    "/Volumes/Backup/Development/llama.cpp/gguf-py",
    "/Volumes/Backup/Development/ggml-llama.cpp/gguf-py",
]
MODELS = [
    ("Ornith-1.5-9B-MTP (IQ4_XS)", "/Users/cyberdeth/.cache/aero-models/Ornith-1.5-9B-MTP-IQ4_XS.gguf"),
    ("Qwen3.8-4B (Q4_K_M)", "/Users/cyberdeth/.cache/aero-models/Qwen3.8-4B-Q4_K_M.gguf"),
]
LABEL = {23: "IQ4_XS(23)", 13: "Q5_K(13)", 8: "Q8_0(8)", 14: "Q6_K(14)", 12: "Q4_K(12)"}

gguf_py = next((p for p in CANDIDATES if os.path.isfile(os.path.join(p, "gguf", "quants.py"))), None)
if gguf_py is None:
    print("=== 0 checks, 0 failures (SKIPPED: gguf-py not found) ===")
    sys.exit(0)

sys.path.insert(0, gguf_py)
import numpy as np                                        # noqa: E402
import gguf.quants as QQ                                  # noqa: E402
from gguf.constants import GGMLQuantizationType as QT      # noqa: E402
from engine.gguf import GGUFReader                         # noqa: E402
from engine import native                                 # noqa: E402
from engine.quant import dequantize as _EQdeq             # noqa: E402

EQ = sys.modules['engine.quant']

print("oracle: %s" % gguf_py)
for label, path in MODELS:
    if not os.path.isfile(path):
        SKIPS += 1
        print("  skip %s (absent)" % label)
        continue
    print("\n[%s]" % label)
    with GGUFReader(path) as r:
        first = {}
        for nm, ti in r.tensors.items():
            if ti.type_id not in first:
                first[ti.type_id] = (nm, ti)
        checked = 0
        for tid in sorted(first):
            if tid not in LABEL:
                continue
            name, ti = first[tid]
            # BOUNDED ON PURPOSE. Dequantizing a 1 GB tensor with numpy costs minutes and
            # the gate deserves to stay fast. Four windows spread across the tensor (start,
            # thirds, end) hit every 64-group alignment and every scale slot; the
            # exhaustive fingerprint test covers the layout itself. BOTH of our paths are
            # checked -- the Python reference and the C++ kernel -- because they were
            # written from the same belief and only this oracle can tell.
            blk_bytes = {23: 136, 13: 176, 12: 144, 8: 34, 14: 210}[tid]
            vals = {23: 256, 13: 256, 12: 256, 8: 32, 14: 256}[tid]
            whole = np.frombuffer(r.tensor_bytes(ti), dtype=np.uint8)
            nblocks = len(whole) // blk_bytes
            span = 96
            starts = sorted({0, max(0, nblocks // 3), max(0, 2 * nblocks // 3), max(0, nblocks - span)})
            worst_py = worst_cpp = 0.0
            for b0 in starts:
                b1 = min(nblocks, b0 + span)
                win = whole[b0 * blk_bytes:b1 * blk_bytes]
                shape = ((b1 - b0) * vals,)
                ref = np.asarray(QQ.dequantize(win, QT(tid))).reshape(-1)
                py = np.asarray(EQ.dequantize(bytes(win), tid, shape), dtype=np.float32).reshape(-1)
                cpp = np.asarray(native.dequant(tid, bytes(win), shape), dtype=np.float32).reshape(-1)
                worst_py = max(worst_py, float(np.abs(ref - py).max()))
                worst_cpp = max(worst_cpp, float(np.abs(ref - cpp).max()))
            check(worst_py == 0.0,
                  "%-11s %-22s python  %d win x %d blk  max|d|=%.2e (must be 0)"
                  % (LABEL[tid], name[:22], len(starts), span, worst_py))
            check(worst_cpp == 0.0,
                  "%-11s %-22s C++     %d win x %d blk  max|d|=%.2e (must be 0)"
                  % (LABEL[tid], name[:22], len(starts), span, worst_cpp))
            checked += 1
        check(checked >= 1, "at least one type cross-checked on real weights")

print("\n=== %d checks, %d failures%s ==="
      % (CHECKS, FAILS, (", %d skipped" % SKIPS) if SKIPS else ""))
sys.exit(1 if FAILS else 0)
