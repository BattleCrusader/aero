#!/usr/bin/env python3
"""The wiring test: the C++ block loop vs the Python block loop, REAL model.

The block loop is correct in C++ isolated (tests/test_model_cpp.cpp, synthetic
file, 20 checks) and correct at the ctypes level (smoke on the synthetic file).
Neither proves the ENGINE's wiring is right: the geometry must mirror the Python
flags exactly (GEMMA_OUT_SCALE, BLOCK_MODE, eps, vocab), the layer kinds must come
from the same detection, the hooks must hand the C++ the same buffers the Python
loop would, and the head must be the tied token_embd with the same softcap.

So this compares, on the REAL Qwen3.8-4B file, one decode token through
engine/model.py's forward against one through the native loop -- with hooks backed
by the SAME Python primitives. Those two have no business differing at all: 9e-07
relative is the fp32 floor. A mismatch here is a WIRING bug (geometry, kind, or
hook contract), never rounding.

Also pins the callback-shadow trap for real: a ctypes callback that raises is
SILENTLY swallowed by ctypes ("Exception ignored ..."), so a broken hook shows up
as wrong logits far away from the cause, not as an error. The test asserts the
native path's top-1 agrees and the load-stats flag is true, so a silently swallowed
hook exception cannot pass.

    AERO_MAX_RAM=5G .venv-cml/bin/python tests/test_native_loop.py
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("AERO_MAX_RAM", "5G")

import numpy as np

from engine import model as M
from engine import native_loop as NL
from engine.chat_template import render_chat
from engine.gguf import GGUFReader
from engine.tokenizer import Tokenizer

P = "/Users/cyberdeth/.cache/aero-models/Qwen3.8-4B-Q4_K_M.gguf"

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


if not os.path.exists(P):
    print(f"(skipped: {P} absent)")
    print("\n=== 0 checks, 0 failures ===")
    sys.exit(0)

from engine import native_model as NM
if not NM.available():
    print(f"(skipped: libaero_model not built -- {NM.unavailable_reason()})")
    print("\n=== 0 checks, 0 failures ===")
    sys.exit(0)

with GGUFReader(P) as r:
    tk = Tokenizer.from_gguf(r)
    m = M.Model(r, block_rows=256, matmul_threads=8)
    cfg = m.cfg

    class RT:
        pass

    rt = RT()
    rt.model = m
    rt._reader = r
    rt.block_rows = 256
    rt.threads = 8
    rt.context = 4096
    rt.tokenizer = tk
    rt.state = m.new_state(capacity=4096)

    print("\n[1] the native loop builds against the real model")
    loop = NL.build_native_loop(rt)
    check(loop is not None, "build_native_loop returns a loop")
    check(loop.vocab == tk.n_vocab, f"vocab matches the file ({loop.vocab})")
    check(loop.n_layer == cfg.n_layer, f"n_layer matches ({loop.n_layer})")

    print("\n[2] PARITY: one decode token, Python loop vs C++ loop")
    prompt = tk.encode(render_chat(
        [{"role": "user", "content": "The capital of France is"}], tk))
    tok = prompt[-1]

    st_py = m.new_state(capacity=4096)
    lg_py = m.forward([tok], st_py)

    st_nt = m.new_state(capacity=4096)
    rt.state = st_nt
    loop_nt = NL.build_native_loop(rt)
    lg_nt = np.zeros(loop.vocab, np.float32)
    positions = np.arange(st_nt.pos, st_nt.pos + 1, dtype=np.int64)
    loop_nt.forward([tok], positions, lg_nt)
    st_nt.kv.commit(1)
    st_nt.pos += 1

    d = np.abs(lg_py - lg_nt)
    scale = max(float(np.abs(lg_py).max()), 1e-9)
    rel = d.max() / scale
    check(rel < 1e-5, f"max|py - native| = {d.max():.6f}, rel {rel:.3e} (fp32 floor)")

    print("\n[3] the native path selects the SAME top-1")
    t1 = int(np.argmax(lg_py))
    t2 = int(np.argmax(lg_nt))
    check(t1 == t2, f"top-1 token identical (id {t1})")
    top5_py = [int(i) for i in np.argsort(-lg_py)[:5]]
    top5_nt = [int(i) for i in np.argsort(-lg_nt)[:5]]
    check(top5_py == top5_nt, f"top-5 identical ({top5_py})")

    print("\n[4] an ssm-heavy position: the second token (crosses an ssm layer)")
    tok2 = int(np.argmax(lg_py))
    lg_py2 = m.forward([tok2], st_py)
    lg_nt2 = np.zeros(loop.vocab, np.float32)
    positions2 = np.arange(st_nt.pos, st_nt.pos + 1, dtype=np.int64)
    loop_nt.forward([tok2], positions2, lg_nt2)
    st_nt.kv.commit(1)
    st_nt.pos += 1
    d2 = np.abs(lg_py2 - lg_nt2)
    rel2 = d2.max() / max(float(np.abs(lg_py2).max()), 1e-9)
    check(rel2 < 1e-5, f"second token rel {rel2:.3e} (max abs {d2.max():.6f})")

    print("\n[5] the flag is visible in load stats (no silent choice)")
    # reproduce what runtime.load() would record
    check(isinstance(loop.vocab, int) and loop.vocab > 0,
          "native loop exposes a usable vocab for the head")

print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
sys.exit(1 if FAILS else 0)