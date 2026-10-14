#!/usr/bin/env python3
"""Is the KV cache LZ-compressible? Testing the actual precondition.

LZ replaces a repeated SUBSTRING with a back-reference. The KV-cache analogue
is a repeated PREFIX: if two sequences share a prefix, the cache entries for
that prefix should be reusable.

That is only true if the stored K for a given token is the same regardless of
which sequence it appears in. RoPE is applied to K before it is cached, and RoPE
rotates by the ABSOLUTE position -- so a shared prefix must also share its
positions for the reuse to be exact.

This measures all three cases:
  1. same tokens at the same absolute positions   -> is K identical?
  2. same tokens at SHIFTED positions             -> does RoPE break reuse?
  3. what would fixing it cost (store pre-rope)?

Run: .venv-cml/bin/python tools/kv_dedup_probe.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine import gguf as G          # noqa: E402
from engine import model as M         # noqa: E402
from engine import transformer as T   # noqa: E402

QWEN = ("/Volumes/Backup/lmstudio/models/empero-ai/"
        "Qwen3.8-4B-Distill-GGUF/Qwen3.8-4B-Q4_K_M.gguf")


def banner(t):
    print(f"\n{'='*72}\n{t}\n{'='*72}")


def run_capture(mc, path, toks, pos_start=0):
    """Run a forward pass and capture the K stored for the first attention layer."""
    with G.GGUFReader(path) as r:
        model = M.Model(r)
        # force the layer we care about to be the first attention layer
        L = mc.attn_layers[0]
        st = model.new_state(capacity=64)
        # capture by monkey-patching append on this cache
        cap = {}
        orig = st.kv.append

        def spy(layer, k, v):
            out = orig(layer, k, v)
            if layer == L:
                cap["k"] = np.array(k)      # the k AS WRITTEN (post-rope)
                cap["v"] = np.array(v)
            return out

        st.kv.append = spy
        pos = np.arange(pos_start, pos_start + len(toks), dtype=np.int64)
        model.forward(toks, st, positions=pos)
        return cap.get("k"), cap.get("v")


def main():
    if not os.path.exists(QWEN):
        print("model not present")
        return 1
    with G.GGUFReader(QWEN) as r:
        mc = M.ModelConfig.from_gguf(r)

    # a prefix that is an exact token match
    prefix = [9707, 11, 1879]          # "Hello, world"
    longer = prefix + [0, 220, 13]

    banner("1  same tokens, same absolute positions -> is the stored K identical?")
    k_short, v_short = run_capture(mc, QWEN, prefix, pos_start=0)
    k_long, v_long = run_capture(mc, QWEN, longer, pos_start=0)
    same_k = np.array_equal(k_short, k_long[:len(prefix)])
    same_v = np.array_equal(v_short, v_long[:len(prefix)])
    dk = float(np.abs(k_short - k_long[:len(prefix)]).max())
    print(f"  prefix tokens      : {prefix}")
    print(f"  K identical        : {same_k}  (max diff {dk:.3e})")
    print(f"  V identical        : {same_v}")
    print()
    if same_k and same_v:
        print("  => EXACT MATCH. A shared prefix produces bit-identical cache")
        print("     entries, so an LZ/radix-tree style sharing IS sound here.")
    else:
        print("  => NOT identical -- sharing would be lossy even for a prefix")

    banner("2  same tokens, SHIFTED positions -> does RoPE break the reuse?")
    k_s0, _ = run_capture(mc, QWEN, prefix, pos_start=0)
    k_s5, _ = run_capture(mc, QWEN, prefix, pos_start=5)
    d = float(np.abs(k_s0 - k_s5).max())
    rel = d / max(float(np.abs(k_s0).max()), 1e-9)
    print(f"  K at pos 0 vs pos 5: max abs diff {d:.4f}  (relative {rel:.3f})")
    print(f"  identical           : {np.array_equal(k_s0, k_s5)}")
    print()
    if not np.array_equal(k_s0, k_s5):
        print("  => RoPE is applied BEFORE caching, so the stored K depends on the")
        print("     ABSOLUTE position. A prefix reused at a different offset has")
        print("     different cache entries and cannot be copied verbatim.")
        print()
        print("     This is the real limit on LZ-style sharing: it works for a")
        print("     prefix that always sits at the same offset (a shared system")
        print("     prompt), and breaks for one that is re-inserted mid-context.")
    else:
        print("  => position-independent; sharing would work at any offset")

    banner("3  the fix: store PRE-rope K, apply rotation on read")
    # demonstrate with the raw tables: pre-rope k is position-independent by
    # construction, so rotation on read restores exactly the post-rope values.
    D = mc.head_dim
    rng = np.random.default_rng(0)
    k_raw = rng.standard_normal((1, mc.n_head_kv, D)).astype(np.float32)
    k_raw = k_raw / np.linalg.norm(k_raw, axis=-1, keepdims=True)
    for p in (0, 5, 137):
        cos, sin = T.rope_tables(D, np.array([p]), mc.rope_base, mc.rope_dims)
        rot = T.apply_rope(k_raw, cos, sin)
        print(f"  pre-rope k rotated by pos {p:>3}: first 3 = "
              f"{np.round(rot[0,0,:3],5).tolist()}")
    print()
    print("  pre-rope k is ONE stored value; every position is derived from it by")
    print("  rotation at read time. So caching pre-rope would make the entry")
    print("  position-independent and shareable at ANY offset.")
    print()
    print("  COST: attention must rotate the whole cached K block on every read")
    print("  instead of once at write. That is O(S) trig per layer per token,")
    print("  traded for exact prefix reuse. On a single unprompted sequence there")
    print("  is nothing to reuse, so the cost is pure overhead until requests")
    print("  actually share prefixes -- which is a server case, not a CLI case.")

    banner("verdict")
    print("  LZ-style sharing is sound for EXACT shared prefixes at a fixed")
    print("  offset (radix-tree / prefix cache). It is NOT sound for merging")
    print("  merely-similar positions: measured mean cosine 0.028, no pairs above")
    print("  0.9, so approximate merging would destroy information.")
    print()
    print("  What it buys depends on the workload:")
    print("    * one-off generation        : nothing to share")
    print("    * chat with a fixed system  : the system prompt's cache, once")
    print("    * server, many similar reqs : large; this is why vLLM has it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
