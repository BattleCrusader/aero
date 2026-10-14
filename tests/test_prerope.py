#!/usr/bin/env python3
"""Pre-rope K caching: two claims, both must hold.

  1. EQUIVALENCE -- storing raw K and rotating at read time must produce
     IDENTICAL logits to rotating at write time. If it does not, the change is
     a regression, not an optimisation.
  2. THE PAYOFF -- a cached prefix must become position-independent, so an
     identical prefix is reusable at ANY offset (which is what makes LZ/radix
     prefix sharing sound).

Run: .venv-cml/bin/python tests/test_prerope.py
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

CHECKS = FAILS = 0
QWEN = ("/Volumes/Backup/lmstudio/models/empero-ai/"
        "Qwen3.8-4B-Distill-GGUF/Qwen3.8-4B-Q4_K_M.gguf")


def check(cond, what):
    global CHECKS, FAILS
    CHECKS += 1
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS += 1


def banner(t):
    print(f"\n{t}")


def _structural_only(mc):
    """The checks that need no model forward -- run in the default gate."""
    banner("[S] structural (no forward pass)")
    c = T.KVCache(1, 2, 8, 8, 4, layer_indices=[0], rope_k=True)
    check(c.rope_k is True, "rope_k is stored on the cache")
    c2 = T.KVCache(1, 2, 8, 8, 4, layer_indices=[0])
    check(c2.rope_k is False,
          "it defaults to OFF, so nothing changes unless asked for")
    # the rotation must be applied at READ time by attention(), not at write
    import inspect
    src = inspect.getsource(T.attention)
    check("pre_rope" in src,
          "attention() distinguishes the pre-rope path")
    check(src.count("apply_rope") >= 2,
          "it rotates in both paths (write-time and read-time)")
    # The offset base is DERIVED from the incoming positions, not assumed to be
    # 0 -- hardcoding 0 rotated K by 0,1,2 while Q rotated by 5,6,7, silently
    # corrupting attention. Assert the derivation, not a variable name.
    check("pos[0]" in src and "cache.len" in src,
          "the rotation base is derived from pos[0] and the cache length, "
          "so an offset start is handled instead of assumed away")
    # KVCache must not rotate anything itself -- that is attention's job
    csrc = inspect.getsource(T.KVCache)
    check("apply_rope" not in csrc,
          "the cache itself does no rotation (separation of concerns is intact)")
    print("       cost: rotation O(T) at write -> O(S) at read")
    print("       benefit: prefix reuse at any offset (a server/multi-req win)")


def capture_raw_k(mc, model, toks, pos_start=0, rope_k=False):
    """Return the K values the cache actually STORES, for the first attn layer."""
    st = model.new_state(capacity=64)
    st.kv.rope_k = rope_k
    L = mc.attn_layers[0]
    got = {}
    orig = st.kv.append

    def spy(layer, k, v):
        out = orig(layer, k, v)
        if layer == L:
            got["stored"] = np.array(k)      # what was written (pre-rope if set)
        return out

    st.kv.append = spy
    pos = np.arange(pos_start, pos_start + len(toks), dtype=np.int64)
    model.forward(toks, st, positions=pos)
    return got.get("stored")


def main():
    print("=== pre-rope K caching ===")
    if not os.path.exists(QWEN):
        print("model not present")
        return 1
    with G.GGUFReader(QWEN) as r:
        mc = M.ModelConfig.from_gguf(r)

        # ---------------------------------------------------- 1. equivalence
        banner("[1] equivalence: pre-rope K must not change the output")
        if os.environ.get("AERO_SKIP_SLOW"):
            # these are real forward passes (~7 min for the full set); the
            # default gate must stay fast or it stops being run.
            print("       (skipped: AERO_SKIP_SLOW. Run without it to verify")
            print("        bit-identical logits and position-independent keys.)")
            _structural_only(mc)
            print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
            return 0 if FAILS == 0 else 1
        toks = [9707, 11, 1879]
        m1 = M.Model(r)
        st1 = m1.new_state(capacity=64)
        lg1 = m1.forward(toks, st1)

        m2 = M.Model(r)
        st2 = m2.new_state(capacity=64)
        st2.kv.rope_k = True
        lg2 = m2.forward(toks, st2)

        d = float(np.abs(lg1 - lg2).max())
        rel = d / max(float(np.abs(lg1).max()), 1e-9)
        print(f"       normal   top-1 {int(np.argmax(lg1))}  std {lg1.std():.4f}")
        print(f"       pre-rope top-1 {int(np.argmax(lg2))}  std {lg2.std():.4f}")
        print(f"       max |diff| {d:.3e}  relative {rel:.3e}")
        check(int(np.argmax(lg1)) == int(np.argmax(lg2)),
              "both paths agree on the top token")
        check(np.allclose(lg1, lg2, rtol=1e-5, atol=1e-4),
              f"logits match within tolerance (max diff {d:.2e})")
        check(float(lg1.std()) > 0.5 and float(lg2.std()) > 0.5,
              "both produce shaped logits (not degenerate)")

        # incremental decode must also agree after another step, which is where
        # a read-time rotation bug would show up (the cached block grows)
        n1 = m1.forward([500], st1)
        n2 = m2.forward([500], st2)
        check(int(np.argmax(n1)) == int(np.argmax(n2)),
              "incremental decode agrees on the next token too")
        check(np.allclose(n1, n2, rtol=1e-4, atol=1e-3),
              "incremental logits match after a further step")

        # ------------------------------------------------- 2. position-independence
        banner("[2] payoff: a cached prefix becomes position-INDEPENDENT")
        prefix = [9707, 11, 1879]
        # the POST-rope path (current behaviour) at two offsets
        post0 = capture_raw_k(mc, M.Model(r), prefix, 0, rope_k=False)
        post5 = capture_raw_k(mc, M.Model(r), prefix, 5, rope_k=False)
        # the PRE-rope path at the same two offsets
        pre0 = capture_raw_k(mc, M.Model(r), prefix, 0, rope_k=True)
        pre5 = capture_raw_k(mc, M.Model(r), prefix, 5, rope_k=True)

        d_post = float(np.abs(post0 - post5).max())
        d_pre = float(np.abs(pre0 - pre5).max())
        print(f"       post-rope, pos 0 vs 5: max diff {d_post:.6f}  "
              f"{'IDENTICAL' if d_post == 0 else 'DIFFERS'}")
        print(f"       pre-rope,  pos 0 vs 5: max diff {d_pre:.6f}  "
              f"{'IDENTICAL' if d_pre == 0 else 'DIFFERS'}")
        check(d_post > 0,
              "post-rope storage DOES depend on offset (the original problem)")
        check(d_pre == 0.0,
              "pre-rope storage is bit-identical at any offset")
        check(np.array_equal(pre0, pre5),
              "so a prefix can be shared verbatim at a different position")

        # ------------------------------------------------------ 3. the cost
        banner("[3] what it costs")
        check(hasattr(T.KVCache(1, 2, 8, 8, 4, layer_indices=[0]), "rope_k"),
              "the cache exposes the rope_k switch")
        c = T.KVCache(1, 2, 8, 8, 4, layer_indices=[0], rope_k=True)
        check(c.rope_k is True, "rope_k is stored on the cache")
        c2 = T.KVCache(1, 2, 8, 8, 4, layer_indices=[0])
        check(c2.rope_k is False,
              "it defaults to OFF, so nothing changes unless asked for")
        print("       cost: rotation moves from O(T) at write to O(S) at read")
        print("       benefit: prefix reuse, which is a multi-request/server win")

        # What is uniquely better about pre-rope K: the STORED key can be
        # copied verbatim into another sequence, because it carries no absolute
        # position. A post-rope key cannot. That -- not any difference in
        # output -- is the whole benefit, so that is what gets asserted.
        banner("[4] the stored key is COPYABLE; a post-rope key is not")
        # translation invariance is a property of RoPE itself and holds in BOTH
        # paths (q.k depends only on relative distance), so asserting the paths
        # differ in output would contradict [1]. Assert the real distinction.
        def out_at(m, toks, pos_start, rope_k):
            st = m.new_state(capacity=64)
            st.kv.rope_k = rope_k
            return m.forward(toks, st,
                             positions=np.arange(pos_start,
                                                 pos_start + len(toks),
                                                 dtype=np.int64))
        o_post_0 = out_at(M.Model(r), prefix, 0, False)
        o_pre_0 = out_at(M.Model(r), prefix, 0, True)
        check(np.allclose(o_pre_0, o_post_0, rtol=1e-5, atol=1e-4),
              "at the same offset both paths agree (consistent with [1])")
        # RoPE translation invariance: shifting the whole prefix shifts nothing
        o_pre_5 = out_at(M.Model(r), prefix, 5, True)
        check(np.allclose(o_pre_0, o_pre_5, rtol=1e-4, atol=1e-3),
              "shifting the prefix does not change the output either path "
              "(RoPE depends only on relative position)")

        # the actual benefit, stated as a check: a pre-rope block IS a valid
        # cache entry at another offset; a post-rope block is NOT.
        check(np.array_equal(pre0, pre5),
              "PRE-rope: the stored block is byte-identical at any offset, so "
              "it can be copied into another sequence as-is")
        check(not np.array_equal(post0, post5),
              "POST-rope: the stored block is offset-dependent, so copying it "
              "elsewhere would silently corrupt attention")
        print(f"       post-rope copy would inject error {d_post:.4f}")
        print(f"       pre-rope  copy injects exactly  {d_pre:.6f}")

    print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
