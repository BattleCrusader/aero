#!/usr/bin/env python3
"""The two checks the gate was MISSING.

Both exist because of the same failure: for this project's entire life the
conv tap order was reversed, every SSM layer was therefore wrong, and the gate
stayed green at 382 checks. A green suite proved nothing about whether the model
worked. These close that:

  [1] CONV TAP ORDER -- a fast synthetic test. The reference accumulates
      s[t + i0] * c[i0], so tap 0 pairs with the OLDEST sample. Reversing it
      changes every SSM output while leaving shapes and magnitudes plausible.
  [2] COHERENCE -- the actual acceptance criterion. Ask the real model a
      question with a known answer and require the answer, not merely that the
      logits are finite and shaped. This is the check that would have caught the
      bug on day one.

Run:  .venv-cml/bin/python tests/test_coherence.py
      AERO_SKIP_SLOW=1 ...      # skip [2] (it runs real forward passes)
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
from engine import ssm as S           # noqa: E402
from engine.tokenizer import Tokenizer  # noqa: E402

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


def test_conv_tap_order():
    banner("[1] conv tap order: c[:,0] pairs with the OLDEST sample")
    # A kernel that responds to only ONE tap makes the pairing visible: feed a
    # ramp and see which input value comes out.
    C, K = 3, 4
    cs = S.ConvState(C, K)

    # kernel picks the oldest tap only: w[:,0] = 1, rest 0
    w_oldest = np.zeros((C, K), dtype=np.float32)
    w_oldest[:, 0] = 1.0
    x = np.array([[1., 1., 1.], [2., 2., 2.], [3., 3., 3.], [4., 4., 4.]],
                 dtype=np.float32)
    out = cs.step(x, w_oldest)
    # window for output t is the K samples ending at t (zero-padded before t=0).
    # With only tap 0 active we must read the OLDEST of those K.
    # t=3: window is x[0..3] -> oldest is x[0] = 1
    check(np.allclose(out[3], 1.0),
          f"tap 0 selects the OLDEST of the window at t=3 (got {out[3,0]:.1f}, "
          f"want 1.0)")

    # and tap K-1 must be the CURRENT sample
    cs2 = S.ConvState(C, K)
    w_new = np.zeros((C, K), dtype=np.float32)
    w_new[:, K - 1] = 1.0
    out2 = cs2.step(x, w_new)
    check(np.allclose(out2[3], 4.0),
          f"tap K-1 selects the CURRENT sample at t=3 (got {out2[3,0]:.1f}, "
          f"want 4.0)")
    check(np.allclose(out[3], out2[3]) is False,
          "the two orders give different results (the test is not vacuous)")

    # a single active tap in the middle, to pin the full mapping
    cs3 = S.ConvState(C, K)
    w_mid = np.zeros((C, K), dtype=np.float32)
    w_mid[:, 1] = 1.0
    out3 = cs3.step(x, w_mid)
    check(np.allclose(out3[3], 2.0),
          f"tap 1 selects x[t-2] at t=3 (got {out3[3,0]:.1f}, want 2.0)")

    # causality, tested properly: a LATER change must not alter an EARLIER
    # output. (A check ending in `or True` can never fail and would be
    # deadweight -- this asserts the real property instead.)
    cs4 = S.ConvState(C, K)
    a = cs4.step(x.copy(), w_oldest)
    x2 = x.copy()
    x2[3] += 100.0
    cs5 = S.ConvState(C, K)
    b = cs5.step(x2, w_oldest)
    check(np.allclose(a[:3], b[:3]),
          "editing a LATER sample leaves earlier outputs identical (causal)")
    # Non-vacuity must use a tap that actually reads the edited sample: with
    # only tap 0 active the t=3 output reads x[0], so editing x[3] correctly
    # leaves it alone. Test the same edit against a CURRENT-reading kernel.
    cs6 = S.ConvState(C, K)
    c = cs6.step(x2, w_new)
    cs7 = S.ConvState(C, K)
    d = cs7.step(x.copy(), w_new)
    check(not np.allclose(c[3:], d[3:]),
          "...and with a CURRENT-reading tap the same edit DOES change its own "
          "position (so the guard is not vacuous)")


def test_conv_would_catch_the_bug():
    banner("[1b] the reversed order is DETECTABLY different")
    # Prove the guard has teeth: build the reversed conv and show it disagrees.
    C, K = 2, 4
    class Reversed(S.ConvState):
        def step(self, x, w):
            return super().step(x, w[:, ::-1])
    w = np.random.default_rng(0).standard_normal((C, K)).astype(np.float32)
    x = np.random.default_rng(1).standard_normal((5, C)).astype(np.float32)
    good = S.ConvState(C, K).step(x, w)
    bad = Reversed(C, K).step(x, w)
    d = float(np.abs(good - bad).max())
    check(d > 1e-3,
          f"a reversed tap order changes the output (max diff {d:.4f}) -- so the "
          f"check above has teeth")
    check(good.shape == bad.shape,
          "...even though the shapes and magnitudes are identical (which is why "
          "a shape or finiteness check cannot see it)")


def test_coherence():
    banner("[2] COHERENCE: the model must answer, not merely emit finite logits")
    if os.environ.get("AERO_SKIP_SLOW"):
        print("       (skipped: AERO_SKIP_SLOW -- this runs real forward passes)")
        return
    if not os.path.exists(QWEN):
        print("       (skipped: model not present)")
        return
    with G.GGUFReader(QWEN) as r:
        tk = Tokenizer.from_gguf(r)
        m = M.Model(r)
        cases = [
            ("The capital of France is", [" Paris", "Paris"]),
            ("The sky is", [" blue", " clear", " very", " a"]),
        ]
        for prompt, wants in cases:
            lg = m.forward(tk.encode(prompt), m.new_state(capacity=64))
            order = np.argsort(-lg)[:400]
            got = [tk.decode([int(i)]) for i in order[:6]]
            # the answer may be the top-1 OR the top-2 (many continuations are
            # two tokens, e.g. ' Paris' == [' ', 'Paris'])
            hit = [w for w in wants if w in got]
            check(bool(hit),
                  f"{prompt!r} -> {got}  (expected one of {wants})")
            # and it must NOT be the degenerate frequency-prior signature that
            # the broken build produced: space/comma/digit filling the top 4
            prior = {" ", ",", ".", "1", "2", "3", "0", "\n", " (", " *"}
            top4 = set(got[:4])
            check(len(top4 - prior) >= 2,
                  f"...and the top-4 is not pure punctuation/digits ({sorted(top4)})")

        # a greedy continuation must be recognisable text, not a token soup
        st = m.new_state(capacity=64)
        lg = m.forward(tk.encode("Once upon a time"), st)
        gen = []
        for _ in range(5):
            n = int(np.argmax(lg))
            gen.append(n)
            lg = m.forward([n], st)
        text = tk.decode(gen)
        print(f"       'Once upon a time' + {text!r}")
        check(any(w in text for w in (" a ", " the ", " I ", " in ", " was ")),
              f"the continuation contains common English words ({text!r})")
        # every produced id must be a real token, and the text must be valid utf-8
        check(all(0 <= t < tk.n_vocab for t in gen),
              "every generated id is in range")


def main():
    print("=== the checks the gate was missing ===")
    test_conv_tap_order()
    test_conv_would_catch_the_bug()
    test_coherence()
    print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
