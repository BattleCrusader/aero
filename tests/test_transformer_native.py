#!/usr/bin/env python3
"""Native transformer primitives vs the PYTHON reference, function by function.

WHY THIS TEST EXISTS. tests/test_transformer_cpp.cpp checks the C++ against
scalar references written in C++. That is necessary but not sufficient: both were
written from the same reading of the maths, so a shared misreading would agree
with itself and pass. This test cross-checks the C++ against
engine/transformer.py -- an independently written implementation -- on IDENTICAL
random inputs, and prints the observed max abs and REL difference for every
primitive rather than asserting "it matches".

Run:  .venv-cml/bin/python tests/test_transformer_native.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine import transformer as T          # noqa: E402
from engine import transformer_native as N   # noqa: E402

CHECKS = 0
FAILS = 0
ROWS: list[tuple[str, float, float]] = []

# The tolerance is fp32 accumulation order (numpy's einsum/sum may reduce in a
# different order than a serial C loop), NOT algorithmic error. Anything at or
# below this is the same computation; anything above it is a bug.
RTOL = 1e-5


def check(cond: bool, what: str) -> None:
    global CHECKS, FAILS
    CHECKS += 1
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS += 1


def banner(t: str) -> None:
    print(f"\n{t}")


def compare(name: str, want, got) -> tuple[float, float]:
    """Compare two float32 arrays, record and return (max abs, max rel)."""
    a = np.asarray(want, dtype=np.float32).reshape(-1)
    b = np.asarray(got, dtype=np.float32).reshape(-1)
    assert a.shape == b.shape, f"{name}: shape {a.shape} vs {b.shape}"
    absd = float(np.abs(a - b).max()) if a.size else 0.0
    den = max(float(np.abs(a).max()), 1e-9)
    rel = absd / den
    ROWS.append((name, absd, rel))
    return absd, rel


def main() -> int:
    print("=== native transformer primitives vs the Python reference ===")
    if not N.available():
        check(False, f"libaero_transformer present -- {N.unavailable_reason()}")
        print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
        return 1
    check(True, "libaero_transformer loaded")
    rng = np.random.default_rng(20261015)

    # ------------------------------------------------------------- rope_tables
    banner("[1] rope_tables (full, partial, proportional RoPE)")
    for label, nd, ff in (
        ("full", None, None),
        ("partial-32", 32, None),
        ("full + freq_factors", None, np.concatenate(
            [np.ones(32, np.float32), np.full(32, 1e30, np.float32)])),
    ):
        head_dim = 64
        positions = rng.integers(0, 500, size=7).astype(np.int32)
        wc, ws = T.rope_tables(head_dim, positions.astype(np.float32), 10000.0, nd,
                               freq_factors=ff)
        gc, gs = N.rope_tables(head_dim, positions, 10000.0, nd, freq_factors=ff)
        dc = compare(f"rope_tables[{label}] cos", wc, gc)
        ds = compare(f"rope_tables[{label}] sin", ws, gs)
        check(max(dc[1], ds[1]) < RTOL,
              f"{label}: cos {dc[0]:.2e}/{dc[1]:.2e}, sin {ds[0]:.2e}/{ds[1]:.2e} "
              f"(abs/rel)")
        check(gc.shape == wc.shape, f"{label}: table shape matches {wc.shape}")

    # ------------------------------------------------------------- apply_rope
    banner("[2] apply_rope (neox and interleaved, plus partial)")
    Tn, H, D = 5, 3, 16
    x = rng.standard_normal((Tn, H, D)).astype(np.float32)
    positions = np.repeat(np.arange(Tn), H).astype(np.int32)
    ctab, stab = T.rope_tables(D, np.arange(Tn, dtype=np.float32), 10000.0)
    for mode, nmode in (("neox", N.ROPE_NEOX), ("interleaved", N.ROPE_INTERLEAVED)):
        want = T.apply_rope(x, ctab, stab, mode=mode)
        got = N.apply_rope(x, ctab, stab, positions=positions, mode=nmode)
        a, r = compare(f"apply_rope[{mode}]", want, got)
        check(r < RTOL, f"{mode}: max abs {a:.2e}, rel {r:.2e}")
    # partial rotary: table built for 8 dims, only 8 channels rotate
    c8, s8 = T.rope_tables(D, np.arange(Tn, dtype=np.float32), 10000.0, 8)
    want = T.apply_rope(x, c8, s8, mode="neox")
    got = N.apply_rope(x, c8, s8, positions=positions, mode=N.ROPE_NEOX)
    a, r = compare("apply_rope[partial-8]", want, got)
    check(r < RTOL, f"partial (8 of {D} channels): max abs {a:.2e}, rel {r:.2e}")

    # ------------------------------------------------------------- norms
    banner("[3] rms_norm (RMS and L2) and the activations")
    Xn = rng.standard_normal((6, D)).astype(np.float32) * 2.0
    wt = (rng.standard_normal(D).astype(np.float32) + 2.0)
    want = T.rms_norm(Xn, wt, 1e-6, unit_offset=True)
    a, r = compare("rms_norm(RMS)", want, N.rmsnorm_rows(Xn, wt, 1e-6))
    check(r < RTOL, f"rms_norm unit_offset=True: max abs {a:.2e}, rel {r:.2e}")
    want = T.rms_norm(Xn, wt, 1e-6, unit_offset=False)
    a, r = compare("rms_norm(L2)", want, N.l2norm_rows(Xn, wt, 1e-6))
    check(r < RTOL, f"rms_norm unit_offset=False (L2): max abs {a:.2e}, rel {r:.2e}")

    z = (rng.standard_normal(401).astype(np.float32) * 5.0)
    a, r = compare("gelu", T.gelu(z), N.gelu(z))
    check(r < RTOL, f"gelu: max abs {a:.2e}, rel {r:.2e}")
    a, r = compare("silu", T.silu(z), N.silu(z))
    check(r < RTOL, f"silu: max abs {a:.2e}, rel {r:.2e}")
    up = rng.standard_normal(401).astype(np.float32)
    a, r = compare("swiglu", T.swiglu(z, up), N.swiglu(z, up))
    check(r < RTOL, f"swiglu: max abs {a:.2e}, rel {r:.2e}")

    sm = rng.standard_normal((4, 9)).astype(np.float32) * 3.0
    a, r = compare("softmax", T.softmax(sm), N.softmax(sm))
    check(r < RTOL, f"softmax (per row): max abs {a:.2e}, rel {r:.2e}")

    # ------------------------------------------------------------- attention
    banner("[4] attention_tiled (GQA, sliding window, blocking)")
    Tn, H, HK, Dd, DV, S = 3, 4, 2, 8, 8, 11
    q = rng.standard_normal((Tn, H, Dd)).astype(np.float32)
    k = rng.standard_normal((S, HK, Dd)).astype(np.float32)
    v = rng.standard_normal((S, HK, DV)).astype(np.float32)
    scale = 1.0 / math.sqrt(Dd)
    qpos = np.arange(Tn, dtype=np.int32)
    kpos = np.arange(S, dtype=np.int32)

    for label, window, block in (("causal", None, 1024), ("streamed block=3", None, 3),
                                 ("window=4", 4, 1024)):
        want = T.attention_tiled(q, k, v, scale, qpos, kpos=kpos, block=block,
                                 window=window, kv_rep=H // HK)
        got = N.attention_tiled(q, k, v, scale, qpos, kpos, block=block, window=window)
        a, r = compare(f"attention_tiled[{label}]", want, got)
        check(r < RTOL, f"{label}: max abs {a:.2e}, rel {r:.2e}")

    # single-token decode path
    banner("[5] single-token decode (attention vs tiled with one query)")
    qd = rng.standard_normal((1, H, Dd)).astype(np.float32)
    want = T.attention_tiled(qd, k, v, scale, np.array([S - 1], dtype=np.int32),
                             kpos=kpos, block=S, kv_rep=H // HK)
    got = N.attention(qd[0], k, v, scale)
    a, r = compare("attention(decode)", want, got)
    check(r < RTOL, f"decode: max abs {a:.2e}, rel {r:.2e}")

    # ------------------------------------------------------------- summary
    banner("=== per-function observed difference (abs / rel) ===")
    worst = 0.0
    worst_name = ""
    for name, a, r in ROWS:
        print(f"       {name:<38} abs {a:.3e}   rel {r:.3e}")
        if r > worst:
            worst, worst_name = r, name
    print(f"       worst rel {worst:.3e} ({worst_name})")
    check(worst < RTOL,
          f"every primitive is within {RTOL:.0e} relative of the Python reference")

    print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
