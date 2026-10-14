#!/usr/bin/env python3
"""Transformer core self-test.

Every primitive is checked against a NAIVE, LOOP-BASED reference written here,
sharing no code with the implementation. That catches vectorisation and
broadcasting bugs -- which are exactly the kind that produce plausible-looking
garbage rather than an exception.

The two highest-value checks:
  * RoPE must make the attention score depend only on RELATIVE position --
    that is the defining property of the transform, and it is testable without
    any reference implementation.
  * Incremental decoding with the KV cache must equal a full-sequence forward
    pass exactly. This is the classic decoder correctness invariant.

Run:  .venv-cml/bin/python tests/test_transformer.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from engine import transformer as T  # noqa: E402

CHECKS = 0
FAILS = 0


def check(cond: bool, what: str) -> None:
    global CHECKS, FAILS
    CHECKS += 1
    if cond:
        print(f"  ok   {what}")
    else:
        FAILS += 1
        print(f"  FAIL {what}")


rng = np.random.default_rng(1234)


def ref_rms_norm(x, w, eps):
    out = np.empty_like(x, dtype=np.float32)
    for i in range(x.shape[0]):
        acc = 0.0
        for j in range(x.shape[1]):
            acc += float(x[i, j]) * float(x[i, j])
        ms = acc / x.shape[1]
        denom = math.sqrt(ms + eps)
        for j in range(x.shape[1]):
            out[i, j] = float(x[i, j]) / denom * float(w[j])
    return out


def ref_softmax(x):
    out = np.empty_like(x, dtype=np.float32)
    for i in range(x.shape[0]):
        m = -np.inf
        for j in range(x.shape[1]):
            if x[i, j] > m:
                m = float(x[i, j])
        s = 0.0
        for j in range(x.shape[1]):
            s += math.exp(float(x[i, j]) - m)
        for j in range(x.shape[1]):
            out[i, j] = math.exp(float(x[i, j]) - m) / s
    return out


def ref_attention(x, pos, wq, wk, wv, wo, n_head, n_head_kv, head_dim, value_dim,
                  rope_base, scale):
    """Fully naive attention: explicit loops, no einsum, no batching tricks."""
    Tn, E = x.shape
    q = x @ wq.T
    k = x @ wk.T
    v = x @ wv.T
    H, HK = n_head, n_head_kv
    q = q.reshape(Tn, H, head_dim)
    k = k.reshape(Tn, HK, head_dim)
    v = v.reshape(Tn, HK, value_dim)

    # rope, done scalar by scalar, in NEOX (split-half) order -- the convention
    # Qwen uses. Pairs are (vec[i], vec[i + half]), NOT (vec[2i], vec[2i+1]);
    # the reference must match the implementation or it tests the wrong thing.
    def rot(vec, p):
        half = head_dim // 2
        out = np.array(vec, dtype=np.float32)
        for i in range(half):
            inv = 1.0 / (rope_base ** (2.0 * i / head_dim))
            a = p * inv
            c, s = math.cos(a), math.sin(a)
            x0, x1 = float(vec[i]), float(vec[i + half])
            out[i] = x0 * c - x1 * s
            out[i + half] = x0 * s + x1 * c
        return out

    qr = np.stack([[rot(q[t, h], int(pos[t])) for h in range(H)] for t in range(Tn)])
    kr = np.stack([[rot(k[t, h], int(pos[t])) for h in range(HK)] for t in range(Tn)])

    rep = H // HK
    out = np.zeros((Tn, H, value_dim), dtype=np.float32)
    for t in range(Tn):
        for h in range(H):
            kh = h // rep
            s = []
            for j in range(Tn):
                if j > t:
                    s.append(-np.inf)
                    continue
                acc = 0.0
                for d in range(head_dim):
                    acc += float(qr[t, h, d]) * float(kr[j, kh, d])
                s.append(acc * scale)
            m = max(s)
            ex = [math.exp(v2 - m) for v2 in s]
            tot = sum(ex)
            for d in range(value_dim):
                acc = 0.0
                for j in range(Tn):
                    acc += (ex[j] / tot) * float(v[j, kh, d])
                out[t, h, d] = acc
    out = out.reshape(Tn, H * value_dim)
    return out @ wo.T


def main() -> int:
    print("=== transformer core self-test ===\n")

    # ------------------------------------------------------------ rms_norm
    print("[1] RMSNorm vs loop reference")
    x = rng.standard_normal((5, 16)).astype(np.float32) * 3
    w = rng.standard_normal((16,)).astype(np.float32)
    got = T.rms_norm(x, w, 1e-6)
    want = ref_rms_norm(x, w, 1e-6)
    check(np.allclose(got, want, atol=1e-5), "matches the scalar loop")
    # unit-variance input with unit weights -> output norm ~ sqrt(n)
    unit = T.rms_norm(np.ones((1, 16), np.float32), np.ones(16, np.float32), 1e-6)
    check(np.allclose(unit, 1.0, atol=1e-4), "ones in -> ones out (normalisation is right)")

    # ------------------------------------------------------------- softmax
    print("\n[2] softmax vs loop reference")
    z = rng.standard_normal((4, 9)).astype(np.float32) * 2
    check(np.allclose(T.softmax(z), ref_softmax(z), atol=1e-6), "matches the scalar loop")
    check(np.allclose(T.softmax(z).sum(-1), 1.0, atol=1e-6), "rows sum to 1")
    big = np.array([[1000.0, 1000.0, 0.0]], dtype=np.float32)
    check(np.all(np.isfinite(T.softmax(big))), "no overflow on large logits")

    # ---------------------------------------------------------------- rope
    print("\n[3] RoPE: the defining property (score depends only on RELATIVE position)")
    D, base = 8, 10000.0
    v1 = rng.standard_normal((1, 1, D)).astype(np.float32)
    v2 = rng.standard_normal((1, 1, D)).astype(np.float32)

    def score_at(m, n):
        """<rope(v1,m), rope(v2,n)> computed through the library."""
        c1, s1 = T.rope_tables(D, np.array([m]), base)
        c2, s2 = T.rope_tables(D, np.array([n]), base)
        a = T.apply_rope(v1.copy(), c1, s1).reshape(-1)
        b = T.apply_rope(v2.copy(), c2, s2).reshape(-1)
        return float(np.dot(a, b))

    d12 = score_at(3, 1)      # relative +2
    d12b = score_at(103, 101)  # relative +2, far away
    check(abs(d12 - d12b) < 1e-3,
          f"score at relative +2 is the same at positions (3,1) and (103,101): "
          f"{d12:.6f} vs {d12b:.6f}")
    d_rev = score_at(1, 3)
    check(abs(d_rev - d12) > 1e-3, "reversing the order changes the score (not symmetric)")

    # norm preservation
    c, s = T.rope_tables(D, np.array([7]), base)
    rotated = T.apply_rope(v1.copy(), c, s).reshape(-1)
    check(abs(float(np.linalg.norm(rotated)) - float(np.linalg.norm(v1))) < 1e-4,
          "rotation preserves the vector norm")

    # partial rotary: only the first n_dims rotate
    c2, s2 = T.rope_tables(D, np.array([5]), base, n_dims=4)
    pr = T.apply_rope(v1.copy(), c2, s2).reshape(-1)
    orig = v1.reshape(-1)
    check(np.allclose(pr[4:], orig[4:]), "partial rotary leaves the tail untouched")
    check(not np.allclose(pr[:4], orig[:4]), "partial rotary does rotate the head")

    # ----------------------------------------------------------- attention
    print("\n[4] GQA attention vs loop reference")
    Tn, E, H, HK, D, DV = 6, 32, 4, 2, 8, 8
    xa = rng.standard_normal((Tn, E)).astype(np.float32)
    wq = rng.standard_normal((H * D, E)).astype(np.float32) * 0.3
    wk = rng.standard_normal((HK * D, E)).astype(np.float32) * 0.3
    wv = rng.standard_normal((HK * DV, E)).astype(np.float32) * 0.3
    wo = rng.standard_normal((E, H * DV)).astype(np.float32) * 0.3
    pos = np.arange(Tn)

    cfg = T.AttnConfig(n_head=H, n_head_kv=HK, head_dim=D, value_dim=DV,
                       rope_base=base)
    w = T.AttnWeights(wq=wq, wk=wk, wv=wv, wo=wo)
    got = T.attention(xa, pos, w, cfg)
    want = ref_attention(xa, pos, wq, wk, wv, wo, H, HK, D, DV, base,
                         1.0 / math.sqrt(D))
    check(got.shape == (Tn, E), f"output shape is (T, n_embd) = {got.shape}")
    check(np.allclose(got, want, atol=2e-4),
          f"matches the loop reference (max diff {np.abs(got - want).max():.2e})")

    # causality: changing a FUTURE token must not change an EARLIER output
    xb = xa.copy()
    xb[5] += 5.0
    got_b = T.attention(xb, pos, w, cfg)
    check(np.allclose(got[:5], got_b[:5], atol=1e-6),
          "no future leakage: editing token 5 leaves outputs 0..4 identical")
    check(not np.allclose(got[5], got_b[5]),
          "token 5's own output DOES change (the test is not vacuous)")

    # -------------------------------------------------------------- kv cache
    print("\n[5] incremental decoding == full-sequence forward")
    cache = T.KVCache(n_layer=1, n_head_kv=HK, head_dim=D, value_dim=DV,
                      capacity=Tn)
    step = np.empty_like(got)
    for t in range(Tn):
        step[t:t + 1] = T.attention(xa[t:t + 1], np.array([t]), w, cfg,
                                    cache=cache, layer=0)
    check(np.allclose(step, got, atol=2e-4),
          f"token-by-token with KV cache equals one batched pass "
          f"(max diff {np.abs(step - got).max():.2e})")
    check(cache.k[0].shape[0] == Tn, f"cache holds {Tn} keys")

    # -------------------------------------------------------------- swiglu
    print("\n[6] SwiGLU")
    g = rng.standard_normal((3, 5)).astype(np.float32)
    u = rng.standard_normal((3, 5)).astype(np.float32)
    ref = np.array([[float(g[i, j]) / (1.0 + math.exp(-float(g[i, j]))) * float(u[i, j])
                     for j in range(5)] for i in range(3)], dtype=np.float32)
    check(np.allclose(T.swiglu(g, u), ref, atol=1e-6), "silu(gate)*up matches the scalar form")
    check(np.allclose(T.swiglu(np.zeros((2, 2)), np.ones((2, 2))), 0.0),
          "silu(0)=0, so an all-zero gate gives all-zero output")

    # ------------------------------------------------------------ sampling
    print("\n[7] sampling")
    logits = np.array([0.0, 5.0, 1.0, -3.0], dtype=np.float32)
    check(T.sample(logits, T.SampleParams(temperature=0.0)) == 1,
          "temperature 0 is greedy -> argmax")
    check(T.sample(logits, T.SampleParams(temperature=0.0, top_k=2)) == 1,
          "top_k does not break greedy argmax")
    r = np.random.default_rng(0)
    hits = {T.sample(logits, T.SampleParams(temperature=1.0, top_k=1), r) for _ in range(20)}
    check(hits == {1}, f"top_k=1 always picks the max (got {hits})")

    # temperature must shape the distribution QUANTITATIVELY, not just "work":
    # logits [0,5,1,-3] at T=2 give z=[0,2.5,0.5,-1.5] -> P(token 1) = 0.809
    import collections
    N = 3000
    counts = collections.Counter(
        T.sample(logits, T.SampleParams(temperature=2.0, seed=7), r) for _ in range(N))
    freq1 = counts.get(1, 0) / N
    check(counts.most_common(1)[0][0] == 1, "the dominant logit is the most frequent")
    check(abs(freq1 - 0.809) < 0.05,
          f"P(token 1) ~= 0.809 at T=2 (measured {freq1:.3f}) -- temperature scales correctly")
    check(len(counts) > 1, "at T=2 other tokens do occur (not a degenerate argmax)")

    # a lower temperature must concentrate the distribution
    c2 = collections.Counter(
        T.sample(logits, T.SampleParams(temperature=0.5, seed=11), r) for _ in range(N))
    check(c2[1] / N > freq1, f"T=0.5 is sharper than T=2 ({c2[1]/N:.3f} > {freq1:.3f})")
    check(all(0 <= h < 4 for h in counts), "sampled ids are in range")

    print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
