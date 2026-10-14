#!/usr/bin/env python3
"""SSM (gated delta net) self-test.

The strongest available checks, none of which require running the model:

  1. CONFIG vs THE REAL FILE -- derive SsmConfig from Qwen3.8-4B's hparams and
     assert every weight shape in blk.0 matches. This validates that my reading
     of the layout is right against actual data, not against my own arithmetic.
  2. RECURRENCE IDENTITY -- the vectorised step against a naive triple-loop.
  3. CAUSALITY -- a future token cannot change an earlier output.
  4. STATE IS CONSTANT SIZE -- independent of sequence length, which is the whole
     point of these layers.
  5. GATE LIMITS -- decay->0 flushes the state; beta->0 writes nothing.

NOT checked here: numerical agreement with the real model's output. That needs
the full forward pass wired end to end (see the ASSUMPTIONS block in ssm.py for
the specific conventions that remain unverified).

Run:  .venv-cml/bin/python tests/test_ssm.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine import gguf as G  # noqa: E402
from engine import ssm as S  # noqa: E402

CHECKS = 0
FAILS = 0
rng = np.random.default_rng(7)


def check(cond: bool, what: str) -> None:
    global CHECKS, FAILS
    CHECKS += 1
    if cond:
        print(f"  ok   {what}")
    else:
        FAILS += 1
        print(f"  FAIL {what}")


def naive_step(q, k, v, alpha, beta, a_log, dt_bias, state, scale):
    """Triple-loop reference for the gated delta rule. No einsum, no vector ops."""
    nv, d, _ = state.shape
    def sp(x):
        return np.log1p(np.exp(-abs(float(x)))) + max(float(x), 0.0)
    def l2(vec):
        n = 0.0
        for e in vec:
            n += float(e) * float(e)
        n = max(n ** 0.5, 1e-6)
        return [float(e) / n for e in vec]

    out = np.zeros((nv, d), dtype=np.float32)
    new = state.copy()
    for h in range(nv):
        qn, kn = l2(q[h]), l2(k[h])
        # ssm_a is already -exp(A_log) -- multiply in directly, no exp()
        # clamp identically to the engine: a decay cannot exceed 1
        g = min(float(a_log[h]) * sp(float(alpha[h]) + float(dt_bias[h])), 0.0)
        dec = np.exp(g)
        b = 1.0 / (1.0 + np.exp(-float(beta[h])))
        for i in range(d):
            for j in range(d):
                new[h, i, j] = state[h, i, j] * dec
        # v_pred[j] = sum_i kn[i] * S[i, j]
        vpred = [sum(kn[i] * float(new[h, i, j]) for i in range(d)) for j in range(d)]
        vnew = [(float(v[h, j]) - vpred[j]) * b for j in range(d)]
        for i in range(d):
            for j in range(d):
                new[h, i, j] += kn[i] * vnew[j]
        for j in range(d):
            out[h, j] = scale * sum(qn[i] * float(new[h, i, j]) for i in range(d))
    return out, new


def main() -> int:
    print("=== SSM (gated delta net) self-test ===\n")

    # --------------------------------------------- 1. config vs the real file
    print("[1] SsmConfig derived from the real model, checked against real shapes")
    real = ("/Volumes/Backup/lmstudio/models/empero-ai/"
            "Qwen3.8-4B-Distill-GGUF/Qwen3.8-4B-Q4_K_M.gguf")
    if not os.path.exists(real):
        print("       (skipped: model not present)")
    else:
        with G.GGUFReader(real) as r:
            cfg = S.SsmConfig(
                n_embd=int(r.hp("embedding_length")),
                inner_size=int(r.hp("ssm.inner_size")),
                state_size=int(r.hp("ssm.state_size")),
                n_group=int(r.hp("ssm.group_count")),
                conv_kernel=int(r.hp("ssm.conv_kernel")),
                time_step_rank=int(r.hp("ssm.time_step_rank")),
                eps=float(r.hp("attention.layer_norm_rms_epsilon")),
            )
            print(f"       n_embd={cfg.n_embd} inner={cfg.inner_size} "
                  f"state={cfg.state_size} groups={cfg.n_group} "
                  f"conv={cfg.conv_kernel} tsr={cfg.time_step_rank}")
            nv, nk, d = cfg.n_v_heads, cfg.n_k_heads, cfg.state_size
            print(f"       -> n_v_heads={nv}  n_k_heads={nk}  head_dim={d}  ratio={cfg.ratio}")
            print(f"       -> q_width={cfg.q_width} kv_width={cfg.kv_width} "
                  f"qkv_width={cfg.qkv_width}")

            exp = {
                "blk.0.attn_qkv.weight": (cfg.qkv_width, cfg.n_embd),
                "blk.0.attn_gate.weight": (cfg.inner_size, cfg.n_embd),
                "blk.0.ssm_alpha.weight": (cfg.time_step_rank, cfg.n_embd),
                "blk.0.ssm_beta.weight": (cfg.time_step_rank, cfg.n_embd),
                "blk.0.ssm_a": (cfg.time_step_rank,),
                "blk.0.ssm_dt.bias": (cfg.time_step_rank,),
                "blk.0.ssm_conv1d.weight": (cfg.qkv_width, cfg.conv_kernel),
                "blk.0.ssm_norm.weight": (cfg.state_size,),
                "blk.0.ssm_out.weight": (cfg.n_embd, cfg.inner_size),
            }
            for name, shape in exp.items():
                got = r.tensors[name].shape
                check(got == shape, f"{name:28s} expected {shape}, file has {got}")

            check(cfg.qkv_width == cfg.q_width + cfg.kv_width + cfg.inner_size,
                  "qkv width == q + k + v exactly (the layout closes)")

            # ssm_a in log space: exp() must be a sane positive decay magnitude
            a_log = r.tensor("blk.0.ssm_a")
            check(np.all(np.isfinite(a_log)), "ssm_a is finite")
            print(f"       ssm_a: min={a_log.min():.3f} max={a_log.max():.3f} "
                  f"exp range=[{np.exp(a_log).min():.4f},{np.exp(a_log).max():.4f}]")

            dt = r.tensor("blk.0.ssm_dt.bias")
            print(f"       dt_bias: min={dt.min():.3f} max={dt.max():.3f}")

            # a real per-token decay must be in (0,1]
            a = r.tensor("blk.0.ssm_alpha.weight")[:1, :8]
            # ssm_a is stored already as -exp(A_log): multiply it in directly
            g = a_log[:1] * S.softplus(a + dt[:1])
            check(np.all(np.exp(g) > 0) and np.all(np.exp(g) <= 1.0 + 1e-6),
                  "exp(g) is a valid decay in (0,1] for real weights")

    # ------------------------------------------------ 2. recurrence identity
    print("\n[2] recurrence: vectorised step vs a naive triple-loop")
    nv, nk, d, Tn = 4, 2, 6, 3
    rep = nv // nk
    q = rng.standard_normal((nk, d)).astype(np.float32)
    k = rng.standard_normal((nk, d)).astype(np.float32)
    v = rng.standard_normal((nv, d)).astype(np.float32)
    alpha = rng.standard_normal(nv).astype(np.float32)
    beta = rng.standard_normal(nv).astype(np.float32)
    # ssm_a is stored as -exp(A_log): ALWAYS NEGATIVE. Randomising its sign would
    # test a decay > 1, which cannot occur and which the engine clamps.
    a_log = -np.abs(rng.standard_normal(nv).astype(np.float32)) * 0.5
    dt_bias = rng.standard_normal(nv).astype(np.float32)
    state = (rng.standard_normal((nv, d, d)).astype(np.float32) * 0.1)
    scale = 1.0 / np.sqrt(d)

    # TILE, not repeat: v-head h reads k-head h % n_k (the reference tiles the
    # k heads up to the v-head count). np.repeat would be h // ratio instead.
    qe = np.tile(q, (rep, 1))
    ke = np.tile(k, (rep, 1))
    got_o, got_s = S.delta_rule_step(qe, ke, v, alpha, beta, a_log, dt_bias,
                                     state.copy(), scale)
    want_o, want_s = naive_step(qe, ke, v, alpha, beta, a_log, dt_bias,
                                state.copy(), scale)
    do = float(np.abs(got_o - want_o).max())
    ds = float(np.abs(got_s - want_s).max())
    check(do < 1e-5, f"output matches the loop reference (max diff {do:.2e})")
    check(ds < 1e-5, f"state matches the loop reference (max diff {ds:.2e})")

    # ------------------------------------------------------- 3. causality
    print("\n[3] causality: a future token cannot affect an earlier output")
    cfg = S.SsmConfig(n_embd=16, inner_size=8, state_size=4, n_group=1,
                      conv_kernel=3, time_step_rank=2)
    W = S.SsmWeights(
        qkv=rng.standard_normal((cfg.qkv_width, cfg.n_embd)).astype(np.float32) * 0.2,
        gate=rng.standard_normal((cfg.inner_size, cfg.n_embd)).astype(np.float32) * 0.2,
        alpha=rng.standard_normal((cfg.time_step_rank, cfg.n_embd)).astype(np.float32) * 0.2,
        beta=rng.standard_normal((cfg.time_step_rank, cfg.n_embd)).astype(np.float32) * 0.2,
        a_log=rng.standard_normal((cfg.time_step_rank,)).astype(np.float32) * 0.3,
        dt_bias=rng.standard_normal((cfg.time_step_rank,)).astype(np.float32),
        conv1d=rng.standard_normal((cfg.qkv_width, cfg.conv_kernel)).astype(np.float32) * 0.3,
        norm=np.ones((cfg.state_size,), np.float32),
        out=rng.standard_normal((cfg.n_embd, cfg.inner_size)).astype(np.float32) * 0.2,
    )
    X = rng.standard_normal((6, cfg.n_embd)).astype(np.float32)
    y1, s1 = S.ssm_forward(X, W, cfg)
    Xb = X.copy()
    Xb[5] += 3.0                       # perturb ONLY the last token
    y2, s2 = S.ssm_forward(Xb, W, cfg)
    check(np.allclose(y1[:5], y2[:5], atol=1e-6),
          "editing token 5 leaves outputs 0..4 identical")
    check(not np.allclose(y1[5], y2[5]), "token 5's own output does change")
    check(y1.shape == (6, cfg.n_embd), f"output shape is (T, n_embd) = {y1.shape}")

    # ------------------------------------------ 4. state is constant size
    print("\n[4] recurrent state is constant size (the point of these layers)")
    st = S.SsmState.create(cfg)
    check(st.nbytes == cfg.n_v_heads * cfg.state_size * cfg.state_size * 4,
          f"state is {st.nbytes} bytes, independent of context length")
    for Tn2 in (1, 64, 256):
        _, sn = S.ssm_forward(rng.standard_normal((Tn2, cfg.n_embd)).astype(np.float32),
                              W, cfg)
        check(sn.shape == (cfg.n_v_heads, cfg.state_size, cfg.state_size),
              f"after {Tn2:4d} tokens the state is still {sn.shape}")
    check(st.nbytes < 64 * 64 * 4 * 4,
          f"state stays tiny ({st.nbytes} B) -- no growing KV for these layers")

    # ------------------------------------------------------ 5. gate limits
    print("\n[5] gate limits behave as the maths says")
    zero_beta = np.full(nv, -60.0, np.float32)      # sigmoid(-60) ~ 0
    o0, s0 = S.delta_rule_step(qe, ke, v, alpha, zero_beta, a_log, dt_bias,
                               state.copy(), scale)
    check(np.allclose(s0, state * np.exp(a_log * S.softplus(
        alpha + dt_bias))[:, None, None], atol=1e-5),
          "beta~0 writes nothing: the state only decays")

    # a full flush must ERASE THE PAST: two different initial states must give
    # the same result. (Asserting the state merely "gets smaller" is wrong --
    # a flushed state is replaced by the new rank-1 write, whose magnitude comes
    # from |v|, not from the old state.)
    flush = np.full(nv, 60.0, np.float32)
    # ssm_a is already -exp(A_log), so a strongly NEGATIVE value is what forces
    # g << 0 and therefore a full decay. (Under the old misreading the sign was
    # inverted and a large POSITIVE entry was the one that flushed.)
    big_a = np.full(nv, -60.0, np.float32)
    huge_dt = np.full(nv, 50.0, np.float32)
    st_a = state.copy()
    st_b = (rng.standard_normal(state.shape).astype(np.float32) * 5.0)
    o_a, s_a = S.delta_rule_step(qe, ke, v, flush, beta, big_a, huge_dt, st_a, scale)
    o_b, s_b = S.delta_rule_step(qe, ke, v, flush, beta, big_a, huge_dt, st_b, scale)
    check(np.allclose(s_a, s_b, atol=1e-6),
          "full decay erases the past: two different initial states converge")
    check(np.allclose(o_a, o_b, atol=1e-6), "and produce the same output")
    check(not np.allclose(st_a, st_b),
          "the two initial states really were different (the test is not vacuous)")

    print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
