#!/usr/bin/env python3
"""Native SSM/MoE kernels vs the PYTHON reference, on identical inputs.

WHY THIS EXISTS. The C++ self-test (build/test_ssm_cpp) compares the vectorised
C++ against a C++ loop reference. Both were written from the same reading of the
conventions, so they can agree with each other while both being wrong. This suite
instead runs the PUBLIC Python entry points twice -- once down the native path
(src/ssm.cpp, src/moe.cpp) and once down the numpy path -- and prints the observed
max absolute and relative difference. It also checks the recurrence against an
independent triple-loop reference, not only against the numpy implementation.

Run:  .venv-cml/bin/python tests/test_ssm_native.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine import moe as M        # noqa: E402
from engine import native_ssm as NS  # noqa: E402
from engine import ssm as S        # noqa: E402

CHECKS = 0
FAILS = 0
rng = np.random.default_rng(2026)

# fp32 rounding through a small matmul/einsum chain lands here; anything above it
# is a convention mismatch, not noise.
TOL = 1e-5


def check(cond: bool, what: str) -> None:
    global CHECKS, FAILS
    CHECKS += 1
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS += 1


def diff(a: np.ndarray, b: np.ndarray):
    """(max abs, max rel-to-peak) between two arrays."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mabs = float(np.abs(a - b).max())
    peak = max(float(np.abs(b).max()), 1e-12)
    return mabs, mabs / peak


# ---- forcing the pure-numpy path so the two can be compared on the same input --
def _force_numpy(mod):
    saved = mod._native_ssm
    mod._native_ssm = None
    return saved


def py_delta_step(q, k, v, alpha, beta, a_log, dt_bias, state, scale, eps=1e-6):
    saved = _force_numpy(S)
    try:
        return S.delta_rule_step(q, k, v, alpha, beta, a_log, dt_bias, state, scale, eps)
    finally:
        S._native_ssm = saved


def py_ssm_forward(x, w, cfg, state=None, gdn_state=None):
    saved = _force_numpy(S)
    try:
        return S.ssm_forward(x, w, cfg, state, gdn_state)
    finally:
        S._native_ssm = saved


def py_route(logits, cfg):
    saved = _force_numpy(M)
    try:
        return M.route(logits, cfg)
    finally:
        M._native_ssm = saved


def py_moe_ffn(h, cfg, rows, shexp=None, block_rows=64, router_fn=None):
    saved = _force_numpy(M)
    try:
        return M.moe_ffn(h, cfg, rows, shexp, block_rows, router_fn=router_fn)
    finally:
        M._native_ssm = saved


# ---- an INDEPENDENT reference: the maths, as explicit loops -------------------
def naive_step(q, k, v, alpha, beta, a_log, dt_bias, state, scale):
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
        g = min(float(a_log[h]) * sp(float(alpha[h]) + float(dt_bias[h])), 0.0)
        dec = np.exp(g)
        b = 1.0 / (1.0 + np.exp(-float(beta[h])))
        for i in range(d):
            for j in range(d):
                new[h, i, j] = state[h, i, j] * dec
        vpred = [sum(kn[i] * float(new[h, i, j]) for i in range(d)) for j in range(d)]
        vnew = [(float(v[h, j]) - vpred[j]) * b for j in range(d)]
        for i in range(d):
            for j in range(d):
                new[h, i, j] += kn[i] * vnew[j]
        for j in range(d):
            out[h, j] = scale * sum(qn[i] * float(new[h, i, j]) for i in range(d))
    return out, new


def py_conv(x, w, tail):
    """ConvState.step, transcribed (causal depthwise, w[:,0] pairs oldest)."""
    T, C = x.shape
    kernel = w.shape[1]
    buf = np.concatenate([tail, x.astype(np.float32)], axis=0)
    n = buf.shape[0]
    out = np.zeros((T, C), dtype=np.float32)
    for t in range(T):
        end = n - T + t + 1
        start = end - kernel
        acc = np.zeros(C, dtype=np.float32)
        for j in range(kernel):
            idx = start + j
            if 0 <= idx < n:
                acc += buf[idx] * w[:, j]
        out[t] = acc
    keep = max(0, kernel - 1)
    return out, (buf[-keep:] if keep else np.zeros((0, C), np.float32))


def main() -> int:
    print("=== SSM + MoE native path vs the Python reference ===\n")

    # ---------------------------------------------- 1. library and symbols
    print("[1] the native library is built and exports every kernel")
    ok = NS.available()
    check(ok, f"libaero_ssm loaded -- {NS.unavailable_reason() or 'ok'}")
    if not ok:
        print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
        return 1
    lib = NS._load()
    for sym in ("aero_ssm_forward", "aero_ssm_delta_step", "aero_ssm_scan",
                "aero_ssm_conv", "aero_ssm_softplus", "aero_moe_route",
                "aero_moe_expert_accum", "aero_moe_shared_accum",
                "aero_moe_ffn_block"):
        check(hasattr(lib, sym), f"exports {sym}")

    # ------------------------------------------------------------ 2. softplus
    print("\n[2] softplus matches the numpy form")
    xs = np.linspace(-30, 30, 401).astype(np.float32)
    want = S.softplus(xs)
    got = NS.softplus(xs)
    a, r = diff(got, want)
    print(f"       max abs diff = {a:.3e}   rel = {r:.3e}")
    check(a < 1e-6, "softplus agrees")

    # ------------------------------- 3. delta rule: C++ vs numpy vs a loop
    print("\n[3] gated delta rule: native path vs numpy path vs an independent loop")
    nv, nk, d = 4, 2, 6
    rep = nv // nk
    q = rng.standard_normal((nk, d)).astype(np.float32)
    k = rng.standard_normal((nk, d)).astype(np.float32)
    v = rng.standard_normal((nv, d)).astype(np.float32)
    alpha = rng.standard_normal(nv).astype(np.float32)
    beta = rng.standard_normal(nv).astype(np.float32)
    a_log = (-np.abs(rng.standard_normal(nv)) * 0.5).astype(np.float32)  # ssm_a = -exp(A_log)
    dt_bias = rng.standard_normal(nv).astype(np.float32)
    state = (rng.standard_normal((nv, d, d)).astype(np.float32) * 0.1)
    scale = 1.0 / np.sqrt(d)
    # tile k/q up to the v-head count: v-head h reads k-head h % n_k
    qe, ke = np.tile(q, (rep, 1)), np.tile(k, (rep, 1))

    n_o, n_s = NS.delta_step(qe, ke, v, alpha, beta, a_log, dt_bias, state, scale)
    p_o, p_s = py_delta_step(qe, ke, v, alpha, beta, a_log, dt_bias, state.copy(), scale)
    ref_o, ref_s = naive_step(qe, ke, v, alpha, beta, a_log, dt_bias, state.copy(), scale)
    a1, r1 = diff(n_o, p_o)
    a2, r2 = diff(n_o, ref_o)
    a3, r3 = diff(p_o, ref_o)
    print(f"       native vs numpy   : out abs={a1:.3e} rel={r1:.3e}  "
          f"state abs={float(np.abs(n_s - p_s).max()):.3e}")
    print(f"       native vs loop-ref: out abs={a2:.3e} rel={r2:.3e}")
    print(f"       numpy  vs loop-ref: out abs={a3:.3e} rel={r3:.3e}")
    check(a1 < TOL and float(np.abs(n_s - p_s).max()) < TOL,
          "native delta step matches the numpy path")
    check(a2 < TOL, "native delta step matches the independent loop reference")

    # --------------------------------------- 4. the per-head scan vs a loop
    print("\n[4] per-head scan (T tokens) vs a Python token loop")
    Tt = 5
    qh = rng.standard_normal((Tt, nk, d)).astype(np.float32)
    kh = rng.standard_normal((Tt, nk, d)).astype(np.float32)
    vh = rng.standard_normal((Tt, nv, d)).astype(np.float32)
    al = rng.standard_normal((Tt, nv)).astype(np.float32)
    bt = rng.standard_normal((Tt, nv)).astype(np.float32)
    st0 = (rng.standard_normal((nv, d, d)).astype(np.float32) * 0.1)
    n_outs, n_state = NS.scan(qh, kh, vh, al, bt, a_log, dt_bias, st0, scale)
    # reference: the same loop ssm_forward runs, stepping the numpy delta rule
    Sref = st0.copy()
    ref_outs = np.empty((Tt, nv, d), dtype=np.float32)
    for t in range(Tt):
        o, Sref = py_delta_step(np.tile(qh[t], (rep, 1)), np.tile(kh[t], (rep, 1)),
                                vh[t], al[t], bt[t], a_log, dt_bias, Sref, scale)
        ref_outs[t] = o
    a4, r4 = diff(n_outs, ref_outs)
    print(f"       max abs diff = {a4:.3e}   rel = {r4:.3e}   "
          f"state abs = {float(np.abs(n_state - Sref).max()):.3e}")
    check(a4 < TOL, "scan matches the per-token loop")

    # --------------------------------------------------- 5. causal conv
    print("\n[5] causal depthwise conv vs ConvState.step")
    C, K, Tc = 6, 4, 7
    x = rng.standard_normal((Tc, C)).astype(np.float32)
    w = rng.standard_normal((C, K)).astype(np.float32)
    tail = np.zeros((0, C), np.float32)
    n_out, n_tail, n_tl = NS.conv(x, w, tail, 0)
    p_out, p_tail = py_conv(x, w, tail)
    a5, r5 = diff(n_out, p_out)
    print(f"       max abs diff = {a5:.3e}   rel = {r5:.3e}")
    check(a5 < TOL, "conv step matches ConvState.step")
    check(n_tl == K - 1 and np.allclose(n_tail, p_tail, atol=1e-6),
          "conv tail matches in shape and value")
    # a second chunk must continue the filter (persistent history)
    x2 = rng.standard_normal((3, C)).astype(np.float32)
    n_out2, _, _ = NS.conv(x2, w, n_tail, n_tl)
    p_out2, _ = py_conv(x2, w, p_tail)
    a5b, _ = diff(n_out2, p_out2)
    print(f"       second chunk max abs diff = {a5b:.3e}")
    check(a5b < TOL, "conv continues across a chunk boundary identically")

    # --------------------------------- 6. the FULL layer: native vs numpy
    print("\n[6] full SSM layer: public ssm_forward down both paths")
    cfg = S.SsmConfig(n_embd=24, inner_size=16, state_size=4, n_group=2,
                      conv_kernel=3, time_step_rank=4)
    W = S.SsmWeights(
        qkv=(rng.standard_normal((cfg.qkv_width, cfg.n_embd)).astype(np.float32) * 0.2),
        gate=(rng.standard_normal((cfg.inner_size, cfg.n_embd)).astype(np.float32) * 0.2),
        alpha=(rng.standard_normal((cfg.time_step_rank, cfg.n_embd)).astype(np.float32) * 0.2),
        beta=(rng.standard_normal((cfg.time_step_rank, cfg.n_embd)).astype(np.float32) * 0.2),
        a_log=(-np.abs(rng.standard_normal(cfg.time_step_rank)) * 0.3).astype(np.float32),
        dt_bias=rng.standard_normal(cfg.time_step_rank).astype(np.float32),
        conv1d=(rng.standard_normal((cfg.qkv_width, cfg.conv_kernel)).astype(np.float32) * 0.3),
        norm=np.ones(cfg.state_size, np.float32),
        out=(rng.standard_normal((cfg.n_embd, cfg.inner_size)).astype(np.float32) * 0.2),
    )
    X = rng.standard_normal((8, cfg.n_embd)).astype(np.float32)
    nat_out, nat_state = S.ssm_forward(X, W, cfg)          # native path
    ref_out, ref_state = py_ssm_forward(X, W, cfg)         # numpy path
    a6, r6 = diff(nat_out, ref_out)
    a6s, r6s = diff(nat_state, ref_state)
    print(f"       n_v_heads={cfg.n_v_heads} n_k_heads={cfg.n_k_heads} "
          f"ratio={cfg.ratio} state_size={cfg.state_size} conv={cfg.conv_kernel}")
    print(f"       output   max abs diff = {a6:.3e}   rel = {r6:.3e}")
    print(f"       gdn state max abs diff = {a6s:.3e}   rel = {r6s:.3e}")
    check(nat_out.shape == (8, cfg.n_embd), f"output shape {nat_out.shape}")
    check(a6 < TOL, "native full-layer output matches the numpy path")
    check(a6s < TOL, "native gdn state matches the numpy path")

    # also carry conv history across two calls, so the tail plumbing is exercised
    st = S.SsmState.create(cfg)
    g0 = st.zeros()
    o1n, g1n = S.ssm_forward(X[:4], W, cfg, st, g0)        # native
    o2n, _ = S.ssm_forward(X[4:], W, cfg, st, g1n)
    st2 = S.SsmState.create(cfg)
    g0b = st2.zeros()
    o1p, g1p = py_ssm_forward(X[:4], W, cfg, st2, g0b)     # numpy
    o2p, _ = py_ssm_forward(X[4:], W, cfg, st2, g1p)
    a6c, _ = diff(np.concatenate([o2n]), np.concatenate([o2p]))
    print(f"       chunked (two calls, conv tail carried) max abs diff = {a6c:.3e}")
    check(a6c < TOL, "chunked forward keeps the conv history identical")

    # ------------------------------------------------------- 7. MoE routing
    print("\n[7] MoE top-k routing: native vs numpy")
    mc = M.MoeConfig(n_expert=8, n_expert_used=3, n_embd=12, ffn_len=6)
    logits = (rng.standard_normal(8).astype(np.float32) * 2.0)
    ni, nw = NS.route(logits, mc.n_expert, mc.n_expert_used, mc.norm_topk_prob)
    pi, pw = py_route(logits, mc)
    print(f"       native idx {ni.tolist()}  numpy idx {pi.tolist()}")
    aw, rw = diff(nw, pw)
    print(f"       weight max abs diff = {aw:.3e}   rel = {rw:.3e}")
    check(list(ni) == list(pi), "the selected experts agree")
    check(aw < TOL, "the routing weights agree")

    # ------------------------------- 8. MoE FFN: native block vs numpy block
    print("\n[8] MoE FFN (routed + shared): public moe_ffn down both paths")
    E, F, NX = 12, 6, 8
    Gt = rng.standard_normal((NX, F, E)).astype(np.float32) * 0.4
    Ut = rng.standard_normal((NX, F, E)).astype(np.float32) * 0.4
    Dt = rng.standard_normal((NX, E, F)).astype(np.float32) * 0.4

    def rows(name, expert, block_rows):
        return {"ffn_gate_exps.weight": Gt, "ffn_up_exps.weight": Ut,
                "ffn_down_exps.weight": Dt}[name][expert]

    Rout = rng.standard_normal((NX, E)).astype(np.float32) * 0.4
    router_fn = lambda hv: (hv @ Rout.T).reshape(-1)     # noqa: E731
    h = rng.standard_normal((4, E)).astype(np.float32)

    nat_moe = M.moe_ffn(h, mc, rows, shexp=None, router_fn=router_fn)      # native
    ref_moe = py_moe_ffn(h, mc, rows, shexp=None, router_fn=router_fn)     # numpy
    a8, r8 = diff(nat_moe, ref_moe)
    print(f"       routed-only  max abs diff = {a8:.3e}   rel = {r8:.3e}")
    check(a8 < TOL, "routed expert accumulation matches the numpy path")

    sh = {"gate_inp": rng.standard_normal((1, E)).astype(np.float32) * 0.4,
          "gate": rng.standard_normal((F, E)).astype(np.float32) * 0.4,
          "up": rng.standard_normal((F, E)).astype(np.float32) * 0.4,
          "down": rng.standard_normal((E, F)).astype(np.float32) * 0.4}
    nat_sh = M.moe_ffn(h, mc, rows, shexp=sh, router_fn=router_fn)
    ref_sh = py_moe_ffn(h, mc, rows, shexp=sh, router_fn=router_fn)
    a9, r9 = diff(nat_sh, ref_sh)
    print(f"       with shared  max abs diff = {a9:.3e}   rel = {r9:.3e}")
    check(a9 < TOL, "shared expert matches the numpy path")
    check(not np.allclose(nat_moe, nat_sh, atol=1e-6),
          "the shared expert actually changes the output")

    # ------------------------------------------- 9. the fallback is honest
    print("\n[9] with the library hidden, the Python paths still run (numpy fallback)")
    s = _force_numpy(S)
    m = _force_numpy(M)
    try:
        f_out, _ = S.ssm_forward(X, W, cfg)
        f_moe = M.moe_ffn(h, mc, rows, shexp=sh, router_fn=router_fn)
        check(f_out.shape == (8, cfg.n_embd) and f_moe.shape == (4, E),
              "numpy fallback produces the right shapes with no native library")
        af, _ = diff(f_out, nat_out)
        check(af < TOL, "the numpy fallback equals the native output (same maths)")
    finally:
        S._native_ssm = s
        M._native_ssm = m

    print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
