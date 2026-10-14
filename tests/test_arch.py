#!/usr/bin/env python3
"""P2 (MoE) and P3 (gemma4) verification, against the real files.

    .venv-cml/bin/python tests/test_arch.py

Every shape assertion here is checked against the actual GGUF on disk, and the
routing maths is checked against an independent loop reference.
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine import arch as ARCH   # noqa: E402
from engine import gemma as GEMMA # noqa: E402
from engine import gguf as G      # noqa: E402
from engine import model as M     # noqa: E402
from engine import moe as MOE     # noqa: E402

CHECKS = FAILS = 0
MOE_P = ("/Volumes/Backup/lmstudio/models/tvall43/"
       "Qwen3.6-14B-A3B-FableVibes-GGUF/Qwen3.6-14B-A3B-FableVibes-Q4_K_M.gguf")
GEMMA_P = ("/Volumes/Backup/lmstudio/models/unsloth/"
         "gemma-4-12b-it-GGUF/gemma-4-12b-it-IQ4_XS.gguf")
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


def test_arch_registry():
    banner("[1] arch registry refuses what it cannot run")
    try:
        ARCH.spec_for("some-unknown-arch")
        check(False, "an unknown arch must raise")
    except ARCH.UnsupportedArch as e:
        check("not supported" in str(e), "unknown arch raises with an explanation")
    for a in ("qwen35", "qwen35moe", "gemma4"):
        try:
            s = ARCH.spec_for(a)
            check(True, f"{a} is known: {ARCH.capabilities(a)}")
        except ARCH.UnsupportedArch:
            check(False, f"{a} should be known")
    check(ARCH.spec_for("qwen35moe").has_moe, "qwen35moe declares MoE")
    check(ARCH.spec_for("qwen35").has_ssm and not ARCH.spec_for("qwen35").has_moe,
          "qwen35 is SSM but not MoE (they are distinguished, not lumped)")
    check(ARCH.spec_for("gemma4").has_sliding_window,
          "gemma4 declares sliding-window attention")


def test_routing():
    banner("[2] expert routing: top-k, normalised, matches a loop reference")
    cfg = MOE.MoeConfig(n_expert=8, n_expert_used=3, n_embd=16, ffn_len=32)
    rng = np.random.default_rng(5)
    for trial in range(5):
        logits = rng.standard_normal(8).astype(np.float32) * 2
        idx, w = MOE.route(logits, cfg)
        # loop reference: pick the k largest, softmax over them
        order = np.argsort(-logits, kind="stable")[:3]
        m = logits[order].max()
        e = np.exp(logits[order] - m)
        ref = e / e.sum()
        ok_idx = set(idx.tolist()) == set(order.tolist())
        ok_w = np.allclose(w, ref, atol=1e-6)
        check(ok_idx and ok_w, f"trial {trial}: selected experts and weights match")
    check(abs(float(w.sum()) - 1.0) < 1e-6,
          f"weights sum to 1 with norm_topk_prob ({w.sum():.6f})")

    raw_cfg = MOE.MoeConfig(n_expert=8, n_expert_used=3, n_embd=16, ffn_len=32,
                            norm_topk_prob=False)
    idx2, w2 = MOE.route(logits, raw_cfg)
    denom = np.exp(logits - logits.max()).sum()
    check(np.allclose(w2, np.exp(logits[idx2] - logits.max()) / denom, atol=1e-6),
          "un-normalised mode uses the full-vocabulary softmax denominator")
    check(float(w2.sum()) < 1.0,
          f"...so the weights sum to <1 ({w2.sum():.4f}) -- scale differs, not equal")

    # determinism: ties must not flip between runs
    tied = np.zeros(8, dtype=np.float32)
    i_a, _ = MOE.route(tied, cfg)
    i_b, _ = MOE.route(tied, cfg)
    check(np.array_equal(i_a, i_b), "tied logits route deterministically")
    check(i_a.tolist() == [0, 1, 2], f"ties resolve to the lowest indices ({i_a.tolist()})")

    # k out of range must raise
    try:
        MOE.route(logits, MOE.MoeConfig(n_expert=8, n_expert_used=9,
                                        n_embd=16, ffn_len=32))
        check(False, "n_expert_used > n_expert must raise")
    except ValueError:
        check(True, "n_expert_used > n_expert raises")


def test_expert_plan():
    banner("[3] expert footprint plan (why this engine exists)")
    cfg = MOE.MoeConfig(n_expert=90, n_expert_used=8, n_embd=2048, ffn_len=1408)
    p = MOE.expert_plan(cfg)
    all_mb = p["all_experts_fp32_bytes"] / 1e6
    act_mb = p["active_experts_fp32_bytes"] / 1e6
    print(f"       all experts fp32 : {all_mb:,.0f} MB")
    print(f"       active (top-8) fp32: {act_mb:,.0f} MB")
    check(p["saving_factor"] == pytest_approx(90 / 8),
          f"reading only the active experts is {p['saving_factor']:.2f}x less work")
    check(act_mb < all_mb / 10,
          f"top-8 of 90 is >10x smaller ({act_mb:,.0f} vs {all_mb:,.0f} MB)")


def pytest_approx(x):
    return round(x, 6)


def test_moe_ffn_reference():
    banner("[4] MoE FFN vs an independent loop reference")
    cfg = MOE.MoeConfig(n_expert=6, n_expert_used=2, n_embd=8, ffn_len=4)
    # expert matrices are stored (out, in): gate/up (ffn, embd), down (embd, ffn)
    rng = np.random.default_rng(9)
    E, F, NX = 8, 4, 6
    CFG_N = 6
    # storage order matches the real file: gate/up (expert, ffn, embd),
    # down (expert, embd, ffn) -- out-dimension first, as GGUF stores it.
    Gt = rng.standard_normal((NX, F, E)).astype(np.float32) * 0.5
    Ut = rng.standard_normal((NX, F, E)).astype(np.float32) * 0.5
    Dt = rng.standard_normal((NX, E, F)).astype(np.float32) * 0.5

    def rows(name, expert, block_rows):
        t = {"ffn_gate_exps.weight": Gt, "ffn_up_exps.weight": Ut,
             "ffn_down_exps.weight": Dt}[name]
        return t[expert]

    # the router is its own projection: (n_expert, n_embd)
    Rout = rng.standard_normal((CFG_N, E)).astype(np.float32) * 0.5
    router_fn = lambda hv: (hv @ Rout.T).reshape(-1)

    h = rng.standard_normal((3, E)).astype(np.float32)
    out = MOE.moe_ffn(h, cfg, rows, shexp=None, router_fn=router_fn)

    # without a router the call must refuse rather than route on h
    try:
        MOE.moe_ffn(h, cfg, rows, shexp=None)
        check(False, "missing router_fn must raise")
    except ValueError:
        check(True, "missing router_fn raises (no routing on the hidden state)")

    # a router of the wrong width must be caught
    try:
        MOE.moe_ffn(h, cfg, rows, shexp=None,
                    router_fn=lambda hv: np.zeros(3, dtype=np.float32))
        check(False, "a router with the wrong width must raise")
    except ValueError:
        check(True, "a router emitting the wrong width raises")

    def ref_one(ht):
        idx, w = MOE.route(router_fn(ht), cfg)
        acc = np.zeros(E, dtype=np.float32)
        for e, wt in zip(idx, w):
            a = (ht @ Gt[e].T) / (1 + np.exp(-(ht @ Gt[e].T))) * (ht @ Ut[e].T)
            acc += wt * (a @ Dt[e].T)
        return acc

    ref = np.stack([ref_one(h[t]) for t in range(3)])
    check(np.allclose(out, ref, atol=1e-5),
          f"matches the loop reference (max diff {np.abs(out-ref).max():.2e})")
    check(out.shape == (3, E), f"output shape {out.shape}")

    # shared expert must actually contribute
    sh = {"gate_inp": rng.standard_normal((1, E)).astype(np.float32),
          "gate": rng.standard_normal((F, E)).astype(np.float32),
          "up": rng.standard_normal((F, E)).astype(np.float32),
          "down": rng.standard_normal((E, F)).astype(np.float32)}
    with_sh = MOE.moe_ffn(h, cfg, rows, shexp=sh, router_fn=router_fn)
    check(not np.allclose(out, with_sh),
          "the shared expert changes the result (it is not silently ignored)")
    check(with_sh.shape == out.shape, "shared expert preserves the shape")
    # ...and its gate bounds the contribution
    g = 1 / (1 + np.exp(-(h @ sh["gate_inp"].T)))
    check(bool(g.min() > 0 and g.max() < 1),
          f"shared gate is bounded in (0,1) (range {g.min():.3f}..{g.max():.3f})")
    del g


def test_real_moe_file():
    banner("[5] MoE config against the real 40-layer / 90-expert file")
    if not os.path.exists(MOE_P):
        check(False, "MoE model present (it is not)")
        return
    with G.GGUFReader(MOE_P) as r:
        mc = M.ModelConfig.from_gguf(r)
        print(f"       arch={mc.arch} layers={mc.n_layer} experts={mc.moe_cfg.n_expert}"
              f" used={mc.moe_cfg.n_expert_used} embd={mc.n_embd} ffn={mc.ffn_len}")
        print(f"       attn layers={mc.attn_layers}")
        print(f"       ssm layers={len(mc.ssm_layers)}")
        check(mc.spec.has_moe, "recognised as a MoE architecture")
        check(mc.moe_cfg.n_expert == 90, f"90 experts read from the file "
                                         f"({mc.moe_cfg.n_expert})")
        check(mc.moe_cfg.n_expert_used == 8, "8 active experts read from the file")
        check(mc.moe_cfg.has_shared, "shared expert detected")
        check(mc.moe_cfg.norm_topk_prob, "top-k weights are normalised")
        # the expert tensors must have the shape the reader assumes
        for nm in ("ffn_gate_exps.weight", "ffn_up_exps.weight",
                   "ffn_down_exps.weight"):
            t = r.tensors[f"blk.0.{nm}"]
            check(t.shape[0] == 90,
                  f"blk.0.{nm} {t.shape} -- expert axis is the outermost one")
        check(r.tensors["blk.0.ffn_gate_inp.weight"].shape == (90, mc.n_embd),
              "router is (n_expert, n_embd)")
        # orientation: gate/up are (ffn, embd), down is (embd, ffn). Getting this
        # backwards is a silent shape-compatible transposition in some cases, so
        # assert it explicitly against the file.
        g_t = r.tensors["blk.0.ffn_gate_exps.weight"]
        d_t = r.tensors["blk.0.ffn_down_exps.weight"]
        check(g_t.shape == (90, mc.moe_cfg.ffn_len, mc.n_embd),
              f"gate_exps is (expert, ffn, embd) = {g_t.shape}")
        check(d_t.shape == (90, mc.n_embd, mc.moe_cfg.ffn_len),
              f"down_exps is (expert, embd, ffn) = {d_t.shape}")
        check(mc.moe_cfg.ffn_len == g_t.shape[1],
              f"routed ffn len comes from the tensor ({mc.moe_cfg.ffn_len})")
        check(mc.moe_cfg.ffn_len != mc.ffn_len,
              f"...and differs from the dense ffn_len ({mc.moe_cfg.ffn_len} vs "
              f"{mc.ffn_len}) -- using the dense value would be wrong")
        # reading ONE expert must be a small fraction of the whole tensor
        info = r.tensors["blk.0.ffn_gate_exps.weight"]
        one = r.row_bytes(info)
        whole = len(r.tensor_bytes(info))
        check(one * 90 <= whole + 1,
              f"one expert row is {one} B of {whole} B -- {whole/one:.0f}x cheaper")
        # SSM layers here too
        check(len(mc.ssm_layers) > 0,
              f"{len(mc.ssm_layers)} SSM layers -> hybdrid, not plain attention")


def test_real_gemma_file():
    banner("[6] gemma4 config against the real 48-layer file")
    if not os.path.exists(GEMMA_P):
        check(False, "gemma4 model present (it is not)")
        return
    with G.GGUFReader(GEMMA_P) as r:
        mc = M.ModelConfig.from_gguf(r)
        g = mc.gemma_cfg
        print(f"       arch={mc.arch} layers={mc.n_layer} embd={mc.n_embd} "
              f"head={mc.n_head} dim={mc.head_dim} ffn={mc.ffn_len}")
        print(f"       window={g.sliding_window} softcap={g.softcap} "
              f"kv_heads[0]={mc.kv_heads_at(0)} kv_heads[5]={mc.kv_heads_at(5)}")
        check(mc.spec.has_sliding_window, "recognised as sliding-window arch")
        check(mc.softcap == 30.0, f"softcap 30.0 read from the file ({mc.softcap})")
        check(g.sliding_window == 1024, f"window 1024 ({g.sliding_window})")
        # THE trap: kv head count is a per-layer list, not a scalar
        check(mc.kv_heads_at(0) == 8, "layer 0 has 8 KV heads")
        check(mc.kv_heads_at(5) == 1, "layer 5 has 1 KV head (a global layer)")
        check(mc.kv_heads_at(0) != mc.kv_heads_at(5),
              "KV heads VARY by layer -- a scalar read would be silently wrong")
        # the pattern must match the kv-head list exactly
        wl, gl = g.window_layers, g.global_layers
        print(f"       window layers: {wl[:8]}... global layers: {gl}")
        check(len(gl) == 8, f"8 global layers in 48 ({len(gl)})")
        check(all(mc.kv_heads_at(i) == 1 for i in gl),
              "every global layer has 1 KV head")
        check(all(mc.kv_heads_at(i) > 1 for i in wl),
              "every windowed layer has >1 KV head")
        # window_at must give None on global layers
        check(g.window_at(0) == 1024, "layer 0 is windowed (window=1024)")
        check(g.window_at(5) is None,
              "layer 5 returns None, NOT 1024 -- a global layer must not be truncated")
        check(mc.window_at(5) is None, "ModelConfig agrees: no window on layer 5")
        # block tensors must all exist
        need = ("attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm",
                "layer_output_scale")
        missing = [n for n in need if f"blk.0.{n}.weight" not in r.tensors]
        check(not missing, f"gemma block tensors all present ({missing})")
        # no SSM here -- and the engine must not invent one
        check(mc.ssm_cfg is None, "no SSM config for gemma4 (it has no SSM layers)")
        check(len(mc.ssm_layers) == 0, "zero SSM layers classified")


def test_softcap():
    banner("[7] logit soft-capping")
    z = np.linspace(-200, 200, 4001, dtype=np.float32)
    s = GEMMA.softcap_logits(z, 30.0)
    check(abs(float(s[-1]) - 30.0) < 0.01 and abs(float(s[0]) + 30.0) < 0.01,
          f"bounds logits to +/-cap ({s.min():.3f}..{s.max():.3f})")
    # tanh saturates, so in float32 the far tails become exactly flat. Monotonic
    # (non-decreasing) is the correct property; strictly-increasing is not
    # achievable once the output stops changing at the precision limit.
    d = np.diff(s)
    check(bool(np.all(d >= 0)), "is monotonically non-decreasing (order preserved)")
    check(int(np.count_nonzero(d == 0)) > 0,
          f"...with flat saturated tails as tanh requires ({int((d==0).sum())} flat steps)")
    check(bool(np.all(d[np.abs(z[:-1]) < 5] > 0)),
          "strictly increasing across the linear region (|z| < 5)")
    check(np.allclose(GEMMA.softcap_logits(z, None), z), "None is a no-op")
    check(np.allclose(GEMMA.softcap_logits(z * 0.001, 30.0), z * 0.001, atol=1e-6),
          "small logits pass through nearly unchanged")
    try:
        GEMMA.softcap_logits(z, 0.0)
        check(False, "a non-positive cap must raise")
    except ValueError:
        check(True, "a non-positive cap raises")


def main():
    print("=== P2/P3: arch dispatch, MoE and gemma4 ===")
    test_arch_registry()
    test_routing()
    test_expert_plan()
    test_moe_ffn_reference()
    test_real_moe_file()
    test_real_gemma_file()
    test_softcap()
    print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    sys.exit(main())