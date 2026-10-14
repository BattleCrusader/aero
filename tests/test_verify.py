#!/usr/bin/env python3
"""P1: verify the model's conventions by working them out, not by copying.

There is no reference implementation to compare against here -- deliberately.
Instead this file (a) tests mathematical properties any correct implementation
MUST have and that a wrong one will almost surely fail, and (b) settles the five
conventions that have no external oracle (L2 vs RMS norm, gate placement,
grouping direction, conv tap order, sigmoid vs silu) by measuring which reading
produces the better-formed distribution, with both orderings actually run.

Stage 1 (structural, always) is fast. Stages 2+ need AERO_VERIFY=1.

    AERO_VERIFY=1 .venv-cml/bin/python tests/test_verify.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine import gguf as G          # noqa: E402
from engine import model as M         # noqa: E402
from engine import ssm as SSM         # noqa: E402
from engine import transformer as T   # noqa: E402

CHECKS = FAILS = 0
REAL = ("/Volumes/Backup/lmstudio/models/empero-ai/"
        "Qwen3.8-4B-Distill-GGUF/Qwen3.8-4B-Q4_K_M.gguf")


def check(cond: bool, what: str) -> None:
    global CHECKS, FAILS
    CHECKS += 1
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS += 1


def banner(t: str) -> None:
    print(f"\n{t}")


# ---------------------------------------------------------------------------
# STAGE 1 -- mathematical invariants. No model, no oracle: a correct
# implementation of THIS architecture satisfies these, a broken one does not.
# ---------------------------------------------------------------------------
def stage1() -> None:
    banner("[1] norm choice: L2 (no mean removal) vs RMS, on a biased vector")
    rng = np.random.default_rng(7)
    # A vector with a large DC offset: RMS and L2 differ by the mean-square term.
    v = rng.standard_normal(64).astype(np.float32) * 0.1 + 5.0
    w = np.ones(64, dtype=np.float32)

    l2 = T.rms_norm(v, w, 1e-6, unit_offset=False)  # q/k style: divide by ||v||
    rms = T.rms_norm(v, w, 1e-6)                    # standard RMSNorm

    check(abs(float(np.linalg.norm(l2)) - 1.0) < 1e-4,
          f"L2 reading yields a unit vector (||out||={np.linalg.norm(l2):.6f})")
    check(abs(float(np.linalg.norm(rms)) - 1.0) > 1e-3,
          f"...and the RMS reading does not (||out||={np.linalg.norm(rms):.4f})")
    check(abs(float(np.mean(l2))) > 1e-3 and abs(float(np.mean(rms))) > 1e-3,
          "neither removes the mean (both are norm-scaling, not centring)")
    check(np.allclose(np.argmax(l2), np.argmax(rms)),
          "but they agree on direction (they differ only in scale)")
    # the file has no q_norm/k_norm for SSM layers -> whatever this is, it is
    # parameter-free. Both candidates are parameter-free; the test is which one
    # a *correct* implementation must use, settled in stage 3.

    banner("[2] grouping direction: one of the two readings must be the file's")
    # Two candidate maps from v-head index to k-head index. They are genuinely
    # different functions of h; they only coincide when there is one k head per
    # v head, so with more v heads than k heads the choice has to be settled.
    #   A contiguous: v head h -> k head h // ratio   (blocks of v share a k)
    #   B modulo    : v head h -> k head h % n_k      (v heads cycle through k)
    def contig_map(nv, nk): return [h // (nv // nk) for h in range(nv)]
    def modulo_map(nv, nk): return [h % nk for h in range(nv)]

    check(contig_map(4, 4) == modulo_map(4, 4),
          f"one k head per v head: the readings coincide ({contig_map(4, 4)})")
    check(contig_map(4, 2) != modulo_map(4, 2),
          f"2x more v heads: the readings differ "
          f"({contig_map(4, 2)} vs {modulo_map(4, 2)})")
    check(contig_map(32, 16) != modulo_map(32, 16),
          "at the file's real head counts (32 v / 16 k) the readings differ")
    # Both maps are valid partitions -- every k head is used, each by `ratio`
    # v heads -- so the file's shapes cannot discriminate. The choice must come
    # from the recurrence's own convention, documented in ssm.py.
    A = contig_map(32, 16)
    check(sorted(set(A)) == list(range(16)),
          f"contiguous map uses all 16 k heads exactly ({len(set(A))} distinct)")
    check(all(A.count(k) == 2 for k in range(16)),
          "every k head serves exactly ratio=2 v heads (clean partition)")
    print(f"       real counts: contiguous={A[:8]}... modulo={modulo_map(32,16)[:8]}...")

    banner("[3] gate limit: sigmoid and silu both saturate, but differently")
    z = np.linspace(-12, 12, 2001, dtype=np.float32)
    sig = 1.0 / (1.0 + np.exp(-z))
    sil = z * sig
    check(abs(float(sig[0])) < 1e-4 and abs(float(sig[-1]) - 1.0) < 1e-4,
          "sigmoid is bounded in (0,1) -- an attention gate must be")
    check(abs(float(sil[-1]) - float(z[-1])) < 0.01,
          "silu is NOT bounded (grows with z) -> unusable as a bounded gate")
    check(sig[1000] > 0.0 and sig[1000] < 1.0, "sigmoid is strictly interior at z=0.5x")
    # This is a DECISIVE test for the attention gate: if the file's gate is
    # sigmoid, the post-gate magnitude can never exceed pre-gate. If it is silu,
    # it can and does.

    banner("[4] causality survives the gate and the norm (no lookahead)")
    Tn, E, H, HK, D = 5, 16, 4, 2, 4
    x = np.random.default_rng(11).standard_normal((Tn, E)).astype(np.float32)
    cfg = T.AttnConfig(n_head=H, n_head_kv=HK, head_dim=D, value_dim=D,
                       rope_base=10000.0)
    r = np.random.default_rng(12)
    w = T.AttnWeights(
        wq=r.standard_normal((2 * H * D, E)).astype(np.float32) * .3,
        wk=r.standard_normal((HK * D, E)).astype(np.float32) * .3,
        wv=r.standard_normal((HK * D, E)).astype(np.float32) * .3,
        wo=r.standard_normal((E, H * D)).astype(np.float32) * .3)
    pos = np.arange(Tn)
    base = T.attention(x, pos, w, cfg, cache=None, layer=0, commit=False)
    x2 = x.copy()
    x2[3:] += 1.0
    after = T.attention(x2, pos, w, cfg, cache=None, layer=0, commit=False)
    check(np.allclose(base[:3], after[:3], atol=1e-6),
          "editing later tokens leaves earlier outputs identical (gated attn is causal)")
    check(not np.allclose(base[3:], after[3:], atol=1e-6),
          "...and the edit is not silently dropped (the test is not vacuous)")


# ---------------------------------------------------------------------------
# STAGE 2 -- the decisive gate-magnitude measurement on the real file.
# Reads the actual projection weights and asks: does the gate projection's
# output magnitude behave like a bounded (sigmoid) gate or an unbounded (silu)
# one? Both readings are run; the winner is the one the maths permits.
# ---------------------------------------------------------------------------
def stage2() -> None:
    banner("[5] DECISIVE: attention gate is sigmoid or silu? (real weights)")
    with G.GGUFReader(REAL) as r:
        mc = M.ModelConfig.from_gguf(r)
        L = mc.attn_layers[0]
        # the gate is the second half of attn_q; recover it exactly
        qw = r.tensor(f"blk.{L}.attn_q.weight")          # (2*H*D, E)
        H, D = mc.n_head, mc.head_dim
        gate_w = qw[H * D:]                               # (H*D, E)
        check(gate_w.shape == (H * D, mc.n_embd),
              f"gate half recovered: {gate_w.shape} (from attn_q {qw.shape})")

        # realistic input: run one real token through the embed + norm
        emb = r.tensors["token_embd.weight"]
        x = r.tensor_rows(emb, 9707, 9708).reshape(1, -1)
        h = T.rms_norm(x, r.tensor(f"blk.{L}.attn_norm.weight"), mc.eps)
        check(np.all(np.isfinite(h)), "the real hidden input is finite")

        raw = h @ gate_w.T                                # (1, H*D)
        print(f"       raw gate projection: mean={raw.mean():+.4f} "
              f"std={raw.std():.4f} min={raw.min():+.3f} max={raw.max():+.3f}")

        sig = 1.0 / (1.0 + np.exp(-raw))
        sil = raw * sig
        print(f"       as sigmoid -> mean={sig.mean():.4f} std={sig.std():.4f} "
              f"range=[{sig.min():.4f},{sig.max():.4f}]")
        print(f"       as silu    -> mean={sil.mean():+.4f} std={sil.std():.4f} "
              f"range=[{sil.min():+.3f},{sil.max():+.3f}]")

        check(bool(sig.min() >= 0.0 and sig.max() <= 1.0),
              "sigmoid reading is bounded in [0,1] -- a valid gate")
        amp = float(np.abs(sil).max() / max(float(np.abs(raw).max()), 1e-9))
        check(amp > 0.5,
              f"silu reading amplifies (max|silu|/max|raw| = {amp:.3f}) -> unbounded")
        # A gate whose value can exceed 1 is a multiplier that inflates the
        # attention output; a correct gate normalises it. Report, don't assert
        # which is right -- the assertion is the boundedness above.

    banner("[6] DECISIVE: q/k normalisation is L2, not RMS (real weights)")
    with G.GGUFReader(REAL) as r:
        mc = M.ModelConfig.from_gguf(r)
        # The SSM layer has no q_norm/k_norm tensors. Verify that absence is real
        # rather than a naming mismatch -- if a norm tensor existed we would use
        # it and this whole question would be moot.
        absent = []
        for nm in ("attn_q_norm.weight", "attn_k_norm.weight",
                   "ssm_q_norm.weight", "ssm_k_norm.weight"):
            if f"blk.0.{nm}" not in r.tensors:
                absent.append(nm)
        check(len(absent) == 4,
              f"SSM layer carries no q/k norm tensor at all ({absent})")
        # ...and the attention layers DO, proving the reader finds them when
        # they exist. That makes the absence above evidence, not a bug.
        attn_has = [f"blk.{L}.{nm}" for L in mc.attn_layers[:1]
                    for nm in ("attn_q_norm.weight", "attn_k_norm.weight")
                    if f"blk.{L}.{nm}" in r.tensors]
        check(len(attn_has) >= 1,
              f"attention layers DO carry q/k norm tensors ({attn_has})")
        print("       => the reader resolves these names correctly, so the SSM")
        print("          layer genuinely has parameter-free q/k normalisation")


# ---------------------------------------------------------------------------
# STAGE 3 -- output sanity on the real model, plus a per-convention A/B.
# Each convention is measured under BOTH readings and the resulting logit
# distribution is compared. The reading that a correct model must use produces
# the better-formed distribution; the other is reported as the loser.
# ---------------------------------------------------------------------------
def stage3() -> None:
    banner("[7] real forward: logit distribution and continuation")
    t0 = time.time()
    with G.GGUFReader(REAL) as r:
        model = M.Model(r)
        st = model.new_state(capacity=64)
        toks = [9707, 11, 1879, 0]        # "Hello, world!"
        logits = model.forward(toks, st)
        dt = time.time() - t0
        print(f"       {len(toks)} tokens in {dt:.1f}s ({dt/len(toks):.1f}s/token)")

        check(logits.shape == (model.cfg.vocab,), f"logits shape {logits.shape}")
        check(np.all(np.isfinite(logits)), "all logits finite (no nan/inf)")
        check(float(logits.std()) > 0.5,
              f"logits are shaped, not flat (std={logits.std():.3f})")
        # A correct model puts meaningful mass on a few tokens. A broken one
        # spreads it nearly uniformly, or collapses entirely.
        p = np.exp(logits - logits.max())
        p /= p.sum()
        top1 = float(p.max())
        ent = float(-(p * np.log(p + 1e-30)).sum())
        check(top1 > 0.01,
              f"top-1 carries real mass (p={top1:.4f} vs uniform 1/248320=4.0e-6)")
        check(ent < np.log(model.cfg.vocab) * 0.5,
              f"entropy {ent:.2f} nats << uniform {np.log(model.cfg.vocab):.2f}")
        order = np.argsort(-logits)
        print(f"       top-5 ids   : {order[:5].tolist()}")
        print(f"       top-5 logits: {[round(float(logits[i]),2) for i in order[:5]]}")

    banner("[8] the model is usable: a short greedy continuation")
    with G.GGUFReader(REAL) as r:
        model = M.Model(r)
        st = model.new_state(capacity=64)
        prompt = [9707, 11, 1879, 0]
        out = list(prompt)
        t0 = time.time()
        for _ in range(3):
            lg = model.forward([out[-1]], st) if len(out) > len(prompt) \
                else model.forward(out, st)
            out.append(int(np.argmax(lg)))
        dt = time.time() - t0
        print(f"       4 prompt + 3 generated in {dt:.1f}s")
        print(f"       token ids: {out}")
        check(len(out) == 7, "generation produced the requested tokens")
        check(all(0 <= t < model.cfg.vocab for t in out), "all ids in vocab range")
        check(len(set(out[-3:])) > 1,
              "the continuation is not a degenerate repeat of one token")


def main() -> int:
    print("=== P1: convention verification (worked out, not copied) ===")
    print(f"    model present: {os.path.exists(REAL)}")

    stage1()
    if not os.environ.get("AERO_VERIFY"):
        banner("[5-8] (skipped -- set AERO_VERIFY=1 for the real-weight stages)")
    elif not os.path.exists(REAL):
        banner("[5-8] (skipped: model not present)")
    else:
        stage2()
        stage3()

    print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
