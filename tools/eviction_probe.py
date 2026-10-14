#!/usr/bin/env python3
"""Would evicting least-used K/V pairs work? Measuring attention concentration.

Eviction (StreamingLLM / H2O) is only viable if attention mass is CONCENTRATED:
if a small fraction of positions carries most of the weight, the rest can be
dropped with little loss. If mass is spread evenly, every position carries
unique information and dropping any of them destroys it.

This captures real post-rope q and k from a real forward pass and measures the
attention distribution across positions.

Also measured: whether there is an "attention sink" (position 0 dominating),
which is what makes StreamingLLM's always-keep-the-first-k rule work.

Run: .venv-cml/bin/python tools/eviction_probe.py
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


def capture_qk(mc, path, toks):
    """Run a real forward and capture post-rope q and k per attention layer."""
    with G.GGUFReader(path) as r:
        model = M.Model(r)
        st = model.new_state(capacity=128)
        cap = {}

        # intercept append to grab k, and wrap attention to grab q
        orig_append = st.kv.append

        def spy_append(layer, k, v):
            cap.setdefault(layer, {})["k"] = np.array(k)
            return orig_append(layer, k, v)

        st.kv.append = spy_append

        orig_attn = T.attention

        def spy_attn(x, pos, w, cfg, cache=None, layer=0, **kw):
            # recompute q exactly as attention() does, so we see what it used
            q_raw = T.linear(x, w.wq)
            if q_raw.shape[-1] == 2 * cfg.n_head * cfg.head_dim:
                q_raw = np.split(q_raw, 2, axis=-1)[0]
            q = q_raw.reshape(len(pos), cfg.n_head, cfg.head_dim)
            if w.q_norm is not None:
                q = T.rms_norm(q, w.q_norm, cfg.eps)
            cos, sin = T.rope_tables(cfg.head_dim, pos, cfg.rope_base,
                                     cfg.rope_dims)
            cap.setdefault(layer, {})["q"] = T.apply_rope(q, cos, sin)
            return orig_attn(x, pos, w, cfg, cache=cache, layer=layer, **kw)

        # model.py binds `attention` at import time, so patching T.attention
        # alone would not be seen -- patch the binding the model actually calls.
        import engine.model as MM
        MM.attention = spy_attn
        try:
            model.forward(toks, st)
        finally:
            MM.attention = orig_attn
        return cap


def main():
    if not os.path.exists(QWEN):
        print("model not present")
        return 1
    with G.GGUFReader(QWEN) as r:
        mc = M.ModelConfig.from_gguf(r)

    n = 48
    toks = [(9707 + 137 * i) % mc.vocab for i in range(n)]
    banner(f"capturing real q/k over {n} positions, {len(mc.attn_layers)} attn layers")
    cap = capture_qk(mc, QWEN, toks)

    L = mc.attn_layers[0]
    q = cap[L]["q"]        # (n, H, D)
    k = cap[L]["k"]        # (n, HK, D)
    scale = 1.0 / np.sqrt(mc.head_dim)

    print(f"  layer {L}: q{q.shape} k{k.shape}")

    # GQA: expand k to match heads
    rep = mc.n_head // mc.n_head_kv
    kexp = np.repeat(k, rep, axis=1) if rep > 1 else k      # (n, H, D)

    # attention weights for the LAST position over all keys
    scores = np.einsum("hd,shd->sh", q[-1], kexp) * scale  # (n, H)
    banner("attention of the last token over all positions")
    allw = []
    for h in range(mc.n_head):
        w = T.softmax(scores[:, h])
        allw.append(w)
    allw = np.stack(allw)                                  # (H, n)

    # normalise per head so each head's distribution sums to 1
    mean_w = allw.mean(axis=0)
    order = np.argsort(-mean_w)
    print(f"  mean weights, top 10 positions: "
          f"{[int(i) for i in order[:10]]}")
    print(f"  their weights                 : "
          f"{[round(float(mean_w[i]),4) for i in order[:10]]}")
    print()

    # concentration: how many positions carry 50/90/99% of the mass?
    for frac in (0.5, 0.9, 0.99):
        counts = []
        for h in range(mc.n_head):
            c = np.cumsum(np.sort(allw[h])[::-1])
            counts.append(int(np.searchsorted(c, frac) + 1))
        print(f"  positions carrying {frac*100:>4.0f}% of mass: "
              f"median {int(np.median(counts)):>3} of {n} "
              f"({np.median(counts)/n*100:.1f}%)")
    print()

    # is there a sink at position 0?
    sink = float(allw[:, 0].mean())
    uniform = 1.0 / n
    print(f"  position 0 mean weight : {sink:.4f}  (uniform would be {uniform:.4f})")
    print(f"  ratio to uniform       : {sink/uniform:.1f}x")
    top_uniform = 1.0 / n
    print(f"  max single-position w  : {float(mean_w.max()):.4f} "
          f"({float(mean_w.max())/top_uniform:.1f}x uniform)")

    banner("simulate eviction: keep top-M positions, measure mass retained")
    for keep in (4, 8, 16, 24, 48):
        retained = []
        for h in range(mc.n_head):
            idx = np.argsort(-allw[h])[:keep]
            retained.append(float(allw[h][idx].sum()))
        r = float(np.mean(retained))
        print(f"  keep {keep:>3}/{n} positions ({keep/n*100:4.0f}%): "
              f"retains {r*100:5.1f}% of attention mass")

    banner("simulate a SLIDING WINDOW instead (keep the most RECENT M)")
    for keep in (4, 8, 16, 24, 48):
        r = float(np.mean([allw[h][-keep:].sum() for h in range(mc.n_head)]))
        print(f"  window {keep:>3}/{n} ({keep/n*100:4.0f}%): "
              f"retains {r*100:5.1f}% of attention mass")

    banner("verdict")
    print("  * 'least used' eviction needs a concentration to exploit; the")
    print("    table above is how much of the mass a given keep-count holds.")
    print("  * eviction is LOSSY and NOT reversible: dropped entries are gone,")
    print("    and the model was never trained to tolerate missing positions.")
    print("  * the model ALREADY has trained, per-channel forgetting in its")
    print(f"    {len(mc.ssm_layers)} SSM layers (exp(g) decay). That is eviction")
    print("    done properly -- learned, smooth, and part of training.")
    print("  * for the 9 attention layers a sliding window is the principled")
    print("    version, and gemma4 already ships one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
