#!/usr/bin/env python3
"""Model assembly self-test.

Fast structural checks by default. The REAL forward pass -- the only test that
can settle the unverified conventions -- is opt-in, because it is slow:

    AERO_REAL=1 .venv-cml/bin/python tests/test_model.py

Run:  .venv-cml/bin/python tests/test_model.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from tests._support import LockedOut, load_model, skip_note

from engine import gguf as G  # noqa: E402
from engine import model as M  # noqa: E402
from engine import transformer as T  # noqa: E402

CHECKS = 0
FAILS = 0

REAL = ("/Volumes/Backup/lmstudio/models/empero-ai/"
        "Qwen3.8-4B-Distill-GGUF/Qwen3.8-4B-Q4_K_M.gguf")


def check(cond: bool, what: str) -> None:
    global CHECKS, FAILS
    CHECKS += 1
    if cond:
        print(f"  ok   {what}")
    else:
        FAILS += 1
        print(f"  FAIL {what}")


rng = np.random.default_rng(3)


def main() -> int:
    print("=== model assembly self-test ===\n")

    # ------------------------------------------------- 1. gated attention
    print("[1] gated attention: [q | gate] split detected from the width")
    Tn, E, H, HK, D, DV = 4, 32, 4, 2, 8, 8
    x = rng.standard_normal((Tn, E)).astype(np.float32)
    cfg = T.AttnConfig(n_head=H, n_head_kv=HK, head_dim=D, value_dim=DV,
                       rope_base=10000.0)
    wk = rng.standard_normal((HK * D, E)).astype(np.float32) * 0.3
    wv = rng.standard_normal((HK * DV, E)).astype(np.float32) * 0.3
    wo = rng.standard_normal((E, H * DV)).astype(np.float32) * 0.3
    wq_plain = rng.standard_normal((H * D, E)).astype(np.float32) * 0.3
    wq_gated = rng.standard_normal((2 * H * D, E)).astype(np.float32) * 0.3
    pos = np.arange(Tn)

    o_plain = T.attention(x, pos, T.AttnWeights(wq=wq_plain, wk=wk, wv=wv, wo=wo), cfg)
    o_gated = T.attention(x, pos, T.AttnWeights(wq=wq_gated, wk=wk, wv=wv, wo=wo), cfg)
    check(o_plain.shape == (Tn, E), f"plain q output {o_plain.shape}")
    check(o_gated.shape == (Tn, E), f"gated q output {o_gated.shape}")
    check(not np.allclose(o_plain, o_gated),
          "the gate actually changes the result (split is real, not ignored)")
    # a gate pinned high must saturate towards the ungated value
    bad = rng.standard_normal((E, 3)).astype(np.float32)
    try:
        T.attention(x, pos, T.AttnWeights(wq=bad, wk=wk, wv=wv, wo=wo), cfg)
        check(False, "a wrong-width q projection must raise")
    except ValueError:
        check(True, "a wrong-width q projection raises instead of silently reshaping")

    # -------------------------------------------- 2. config vs the real file
    if not os.path.exists(REAL):
        print("\n[2] (skipped: model not present)")
    else:
        print("\n[2] ModelConfig from the real model")
        with G.GGUFReader(REAL) as r:
            mc = M.ModelConfig.from_gguf(r)
            print(f"       arch={mc.arch} layers={mc.n_layer} embd={mc.n_embd} "
                  f"head={mc.n_head}/{mc.n_head_kv} dim={mc.head_dim} "
                  f"ffn={mc.ffn_len} vocab={mc.vocab}")
            print(f"       rope base={mc.rope_base:.0f} dims={mc.rope_dims} "
                  f"eps={mc.eps:.1e} tied_output={mc.tied_output}")
            print(f"       attn layers={mc.attn_layers}")
            print(f"       ssm  layers={len(mc.ssm_layers)} of {mc.n_layer}")

            check(mc.tied_output, "output head is TIED (no output.weight in the file)")
            check(len(mc.attn_layers) + len(mc.ssm_layers) == mc.n_layer,
                  "every layer is classified")
            # blk.32 is an MTP/NextN head (it carries nextn.* tensors) and is NOT
            # part of the main decoder pass, so it must not appear here.
            check(mc.attn_layers == [3, 7, 11, 15, 19, 23, 27, 31],
                  f"full-attention DECODE layers match the file "
                  f"({mc.attn_layers})")
            check(mc.n_layer_total == 33 and mc.n_layer == 32,
                  f"33 blocks total, 32 decoded (MTP heads excluded)")
            check(mc.n_nextn == 1, f"one MTP head detected ({mc.n_nextn})")
            check(32 not in mc.attn_layers,
                  "the MTP head is excluded (running it corrupts the output)")
            check(mc.ssm_cfg is not None, "SSM config derived")
            if mc.ssm_cfg:
                check(mc.ssm_cfg.qkv_width ==
                      r.tensors["blk.0.attn_qkv.weight"].shape[0],
                      "ssm qkv width matches the file")
            # every layer's norm tensors must exist (the assembly reads them)
            missing = []
            for spec in mc.layers:
                for suffix in ("attn_norm.weight", "post_attention_norm.weight",
                               "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"):
                    if f"blk.{spec.index}.{suffix}" not in r.tensors:
                        missing.append(f"blk.{spec.index}.{suffix}")
            check(not missing, f"all per-layer tensors the assembly needs exist "
                               f"({len(missing)} missing)")
            # attention layers must have the gated q width
            for i in mc.attn_layers[:3]:
                qw = r.tensors[f"blk.{i}.attn_q.weight"].shape[0]
                check(qw == 2 * mc.n_head * mc.head_dim,
                      f"blk.{i}: attn_q={qw} == 2*{mc.n_head}*{mc.head_dim} (gated)")

            # ------------------------------------------ 3. state allocation
            print("\n[3] state allocation")
            try:
                model = load_model(r)
            except RuntimeError as e:
                if "already holds the model-load lock" in str(e):
                    print("\n=== SKIPPED: another aero process holds the "
                          "model-load lock (one load at a time is deliberate) ===")
                    return 0
                raise
            st = model.new_state(capacity=64)
            check(st.kv is not None, "KV cache allocated (attention layers exist)")
            check(st.kv.capacity == 64, "KV capacity is honoured (footprint known up front)")
            check(len(st.ssm_states) == len(mc.ssm_layers),
                  f"{len(st.ssm_states)} SSM states, one per SSM layer")
            check(len(st.gdn) == len(mc.ssm_layers), "one GDN state per SSM layer")
            print(f"       state bytes: {st.nbytes/1e6:.2f} MB at 64 positions")
            check(st.nbytes > 0, "state has a non-zero, known footprint")

    # ------------------------------------------------- 4. the real forward
    print("\n[4] REAL forward pass (opt-in: AERO_REAL=1)")
    if not os.environ.get("AERO_REAL"):
        print("       (skipped -- set AERO_REAL=1 to run; this is P1, the")
        print("        only test that can settle the unverified conventions)")
    elif not os.path.exists(REAL):
        print("       (skipped: model not present)")
    else:
        with G.GGUFReader(REAL) as r:
            try:
                model = load_model(r)
            except RuntimeError as e:
                if "already holds the model-load lock" in str(e):
                    print("       (skipped: model-load lock held by another process)")
                    return 0
                raise
            st = model.new_state(capacity=32)
            toks = [9707]        # "Hello" in many BPE vocabs; any id works
            t0 = time.time()
            logits = model.forward(toks, st)
            dt = time.time() - t0
            print(f"       {len(toks)} token(s) in {dt:.1f}s "
                  f"({dt/len(toks):.1f}s/token)")
            check(logits.shape == (model.cfg.vocab,), f"logits shape {logits.shape}")
            check(np.all(np.isfinite(logits)), "logits are all finite (no nan/inf)")
            check(float(logits.std()) > 1e-3,
                  f"logits are not degenerate (std={logits.std():.3f})")
            top = np.argsort(-logits)[:5]
            print(f"       top-5 token ids: {top.tolist()}")
            print(f"       top-5 logits   : {[round(float(logits[i]),2) for i in top]}")
            check(len(set(top.tolist())) == 5, "top-5 ids are distinct")

    print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
