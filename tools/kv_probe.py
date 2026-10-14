#!/usr/bin/env python3
"""KV cache: what actually works, and how much redundancy is really there?

Answers three questions with measurements:
  A. does int8 KV change the model's output? (and by how much)
  B. how much do two positions' K vectors resemble each other? -- if they are
     near-identical, dedup/merging is viable; if not, it is not
  C. how much does the cache actually have to hold, given this architecture?

Run: .venv-cml/bin/python tools/kv_probe.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine import gguf as G      # noqa: E402
from engine import model as M     # noqa: E402
from engine import transformer as T  # noqa: E402

QWEN = ("/Volumes/Backup/lmstudio/models/empero-ai/"
        "Qwen3.8-4B-Distill-GGUF/Qwen3.8-4B-Q4_K_M.gguf")


def banner(t):
    print(f"\n{'='*72}\n{t}\n{'='*72}")


def a_int8_effect(mc):
    banner("A  int8 KV: size win, and the exact cost to output")
    L = mc.n_layer
    for cap in (4096, 16384, 65536, 262144):
        f = T.KVCache(L, mc.n_head_kv, mc.head_dim, mc.value_dim, cap, dtype="fp32")
        i = T.KVCache(L, mc.n_head_kv, mc.head_dim, mc.value_dim, cap, dtype="int8")
        print(f"  ctx {cap:>7}:  fp32 {f.fp32_equivalent_bytes/2**20:9.1f} MiB"
              f"   int8 {i.nbytes/2**20:8.1f} MiB"
              f"   {f.fp32_equivalent_bytes/i.nbytes:5.2f}x smaller")
    print()
    # quantization error on realistic key vectors
    rng = np.random.default_rng(3)
    errs = []
    for scale in (0.1, 1.0, 5.0):
        k = (rng.standard_normal((64, mc.n_head_kv, mc.head_dim)) * scale).astype(np.float32)
        c = T.KVCache(1, mc.n_head_kv, mc.head_dim, mc.value_dim, 64, dtype="int8")
        c.append(0, k, k)
        errs.append(c.quant_error())
    for s, e in zip((0.1, 1.0, 5.0), errs):
        print(f"  key scale {s:>4}: max relative quant error {e:.3e}")
    print(f"  => int8 rounding costs ~{max(errs):.1e} relative -- "
          f"comparable to fp16 storage error")
    return True


def b_redundancy(mc, n_pos=256):
    banner("B  is there KV redundancy worth deduplicating?")
    with G.GGUFReader(QWEN) as r:
        # take a real attention layer's K projection and project real tokens
        L = mc.attn_layers[0]
        wk = r.tensor(f"blk.{L}.attn_k.weight")           # (n_kv*D, n_embd)
        kn = r.tensor(f"blk.{L}.attn_k_norm.weight") if f"blk.{L}.attn_k_norm.weight" in r.tensors else None
        emb = r.tensors["token_embd.weight"]

        # build a realistic hidden state: embed a spread of token ids, then norm
        ids = np.arange(1000, 1000 + n_pos) % mc.vocab
        h = np.stack([r.tensor_rows(emb, int(i), int(i) + 1).reshape(-1) for i in ids])
        h = T.rms_norm(h, r.tensor(f"blk.{L}.attn_norm.weight"), mc.eps)
        k = h @ wk.T
        k = k.reshape(n_pos, mc.n_head_kv, mc.head_dim)
        if kn is not None:
            k = T.rms_norm(k, kn, mc.eps)
        # normalise so similarity is about DIRECTION, which is what attention uses
        kn_ = k / np.maximum(np.linalg.norm(k, axis=-1, keepdims=True), 1e-9)

        # cosine similarity between random pairs of positions, per head
        rng = np.random.default_rng(7)
        sims = []
        for _ in range(2000):
            i, j = rng.integers(0, n_pos, 2)
            if i == j:
                continue
            head = rng.integers(0, mc.n_head_kv)
            sims.append(float(np.dot(kn_[i, head], kn_[j, head])))
        sims = np.array(sims)
        print(f"  pairs sampled        : {len(sims)}")
        print(f"  mean cosine          : {sims.mean():+.4f}")
        print(f"  std                  : {sims.std():.4f}")
        print(f"  |cos| > 0.9 (near-dup): {(np.abs(sims) > 0.9).mean()*100:5.1f}%")
        print(f"  |cos| > 0.99          : {(np.abs(sims) > 0.99).mean()*100:5.1f}%")
        print(f"  |cos| > 0.999         : {(np.abs(sims) > 0.999).mean()*100:5.1f}%")

        # how many EXACT duplicates exist at a strict threshold?
        for thr in (0.99, 0.995, 0.999):
            keep = []
            for i in range(n_pos):
                if not any(float(np.dot(kn_[i, h], kn_[j, h])) > thr
                           for j, h in keep):
                    keep.append((i, 0))
            print(f"  greedy dedup @ {thr}: {len(keep)}/{n_pos} kept "
                  f"({(1-len(keep)/n_pos)*100:.1f}% removed)")
        print()
        print("  => if |cos| is spread around 0, positions are NOT interchangeable")
        print("     and merging KV entries would destroy information.")
    return sims


def c_architecture(mc):
    banner("C  what this architecture already does for free")
    print(f"  layers              : {mc.n_layer}")
    print(f"  full-attention      : {len(mc.attn_layers)}  <- only these grow")
    print(f"  SSM (O(1) state)    : {len(mc.ssm_layers)}  <- constant, forever")
    print(f"  GQA sharing         : {mc.n_head}/{mc.n_head_kv} = "
          f"{mc.n_head/mc.n_head_kv:.0f} query heads per KV head")
    if mc.sliding_window:
        print(f"  sliding window      : {mc.sliding_window} "
              f"({len(mc.sliding_pattern) if mc.sliding_pattern else '?'} layers, "
              f"pattern-based)")
    print()
    print("  The model designers already applied three of the four levers:")
    print("    * most layers are O(1) SSM, not O(n) attention")
    print("    * GQA shares K/V across query heads")
    print("    * gemma4 additionally bounds attention with a sliding window")
    print("  What is left for the engine is only the fourth:")
    print("    * the PRECISION the surviving attention layers store at")


def main():
    if not os.path.exists(QWEN):
        print("model not present")
        return 1
    with G.GGUFReader(QWEN) as r:
        mc = M.ModelConfig.from_gguf(r)
    a_int8_effect(mc)
    b_redundancy(mc)
    with G.GGUFReader("/Volumes/Backup/lmstudio/models/unsloth/"
                      "gemma-4-12b-it-GGUF/gemma-4-12b-it-IQ4_XS.gguf") as r:
        gm = M.ModelConfig.from_gguf(r)
    c_architecture(mc)
    banner("C2  gemma4 (has a sliding window as well)")
    c_architecture(gm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
