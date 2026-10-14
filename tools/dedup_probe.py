#!/usr/bin/env python3
"""Dedup investigation: where does the work actually go, and what can be reused?

This MEASURES before proposing anything. Three questions:

  Q1  KV: which layers actually hold a growing cache? (SSM layers do not.)
  Q2  KV: can the SSM layers' constant state replace the attention cache?
  Q3  load: how much is re-read per token that was already read?

Run: .venv-cml/bin/python tools/dedup_probe.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine import gguf as G   # noqa: E402
from engine import model as M  # noqa: E402

QWEN = ("/Volumes/Backup/lmstudio/models/empero-ai/"
        "Qwen3.8-4B-Distill-GGUF/Qwen3.8-4B-Q4_K_M.gguf")
MOE = ("/Volumes/Backup/lmstudio/models/tvall43/"
       "Qwen3.6-14B-A3B-FableVibes-GGUF/Qwen3.6-14B-A3B-FableVibes-Q4_K_M.gguf")


def banner(t):
    print(f"\n{'='*72}\n{t}\n{'='*72}")


def q1_kv_growth(path, label):
    banner(f"Q1  {label}: which layers hold a growing cache?")
    with G.GGUFReader(path) as r:
        mc = M.ModelConfig.from_gguf(r)
        n_attn = len(mc.attn_layers)
        n_ssm = len(mc.ssm_layers)
        kv_per_pos = n_attn * 2 * mc.n_head_kv * mc.head_dim * 4
        ssm_bytes = 0
        if mc.ssm_cfg:
            c = mc.ssm_cfg
            # one state per SSM layer: (n_v_heads, state, state)
            ssm_bytes = n_ssm * c.n_v_heads * c.state_size * c.state_size * 4

        print(f"  layers           : {mc.n_layer} total, {n_attn} attn, {n_ssm} ssm")
        print(f"  KV per position  : {kv_per_pos:>12,} B  ({kv_per_pos/1024:.1f} KiB)")
        print(f"  SSM state total  : {ssm_bytes:>12,} B  ({ssm_bytes/2**20:.2f} MiB) FIXED")
        print()
        print(f"  {'context':>10}  {'KV cache':>12}  {'SSM state':>12}  {'ratio':>7}")
        for ctx in (1024, 4096, 16384, 65536, 262144):
            kv = kv_per_pos * ctx
            print(f"  {ctx:>10}  {kv/2**20:>10.1f}M  {ssm_bytes/2**20:>10.2f}M  "
                  f"{kv/max(ssm_bytes,1):>6.0f}x")
        print()
        # the finding: only a FRACTION of layers grow at all
        frac = n_attn / mc.n_layer
        print(f"  => only {n_attn}/{mc.n_layer} layers ({frac*100:.0f}%) grow with "
              f"context; {n_ssm} are O(1)")
        if n_attn and n_ssm:
            print(f"  => if all layers used KV the cache would be "
                  f"{mc.n_layer/n_attn:.2f}x larger than it is")
        return mc


def q2_kv_sharing(path, label):
    banner(f"Q2  {label}: does the per-layer KV layout waste anything?")
    with G.GGUFReader(path) as r:
        mc = M.ModelConfig.from_gguf(r)
        # GQA already dedups: n_head_kv < n_head means K/V are shared across
        # query heads. Quantify how much that already saves.
        share = mc.n_head / mc.n_head_kv
        per_pos_full = len(mc.attn_layers) * 2 * mc.n_head * mc.head_dim * 4
        per_pos_gqa = len(mc.attn_layers) * 2 * mc.n_head_kv * mc.head_dim * 4
        print(f"  n_head={mc.n_head} n_head_kv={mc.n_head_kv} -> "
              f"{share:.1f} query heads per KV head")
        print(f"  KV WITHOUT sharing : {per_pos_full:>12,} B/pos")
        print(f"  KV WITH GQA sharing: {per_pos_gqa:>12,} B/pos")
        print(f"  => GQA already dedups K/V by {share:.1f}x")
        print()
        # a KV cache is only read at the CURRENT position during decode; history
        # is read but never rewritten. So the win available is in READ volume.
        print(f"  decode read volume per token (attention): "
              f"{per_pos_gqa * 4096 / 2**20:.1f} MiB at 4096 ctx")
        print(f"  ...of which the CURRENT position's row is "
              f"{per_pos_gqa/1024:.1f} KiB ({(per_pos_gqa)/(per_pos_gqa*4096)*100:.3f}%)")
        return mc


def q3_reread(path, label):
    banner(f"Q3  {label}: what does each decode step re-read?")
    with G.GGUFReader(path) as r:
        mc = M.ModelConfig.from_gguf(r)
        # per-layer weights are read EVERY token and released. That is the
        # dominant cost, and it is pure re-read.
        blk = {}
        for name, info in r.tensors.items():
            if name.startswith("blk."):
                b = name.split(".")[1]
                from engine.quant import ggml_type_size
                blk[b] = blk.get(b, 0) + ggml_type_size(info.type_id, info.n_elements)
        total = sum(blk.values())
        emb = r.tensors["token_embd.weight"]
        from engine.quant import ggml_type_size
        emb_b = ggml_type_size(emb.type_id, emb.n_elements)

        print(f"  weights re-read per token: {total/2**20:>8.1f} MiB "
              f"(all {mc.n_layer} layers)")
        print(f"  embedding row read       : {emb_b/2**20:>8.1f} MiB "
              f"(only 1 of {mc.vocab:,} rows is needed)")
        print()
        print(f"  {'cache size':>12}  {'re-read/token':>14}  {'saving':>8}")
        for frac, name in ((0.0, "none"), (0.25, "25%"), (0.5, "50%"),
                           (1.0, "all")):
            cached = total * frac
            red = total - cached
            print(f"  {name:>12}  {red/2**20:>12.1f}M  "
                  f"{(total/max(red,1e-9)):>7.2f}x")
        print()
        print(f"  => a whole-model weight cache eliminates "
              f"{total/2**20:.0f} MiB of re-read per token, but costs "
              f"{total/2**20:.0f} MiB resident. On a 4.5 GB budget that is the "
              f"central tradeoff, not a free win.")
        return mc


def main():
    if not os.path.exists(QWEN):
        print("model not present")
        return 1
    mc = q1_kv_growth(QWEN, "Qwen3.8-4B")
    if os.path.exists(MOE):
        q1_kv_growth(MOE, "Qwen3.6-14B-A3B")
    q2_kv_sharing(QWEN, "Qwen3.8-4B")
    q3_reread(QWEN, "Qwen3.8-4B")
    print("\ndone")
    return 0


if __name__ == "__main__":
    sys.exit(main())
