"""Sweep the un-verified gemma4 structural knobs on the real file, one forward each.

Cheap and decisive: a coherent model puts ' Paris' at/near the top for
'The capital of France is'; a broken one puts punctuation or foreign script.
Each variant is a fresh state, so nothing leaks between them.
"""
import sys, time
sys.path.insert(0, '.')
import numpy as np
from engine.gguf import GGUFReader
from engine import model as M
from engine.tokenizer import Tokenizer

p = '/Users/cyberdeth/.cache/aero-models/gemma-4-12b-it-IQ4_XS.gguf'
PROMPT = 'The capital of France is'

VARIANTS = [
    ("sum",      "full_only", "sqrt_embd"),
    ("none",     "full_only", "sqrt_embd"),
    ("deltas",   "full_only", "sqrt_embd"),
    ("sum",      "none",      "sqrt_embd"),
    ("sum",      "all",       "sqrt_embd"),
    ("sum",      "full_only", "none"),
]

with GGUFReader(p) as r:
    tk = Tokenizer.from_gguf(r)
    m = M.Model(r, block_rows=256, matmul_threads=8)
    ids = tk.encode(PROMPT)
    print('prompt tokens', ids)
    for out_scale, freq, emb in VARIANTS:
        M.GEMMA_OUT_SCALE, M.GEMMA_ROPE_FREQS, M.GEMMA_EMBED_SCALE = out_scale, freq, emb
        M.GEMMA_TRACE = []
        t0 = time.time()
        st = m.new_state(capacity=256)
        lg = m.forward(ids, st)
        M.GEMMA_TRACE = None
        top = [tk.decode([int(i)]) for i in np.argsort(-lg)[:6]]
        print(f"out_scale={out_scale:7s} freqs={freq:9s} emb={emb:10s} "
              f"-> {top}  logit[{lg.min():.1f},{lg.max():.1f}]  {time.time()-t0:.0f}s")
