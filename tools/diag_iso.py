"""Isolate gemma-4-12b's failure: T=1 (pos 0) vs T>1.

At position 0 RoPE is the identity (cos=1, sin=0 for every pair, freq_factors or
not) and attention over a single key is exactly V. So a 1-token prompt exercises
embeddings -> per-layer norms/FFN -> final norm -> tied lm_head, and NOTHING of
the positional/attention machinery. If 1-token output is sensible and 5-token
output is not, the bug is in attention or RoPE.

Run from the repo root:
    AERO_MAX_RAM=5G .venv-cml/bin/python tools/diag_iso.py
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from engine.gguf import GGUFReader
from engine import model as M
from engine.tokenizer import Tokenizer

P = '/Users/cyberdeth/.cache/aero-models/gemma-4-12b-it-IQ4_XS.gguf'
CASES = ['France', 'The', 'Paris', 'The capital of France is',
         'Once upon a time']


def topk(tk, lg, k=6):
    return [(tk.decode([int(i)]), round(float(lg[i]), 2))
            for i in np.argsort(-lg)[:k]]


with GGUFReader(P) as r:
    tk = Tokenizer.from_gguf(r)
    m = M.Model(r, block_rows=256, matmul_threads=8)
    for pr in CASES:
        ids = tk.encode(pr)
        st = m.new_state(capacity=256)
        lg = m.forward(ids, st)
        print(f'\n{pr!r}  T={len(ids)} tokens={ids}')
        print('   pieces:', [tk.decode([i]) for i in ids])
        print('   top6  :', topk(tk, lg))
        # a second forward on a fresh state with the SAME single token appended
        # (position 1) shows what a *second* position does when it is the only
        # thing attention can look at besides itself.
        if len(ids) == 1:
            st2 = m.new_state(capacity=256)
            m.forward(ids, st2)
            lg2 = m.forward(ids, st2)
            print('   pos1 (same tok repeated):', topk(tk, lg2))
