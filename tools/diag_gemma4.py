"""Localise the gemma-4-12b incoherence: per-layer RMS trace + logit stats.

Run from the repo root:
    AERO_MAX_RAM=5G .venv-cml/bin/python tools/diag_gemma4.py
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from engine.gguf import GGUFReader
from engine import model as M
from engine.tokenizer import Tokenizer

P = '/Users/cyberdeth/.cache/aero-models/gemma-4-12b-it-IQ4_XS.gguf'
PROMPT = sys.argv[1] if len(sys.argv) > 1 else 'The capital of France is'

with GGUFReader(P) as r:
    tk = Tokenizer.from_gguf(r)
    ids = tk.encode(PROMPT)
    print('prompt   :', repr(PROMPT))
    print('tokens   :', ids)
    print('pieces   :', [tk.decode([i]) for i in ids], flush=True)
    m = M.Model(r, block_rows=256, matmul_threads=8)
    tr = []
    M.GEMMA_TRACE = tr
    st = m.new_state(capacity=256)
    lg = m.forward(ids, st)
    M.GEMMA_TRACE = None

    print('\nlayer  win   rms_in   rms_att   rms_attn_out  rms_ffn   scale    rms_out')
    for (layer, a, b, c, d, s, e) in tr:
        win = m.cfg.window_at(layer)
        print(f'{layer:5d} {str(win):>5s} {a:9.4f} {b:9.4f} {c:13.4f} {d:9.4f} '
              f'{s:8.4f} {e:9.4f}')
    print('logits: min %.3f max %.3f std %.3f' % (lg.min(), lg.max(), lg.std()))
    o = np.argsort(-lg)
    print('top10:', [(tk.decode([int(i)]), round(float(lg[i]), 3)) for i in o[:10]])
    g = []
    for _ in range(12):
        n = int(np.argmax(lg)); g.append(n); lg = m.forward([n], st)
    print('greedy:', repr(tk.decode(g)))
