"""Reproducible coherence check for gemma-4-12b through the aero engine."""
import sys, time
sys.path.insert(0, '.')
import numpy as np
from engine.gguf import GGUFReader
from engine import model as M
from engine.tokenizer import Tokenizer

p = sys.argv[1] if len(sys.argv) > 1 else \
    '/Users/cyberdeth/.cache/aero-models/gemma-4-12b-it-IQ4_XS.gguf'
PROMPTS = ['The capital of France is', 'Once upon a time']
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12

with GGUFReader(p) as r:
    tk = Tokenizer.from_gguf(r)
    m = M.Model(r, block_rows=256, matmul_threads=8)
    print('cfg: embed_scale=%r rope_freqs=%s' % (
        m.cfg.embed_scale,
        None if m.cfg.rope_freqs is None else m.cfg.rope_freqs.shape))
    for pr in PROMPTS:
        t0 = time.time()
        st = m.new_state(capacity=256)
        ids = tk.encode(pr)
        lg = m.forward(ids, st)
        top = [tk.decode([int(i)]) for i in np.argsort(-lg)[:5]]
        print(f'\nprompt {pr!r} tokens={ids}')
        print('  top5:', top)
        g = []
        for _ in range(N):
            n = int(np.argmax(lg))
            g.append(n)
            lg = m.forward([n], st)
        print('  greedy:', repr(tk.decode(g)))
        print('  %.1fs' % (time.time() - t0))
