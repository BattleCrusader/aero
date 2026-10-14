"""Reproducible coherence check for gemma-4-12b through the aero engine.

    AERO_MAX_RAM=5G .venv-cml/bin/python tools/check_gemma4.py [model.gguf] [N]

The prompt is tokenised with the file's own tokenizer, which now honours
`tokenizer.ggml.add_bos_token` (gemma: true, qwen: false).

Reference point: llama.cpp b11033 on the same file, raw prompt, greedy:
    "The capital of France is" -> "The"      (this is gemma-4-12b-IT, whose
    off-template raw continuation is not 'Paris'; see the note in README/report)
"""
import sys
import time
sys.path.insert(0, '.')
import numpy as np
from engine.gguf import GGUFReader
from engine import model as M
from engine.tokenizer import Tokenizer

P = sys.argv[1] if len(sys.argv) > 1 else \
    '/Users/cyberdeth/.cache/aero-models/gemma-4-12b-it-IQ4_XS.gguf'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
PROMPTS = ['The capital of France is', 'Once upon a time']

with GGUFReader(P) as r:
    tk = Tokenizer.from_gguf(r)
    m = M.Model(r, block_rows=256, matmul_threads=8)
    print('cfg: embed_scale=%r rope_freqs=%s add_bos=%r bos_id=%r' % (
        m.cfg.embed_scale,
        None if m.cfg.rope_freqs is None else m.cfg.rope_freqs.shape,
        tk.add_bos, tk.bos_id))
    for pr in PROMPTS:
        t0 = time.time()
        st = m.new_state(capacity=256)
        ids = tk.encode(pr)
        lg = m.forward(ids, st)
        top = [tk.decode([int(i)]) for i in np.argsort(-lg)[:5]]
        print(f'\nprompt {pr!r} tokens={ids}')
        print('  top5  :', top)
        g = []
        for _ in range(N):
            n = int(np.argmax(lg))
            g.append(n)
            lg = m.forward([n], st)
        print('  greedy:', repr(tk.decode(g)))
        print('  %.1fs' % (time.time() - t0))
