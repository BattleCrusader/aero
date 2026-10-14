"""Chat-path coherence check: feed gemma-4-12b its OWN template and see if it answers.

    AERO_MAX_RAM=5G .venv-cml/bin/python tools/check_gemma4_chat.py [model.gguf] [N]

Why a separate tool from check_gemma4.py: that one sends a RAW prompt, which is the
wrong input for an instruct model. This one goes through engine.chat_template --
the same renderer the server and CLI use -- so what it measures is what a user
actually gets. A tool that renders its own prompt would verify something the engine
never sends.

The prompt is wrapped in a user turn; the template appends '<|turn>model\\n' and, with
thinking disabled, '<|channel>thought\\n<channel|>' to force a direct answer.
"""
import sys
import time

sys.path.insert(0, '.')
import numpy as np
from engine.gguf import GGUFReader
from engine import model as M
from engine.tokenizer import Tokenizer
from engine.chat_template import render_chat, detect_envelope

P = sys.argv[1] if len(sys.argv) > 1 else \
    '/Users/cyberdeth/.cache/aero-models/gemma-4-12b-it-IQ4_XS.gguf'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 16
QUESTIONS = [
    'What is the capital of France?',
    'The capital of France is',
    'Once upon a time',
]

with GGUFReader(P) as r:
    tk = Tokenizer.from_gguf(r)
    m = M.Model(r, block_rows=256, matmul_threads=8)
    print('cfg: envelope=%s add_bos=%r bos_id=%r' % (
        detect_envelope(tk), tk.add_bos, tk.bos_id))
    for q in QUESTIONS:
        t0 = time.time()
        text = render_chat([{'role': 'user', 'content': q}], tk)
        ids = tk.encode(text)
        st = m.new_state(capacity=256)
        lg = m.forward(ids, st)
        prompt_len = len(ids)
        top = [tk.decode([int(i)]) for i in np.argsort(-lg)[:5]]
        print(f'\nQ {q!r}')
        print(f'  rendered tokens ({prompt_len}): {ids[:8]}...{ids[-4:]}')
        print('  top5  :', top)
        g = []
        for _ in range(N):
            n = int(np.argmax(lg))
            if n in (tk.eos_id,):
                break
            g.append(n)
            lg = m.forward([n], st)
        # raw ids matter: decode() drops specials, so a '' can be a correct <|im_end|>
        print('  ids   :', g[:12])
        print('  answer:', repr(tk.decode(g)))
        print('  %.1fs' % (time.time() - t0))
