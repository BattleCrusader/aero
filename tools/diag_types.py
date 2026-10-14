"""Dump every tensor name -> ggml type + quant for the gemma GGUF."""
import sys
sys.path.insert(0, '.')
from collections import Counter
from engine.gguf import GGUFReader

P = '/Users/cyberdeth/.cache/aero-models/gemma-4-12b-it-IQ4_XS.gguf'
with GGUFReader(P) as r:
    c = Counter()
    for name, info in r.tensors.items():
        t = getattr(info, 'ggml_type', None)
        t = t if t is not None else getattr(info, 'type', None)
        c[str(t)] += 1
    print('type histogram:', dict(c))
    print()
    show = ['token_embd.weight', 'output_norm.weight', 'rope_freqs.weight',
            'blk.0.attn_q.weight', 'blk.0.attn_k.weight', 'blk.0.attn_v.weight',
            'blk.0.attn_output.weight', 'blk.0.ffn_gate.weight',
            'blk.0.ffn_down.weight', 'blk.0.attn_q_norm.weight',
            'blk.0.attn_norm.weight', 'blk.5.attn_k.weight',
            'blk.5.attn_q.weight', 'blk.5.attn_output.weight']
    for n in show:
        info = r.tensors.get(n)
        if info is None:
            print(f'{n:32s} MISSING')
            continue
        t = getattr(info, 'ggml_type', None)
        t = t if t is not None else getattr(info, 'type', None)
        print(f'{n:32s} type={t!s:10s} shape={getattr(info, "shape", "?")} '
              f'offset={getattr(info, "offset", "?")}')
    # raw type ids
    print()
    tid = Counter()
    for name, info in r.tensors.items():
        v = getattr(info, 'type_id', None)
        if v is None:
            v = getattr(info, 'type', None)
        tid[v] += 1
    print('type ids:', dict(tid))
