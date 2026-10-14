"""Dump gemma4 hparams + small tensors so bug hypotheses can be ranked by facts."""
import sys
sys.path.insert(0, '.')
import numpy as np
from engine.gguf import GGUFReader
from engine import model as M

p = '/Users/cyberdeth/.cache/aero-models/gemma-4-12b-it-IQ4_XS.gguf'
with GGUFReader(p) as r:
    for k in ('attention.head_count', 'attention.head_count_kv', 'attention.key_length',
              'attention.key_length_swa', 'attention.value_length', 'attention.value_length_swa',
              'attention.sliding_window', 'attention.shared_kv_layers',
              'rope.dimension_count', 'rope.dimension_count_swa',
              'rope.freq_base', 'rope.freq_base_swa',
              'attention.layer_norm_rms_epsilon', 'final_logit_softcapping',
              'embedding_length', 'block_count', 'feed_forward_length'):
        print(f"{k:38s} = {r.hp(k)!r}")
    swp = r.hp('attention.sliding_window_pattern')
    print('pattern[:12] =', swp[:12] if isinstance(swp, (list, tuple)) else swp)
    print('pattern count True/False =', (sum(1 for x in swp if x), sum(1 for x in swp if not x)))

    cfg = M.ModelConfig.from_gguf(r)
    print('arch', cfg.arch, 'n_layer', cfg.n_layer, 'n_embd', cfg.n_embd,
          'n_head', cfg.n_head, 'rope_base', cfg.rope_base, 'rope_dims', cfg.rope_dims)
    g = cfg.gemma_cfg
    for i in range(cfg.n_layer):
        print(f"L{i:2d} hd={cfg.head_dim_at(i):4d} vd={cfg.value_dim_at(i):4d} "
              f"kv={cfg.kv_heads_at(i)} rope_base={g.rope_base_at(i):>9.1f} "
              f"rope_dims={g.rope_dims_at(i):4d} win={cfg.window_at(i)}")
    # small tensors
    rf = r.tensor('rope_freqs.weight')
    rf = np.asarray(rf, dtype=np.float32).reshape(-1)
    print('rope_freqs.shape', rf.shape, 'min', rf.min(), 'max', rf.max())
    print('rope_freqs[:8]', rf[:8])
    print('rope_freqs[-4:]', rf[-4:])
    los = []
    for i in range(cfg.n_layer):
        v = np.asarray(r.tensor(f'blk.{i}.layer_output_scale.weight'), dtype=np.float32).reshape(-1)
        los.append(float(v[0]))
    print('layer_output_scale min/max', min(los), max(los))
    print('layer_output_scale', [round(x, 4) for x in los])
    print('has attn_v:', [i for i in range(cfg.n_layer) if f'blk.{i}.attn_v.weight' in r.tensors])
    print('output.weight present:', 'output.weight' in r.tensors)
