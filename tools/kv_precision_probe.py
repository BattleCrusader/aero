#!/usr/bin/env python3
"""How far can the KV cache be compressed before the OUTPUT breaks?

Vector rounding error is the wrong metric -- what matters is the error in the
ATTENTION output, which is what the rest of the network consumes. This captures
real K/V, quantises at several bit widths, and measures the resulting attention
error end to end.

Also tests ASYMMETRIC precision: V is consumed as a weighted SUM (so rounding
averages out over many positions), while K goes through exp() in the softmax
(so its error is amplified non-linearly). If that asymmetry is real, K needs
more bits than V and the average can be lower than a uniform choice.

Run: .venv-cml/bin/python tools/kv_precision_probe.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine import gguf as G          # noqa: E402
from engine import model as M         # noqa: E402
from engine import transformer as T   # noqa: E402

QWEN = ("/Volumes/Backup/lmstudio/models/empero-ai/"
        "Qwen3.8-4B-Distill-GGUF/Qwen3.8-4B-Q4_K_M.gguf")


def banner(t):
    print(f"\n{'='*72}\n{t}\n{'='*72}")


def qsym(x, bits):
    """Symmetric uniform quantisation to `bits`, per (position, head) vector."""
    levels = (1 << (bits - 1)) - 1          # 127 for 8, 7 for 4, 3 for 2
    xf = x.astype(np.float32)
    amax = np.abs(xf).max(axis=-1, keepdims=True)
    s = np.where(amax > 0, amax / levels, 1.0).astype(np.float32)
    q = np.clip(np.round(xf / s), -levels, levels)
    return (q * s).astype(np.float32)


def qperp(x, bits):
    """Per-position-per-head uniform quantisation (the layout we actually use:
    one scale per (position, head) vector, not per element)."""
    return qsym(x, bits)


def main():
    if not os.path.exists(QWEN):
        print("model not present")
        return 1
    with G.GGUFReader(QWEN) as r:
        mc = M.ModelConfig.from_gguf(r)
        model = M.Model(r)
        st = model.new_state(capacity=192)
        cap = {}
        orig = st.kv.append

        def spy(layer, k, v):
            cap.setdefault(layer, []).append((np.array(k), np.array(v)))
            return orig(layer, k, v)

        st.kv.append = spy
        toks = [(9707 + 137 * i) % mc.vocab for i in range(96)]
        model.forward(toks, st)

    L = mc.attn_layers[0]
    ks = np.concatenate([a for a, _ in cap[L]], axis=0)     # (S, HK, D)
    vs = np.concatenate([b for _, b in cap[L]], axis=0)
    S, HK, D = ks.shape
    banner(f"captured real K/V: layer {L}, {S} positions, {HK} kv heads, dim {D}")

    # attention of the last position over all keys, in fp32
    rng = np.random.default_rng(0)
    q = rng.standard_normal((HK, D)).astype(np.float32)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    scale = 1.0 / np.sqrt(D)

    def attn(k, v):
        s = np.einsum("hd,shd->sh", q, k) * scale          # (S, HK)
        w = T.softmax(s, axis=0)
        return np.einsum("sh,shd->hd", w, v)

    ref = attn(ks, vs)
    ref_n = np.linalg.norm(ref)

    banner("A  uniform precision: error in the ATTENTION OUTPUT")
    print(f"  {'bits':>5} {'bytes/pos':>10} {'K rel err':>11} {'V rel err':>11} "
          f"{'OUT rel err':>12}")
    base_bytes = 2 * HK * D * 2      # fp16-equivalent reference, per position
    for bits in (16, 8, 6, 4, 3, 2):
        kq, vq = qperp(ks, bits), qperp(vs, bits)
        e = np.abs(attn(kq, vq) - ref).max() / ref_n
        ek = np.abs(kq - ks).max() / np.abs(ks).max()
        ev = np.abs(vq - vs).max() / np.abs(vs).max()
        bp = 2 * HK * D * (bits / 8.0)
        print(f"  {bits:>5} {bp:>10.1f} {ek:>11.2e} {ev:>11.2e} {e:>12.2e}")

    banner("B  ASYMMETRIC: does K need more bits than V?")
    print(f"  {'K bits':>7} {'V bits':>7} {'bytes/pos':>10} {'OUT rel err':>12}")
    for kb, vb in ((8, 8), (8, 6), (8, 4), (8, 3), (8, 2), (6, 4),
                   (4, 4), (4, 3), (4, 2)):
        kq, vq = qperp(ks, kb), qperp(vs, vb)
        e = np.abs(attn(kq, vq) - ref).max() / ref_n
        bp = HK * D * ((kb + vb) / 8.0) * 2
        print(f"  {kb:>7} {vb:>7} {bp:>10.1f} {e:>12.2e}")

    banner("C  what does 262144 cost at each setting (9 attn layers)?")
    per_pos_bytes = lambda kb, vb: 9 * HK * D * ((kb + vb) / 8.0)
    for kb, vb in ((32, 32), (16, 16), (8, 8), (8, 4), (4, 4), (4, 2), (2, 2)):
        tot = per_pos_bytes(kb, vb) * 262144
        print(f"  K{kb}/V{vb}: {tot/2**30:>7.2f} GiB   "
              f"{'<-- target ~2 GB' if 1.5 < tot/2**30 < 2.5 else ''}")

    banner("note")
    print("  'OUT rel err' is the max relative error in the attention output for")
    print("  one query, over 96 real positions. The threshold that matters is")
    print("  whether the error survives the rest of the network -- a few percent")
    print("  in attention can still flip a token, which is why the end-to-end")
    print("  token comparison in tests/test_kv.py is the real gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
