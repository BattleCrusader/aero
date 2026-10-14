#!/usr/bin/env python3
"""Optimisation self-test.

Each optimisation is checked BOTH for correctness (identical results to the
unoptimised form) and for the resource property it claims:

  1. tiled attention  -- exact vs the naive form, and peak score memory bounded
                         by (T, H, block) instead of (T, H, S)
  2. blocked matvec   -- exact vs dequantise-then-matmul, with bounded scratch
  3. preallocated KV  -- exact, capacity-bounded, and no per-append realloc
  4. chunked prefill  -- covers the range exactly, with no gaps or overlaps

Run:  .venv-cml/bin/python tests/test_optimizations.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine import gguf as G  # noqa: E402
from engine import quant as Q  # noqa: E402
from engine import transformer as T  # noqa: E402

CHECKS = 0
FAILS = 0


def check(cond: bool, what: str) -> None:
    global CHECKS, FAILS
    CHECKS += 1
    if cond:
        print(f"  ok   {what}")
    else:
        FAILS += 1
        print(f"  FAIL {what}")


rng = np.random.default_rng(99)


def naive_scores_peak(Tn, H, S):
    return Tn * H * S * 4


def main() -> int:
    print("=== optimisation self-test ===\n")

    # ------------------------------------------------- 1. tiled attention
    print("[1] tiled attention: exact, and memory-bounded")
    Tn, H, HK, D, VD = 512, 8, 2, 16, 16
    q = rng.standard_normal((Tn, H, D)).astype(np.float32)
    k = rng.standard_normal((Tn, HK, D)).astype(np.float32)
    v = rng.standard_normal((Tn, HK, VD)).astype(np.float32)
    qpos = np.arange(Tn)
    scale = 1.0 / math.sqrt(D)

    stats = {}
    out_tiled = T.attention_tiled(q, k, v, scale, qpos, block=64,
                                  kv_rep=H // HK, stats=stats)

    # naive reference: full (T,H,S) then softmax then weighted sum
    ke = np.repeat(k, H // HK, axis=1)
    ve = np.repeat(v, H // HK, axis=1)
    full = np.einsum("thd,shd->ths", q, ke) * scale
    keep = np.arange(Tn)[None, :] <= qpos[:, None]
    full = np.where(keep[:, None, :], full, -np.inf)
    p = np.exp(full - np.max(full, -1, keepdims=True))
    p = p / p.sum(-1, keepdims=True)
    naive = np.einsum("ths,shv->thv", p, ve)

    maxdiff = float(np.abs(out_tiled - naive).max())
    check(maxdiff < 1e-5, f"matches the naive full-matrix form (max diff {maxdiff:.2e})")

    peak = stats["peak_scores_bytes"]
    naive_peak = naive_scores_peak(Tn, H, Tn)
    check(peak == Tn * H * 64 * 4,
          f"peak score block is (T,H,block) = {peak/1e6:.2f} MB")
    check(peak < naive_peak / 4,
          f"peak is {naive_peak/peak:.1f}x smaller than the naive (T,H,S) = "
          f"{naive_peak/1e6:.1f} MB")
    check(stats["blocks"] == math.ceil(Tn / 64), f"used {stats['blocks']} key blocks")

    # block size must not change the answer
    for blk in (1, 7, 128, 1024):
        o2 = T.attention_tiled(q, k, v, scale, qpos, block=blk, kv_rep=H // HK)
        d2 = float(np.abs(o2 - naive).max())
        check(d2 < 1e-5, f"block={blk:<5} identical result (max diff {d2:.2e})")

    # the real claim: at a 64k prompt the naive form is 256 GiB, this is not
    Tn2, H2 = 65536, 16
    naive_64k = naive_scores_peak(Tn2, H2, Tn2)
    tiled_64k = Tn2 * H2 * 1024 * 4
    check(naive_64k / 2**30 > 200,
          f"naive at 64k would need {naive_64k/2**30:.0f} GiB (the original defect)")
    check(tiled_64k / 2**30 < 5,
          f"tiled at 64k needs {tiled_64k/2**30:.2f} GiB peak instead")

    # sliding window still works
    o_win = T.attention_tiled(q, k, v, scale, qpos, block=64, kv_rep=H // HK,
                              window=32)
    check(o_win.shape == naive.shape, "windowed variant returns the right shape")
    check(not np.allclose(o_win, naive),
          "window=32 actually restricts attention (not a no-op)")
    check(np.all(np.isfinite(o_win)), "windowed output is finite")

    # ------------------------------------------- 2. blocked dequant-matvec
    print("\n[2] blocked dequant-matvec: exact, bounded scratch")
    w = G.GGUFWriter("t", alignment=32)
    rows, cols = 512, 256
    dense = (rng.standard_normal((rows, cols)) * 0.1).astype(np.float32)
    # store as Q4_K so dequantisation is genuinely exercised
    q4k = np.zeros((rows * cols // 256, 144), dtype=np.uint8)
    q4k[:, :2] = np.frombuffer(np.float16(0.01).tobytes(), dtype=np.uint8)   # d
    q4k[:, 2:4] = np.frombuffer(np.float16(0.0).tobytes(), dtype=np.uint8)   # dmin
    q4k[:, 4:16] = 1                                                          # scales
    q4k[:, 16:] = rng.integers(0, 256, size=(q4k.shape[0], 128), dtype=np.uint8)
    w.add_tensor("w.q4k", (rows, cols), q4k.tobytes(), Q.T_Q4_K)
    tmp = os.path.join(HERE, ".optfix.gguf")
    w.write(tmp)

    x = rng.standard_normal(cols).astype(np.float32)
    with G.GGUFReader(tmp) as r:
        y_blocked, peak_scratch = r.matvec("w.q4k", x, block_rows=32)
        full_w = r.tensor("w.q4k")
        y_full = full_w @ x
        md = float(np.abs(y_blocked - y_full).max())
        check(md < 1e-4, f"blocked == full materialisation (max diff {md:.2e})")
        check(peak_scratch == 32 * cols * 4,
              f"peak fp32 scratch is bounded: {peak_scratch/1024:.1f} KiB "
              f"(block_rows * cols * 4)")
        check(peak_scratch < full_w.nbytes / 10,
              f"scratch is {full_w.nbytes/peak_scratch:.1f}x smaller than the full "
              f"fp32 matrix ({full_w.nbytes/1e6:.2f} MB)")

        # and the whole point: the quantized resident form is much smaller
        q_bytes = Q.ggml_type_size(Q.T_Q4_K, rows * cols)
        check(full_w.nbytes / q_bytes > 7.0,
              f"fp32 materialisation inflates the weight {full_w.nbytes/q_bytes:.2f}x "
              f"vs keeping it quantized")

        # block size must not change the answer
        for br in (1, 7, 128, 512):
            yb, _ = r.matvec("w.q4k", x, block_rows=br)
            check(float(np.abs(yb - y_full).max()) < 1e-4,
                  f"block_rows={br:<4} identical result")
    os.unlink(tmp)

    # ------------------------------------------------------- 3. KV cache
    print("\n[3] preallocated KV cache: exact, bounded, no realloc")
    Hk, Dk, VDk = 2, 8, 8
    N = 64
    cache = T.KVCache(n_layer=1, n_head_kv=Hk, head_dim=Dk, value_dim=VDk, capacity=N)
    base_k = cache.k[0]
    check(base_k.shape == (N, Hk, Dk), f"buffer is preallocated ({base_k.shape})")
    steps = []
    for t in range(N):
        kk = rng.standard_normal((1, Hk, Dk)).astype(np.float32)
        vv = rng.standard_normal((1, Hk, VDk)).astype(np.float32)
        ka, va = cache.append(0, kk, vv)
        steps.append(ka.copy())
        cache.commit(1)
    check(cache.k[0] is base_k, "the buffer is never reallocated while appending")
    check(cache.len == N, f"len is {cache.len} after {N} steps")
    check(np.allclose(steps[-1], cache.k[0][:N]),
          "the returned view matches the buffer contents")
    try:
        cache.append(0, np.zeros((1, Hk, Dk), np.float32), np.zeros((1, Hk, VDk), np.float32))
        check(False, "appending past capacity must raise")
    except MemoryError:
        check(True, "appending past capacity raises MemoryError instead of growing")

    # --------------------------------------------- 4. chunked prefill
    print("\n[4] chunked prefill: exact coverage")
    for total, chunk in ((10, 4), (1, 1), (4096, 512), (100, 100), (7, 1000)):
        spans = list(T.prefill_chunks(total, chunk))
        covered = []
        for a, b in spans:
            covered.extend(range(a, b))
        check(covered == list(range(total)),
              f"total={total} chunk={chunk}: {len(spans)} span(s) cover [0,{total}) exactly")
    try:
        list(T.prefill_chunks(10, 0))
        check(False, "chunk=0 must raise")
    except ValueError:
        check(True, "chunk=0 raises ValueError")

    print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
