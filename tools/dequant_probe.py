#!/usr/bin/env python3
"""Is dequantisation avoidable, and how much does it cost?

Q4_K stores 256 weights in 144 bytes with a 6-bit scale and 6-bit min per
32-weight sub-block. The standard path is:

    read 144 B -> dequantise to 256 x fp32 (1024 B) -> matmul

which is a 7.1x memory expansion followed by a GEMM. The alternative is to fold
the scale/min into the accumulation so no fp32 weight matrix is ever built:

    y[j] = sum_i w[i,j] * x[i]
         = sum_sb sum_i (d_sb * q[i] + m_sb) * x[i]
         = sum_sb ( d_sb * sum_i q[i]*x[i]  +  m_sb * sum_i x[i] )

so per sub-block only q (6-bit ints) and one x partial sum are needed. This
measures both paths against each other on real file data.

Run: .venv-cml/bin/python tools/dequant_probe.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine import gguf as G      # noqa: E402
from engine import quant as Q     # noqa: E402

QWEN = ("/Volumes/Backup/lmstudio/models/empero-ai/"
        "Qwen3.8-4B-Distill-GGUF/Qwen3.8-4B-Q4_K_M.gguf")


def banner(t):
    print(f"\n{'='*72}\n{t}\n{'='*72}")


def main():
    if not os.path.exists(QWEN):
        print("model not present")
        return 1

    with G.GGUFReader(QWEN) as r:
        # pick a real big matrix: the FFN gate of a middle layer
        name = "blk.16.ffn_gate.weight"
        info = r.tensors[name]
        banner(f"target: {name} {info.shape} {info.type_name}")
        raw = r.tensor_bytes(info)
        rows, cols = info.shape
        print(f"  on-disk  : {len(raw):,} B ({len(raw)/2**20:.2f} MiB)")
        print(f"  as fp32  : {rows*cols*4:,} B ({rows*cols*4/2**20:.2f} MiB)")
        print(f"  expansion: {(rows*cols*4)/len(raw):.2f}x")

        # ---- path A: dequantise the whole thing, then matmul
        banner("path A -- dequantise to fp32, then matmul (what we do now)")
        t0 = time.perf_counter()
        W = Q.dequantize(raw, info.type_id, info.shape)
        t_deq = time.perf_counter() - t0
        print(f"  dequantise whole matrix : {t_deq*1000:8.1f} ms")
        x = np.random.default_rng(0).standard_normal(cols).astype(np.float32)
        t0 = time.perf_counter()
        for _ in range(5):
            yA = W @ x
        t_gemm = (time.perf_counter() - t0) / 5
        print(f"  matmul (fp32 GEMM)      : {t_gemm*1000:8.1f} ms")
        print(f"  TOTAL per matvec        : {(t_deq+t_gemm)*1000:8.1f} ms")
        print(f"  peak resident           : {W.nbytes/2**20:.2f} MiB (fp32 weights)")

        # ---- path B: blocked dequant + matmul (the current engine path)
        banner("path B -- blocked dequant+matmul (GGUFReader.matmul, block_rows)")
        for br in (16, 64, 256, 1024):
            t0 = time.perf_counter()
            yB = r.matmul(name, x.reshape(1, -1), br)[0].reshape(-1)
            t = time.perf_counter() - t0
            print(f"  block_rows={br:<5} {t*1000:8.1f} ms   "
                  f"scratch {br*cols*4/2**20:6.2f} MiB")

        # ---- path C: quantised-domain accumulation (NO fp32 weight ever built)
        banner("path C -- accumulate in the quantised domain (no fp32 weight)")
        t0 = time.perf_counter()
        yC = Q.matvec_q4k_direct(raw, info.shape, x)
        t_direct = time.perf_counter() - t0
        err = float(np.abs(yC - yA).max() / max(float(np.abs(yA).max()), 1e-9))
        print(f"  direct (no dequant)     : {t_direct*1000:8.1f} ms")
        print(f"  max relative error vs A : {err:.3e}")
        print(f"  peak resident           : {len(raw)/2**20:.2f} MiB (quantised only)")
        print()
        print(f"  speedup vs A            : {(t_deq+t_gemm)/t_direct:6.2f}x")
        print(f"  memory vs A             : {W.nbytes/len(raw):6.2f}x smaller")

    return 0


if __name__ == "__main__":
    sys.exit(main())
