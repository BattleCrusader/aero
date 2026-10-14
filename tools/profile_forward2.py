#!/usr/bin/env python3
"""Clean attribution of one decode step. Replaces the earlier profiler, which
printed a BYTE COUNT in the seconds column and filtered out the one row that
mattered -- so its "87.6% dequantise" headline was not trustworthy.

This times the real call sites: tensor_rows (read+dequantise), matmul (the
matvec itself), SSM and attention -- each accumulated separately, and reports
bytes actually read so throughput can be checked against the bus limit.

Run: .venv-cml/bin/python tools/profile_forward2.py
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine import gguf as G          # noqa: E402
from engine import model as M         # noqa: E402
from engine import ssm as S           # noqa: E402
from engine.tokenizer import Tokenizer  # noqa: E402

QWEN = ("/Volumes/Backup/lmstudio/models/empero-ai/"
        "Qwen3.8-4B-Distill-GGUF/Qwen3.8-4B-Q4_K_M.gguf")

S_TIME = defaultdict(float)
S_N = defaultdict(int)
S_BYTES = defaultdict(int)


def main():
    if not os.path.exists(QWEN):
        print("model not present")
        return 1

    with G.GGUFReader(QWEN) as r:
        tk = Tokenizer.from_gguf(r)

        # --- instrument tensor_rows (the read+dequantise primitive)
        orig_tr = G.GGUFReader.tensor_rows
        def timed_tr(self, info, r0, r1):
            t0 = time.perf_counter()
            out = orig_tr(self, info, r0, r1)
            S_TIME["tensor_rows"] += time.perf_counter() - t0
            S_N["tensor_rows"] += 1
            S_BYTES["tensor_rows"] += out.nbytes
            return out
        G.GGUFReader.tensor_rows = timed_tr

        # --- instrument tensor_bytes (the whole-tensor read path)
        orig_tb = G.GGUFReader.tensor_bytes
        def timed_tb(self, info):
            t0 = time.perf_counter()
            out = orig_tb(self, info)
            S_TIME["tensor_bytes"] += time.perf_counter() - t0
            S_N["tensor_bytes"] += 1
            S_BYTES["tensor_bytes"] += len(out)
            return out
        G.GGUFReader.tensor_bytes = timed_tb

        # --- instrument matmul (total, correctly in SECONDS)
        orig_mm = G.GGUFReader.matmul
        def timed_mm(self, name, X, block_rows=128, read_once=True, threads=1):
            t0 = time.perf_counter()
            out = orig_mm(self, name, X, block_rows=block_rows,
                          read_once=read_once, threads=threads)
            S_TIME["matmul_total"] += time.perf_counter() - t0
            S_N["matmul_total"] += 1
            S_BYTES["matmul_total"] += out[0].nbytes
            return out
        G.GGUFReader.matmul = timed_mm

        # --- ssm / attention
        orig_ssm = S.ssm_forward
        def timed_ssm(*a, **k):
            t0 = time.perf_counter()
            out = orig_ssm(*a, **k)
            S_TIME["ssm"] += time.perf_counter() - t0
            S_N["ssm"] += 1
            return out
        S.ssm_forward = timed_ssm

        orig_attn = M.attention
        def timed_attn(*a, **k):
            t0 = time.perf_counter()
            out = orig_attn(*a, **k)
            S_TIME["attention"] += time.perf_counter() - t0
            S_N["attention"] += 1
            return out
        M.attention = timed_attn

        m = M.Model(r, block_rows=256)
        ids = tk.encode("The capital of France is")
        st = m.new_state(capacity=128)

        # warm, then measure a DECODE step
        m.forward(ids, st)
        for k in list(S_TIME):
            S_TIME[k] = 0.0
        for k in list(S_N):
            S_N[k] = 0
        for k in list(S_BYTES):
            S_BYTES[k] = 0

        t0 = time.perf_counter()
        lg = m.forward([220], st)
        total = time.perf_counter() - t0

    print(f"decode 1 token: {total:.2f} s\n")
    keys = ["matmul_total", "tensor_rows", "tensor_bytes", "ssm", "attention"]
    print(f"{'component':<16}{'seconds':>10}{'calls':>8}{'% total':>9}{'bytes':>12}")
    for k in keys:
        if k in S_TIME:
            print(f"{k:<16}{S_TIME[k]:>10.3f}{S_N[k]:>8}"
                  f"{S_TIME[k]/total*100:>8.1f}%{S_BYTES.get(k,0)/1e6:>11.1f}M")
    acc = sum(S_TIME[k] for k in ("matmul_total", "ssm", "attention") if k in S_TIME)
    print(f"{'TOTAL accounted':<16}{acc:>10.3f}{'':>8}{acc/total*100:>8.1f}%")
    print()
    mr = S_TIME.get("matmul_total", 0.0)
    print(f"of the {mr:.2f}s in matmul, tensor_rows/dequant accounts for "
          f"{S_TIME.get('tensor_rows',0):+.2f}s and tensor_bytes "
          f"{S_TIME.get('tensor_bytes',0):+.2f}s")
    print(f"  (they sum to more than matmul_total because tensor_rows is called "
          f"from inside it)")
    print()
    tr = S_TIME.get("tensor_rows", 0.0)
    tb = S_TIME.get("tensor_bytes", 0.0)
    io = tr + tb
    print(f"read+dequantise : {io:6.2f} s   ({io/total*100:.1f}% of the token)")
    print(f"actual arithmetic: {mr - io + S_TIME.get('ssm',0) + S_TIME.get('attention',0):6.2f} s")
    print()
    by = S_BYTES.get("tensor_rows", 0) + S_BYTES.get("tensor_bytes", 0)
    print(f"fp32 bytes produced: {by/1e6:.0f} MB in {io:.2f}s -> {by/io/1e6:.0f} MB/s")
    print("  (a memory-bound dequantise should hit GB/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
