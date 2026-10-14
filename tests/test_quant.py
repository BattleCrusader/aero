#!/usr/bin/env python3
"""GGUF + dequantisation self-test.

Three independent kinds of evidence:

  1. STRUCTURAL -- parse a real model file and check that the sum of every
     tensor's byte size plus the data offset equals the file size. That validates
     the parser AND every type-size in one shot; an error in any block size
     shows up as a mismatch.

  2. HAND-BUILT BLOCKS -- construct K-quant super-blocks field by field with
     values chosen so the correct output is known analytically. With d=1,
     dmin=0 and every sub-scale forced to 1, the dequantised values must BE the
     raw quant values. Any bit-layout error breaks this immediately.

  3. WRITER/READER ROUND-TRIP -- write a fixture, read it back, compare.

Run:  .venv-cml/bin/python tests/test_quant.py
"""

from __future__ import annotations

import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine import gguf as G  # noqa: E402
from engine import quant as Q  # noqa: E402

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


def f16(v: float) -> bytes:
    return struct.pack("<e", v)


def main() -> int:
    print("=== gguf + quant self-test ===\n")

    # ------------------------------------------------------- type sizes
    print("[1] block-size arithmetic")
    cases = [
        (Q.T_F32, 1, 4), (Q.T_F16, 1, 2), (Q.T_BF16, 1, 2),
        (Q.T_Q4_0, 32, 18), (Q.T_Q4_1, 32, 20), (Q.T_Q5_0, 32, 22),
        (Q.T_Q5_1, 32, 24), (Q.T_Q8_0, 32, 34),
        (Q.T_Q2_K, 256, 84), (Q.T_Q4_K, 256, 144), (Q.T_Q5_K, 256, 176),
        (Q.T_Q6_K, 256, 210),
    ]
    for t, per, expect in cases:
        got = Q.ggml_type_size(t, per)
        check(got == expect, f"type {t}: {per} values -> {got} bytes (expect {expect})")
    check(Q.ggml_type_size(Q.T_Q4_K, 512) == 288, "Q4_K scales linearly (512 -> 288)")

    # ---------------------------------------------------- hand-built Q4_K
    print("\n[2] Q4_K: hand-built block, analytically known output")
    # d=1, dmin=0, all effective sub-scales = 1 -> output == the quant values
    scales = bytes([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
    qs = bytes([(i & 0x0F) | (((31 - i) & 0x0F) << 4) for i in range(128)])
    block = f16(1.0) + f16(0.0) + scales + qs
    out = Q.dequantize(block, Q.T_Q4_K, (256,))

    exp_lo = np.array([i & 0x0F for i in range(128)], dtype=np.float32)
    exp_hi = np.array([(31 - i) & 0x0F for i in range(128)], dtype=np.float32)
    expect = np.empty(256, dtype=np.float32)
    for j in range(4):
        expect[j * 64:j * 64 + 32] = exp_lo[j * 32:(j + 1) * 32]
        expect[j * 64 + 32:j * 64 + 64] = exp_hi[j * 32:(j + 1) * 32]
    check(np.allclose(out, expect), "every value matches the known quant (d=1, dmin=0, sc=1)")
    check(out.shape == (256,), "shape is 256")
    print(f"       first 8: {out[:8].tolist()}")

    # with dmin != 0 the min must actually subtract
    block2 = f16(1.0) + f16(2.0) + scales + qs
    out2 = Q.dequantize(block2, Q.T_Q4_K, (256,))
    # mins: scales[4..7] = 0 -> m = 0 for j<4; for j>=4 m = (scales[j+4]>>4)|... = 0
    check(np.allclose(out2, out), "dmin with all-zero mins changes nothing (correct)")

    # ---------------------------------------------------- hand-built Q6_K
    print("\n[3] Q6_K: hand-built block, analytically known output")
    # d=1, all scales=1 -> output == (quant - 32) where quant is the 6-bit value
    ql = bytes([0x00] * 128)
    qh = bytes([0x00] * 64)
    sc = bytes([1] * 16)
    blk6 = ql + qh + sc + f16(1.0)
    out6 = Q.dequantize(blk6, Q.T_Q6_K, (256,))
    check(np.allclose(out6, -32.0), "all-zero quant bits -> every value = 0 - 32 = -32")

    ql2 = bytes([0xFF] * 128)
    qh2 = bytes([0xFF] * 64)
    blk6b = ql2 + qh2 + sc + f16(1.0)
    out6b = Q.dequantize(blk6b, Q.T_Q6_K, (256,))
    check(np.allclose(out6b, 63.0 - 32.0), "all-one quant bits -> every value = 63 - 32 = 31")

    # partial: low nibble set, high nibble zero. q1/q2 read the LOW nibble of
    # ql[l] and ql[l+32]; q3/q4 read the HIGH nibble of the same bytes.
    ql3 = bytes([0x0F] * 128)
    qh3 = bytes([0x00] * 64)
    out6c = Q.dequantize(ql3 + qh3 + sc + f16(1.0), Q.T_Q6_K, (256,))
    expect6 = np.empty(256, dtype=np.float32)
    for half in range(2):
        b = half * 128
        expect6[b:b + 32] = 15.0 - 32.0        # q1: low nibble
        expect6[b + 32:b + 64] = 15.0 - 32.0   # q2: low nibble
        expect6[b + 64:b + 96] = 0.0 - 32.0    # q3: HIGH nibble (= 0)
        expect6[b + 96:b + 128] = 0.0 - 32.0   # q4: HIGH nibble (= 0)
    check(np.allclose(out6c, expect6),
          "low nibble only -> q1/q2 = -17, q3/q4 = -32 (high nibble is zero)")

    # --------------------------------------------------------- Q8_0 / Q4_0
    print("\n[4] simple formats")
    d = f16(0.5)
    q8 = d + bytes(range(32))
    out8 = Q.dequantize(q8, Q.T_Q8_0, (32,))
    check(np.allclose(out8, np.arange(32, dtype=np.float32) * 0.5), "Q8_0: scale * index")

    q4 = f16(1.0) + bytes([0x00] * 16)
    out4 = Q.dequantize(q4, Q.T_Q4_0, (32,))
    check(np.allclose(out4, -8.0), "Q4_0: nibble 0 -> 0 - 8 = -8")

    # bf16 must be a bit shift, not a value cast
    bf = struct.pack("<H", 0x3F80)      # bf16 1.0
    check(np.allclose(Q.dequantize(bf, Q.T_BF16, (1,)), [1.0]), "BF16 0x3F80 -> 1.0")
    bf2 = struct.pack("<H", 0xC000)     # bf16 -2.0
    check(np.allclose(Q.dequantize(bf2, Q.T_BF16, (1,)), [-2.0]), "BF16 0xC000 -> -2.0")

    # ---------------------------------------------- writer/reader round-trip
    print("\n[5] writer/reader round-trip")
    w = G.GGUFWriter("testarch", alignment=32)
    w.add_scalar("testarch.block_count", 2)
    w.add_scalar("testarch.embedding_length", 8)
    f32 = np.arange(24, dtype=np.float32).reshape(4, 6)
    w.add_tensor("w.f32", (4, 6), f32.tobytes(), Q.T_F32)
    f16t = np.array([[1.5, -2.5, 3.0], [0.0, 4.0, -1.0]], dtype=np.float16)
    w.add_tensor("w.f16", (2, 3), f16t.tobytes(), Q.T_F16)
    tmp = os.path.join(HERE, ".fixture.gguf")
    w.write(tmp)

    with G.GGUFReader(tmp) as r:
        check(r.arch() == "testarch", "arch round-trips")
        check(r.hp("block_count") == 2, "uint32 metadata round-trips")
        check(r.hp("embedding_length") == 8, "second uint32 round-trips")
        check(set(r.tensors) == {"w.f32", "w.f16"}, "both tensors present")
        inf = r.tensors["w.f32"]
        check(inf.shape == (4, 6), f"F32 shape round-trips (got {inf.shape})")
        check(inf.n_elements == 24, "element count is 24")
        back = r.tensor("w.f32")
        check(np.allclose(back, f32), "F32 values round-trip exactly")
        back16 = r.tensor("w.f16")
        check(np.allclose(back16, f16t.astype(np.float32)), "F16 values round-trip")
        check(r.row_bytes(inf) == 6 * 4, "row_bytes for F32 4x6 row = 24")
    os.unlink(tmp)

    # ------------------------------------------------- real model structure
    print("\n[6] structural validation against a real model file")
    real = ("/Volumes/Backup/lmstudio/models/empero-ai/"
            "Qwen3.8-4B-Distill-GGUF/Qwen3.8-4B-Q4_K_M.gguf")
    if not os.path.exists(real):
        print("       (skipped: model not present)")
    else:
        with G.GGUFReader(real) as r:
            fsize = os.path.getsize(real)
            print(r.summary())
            # sum every tensor's size; the largest end must be within one
            # alignment of the file end
            end = 0
            unsupported = set()
            for name, inf in r.tensors.items():
                try:
                    sz = Q.ggml_type_size(inf.type_id, inf.n_elements)
                except NotImplementedError:
                    unsupported.add(inf.type_name)
                    continue
                end = max(end, inf.offset + sz)
            total_end = r.data_offset + end
            slack = fsize - total_end
            check(0 <= slack < 2 * r.alignment,
                  f"tensor bytes account for the file: end={total_end} "
                  f"file={fsize} slack={slack}")
            check(not unsupported,
                  f"every tensor type is supported (unsupported={unsupported or 'none'})")

            # a K-quant tensor must dequantise to finite, non-degenerate values
            kq = [n for n, i in r.tensors.items() if i.type_id in
                  (Q.T_Q4_K, Q.T_Q5_K, Q.T_Q6_K, Q.T_Q2_K)]
            check(bool(kq), f"found {len(kq)} K-quant tensors")
            if kq:
                # pick a small one to keep this fast
                smallest = min(kq, key=lambda n: r.tensors[n].n_elements)
                t = r.tensor(smallest)
                check(np.all(np.isfinite(t)), f"{smallest}: all values finite")
                check(float(t.std()) > 1e-6, f"{smallest}: not degenerate (std={t.std():.4g})")
                check(abs(float(t.mean())) < 5 * float(t.std()) + 1e-9,
                      f"{smallest}: mean is sane relative to std")
                print(f"       {smallest} {t.shape} {r.tensors[smallest].type_name}: "
                      f"mean={t.mean():.4g} std={t.std():.4g} "
                      f"min={t.min():.4g} max={t.max():.4g}")

    print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
