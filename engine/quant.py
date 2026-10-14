"""Dequantisation for the ggml tensor types we need.

Layouts follow the ggml block definitions exactly; the K-quants in particular
have fiddly 6-bit scale/min packing, so each one is implemented against the
documented bit layout rather than by analogy.

All functions return float32 numpy arrays in row-major shape.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# ---- type ids (mirror gguf.py) -------------------------------------------
T_F32, T_F16 = 0, 1
T_Q4_0, T_Q4_1 = 2, 3
T_Q5_0, T_Q5_1 = 6, 7
T_Q8_0 = 8
T_Q2_K, T_Q3_K, T_Q4_K, T_Q5_K, T_Q6_K = 10, 11, 12, 13, 14
T_IQ4_XS = 23
T_BF16 = 30

QK_K = 256   # K-quant super-block size

# (block_bytes, values_per_block)
_BLOCK: Dict[int, Tuple[int, int]] = {
    T_F32: (4, 1),
    T_F16: (2, 1),
    T_BF16: (2, 1),
    T_Q4_0: (18, 32),
    T_Q4_1: (20, 32),
    T_Q5_0: (22, 32),
    T_Q5_1: (24, 32),
    T_Q8_0: (34, 32),
    T_Q2_K: (84, QK_K),
    T_Q4_K: (144, QK_K),
    T_Q5_K: (176, QK_K),
    T_Q6_K: (210, QK_K),
    T_IQ4_XS: (136, QK_K),
}


def ggml_type_size(type_id: int, n_elements: int) -> int:
    """Bytes occupied by `n_elements` values of `type_id`."""
    if type_id not in _BLOCK:
        raise NotImplementedError(f"unsupported ggml type {type_id}")
    block_bytes, per_block = _BLOCK[type_id]
    if n_elements % per_block:
        raise ValueError(f"type {type_id}: {n_elements} is not a multiple of {per_block}")
    return (n_elements // per_block) * block_bytes


def _f16(buf: np.ndarray) -> np.ndarray:
    """fp16 bytes -> float32, 1-D over blocks.

    Returns shape (nblocks,) -- NOT (nblocks, 1). Getting this wrong is silent:
    with a single block an (nb,1) array broadcasts fine, but with many blocks
    `d * scales` broadcasts to (nb,nb) and only fails on real files.
    """
    return buf.view(np.float16).astype(np.float32).reshape(-1)


def _bf16(raw: np.ndarray) -> np.ndarray:
    """bf16 -> f32 by shifting into the high half of a u32 (not a value cast)."""
    u16 = raw.view(np.uint16).astype(np.uint32)
    return (u16 << 16).view(np.float32)


# ---- simple (non-K) formats ----------------------------------------------

def _deq_q4_0(raw: bytes, n: int) -> np.ndarray:
    a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 18)
    d = _f16(a[:, :2].copy())[:, None]
    qs = a[:, 2:]
    lo = (qs & 0x0F).astype(np.int8) - 8
    hi = (qs >> 4).astype(np.int8) - 8
    out = np.empty((a.shape[0], 32), dtype=np.float32)
    out[:, :16] = lo * d
    out[:, 16:] = hi * d
    return out.reshape(-1)[:n]


def _deq_q4_1(raw: bytes, n: int) -> np.ndarray:
    a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 20)
    d = _f16(a[:, :2].copy())[:, None]
    m = _f16(a[:, 2:4].copy())[:, None]
    qs = a[:, 4:]
    lo = (qs & 0x0F).astype(np.float32)
    hi = (qs >> 4).astype(np.float32)
    out = np.empty((a.shape[0], 32), dtype=np.float32)
    out[:, :16] = lo * d + m
    out[:, 16:] = hi * d + m
    return out.reshape(-1)[:n]


def _deq_q5_0(raw: bytes, n: int) -> np.ndarray:
    a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 22)
    d = _f16(a[:, :2].copy())
    qh = a[:, 2:6].view(np.uint32).reshape(-1)          # 32 high bits
    qs = a[:, 6:]
    out = np.empty((a.shape[0], 32), dtype=np.float32)
    for b in range(32):
        bit = ((qh >> b) & 1).astype(np.uint8)
        lo = (qs[:, b] & 0x0F).astype(np.uint8) | (bit << 4)
        hi = (qs[:, b] >> 4).astype(np.uint8) | (bit << 4)
        if b < 16:
            out[:, b] = lo.astype(np.int8) - 16
        else:
            out[:, b] = hi.astype(np.int8) - 16
    return (out * d[:, None]).reshape(-1)[:n]


def _deq_q5_1(raw: bytes, n: int) -> np.ndarray:
    a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 24)
    d = _f16(a[:, :2].copy())
    m = _f16(a[:, 2:4].copy())
    qh = a[:, 4:8].view(np.uint32).reshape(-1)
    qs = a[:, 8:]
    out = np.empty((a.shape[0], 32), dtype=np.float32)
    for b in range(32):
        bit = ((qh >> b) & 1).astype(np.uint8)
        lo = (qs[:, b] & 0x0F).astype(np.uint8) | (bit << 4)
        hi = (qs[:, b] >> 4).astype(np.uint8) | (bit << 4)
        if b < 16:
            out[:, b] = lo.astype(np.float32)
        else:
            out[:, b] = hi.astype(np.float32)
    return (out * d[:, None] + m[:, None]).reshape(-1)[:n]


def _deq_q8_0(raw: bytes, n: int) -> np.ndarray:
    a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
    d = _f16(a[:, :2].copy())[:, None]
    qs = a[:, 2:].view(np.int8).astype(np.float32)
    return (qs * d).reshape(-1)[:n]


# ---- K-quants ------------------------------------------------------------

def _get_scale_min_k4(j: int, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """6-bit scale/min extraction, vectorised over super-blocks.

    j < 4:  d = q[j]     & 63,           m = q[j+4] & 63
    j >= 4: d = (q[j+4] & 0xF) | (q[j-4] >> 6) << 4
            m = (q[j+4] >> 4)  | (q[j]   >> 6) << 4
    q is (nblocks, 12) uint8. Returns (d, m) each (nblocks,) uint8.
    """
    if j < 4:
        d = q[:, j] & 63
        m = q[:, j + 4] & 63
    else:
        d = (q[:, j + 4] & 0x0F) | ((q[:, j - 4] >> 6) << 4)
        m = (q[:, j + 4] >> 4) | ((q[:, j] >> 6) << 4)
    return d, m


def _deq_q4_k(raw: bytes, n: int) -> np.ndarray:
    a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 144)
    d = _f16(a[:, :2].copy())
    dmin = _f16(a[:, 2:4].copy())
    scales = a[:, 4:16]
    qs = a[:, 16:]                       # (nb, 128)

    out = np.empty((a.shape[0], QK_K), dtype=np.float32)
    is_ = 0
    col = 0
    for j in range(0, QK_K, 64):
        d1, m1 = _get_scale_min_k4(is_ + 0, scales)
        d2, m2 = _get_scale_min_k4(is_ + 1, scales)
        f1 = (d * d1.astype(np.float32))[:, None]
        g1 = (dmin * m1.astype(np.float32))[:, None]
        f2 = (d * d2.astype(np.float32))[:, None]
        g2 = (dmin * m2.astype(np.float32))[:, None]
        blk = qs[:, col:col + 32]
        out[:, j:j + 32] = f1 * (blk & 0x0F) - g1
        out[:, j + 32:j + 64] = f2 * (blk >> 4) - g2
        col += 32
        is_ += 2
    return out.reshape(-1)[:n]


def _deq_q5_k(raw: bytes, n: int) -> np.ndarray:
    a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 176)
    d = _f16(a[:, :2].copy())
    dmin = _f16(a[:, 2:4].copy())
    scales = a[:, 4:16]
    qh = a[:, 16:48]
    qs = a[:, 48:]

    out = np.empty((a.shape[0], QK_K), dtype=np.float32)
    is_ = 0
    col = 0
    for j in range(0, QK_K, 64):
        d1, m1 = _get_scale_min_k4(is_ + 0, scales)
        d2, m2 = _get_scale_min_k4(is_ + 1, scales)
        f1 = (d * d1.astype(np.float32))[:, None]
        g1 = (dmin * m1.astype(np.float32))[:, None]
        f2 = (d * d2.astype(np.float32))[:, None]
        g2 = (dmin * m2.astype(np.float32))[:, None]
        blk = qs[:, col:col + 32]
        # 5th bit for the two halves of this 64-block
        h1 = ((qh >> (2 * is_)) & 1).astype(np.uint8)
        h2 = ((qh >> (2 * is_ + 1)) & 1).astype(np.uint8)
        out[:, j:j + 32] = f1 * ((blk & 0x0F) | (h1 << 4)) - g1
        out[:, j + 32:j + 64] = f2 * ((blk >> 4) | (h2 << 4)) - g2
        col += 32
        is_ += 2
    return out.reshape(-1)[:n]


def _deq_q6_k(raw: bytes, n: int) -> np.ndarray:
    a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 210)
    ql = a[:, :128]
    qh = a[:, 128:192]
    sc = a[:, 192:208].view(np.int8).astype(np.float32)
    d = _f16(a[:, 208:210].copy())

    out = np.empty((a.shape[0], QK_K), dtype=np.float32)
    for half in range(2):
        ql_h = ql[:, half * 64:(half + 1) * 64]
        qh_h = qh[:, half * 32:(half + 1) * 32]
        sc_h = sc[:, half * 8:(half + 1) * 8]
        base = half * 128
        for l in range(32):
            is_ = l // 16
            q1 = ((ql_h[:, l] & 0x0F) | (((qh_h[:, l] >> 0) & 3) << 4)).astype(np.int8) - 32
            q2 = ((ql_h[:, l + 32] & 0x0F) | (((qh_h[:, l] >> 2) & 3) << 4)).astype(np.int8) - 32
            q3 = ((ql_h[:, l] >> 4) | (((qh_h[:, l] >> 4) & 3) << 4)).astype(np.int8) - 32
            q4 = ((ql_h[:, l + 32] >> 4) | (((qh_h[:, l] >> 6) & 3) << 4)).astype(np.int8) - 32
            out[:, base + l + 0] = d * sc_h[:, is_ + 0] * q1
            out[:, base + l + 32] = d * sc_h[:, is_ + 2] * q2
            out[:, base + l + 64] = d * sc_h[:, is_ + 4] * q3
            out[:, base + l + 96] = d * sc_h[:, is_ + 6] * q4
    return out.reshape(-1)[:n]


def _deq_q2_k(raw: bytes, n: int) -> np.ndarray:
    a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 84)
    scales = a[:, :16]
    qs = a[:, 16:80]
    d = _f16(a[:, 80:82].copy())
    dmin = _f16(a[:, 82:84].copy())

    sc = np.empty((a.shape[0], 16), dtype=np.float32)
    mn = np.empty((a.shape[0], 16), dtype=np.float32)
    for j in range(16):
        sc[:, j] = (scales[:, j] & 0x0F).astype(np.float32)
        mn[:, j] = (scales[:, j] >> 4).astype(np.float32)
    sc *= d[:, None]
    mn *= dmin[:, None]

    out = np.empty((a.shape[0], QK_K), dtype=np.float32)
    for i in range(2):
        for j in range(4):
            q = qs[:, i * 32 + j * 8:i * 32 + j * 8 + 8]
            base = i * 128 + j * 32
            for k in range(4):
                shift = 2 * k
                v = (q >> shift) & 3
                cols = np.arange(base + k * 8, base + k * 8 + 8)
                is_ = 2 * i + k // 2
                out[:, cols] = sc[:, is_][:, None] * v.astype(np.float32) - mn[:, is_][:, None]
    return out.reshape(-1)[:n]


def matvec_q4k_direct(raw: bytes, shape: Tuple[int, ...],
                      x: np.ndarray) -> np.ndarray:
    """y = W x without ever materialising W in fp32.

    The point: dequantising Q4_K expands the matrix 7.1x (144 B -> 1024 B per
    256 weights). If all we want is W @ x, that expansion is entirely avoidable.

    Each 32-weight sub-block stores  w[i] = d_sb * q[i] - m_sb   (q = 4-bit)
    So     sum_i w[i]*x[i] = d_sb * sum_i q[i]*x[i] - m_sb * sum_i x[i]

    Both sums are over the same 32 x-values, so the weights never need to exist
    as floats: only the 4-bit q values, plus ONE x partial sum per sub-block.
    The 6-bit scales are unpacked exactly as the dequantiser does, so the result
    is bit-identical to the fp32 path up to accumulation order.

    Costs: 144 B of reads per 256 weights instead of 1024 B, and no fp32 matrix
    in memory at all.
    """
    if shape[-1] % QK_K != 0:
        raise ValueError(f"Q4_K needs a multiple of {QK_K}, got {shape[-1]}")
    a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 144)
    nb = a.shape[0]                       # total 256-weight blocks (rows x cols/256)
    xf = np.asarray(x, dtype=np.float32).reshape(-1)
    if xf.size != shape[-1]:
        raise ValueError(
            f"x must have {shape[-1]} elements (one per column); got {xf.size}")

    # x is per-COLUMN. The blocks run down the flattened matrix, so x tiles:
    # block b covers columns (b*QK_K % cols) .. +QK_K.
    reps = nb // (shape[-1] // QK_K)
    if reps * (shape[-1] // QK_K) != nb:
        raise ValueError(f"block count {nb} does not tile {shape[-1]} columns")
    X = np.tile(xf, reps).reshape(nb, QK_K)

    d = _f16(a[:, :2].copy())            # (nb,) super-block scale
    dmin = _f16(a[:, 2:4].copy())        # (nb,) super-block min scale
    scales = a[:, 4:16]
    qs = a[:, 16:]                       # (nb, 128) packed 4-bit

    acc = np.zeros(nb, dtype=np.float32)
    is_ = 0
    col = 0
    for j in range(0, QK_K, 64):
        d1, m1 = _get_scale_min_k4(is_ + 0, scales)
        d2, m2 = _get_scale_min_k4(is_ + 1, scales)
        blk = qs[:, col:col + 32]
        xlo = X[:, j:j + 32]
        xhi = X[:, j + 32:j + 64]
        # low nibbles: sum of q*x, and the sum of x for the min term
        acc += (d * d1.astype(np.float32)) * np.einsum(
            "ij,ij->i", (blk & 0x0F).astype(np.float32), xlo)
        acc -= (dmin * m1.astype(np.float32)) * xlo.sum(axis=1)
        acc += (d * d2.astype(np.float32)) * np.einsum(
            "ij,ij->i", (blk >> 4).astype(np.float32), xhi)
        acc -= (dmin * m2.astype(np.float32)) * xhi.sum(axis=1)
        col += 32
        is_ += 2

    # acc is currently per 256-weight BLOCK; fold the blocks belonging to each
    # row. blocks_per_row = cols / QK_K, and blocks run row-major, so a plain
    # reshape-and-sum recovers one value per output row.
    bpr = shape[-1] // QK_K
    return acc.reshape(shape[0], bpr).sum(axis=1)


# ---------------------------------------------------------------- IQ4_XS
# 256 values in 136 bytes: d(2 fp16) scales_h(2) scales_l(4) qs(128).
# The 4-bit value indexes a fixed non-linear codebook, so this is a lookup, not
# an affine dequant -- unlike Q4_K. The 6-bit per-sub-block scale is assembled
# from scales_l (low 4 bits) and scales_h (high 2 bits) and biased by -32.
_KVALUES_IQ4NL = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=np.float32)


def _deq_iq4_xs(raw: bytes, n: int) -> np.ndarray:
    a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 136)
    d = _f16(a[:, :2].copy())
    sh = a[:, 2].astype(np.uint32) | (a[:, 3].astype(np.uint32) << 8)
    sl = a[:, 4:8]
    qs = a[:, 8:]                                   # (nb, 128)

    out = np.empty((a.shape[0], QK_K), dtype=np.float32)
    for ib in range(8):                             # 8 sub-blocks of 32
        ls = ((sl[:, ib >> 1] >> (4 * (ib & 1))) & 0x0F) |              (((sh >> (2 * ib)) & 3) << 4)
        dl = (d * (ls.astype(np.float32) - 32.0))[:, None]
        q = qs[:, ib * 16:(ib + 1) * 16]
        out[:, ib * 32:ib * 32 + 16] = dl * _KVALUES_IQ4NL[q & 0x0F]
        out[:, ib * 32 + 16:ib * 32 + 32] = dl * _KVALUES_IQ4NL[q >> 4]
    return out.reshape(-1)[:n]


def dequantize(raw: bytes, type_id: int, shape: Tuple[int, ...]) -> np.ndarray:
    """Dequantise `raw` into a float32 array of `shape` (row-major)."""
    n = 1
    for d in shape:
        n *= d

    if type_id == T_F32:
        out = np.frombuffer(raw, dtype=np.float32).copy()
    elif type_id == T_F16:
        out = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif type_id == T_BF16:
        out = _bf16(np.frombuffer(raw, dtype=np.uint8))
    elif type_id == T_Q4_0:
        out = _deq_q4_0(raw, n)
    elif type_id == T_Q4_1:
        out = _deq_q4_1(raw, n)
    elif type_id == T_Q5_0:
        out = _deq_q5_0(raw, n)
    elif type_id == T_Q5_1:
        out = _deq_q5_1(raw, n)
    elif type_id == T_Q8_0:
        out = _deq_q8_0(raw, n)
    elif type_id == T_Q2_K:
        out = _deq_q2_k(raw, n)
    elif type_id == T_Q4_K:
        out = _deq_q4_k(raw, n)
    elif type_id == T_Q5_K:
        out = _deq_q5_k(raw, n)
    elif type_id == T_IQ4_XS:
        out = _deq_iq4_xs(raw, n)
    elif type_id == T_Q6_K:
        out = _deq_q6_k(raw, n)
    else:
        raise NotImplementedError(f"dequantise: unsupported ggml type {type_id}")

    return np.ascontiguousarray(out.reshape(shape))
