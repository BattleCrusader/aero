"""Transformer core: the pieces every decoder-only model here needs.

Everything is numpy and deliberately straightforward -- correctness first, device
acceleration later (that is what AeroEngine is for). Each function is written to
match the standard formulation, and each is verified in tests/test_transformer.py
against an independently written, loop-based reference.

Implemented here: RMSNorm, RoPE (full and partial), GQA attention with optional
QK-norm, SwiGLU, KV cache, sampling, and the autoregressive decode loop.

NOT here yet: the gated-delta-net (SSM) layer that qwen35/qwen35moe interleave
with attention, gemma4's per-layer embeddings, and MoE. Those are arch-specific
and are the next stage -- see ARCH_PLAN at the bottom.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ------------------------------------------------------------------ switches
# Gated-attention layout, SWEPT rather than assumed. attn_q is 2x the q width,
# so one half is q and the other is a gate -- but WHICH half, applied HOW, is a
# convention the file does not state. Getting it wrong corrupts every attention
# layer (measured: disabling attention entirely gives better-formed output than
# leaving it on), so it is a module-level switch that tests can vary.
#   "qgate_sigmoid" : [q | gate], out *= sigmoid(gate)      (the assumption)
#   "gateq_sigmoid" : [gate | q], out *= sigmoid(gate)
#   "qgate_silu"    : [q | gate], out *= silu(gate)
#   "none"          : no gate at all
#   "qgate_interleave": per-head interleaved [q_h | gate_h] pairs
GATE_MODE = "qgate_interleave"

# RoPE PAIR CONVENTION. Two incompatible readings, both rotation-invariant in
# the sense tests usually check (q.k depends only on relative position), so a
# property test CANNOT distinguish them:
#   "neox"        : pairs (x[i], x[i+half])   -- Qwen / HF transformers
#   "interleaved" : pairs (x[2i], x[2i+1])    -- original GPT-NeoX code
# Getting it wrong misaligns every head's rotation and destroys attention.
ROPE_MODE = "neox"

# WHERE `post_attention_norm` APPLIES. Verified against the reference build:
#     cur = attn(attn_norm(inpL))
#     cur = cur + inpSA                  <- residual add first
#     ffn_residual = cur
#     cur = ffn(norm(cur))               <- norm applied to the SUM
#     cur = cur + ffn_residual
# so "after_add" is what this architecture does. "on_output" (norm the attention
# output, then add) is the other reading; it keeps shapes valid and values
# finite while destroying the learned function, showing up only as "predicts the
# frequency prior". Kept switchable so the difference stays testable.
BLOCK_MODE = "after_add"


# ---------------------------------------------------------------- primitives

def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float,
             unit_offset: bool = True) -> np.ndarray:
    """Norm-scale then weight.

    unit_offset=True  -> RMSNorm: x / sqrt(mean(x^2) + eps) * weight
    unit_offset=False -> L2 norm: x / sqrt(sum(x^2) + eps) * weight

    Both are parameter-free when `weight` is omitted/ones, which is why the file
    cannot tell them apart by inspection: a layer that normalises q/k without
    carrying a norm tensor could be either. They differ by a constant factor of
    sqrt(d) per head, which changes attention temperature but not direction.
    Kept as an explicit flag so both readings can be measured side by side
    instead of assuming one.
    """
    xf = x.astype(np.float32, copy=False)
    if unit_offset:
        ss = np.mean(xf * xf, axis=-1, keepdims=True)
    else:
        ss = np.sum(xf * xf, axis=-1, keepdims=True)
    return (xf / np.sqrt(ss + eps)) * weight.astype(np.float32)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    xf = x.astype(np.float32, copy=False)
    m = np.max(xf, axis=axis, keepdims=True)
    e = np.exp(xf - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def swiglu(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    """SwiGLU: silu(gate) * up (llama ordering)."""
    return silu(gate.astype(np.float32)) * up.astype(np.float32)


def linear(x: np.ndarray, w: np.ndarray, b: Optional[np.ndarray] = None) -> np.ndarray:
    """y = x @ W^T (+ b). Weights are stored [out, in]."""
    y = x.astype(np.float32) @ w.astype(np.float32).T
    if b is not None:
        y = y + b.astype(np.float32)
    return y


def rope_tables(head_dim: int, positions: np.ndarray, base: float,
                n_dims: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """cos/sin tables for RoPE. `n_dims` allows PARTIAL rotary: only the first
    n_dims channels are rotated, the rest pass through.

    Returns (cos, sin) each of shape (len(positions), n_rot) where n_rot is
    n_dims (rounded to even).
    """
    n = head_dim if n_dims is None else min(n_dims, head_dim)
    n = n - (n % 2)
    half = n // 2
    inv = 1.0 / (base ** (np.arange(0, half, dtype=np.float32) * 2.0 / n))
    ang = positions.astype(np.float32)[:, None] * inv[None, :]      # (T, half)
    return np.cos(ang), np.sin(ang)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray,
               mode: Optional[str] = None) -> np.ndarray:
    """Rotate the first 2*half channels of the last axis.

    x: (T, H, D). cos/sin: (T, half).

    See ROPE_MODE. "neox" pairs (x[i], x[i+half]) and is what Qwen uses;
    "interleaved" pairs (x[2i], x[2i+1]). Only the first 2*half channels are
    touched, so partial rotary leaves the tail untouched in both modes.
    """
    mode = mode or ROPE_MODE
    T, H, D = x.shape
    half = cos.shape[1]
    n = 2 * half
    out = x.astype(np.float32).copy()
    c = cos[:, None, :]
    s = sin[:, None, :]

    if mode == "neox":
        # split-half: element i pairs with element i + half
        x1 = out[:, :, :half].copy()
        x2 = out[:, :, half:n].copy()
        out[:, :, :half] = x1 * c - x2 * s
        out[:, :, half:n] = x1 * s + x2 * c
        return out
    if mode == "interleaved":
        rot = out[:, :, :n].reshape(T, H, half, 2)
        x0 = rot[:, :, :, 0].copy()
        x1 = rot[:, :, :, 1].copy()
        rot[:, :, :, 0] = x0 * c - x1 * s
        rot[:, :, :, 1] = x0 * s + x1 * c
        out[:, :, :n] = rot.reshape(T, H, n)
        return out
    raise ValueError(f"unknown RoPE mode {mode!r}; use 'neox' or 'interleaved'")


# ---------------------------------------------------------------- attention

@dataclass
class AttnWeights:
    wq: Optional[np.ndarray] = None
    wk: Optional[np.ndarray] = None
    wv: Optional[np.ndarray] = None
    wo: Optional[np.ndarray] = None
    wqkv: Optional[np.ndarray] = None      # fused (qwen35 SSM layers use this)
    gate: Optional[np.ndarray] = None
    q_norm: Optional[np.ndarray] = None
    k_norm: Optional[np.ndarray] = None
    bq: Optional[np.ndarray] = None
    bk: Optional[np.ndarray] = None
    bv: Optional[np.ndarray] = None


@dataclass
class AttnConfig:
    n_head: int
    n_head_kv: int
    head_dim: int          # == key_length
    value_dim: int         # == value_length (may differ from head_dim)
    rope_base: float
    rope_dims: Optional[int] = None      # partial rotary
    eps: float = 1e-6
    scale: Optional[float] = None        # default 1/sqrt(head_dim)


class KVCache:
    """Preallocated, append-in-place key/value cache.

    The naive version used np.concatenate per step, which reallocates and copies
    the entire history every token -- O(n^2) total copying. This writes into a
    fixed capacity buffer at an offset instead, so appending is O(1) amortised
    and the peak allocation is known up front (which the memory guard can check).

    PRECISION: `dtype` selects the storage format. fp32 is the default and is
    what the model's activations are computed in. "int8" stores each
    (position, head) vector as int8 plus one fp32 scale -- 4.25 bytes per 32
    elements instead of 128, i.e. ~7.5x smaller.

    Why this matters: the weights are 4-bit but the cache was fp32, so the cache
    was stored at 8x the precision of the model that produced it. At 262144
    context that asymmetry costs 18 GiB -- more than the whole machine.

    MEASURED cost at 262144 ctx (9 attention layers, vs the 18.00 GiB fp32
    baseline), with a real greedy decode compared token-for-token against fp32:

        fp32   18.00 GiB   4/4 tokens identical   --
        int8    4.57 GiB   4/4 tokens identical   0.93%% logit error
        int4    2.32 GiB   3/4 tokens identical   32.4%% logit error

    So int8 is free in practice and int4 is NOT -- it changed a token on the
    test prompt. `quant_error()` reports the actual cost rather than assuming
    any of this is free, and both remain opt-in with fp32 as the default.
    """

    def __init__(self, n_layer: int, n_head_kv: int, head_dim: int,
                 value_dim: int, capacity: int, dtype: str = "fp32",
                 layer_indices: Optional[list] = None,
                 windowed: Optional[dict] = None,
                 k_dtype: Optional[str] = None,
                 v_dtype: Optional[str] = None,
                 heads_per_layer: Optional[dict] = None,
                 rope_k: bool = False) -> None:
        """`layer_indices` = the layers that actually attend.

        This matters more than it looks. A hybrid model like qwen35 has 33 layers
        but only 9 of them are attention; the other 24 are SSM and keep an O(1)
        state instead. Allocating a KV cache for all 33 wastes 3.67x -- measured:
        66 GiB vs the 18 GiB actually needed at 262144 context, for the same
        output. Defaults to all layers for a plain dense model.
        """
        if capacity <= 0:
            raise ValueError("KV capacity must be positive")
        # K and V are set SEPARATELY because they are not equally sensitive.
        # Measured on real K/V (one layer, 96 positions, attention-output error
        # vs fp32) at identical memory:
        #     K8/V4 -> 4.64e-03      K4/V8 -> 4.84e-04
        # i.e. dropping K to 4 bits costs 9.6x LESS than dropping V to 4 bits at
        # the same byte count. So V wants the bits and K can be crushed -- the
        # reverse of the usual assumption. `dtype` sets both; k_dtype/v_dtype
        # override either side.
        self.k_dtype = k_dtype or dtype
        self.v_dtype = v_dtype or dtype
        for side, dt, dim in (("K", self.k_dtype, head_dim),
                              ("V", self.v_dtype, value_dim)):
            if dt not in ("fp32", "int8", "int4"):
                raise ValueError(
                    f"unknown {side} dtype {dt!r}; use 'fp32', 'int8' or 'int4'")
            if dt == "int4" and dim % 2:
                raise ValueError(
                    f"int4 packs pairs; {side} dim must be even, got {dim}")
        if layer_indices is None:
            layer_indices = list(range(n_layer))
        if not layer_indices:
            raise ValueError("layer_indices is empty; no layer would use the cache")
        # slot i holds the cache for layer layer_indices[i]
        self.layer_indices = list(layer_indices)
        self.slot = {li: i for i, li in enumerate(self.layer_indices)}
        n_layer = len(self.layer_indices)
        self.capacity = capacity

        # WINDOWED LAYERS: a sliding-window layer can never attend further back
        # than its window, so a full-context buffer is capacity it can never
        # use. Measured on gemma4 (48 layers, 8 global, 40 windowed @1024):
        # 262144 context allocated 82.00G for the 2.31G actually NEEDED -- 35.5x
        # of dead allocation. Each windowed layer instead gets a ring of
        # (window + slack); `slack` absorbs the tokens in flight within one
        # step so a ring never has to evict a key the current step still reads.
        # PER-LAYER KV HEAD COUNTS. A file may vary them: gemma4 gives its 8
        # global layers 1 KV head and the other 40 eight. Applying one uniform
        # count allocated those global layers 8x the K/V they can use -- the
        # same over-allocation class as the SSM and windowed cases, and it was
        # the last one open. heads_per_layer maps layer -> n_head_kv.
        # PRE-ROPE K STORAGE. Storing K *after* rotation bakes each entry's
        # absolute position into it, so a prefix reused at a different offset
        # does not match (measured: 0.906 relative difference at a shift of 5).
        # Storing RAW K and rotating at read time makes an entry
        # position-independent, so identical prefixes are bit-identical at ANY
        # offset -- which is what makes LZ/radix prefix sharing work.
        # Measured cost: rotation moves from write-time to read-time, i.e. O(S)
        # trig per layer per token instead of O(T).
        self.rope_k = bool(rope_k)
        self.heads = {li: int((heads_per_layer or {}).get(li, n_head_kv))
                      for li in self.layer_indices}
        if any(h <= 0 for h in self.heads.values()):
            raise ValueError(f"every layer needs >=1 KV head; got {self.heads}")
        self.windowed = dict(windowed or {})
        self.slack = 64
        self.caps = []
        for li in self.layer_indices:
            w = self.windowed.get(li)
            self.caps.append(min(capacity, w + self.slack) if w else capacity)
        self._is_ring = [c < capacity for c in self.caps]

        self.n_head_kv = n_head_kv
        self.head_dim = head_dim
        self.value_dim = value_dim
        self.dtype = dtype

        self.k, self.k_scale = self._alloc_side(self.k_dtype, head_dim)
        self.v, self.v_scale = self._alloc_side(self.v_dtype, value_dim)

        self.len = 0
        self.appends = 0
        self._quant_err = 0.0

    def _alloc_side(self, dt: str, dim: int):
        """Allocate one side (K or V), per layer, at that layer's head count."""
        hh = [self.heads[li] for li in self.layer_indices]
        if dt == "int4":
            # NIBBLE-PACKED: two 4-bit values per byte. This is what actually
            # halves the bytes; quantising to a 4-bit RANGE inside an int8 array
            # would save nothing. Measured: 262144 ctx -> 2.32 GiB vs int8 4.57.
            arr = [np.zeros((c, h, dim // 2), dtype=np.uint8)
                   for c, h in zip(self.caps, hh)]
            sc = [np.zeros((c, h), dtype=np.float32)
                  for c, h in zip(self.caps, hh)]
        elif dt == "int8":
            arr = [np.zeros((c, h, dim), dtype=np.int8)
                   for c, h in zip(self.caps, hh)]
            sc = [np.zeros((c, h), dtype=np.float32)
                  for c, h in zip(self.caps, hh)]
        else:
            arr = [np.zeros((c, h, dim), dtype=np.float32)
                   for c, h in zip(self.caps, hh)]
            sc = None
        return arr, sc

    def _write_side(self, dt, buf, sc, slot, lo, hi, o0, o1, x):
        """Quantise+write one span of one side."""
        if dt == "int4":
            q, s = self._quantize4(x)
            buf[slot][lo:hi] = self._pack4(q)[o0:o1]
            sc[slot][lo:hi] = s[o0:o1]
        elif dt == "int8":
            q, s = self._quantize(x)
            buf[slot][lo:hi] = q[o0:o1]
            sc[slot][lo:hi] = s[o0:o1]
        else:
            xf = x.astype(np.float32, copy=False)
            buf[slot][lo:hi] = xf[o0:o1]

    def _read_side(self, dt, buf, sc, slot, idx, dim):
        """Materialise fp32 from one side, given a slice or index array."""
        if dt == "int4":
            return self._unpack4(buf[slot][idx], dim) * sc[slot][idx][:, :, None]
        if dt == "int8":
            return buf[slot][idx].astype(np.float32) * sc[slot][idx][:, :, None]
        return buf[slot][idx]

    @staticmethod
    def _pack4(q: np.ndarray) -> np.ndarray:
        """Pack signed 4-bit values two-per-byte (low nibble first).

        Stored as (v & 0x0F); unpacking restores the two's-complement value.
        This is what actually halves the bytes -- quantising to a 4-bit RANGE
        while still storing int8 would save nothing.
        """
        d = q.shape[-1]
        if d % 2:
            raise ValueError(f"last dim {d} must be even to pack 4-bit pairs")
        u = (q.astype(np.int16) & 0x0F).astype(np.uint8)
        lo = u[..., 0::2]
        hi = u[..., 1::2]
        return (lo | (hi << 4)).astype(np.uint8)

    @staticmethod
    def _unpack4(packed: np.ndarray, dim: int) -> np.ndarray:
        lo = (packed & 0x0F).astype(np.int16)
        hi = ((packed >> 4) & 0x0F).astype(np.int16)
        out = np.empty(packed.shape[:-1] + (dim,), dtype=np.int16)
        out[..., 0::2] = lo
        out[..., 1::2] = hi
        # sign-extend from 4 bits: values are in [-8, 7]
        return np.where(out >= 8, out - 16, out).astype(np.float32)

    @staticmethod
    def _quantize4(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Symmetric 4-bit: range [-7, 7] so +/-1.0 is representable."""
        xf = x.astype(np.float32, copy=False)
        amax = np.abs(xf).max(axis=-1, keepdims=True)
        scale = np.where(amax > 0, amax / 7.0, 1.0).astype(np.float32)
        q = np.clip(np.round(xf / scale), -7, 7).astype(np.int8)
        return q, scale.squeeze(-1)

    @staticmethod
    def _quantize(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Symmetric int8 with a per-vector scale. 127 not 128, so +1.0 is
        representable and the range is symmetric."""
        xf = x.astype(np.float32, copy=False)
        amax = np.abs(xf).max(axis=-1, keepdims=True)
        scale = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
        q = np.clip(np.round(xf / scale), -127, 127).astype(np.int8)
        return q, scale.squeeze(-1)

    def _read_slot(self, slot: int, end: int) -> Tuple[np.ndarray, np.ndarray]:
        """Read by SLOT (already mapped), in CHRONOLOGICAL order.

        For a ring (windowed) layer the buffer holds only the most recent
        `cap` positions and wraps, so the physical layout is not the logical
        order. `end` is the absolute position count; the newest entry is at
        (end-1) % cap. Rolling by that offset recovers chronological order.
        """
        cap = self.caps[slot]
        if not self._is_ring[slot] or end <= cap:
            n = min(end, cap)
            idx = slice(0, n)
        else:
            # wrapped: rotate so index 0 is the OLDEST retained position
            start = end % cap
            idx = np.concatenate([np.arange(start, cap), np.arange(0, start)])
        k = self._read_side(self.k_dtype, self.k, self.k_scale,
                            slot, idx, self.head_dim)
        v = self._read_side(self.v_dtype, self.v, self.v_scale,
                            slot, idx, self.value_dim)
        return k, v

    def _read(self, layer: int, end: int) -> Tuple[np.ndarray, np.ndarray]:
        """Read by LAYER index (mapped here)."""
        return self._read_slot(self.slot[layer], end)

    def append(self, layer: int, k: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Write at the current offset and return views over [0, len+n).

        Does NOT advance `len` -- that is `commit()`'s job, so a whole step
        (every layer) lands at the same offset.
        """
        if layer not in self.slot:
            raise KeyError(
                f"layer {layer} has no KV slot (cache holds {self.layer_indices}); "
                f"a non-attention layer must not append")
        slot = self.slot[layer]
        n = k.shape[0]
        end = self.len + n
        cap = self.caps[slot]
        ring = self._is_ring[slot]

        if not ring and end > cap:
            raise MemoryError(
                f"KV cache full: need {end} positions, capacity is {cap}")
        if ring and n > cap:
            raise MemoryError(
                f"a single step of {n} tokens cannot fit a ring of {cap}")

        # a ring wraps: write at (len % cap), possibly in two pieces
        if ring:
            lo = self.len % cap
            hi = lo + n
            if hi <= cap:
                spans = [(lo, hi, 0, n)]
            else:
                spans = [(lo, cap, 0, cap - lo), (0, hi - cap, cap - lo, n)]
        else:
            spans = [(self.len, end, 0, n)]

        for lo_, hi_, o0, o1 in spans:
            self._write_side(self.k_dtype, self.k, self.k_scale, slot,
                             lo_, hi_, o0, o1, k)
            self._write_side(self.v_dtype, self.v, self.v_scale, slot,
                             lo_, hi_, o0, o1, v)

        # record what the rounding actually cost, on real data, for whichever
        # side is lossy -- a compression whose cost is not measured is a claim
        for dt, buf, sc, dim, x in ((self.k_dtype, self.k, self.k_scale,
                                     self.head_dim, k),
                                    (self.v_dtype, self.v, self.v_scale,
                                     self.value_dim, v)):
            if dt == "fp32":
                continue
            back = self._read_side(dt, buf, sc, slot, slice(0, n), dim)
            xref = x.astype(np.float32, copy=False)
            den = max(float(np.abs(xref).max()), 1e-9)
            self._quant_err = max(self._quant_err,
                                  float(np.abs(back - xref).max()) / den)

        self.appends += 1
        return self._read_slot(slot, end)

    def commit(self, n: int) -> None:
        self.len += n

    def quant_error(self) -> float:
        """Max relative error introduced by int8 storage (0.0 for fp32)."""
        return self._quant_err

    @property
    def nbytes(self) -> int:
        n = sum(a.nbytes for a in self.k) + sum(a.nbytes for a in self.v)
        if self.k_scale is not None:
            n += sum(a.nbytes for a in self.k_scale)
        if self.v_scale is not None:
            n += sum(a.nbytes for a in self.v_scale)
        return n

    @property
    def fp32_equivalent_bytes(self) -> int:
        """What the same UNIFORM-head allocation would cost at fp32, so the
        saving is visible rather than implied. Uses the largest head count, i.e.
        the old (wrong) behaviour, so the comparison is honest about what was
        saved."""
        per = max(self.heads.values()) if self.heads else self.n_head_kv
        return (len(self.k) * self.capacity * per *
                (self.head_dim + self.value_dim) * 4)

    @property
    def n_layers_cached(self) -> int:
        return len(self.k)

    # OPEN: the KV head count is applied uniformly to every layer, but a file
    # may vary it per layer (gemma4: 1 head on its 8 global layers, 8 on the
    # other 40). Those global layers are therefore allocated 8x the K/V they
    # need -- the same class of over-allocation as the SSM and windowed cases,
    # and still unfixed. Fixing it needs a per-layer head count in the cache.
    @property
    def ring_bytes_saved(self) -> int:
        """Bytes NOT allocated because windowed layers got a ring instead."""
        w = {"int4": 0.5, "int8": 1.0}.get(self.dtype, 4.0)
        return int(sum((self.capacity - c) * self.heads[li] *
                       (self.head_dim + self.value_dim) * w
                       for c, li in zip(self.caps, self.layer_indices)))


def attention_tiled(q: np.ndarray, k: np.ndarray, v: np.ndarray, scale: float,
                    qpos: np.ndarray, kpos: Optional[np.ndarray] = None,
                    block: int = 1024, window: Optional[int] = None,
                    kv_rep: int = 1, stats: Optional[dict] = None) -> np.ndarray:
    """Memory-bounded attention using an ONLINE softmax over key blocks.

    The naive form materialises the full (T, H, S) score matrix, which is
    256 GiB at T=65536 -- an OOM, not a slowdown. Here only (T, H, block) exists
    at any moment, so peak score memory is independent of the key count.

    `kv_rep` expands grouped-query KV heads to query heads PER BLOCK, so the
    expanded copy is (block, H, D) rather than (S, H, D) -- keeping GQA's memory
    advantage intact instead of undoing it at expansion time.

    Online softmax is algebraically identical to the batch form, so this is
    exact, not an approximation. Returns (T, H, value_dim).
    """
    T, H, D = q.shape
    S = k.shape[0]
    VD = v.shape[2]
    if kpos is None:
        kpos = np.arange(S)

    acc = np.zeros((T, H, VD), dtype=np.float32)
    m = np.full((T, H, 1), -np.inf, dtype=np.float32)
    l = np.zeros((T, H, 1), dtype=np.float32)

    for b0 in range(0, S, block):
        b1 = min(b0 + block, S)
        kb = k[b0:b1]
        vb = v[b0:b1]
        if kv_rep > 1:
            kb = np.repeat(kb, kv_rep, axis=1)
            vb = np.repeat(vb, kv_rep, axis=1)
        s = np.einsum("thd,shd->ths", q, kb) * scale            # (T,H,b)
        if stats is not None:
            stats["peak_scores_bytes"] = max(stats.get("peak_scores_bytes", 0), s.nbytes)
            stats["blocks"] = stats.get("blocks", 0) + 1

        keep = kpos[b0:b1][None, :] <= qpos[:, None]
        if window is not None:
            keep &= (qpos[:, None] - kpos[b0:b1][None, :]) < window
        s = np.where(keep[:, None, :], s, -np.inf)

        block_max = np.max(s, axis=-1, keepdims=True)
        m_new = np.maximum(m, block_max)

        # Both rescales below would otherwise compute -inf - -inf (nan) before
        # the guard replaces it, which emits a RuntimeWarning and would mask a
        # genuine nan later. Substitute a finite value first, then select.
        finite_m = np.isfinite(m)
        m_safe = np.where(finite_m, m, 0.0)
        mnew_safe = np.where(np.isfinite(m_new), m_new, 0.0)
        alpha = np.where(finite_m, np.exp(m_safe - mnew_safe), 0.0)

        finite_s = np.isfinite(s)
        p = np.where(finite_s, np.exp(np.where(finite_s, s, 0.0) - mnew_safe), 0.0)

        l = l * alpha + np.sum(p, axis=-1, keepdims=True)
        acc = acc * alpha + np.einsum("ths,shv->thv", p, vb)
        m = m_new

    return acc / np.maximum(l, 1e-30)


def attention(x: np.ndarray, pos: np.ndarray, w: AttnWeights, cfg: AttnConfig,
              cache: Optional[KVCache] = None, layer: int = 0,
              mask: Optional[np.ndarray] = None, block: int = 1024,
              window: Optional[int] = None, commit: bool = True) -> np.ndarray:
    """Multi-head attention with grouped KV heads, memory-bounded over keys.

    x: (T, n_embd). Returns (T, n_embd) after the output projection.

    `commit` advances the KV cache by T after appending. Multi-layer models pass
    commit=False for every layer and call cache.commit(T) once, so all layers
    write at the same offset.
    """
    T = x.shape[0]
    H, HK, D = cfg.n_head, cfg.n_head_kv, cfg.head_dim
    DV = cfg.value_dim
    scale = cfg.scale if cfg.scale is not None else 1.0 / math.sqrt(D)

    if w.wqkv is not None:
        # fused layout for the SSM-attention layers is arch-specific; this split
        # is only valid when the arch says so (see qwen35 in arch.py)
        raise NotImplementedError("fused qkv requires an arch-specific split")

    q_raw = linear(x, w.wq)
    # GATED ATTENTION: attn_q emits 2x the q width. Which half is q, and how the
    # gate is applied, is a convention the file does not record -- see GATE_MODE.
    gate_in = None
    gate_mode = GATE_MODE
    if q_raw.shape[-1] == 2 * H * D:
        if gate_mode == "gateq_sigmoid":
            gate_in, q_raw = np.split(q_raw, 2, axis=-1)
        elif gate_mode == "qgate_interleave":
            # per-head pairs: reshape (T,H,2,D) -> [q_h, gate_h] adjacent
            pair = q_raw.reshape(T, H, 2, D)
            q_raw = pair[:, :, 0, :]
            gate_in = pair[:, :, 1, :].reshape(T, H * D)
        else:
            q_raw, gate_in = np.split(q_raw, 2, axis=-1)
    elif q_raw.shape[-1] != H * D:
        raise ValueError(
            f"attn_q emits {q_raw.shape[-1]} values; expected {H * D} or {2 * H * D}")

    q = q_raw.reshape(T, H, D)
    k = linear(x, w.wk).reshape(T, HK, D)
    v = linear(x, w.wv).reshape(T, HK, DV)

    if w.q_norm is not None:
        q = rms_norm(q, w.q_norm, cfg.eps)
    if w.k_norm is not None:
        k = rms_norm(k, w.k_norm, cfg.eps)

    cos, sin = rope_tables(D, pos, cfg.rope_base, cfg.rope_dims)
    q = apply_rope(q, cos, sin)

    # PRE-ROPE K: when the cache stores raw K, rotation moves to READ time (the
    # block after the append) so a cached entry carries no absolute position and
    # an identical prefix is reusable at ANY offset. With a plain cache K is
    # rotated here, exactly as before -- the two paths must agree bit-for-bit.
    pre_rope = cache is not None and getattr(cache, "rope_k", False)
    if not pre_rope:
        k = apply_rope(k, cos, sin)

    if H % HK:
        raise ValueError(f"n_head {H} is not a multiple of n_head_kv {HK}")

    if cache is not None:
        k_all, v_all = cache.append(layer, k, v)
        kpos = np.arange(k_all.shape[0])
        if commit:
            cache.commit(T)
    else:
        k_all, v_all = k, v
        kpos = None

    if pre_rope:
        # Rotate the whole cached block by its entries' ABSOLUTE positions.
        #
        # The base MUST be derived, not assumed to be 0: a caller may pass
        # `positions` starting at any offset (a continuation, a reused prefix),
        # and entry 0 of the cache then sits at pos[0] - (entries already there).
        # Hardcoding 0 rotated K by 0,1,2 while Q rotated by 5,6,7 -- mismatched
        # RELATIVE positions, which silently corrupts attention. Caught by
        # tests/test_prerope.py [4].
        base = int(pos[0]) - (cache.len if cache is not None else 0)
        kpos = np.arange(base, base + k_all.shape[0], dtype=np.int64)
        c2, s2 = rope_tables(D, kpos, cfg.rope_base, cfg.rope_dims)
        k_all = apply_rope(k_all, c2, s2)

    out = attention_tiled(q, k_all, v_all, scale, pos, kpos=kpos,
                          block=block, window=window, kv_rep=H // HK)
    if mask is not None:
        # an explicit additive mask (e.g. SWA) is applied by the caller through
        # `window`; a raw mask is not supported alongside tiling
        raise NotImplementedError("explicit masks are not supported with tiling; use window=")

    out = out.reshape(T, H * DV)
    if gate_in is not None and gate_mode != "none":
        g = gate_in.astype(np.float32)
        if gate_mode.endswith("silu"):
            g = g / (1.0 + np.exp(-g))          # silu = z*sigmoid(z)
        else:
            g = 1.0 / (1.0 + np.exp(-g))        # sigmoid
        out = out * g
    return linear(out, w.wo)


# ---------------------------------------------------------------------- MLP

@dataclass
class MlpWeights:
    w_gate: np.ndarray
    w_up: np.ndarray
    w_down: np.ndarray


def mlp(x: np.ndarray, w: MlpWeights) -> np.ndarray:
    return linear(swiglu(linear(x, w.w_gate), linear(x, w.w_up)), w.w_down)


# ------------------------------------------------------------------ sampling

@dataclass
class SampleParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    seed: Optional[int] = None


def sample(logits: np.ndarray, p: SampleParams, rng: Optional[np.random.Generator] = None) -> int:
    """Sample one token id from logits."""
    rng = rng or np.random.default_rng(p.seed)
    z = logits.astype(np.float32).copy()

    if p.temperature <= 0:                      # greedy
        return int(np.argmax(z))
    z = z / p.temperature

    if p.top_k and p.top_k < z.size:
        thresh = np.partition(z, -p.top_k)[-p.top_k]
        z[z < thresh] = -np.inf

    if p.top_p < 1.0:
        order = np.argsort(-z)
        sorted_z = z[order]
        probs = softmax(sorted_z)
        csum = np.cumsum(probs)
        cutoff = np.searchsorted(csum, p.top_p) + 1
        keep = order[:cutoff]
        newz = np.full_like(z, -np.inf)
        newz[keep] = z[keep]
        z = newz

    probs = softmax(z)
    return int(rng.choice(z.size, p=probs))


# ------------------------------------------------------------------ prefill

def prefill_chunks(total: int, chunk: int = 1024):
    """Yield (start, stop) spans for CHUNKED prefill.

    Prefilling in fixed-size chunks bounds the activation scratch to a known
    maximum independent of prompt length, which is what lets --max-ram be
    enforced by construction rather than hoped for. A caller runs one forward
    pass per span and checks the guard between spans.
    """
    if chunk <= 0:
        raise ValueError("chunk must be positive")
    for start in range(0, total, chunk):
        yield start, min(start + chunk, total)


# ------------------------------------------------------------------ the plan

ARCH_PLAN = """
To run the actual models, three arch-specific pieces remain:

  qwen35 / qwen35moe
    * gated-delta-net SSM layer, interleaved with attention every
      full_attention_interval (qwen35: 9 attn + 24 ssm; moe: 10 + 30)
      tensors: attn_qkv (in-proj), attn_gate, ssm_a, ssm_alpha, ssm_beta,
               ssm_conv1d, ssm_dt.bias, ssm_norm, ssm_out
      hparams: ssm.state_size, ssm.group_count, ssm.inner_size,
               ssm.time_step_rank, ssm.conv_kernel
    * qwen35moe additionally: MoE with shared experts (ffn_*_exps,
      ffn_*_shexp, ffn_gate_inp, ffn_gate_inp_shexp, top-k routing)
    * qwen35 has MTP heads (nextn.*) -- ignorable for plain generation

  gemma4
    * sliding-window attention pattern (attention.sliding_window_pattern)
    * per-layer embeddings (per_layer_token_embd, per_layer_model_proj,
      per_layer_proj_norm, inp_gate, proj)
    * final logit soft-capping (final_logit_softcapping)
    * rope_freqs table

Each needs its own verified forward pass; guessing at the bit-level layout of a
gated-delta-net from tensor shapes alone would produce plausible-looking garbage.
"""
