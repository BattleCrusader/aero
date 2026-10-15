"""Gated DeltaNet (SSM) layer for qwen35 / qwen35moe.

These models interleave linear-attention layers (`ssm_*`) with full-attention
layers. 24 of Qwen3.8-4B's 33 layers are SSM, so nothing runs without this.

THE RECURRENCE (gated delta rule, Yang/Kautz/Hatamizadeh):

    g       = -exp(A_log) * softplus(alpha + dt_bias)     log-space decay
    beta    = sigmoid(beta_proj)
    S       = S * exp(g)                                  decay the state
    v_new   = (v - k @ S) * beta                          delta rule
    S       = S + outer(k, v_new)                         rank-1 write
    o       = scale * (q @ S)                             read

Derived from the published formulation, not from any existing implementation.

=== ASSUMPTIONS -- these are the parts I could NOT verify from the file alone ===

The tensor SHAPES are read from the real model and the arithmetic closes exactly
(q 16x128 + k 16x128 + v 32x128 = 8192 = the qkv output width). What the file does
NOT specify, and which would silently produce garbage if wrong:

  1. **q and k are L2-normalised per head** before the recurrence, with the scale
     applied to q. This is standard for GDN (and matches the absence of any
     q_norm/k_norm weights on SSM layers -- normalisation with no learnable
     parameters leaves no trace in the file).
  2. **The output gate is silu(gate) * norm(o)**, applied before `ssm_out`.
  3. **The conv is causal and depthwise** over the fused qkv channels, with
     kernel = ssm.conv_kernel.
  4. **v-head h reads k-head h % n_k_heads** -- llama.cpp TILES q/k up to the v-head
     count (ggml_repeat_4d), which is CYCLIC. h // ratio is a different function
     (it was the original bug). VERIFIED against PrismML-llama.cpp
     src/models/qwen35.cpp + delta-net-base.cpp: no longer an assumption.
     (which v-heads share a k-head) is convention, not data.

Each is isolated in a small named function so it can be flipped in one place once
a real-model check is possible. Until then this layer is STRUCTURALLY verified
(shapes, causality, state boundedness, recurrence identity vs a loop reference),
NOT numerically verified against a real model's output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .transformer import linear, rms_norm, silu

try:                                    # native inner loops (src/ssm.cpp)
    from . import native_ssm as _native_ssm
except ImportError:                     # pragma: no cover - lib not built
    _native_ssm = None


def _native_mod():
    """The native SSM kernels, or None when the library is not built."""
    if _native_ssm is not None and _native_ssm.available():
        return _native_ssm
    return None



def softplus(x: np.ndarray) -> np.ndarray:
    """log(1 + e^x), stable for large |x|."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    """ASSUMPTION 1. Unit-norm each vector; no learnable parameters."""
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


# --------------------------------------------------------------------- conv

class ConvState:
    """Causal depthwise conv1d state, holding the last (kernel-1) inputs."""

    def __init__(self, channels: int, kernel: int) -> None:
        self.channels = channels
        self.kernel = kernel
        self.tail = np.zeros((0, channels), dtype=np.float32)

    def step(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        """x: (T, channels), w: (channels, kernel). Returns (T, channels).

        TAP ORDER: w[:, 0] multiplies the OLDEST input in the window and
        w[:, kernel-1] the CURRENT one. Verified from the reference's own index
        arithmetic -- for output t it accumulates s[t + i0] * c[i0], so tap 0
        pairs with the earliest sample. Having it the other way round reverses
        the filter and corrupts every SSM layer while leaving shapes and
        magnitudes entirely plausible.
        """
        T, C = x.shape
        if w.shape != (C, self.kernel):
            raise ValueError(f"conv weight {w.shape} does not match ({C}, {self.kernel})")
        buf = np.concatenate([self.tail, x.astype(np.float32)], axis=0)
        n = buf.shape[0]
        out = np.zeros((T, C), dtype=np.float32)
        for t in range(T):
            end = n - T + t + 1                 # inclusive end of the window
            start = end - self.kernel
            acc = np.zeros(C, dtype=np.float32)
            for j in range(self.kernel):
                idx = start + j
                if 0 <= idx < n:
                    # j = 0 is the OLDEST input in the window -> w[:, 0]
                    acc += buf[idx] * w[:, j]
            out[t] = acc
        keep = max(0, self.kernel - 1)
        self.tail = buf[-keep:] if keep else np.zeros((0, C), np.float32)
        return out


# ---------------------------------------------------------------------- GDN

@dataclass
class SsmConfig:
    n_embd: int
    inner_size: int          # total v width
    state_size: int          # head_k_dim == head_v_dim
    n_group: int             # number of k heads
    conv_kernel: int
    time_step_rank: int
    eps: float = 1e-6
    norm_eps: float = 1e-6

    @property
    def n_v_heads(self) -> int:
        if self.inner_size % self.state_size:
            raise ValueError("inner_size must be a multiple of state_size")
        return self.inner_size // self.state_size

    @property
    def n_k_heads(self) -> int:
        return self.n_group

    @property
    def ratio(self) -> int:
        if self.n_v_heads % self.n_k_heads:
            raise ValueError("n_v_heads must be a multiple of n_k_heads")
        return self.n_v_heads // self.n_k_heads

    @property
    def q_width(self) -> int:
        return self.n_k_heads * self.state_size

    @property
    def kv_width(self) -> int:
        return self.n_k_heads * self.state_size

    @property
    def qkv_width(self) -> int:
        return self.q_width + self.kv_width + self.inner_size

    @property
    def scale(self) -> float:
        return 1.0 / math.sqrt(self.state_size)


@dataclass
class SsmWeights:
    qkv: np.ndarray
    gate: np.ndarray          # (inner_size, n_embd)
    alpha: np.ndarray         # (time_step_rank, n_embd)
    beta: np.ndarray          # (time_step_rank, n_embd)
    a_log: np.ndarray         # (time_step_rank,)
    dt_bias: np.ndarray       # (time_step_rank,)
    conv1d: np.ndarray        # (qkv_width, conv_kernel)
    norm: np.ndarray          # (state_size,)
    out: np.ndarray           # (n_embd, inner_size)


@dataclass
class SsmState:
    """Per-layer recurrent state: fixed size, independent of context length.

    This is why these models tolerate long context cheaply -- 24 of 33 layers
    hold a constant-size state instead of a growing KV cache.
    """
    n_v_heads: int
    state_size: int
    inner_size: int
    conv: ConvState

    @classmethod
    def create(cls, cfg: SsmConfig) -> "SsmState":
        return cls(cfg.n_v_heads, cfg.state_size, cfg.inner_size,
                   ConvState(cfg.qkv_width, cfg.conv_kernel))

    def zeros(self) -> np.ndarray:
        return np.zeros((self.n_v_heads, self.state_size, self.state_size),
                        dtype=np.float32)

    @property
    def nbytes(self) -> int:
        return self.n_v_heads * self.state_size * self.state_size * 4


def delta_rule_step(q, k, v, alpha, beta, a_log, dt_bias, state, scale, eps=1e-6):
    """ONE time step of the gated delta rule.

    q,k:  (n_heads, d)     unit-normalised, grouped
    v:    (n_v_heads, d)
    alpha,beta: (n_v_heads,)   raw projections
    a_log,dt_bias: (time_step_rank,) broadcast to n_v_heads
    state: (n_v_heads, d, d)
    Returns (out (n_v_heads, d), new_state)
    """
    # Native path: the same recurrence in C++. Falls through to numpy if the
    # library is not built, so a source checkout without `make all` still runs.
    nm = _native_mod()
    if nm is not None:
        try:
            return nm.delta_step(q, k, v, alpha, beta, a_log, dt_bias,
                                 state, scale, eps)
        except nm.SsmUnavailable:
            pass

    # ASSUMPTION 1: L2-normalise q and k per head
    qn = l2_normalize(q.astype(np.float32), axis=-1, eps=eps)
    kn = l2_normalize(k.astype(np.float32), axis=-1, eps=eps)

    # Decay gate. `a_log` here is NOT a log: the file stores ssm_a already as
    # -exp(A_log) (it is negative for every entry, min -0.381 max -0.015), and
    # it is multiplied into softplus(alpha + dt_bias) DIRECTLY. Applying -exp()
    # to it again double-transforms the decay and is what flattened the state.
    g = a_log.astype(np.float32) * softplus(
        alpha.astype(np.float32) + dt_bias.astype(np.float32))
    # A decay factor cannot exceed 1, so g must be <= 0; with ssm_a negative (as
    # the file stores it) that holds by construction. Clamp anyway so a caller
    # passing a positive ssm_a gets a saturated decay instead of an exp overflow
    # and a nan downstream -- silently overflowing here is how this class of bug
    # hides.
    decay = np.exp(np.minimum(g, 0.0))                    # (n_v_heads,) in (0,1]

    b = 1.0 / (1.0 + np.exp(-beta.astype(np.float32)))    # sigmoid

    state = state * decay[:, None, None]
    v_pred = np.einsum("hd,hde->he", kn, state)           # what memory predicts
    v_new = (v.astype(np.float32) - v_pred) * b[:, None]  # delta rule
    state = state + np.einsum("hd,he->hde", kn, v_new)    # rank-1 write
    out = scale * np.einsum("hd,hde->he", qn, state)      # read
    return out, state


def _ssm_forward_native(nm, x, w, cfg, state, gdn_state):
    """Native full-layer forward. Returns (out, new_gdn_state), or None to fall
    back to numpy when the dimension combination is one the C++ refuses.

    `state.conv.tail` carries the conv history across calls exactly as the numpy
    ConvState does; gdn_state is stepped on a copy so the caller's array is not
    mutated (the numpy path returns a fresh state too).
    """
    nv, d = cfg.n_v_heads, cfg.state_size
    if cfg.time_step_rank != nv or cfg.inner_size != nv * d:
        return None
    if state is None:
        state = SsmState.create(cfg)
    if gdn_state is None:
        gdn_state = state.zeros()
    gdn_state = np.asarray(gdn_state, dtype=np.float32)
    if gdn_state.shape != (nv, d, d):
        return None

    keep = max(cfg.conv_kernel - 1, 0)
    tailbuf = np.zeros((keep, cfg.qkv_width), dtype=np.float32)
    tl = int(state.conv.tail.shape[0])
    if tl > 0:
        tailbuf[:tl] = state.conv.tail
    st = np.ascontiguousarray(gdn_state).copy()

    res = nm.forward(
        np.ascontiguousarray(x, dtype=np.float32),
        w.qkv, w.gate, w.alpha, w.beta, w.a_log, w.dt_bias, w.conv1d, w.norm,
        w.out, cfg.n_embd, cfg.inner_size, cfg.state_size, cfg.n_group,
        cfg.conv_kernel, cfg.time_step_rank, cfg.eps, cfg.norm_eps, tailbuf, tl, st)
    if res is None:
        return None
    out, new_tl = res
    state.conv.tail = tailbuf[:new_tl].copy()
    return out, st


def ssm_forward(x: np.ndarray, w: SsmWeights, cfg: SsmConfig,
                state: Optional[SsmState] = None,
                gdn_state: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Full SSM layer for T tokens.

    x: (T, n_embd). Returns (out (T, n_embd), new_gdn_state).
    `gdn_state` is (n_v_heads, state_size, state_size); one per layer.
    """
    nm = _native_mod()
    if nm is not None:
        try:
            r = _ssm_forward_native(nm, x, w, cfg, state, gdn_state)
            if r is not None:
                return r
        except nm.SsmUnavailable:
            pass

    T = x.shape[0]
    if state is None:
        state = SsmState.create(cfg)
    if gdn_state is None:
        gdn_state = state.zeros()

    # 1. fused projection + causal depthwise conv
    qkv = linear(x, w.qkv)                                  # (T, qkv_width)
    qkv = state.conv.step(qkv, w.conv1d)
    # SILU AFTER THE CONV. Without it the q/k/v fed to the recurrence are a
    # linear convolution of the input rather than a gated one, which changes
    # every value that follows by a large factor.
    qkv = silu(qkv)

    q, k, v = np.split(qkv, [cfg.q_width, cfg.q_width + cfg.kv_width], axis=-1)

    # 2. gates: alpha/beta are time_step_rank wide == n_v_heads here
    alpha = linear(x, w.alpha)                              # (T, tsr)
    beta = linear(x, w.beta)
    gate = linear(x, w.gate)                                # (T, inner_size)

    nk, d, nv = cfg.n_k_heads, cfg.state_size, cfg.n_v_heads
    rep = cfg.ratio
    qh = q.reshape(T, nk, d)
    kh = k.reshape(T, nk, d)
    vh = v.reshape(T, nv, d)

    outs = np.empty((T, nv, d), dtype=np.float32)
    S = gdn_state
    for t in range(T):
        # v-head h reads k-head h % n_k. The reference TILES the k/q heads up to
        # the v-head count (ggml_repeat_4d), so head h maps to h mod n_k. The
        # alternative (contiguous blocks, h // ratio) is a different function
        # and was wrong.
        k_exp = np.tile(kh[t], (rep, 1)) if rep > 1 else kh[t]
        q_exp = np.tile(qh[t], (rep, 1)) if rep > 1 else qh[t]
        o, S = delta_rule_step(q_exp, k_exp, vh[t], alpha[t], beta[t],
                               w.a_log, w.dt_bias, S, cfg.scale, cfg.eps)
        outs[t] = o

    # 3. ASSUMPTION 2: gated RMSNorm over the head dim, then silu gate
    o = rms_norm(outs.reshape(T, nv * d), np.tile(w.norm, nv), cfg.norm_eps)
    o = o * silu(gate)

    # 4. output projection
    return linear(o, w.out), S


def ssm_weights_from(reader, layer: int) -> SsmWeights:
    """Load one SSM layer's weights from a GGUF reader."""
    p = f"blk.{layer}."
    return SsmWeights(
        qkv=reader.tensor(p + "attn_qkv.weight"),
        gate=reader.tensor(p + "attn_gate.weight"),
        alpha=reader.tensor(p + "ssm_alpha.weight"),
        beta=reader.tensor(p + "ssm_beta.weight"),
        a_log=reader.tensor(p + "ssm_a"),
        dt_bias=reader.tensor(p + "ssm_dt.bias"),
        conv1d=reader.tensor(p + "ssm_conv1d.weight"),
        norm=reader.tensor(p + "ssm_norm.weight"),
        out=reader.tensor(p + "ssm_out.weight"),
    )
