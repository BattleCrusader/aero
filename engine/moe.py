"""Mixture-of-Experts FFN, as it appears in qwen35moe.

Read off the real file (`Qwen3.6-14B-A3B`, 40 layers, 90 experts, 8 active):

    ffn_gate_inp.weight       (n_expert, n_embd)      router logits
    ffn_gate_inp_shexp.weight (1, n_embd)             shared-expert gate
    ffn_gate_exps.weight      (n_expert, n_embd, ffn) routed experts, gate half
    ffn_up_exps.weight        (n_expert, n_embd, ffn) routed experts, up half
    ffn_down_exps.weight      (n_expert, ffn, n_embd) routed experts, down half
    ffn_gate_shexp.weight     (ffn, n_embd)           the one shared expert
    ffn_up_shexp.weight       (ffn, n_embd)
    ffn_down_shexp.weight     (n_embd, ffn)

The expert count and the active count come from the file (`expert_count`,
`expert_used_count`), not from anything hard-coded here.

MEMORY NOTE: the expert tensors are the reason this design exists. For 14B-A3B
the gate/up/down stack is ~90 experts x 3 matrices x (2048 x 1408) per layer --
the whole point of the blocked reader is that only the SELECTED experts ever get
dequantised. This module therefore asks the reader for individual expert rows
rather than materialising an expert tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .transformer import silu

try:                                    # native inner loops (src/moe.cpp)
    from . import native_ssm as _native_ssm
except ImportError:                     # pragma: no cover - lib not built
    _native_ssm = None


def _native_mod():
    """The native MoE kernels, or None when the library is not built."""
    if _native_ssm is not None and _native_ssm.available():
        return _native_ssm
    return None


@dataclass
class MoeConfig:
    n_expert: int
    n_expert_used: int
    n_embd: int
    ffn_len: int
    has_shared: bool = True
    norm_topk_prob: bool = True

    @property
    def routed_width(self) -> int:
        return self.n_expert_used * self.ffn_len


def route(logits: np.ndarray, cfg: MoeConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Top-k routing over a single token's router logits.

    Returns (expert_index (k,), weight (k,)). Weights are softmax-normalised
    over the SELECTED experts only when norm_topk_prob is set -- otherwise the
    raw softmax probability of each selected expert is used, which makes the
    summed contribution smaller than one expert and changes the output scale.
    """
    k = cfg.n_expert_used
    if k <= 0 or k > cfg.n_expert:
        raise ValueError(f"n_expert_used={k} out of range for {cfg.n_expert} experts")

    # Native path: the same top-k + softmax in C++.
    nm = _native_mod()
    if nm is not None:
        try:
            return nm.route(logits, cfg.n_expert, k, cfg.norm_topk_prob)
        except nm.SsmUnavailable:
            pass

    # top-k by logit; ties resolved by lowest index for determinism
    idx = np.argpartition(-logits, k - 1)[:k]
    idx = idx[np.argsort(-logits[idx], kind="stable")]

    sel = logits[idx].astype(np.float32)
    if cfg.norm_topk_prob:
        m = sel.max()
        e = np.exp(sel - m)
        w = e / e.sum()
    else:
        # raw softmax denominator is over ALL experts
        m = float(np.max(logits))
        denom = float(np.exp(logits - m).sum(dtype=np.float64))
        w = np.exp(sel - m) / denom
    return idx.astype(np.int64), w.astype(np.float32)


def _moe_ffn_native(nm, h, cfg, expert_rows, shexp, block_rows, router_fn):
    """Native MoE FFN block. Raises on a bad router width (same contract as the
    numpy loop); the caller falls back only on SsmUnavailable."""
    h = np.ascontiguousarray(h, dtype=np.float32)
    T = h.shape[0]
    # The router consumes the ROUTER's logits, which are n_expert wide -- a
    # separate projection from the hidden state. Compute them per token here.
    router_logits = np.empty((T, cfg.n_expert), dtype=np.float32)
    for t in range(T):
        rl = np.asarray(router_fn(h[t]), dtype=np.float32).reshape(-1)
        if rl.shape[-1] != cfg.n_expert:
            raise ValueError(
                f"router produced {rl.shape[-1]} logits for {cfg.n_expert} experts")
        router_logits[t] = rl
    use_shared = bool(cfg.has_shared and shexp)
    return nm.ffn_block(h, router_logits, cfg.n_expert, cfg.n_expert_used,
                        cfg.ffn_len, cfg.norm_topk_prob,
                        shexp if use_shared else None, expert_rows, block_rows)


def moe_ffn(h: np.ndarray, cfg: MoeConfig, expert_rows,
            shexp: Optional[dict] = None,
            block_rows: int = 64,
            router_fn=None) -> np.ndarray:
    """The MoE FFN for a (T, n_embd) block.

    `expert_rows(name, expert, r0, r1)` must return dequantised rows [r0,r1) of
    one expert's matrix -- this is what keeps the footprint bounded.
    """
    T, E = h.shape
    if E != cfg.n_embd:
        raise ValueError(f"hidden width {E} != n_embd {cfg.n_embd}")
    if router_fn is None:
        # default: the router is a separate projection, so without one there is
        # nothing to route on. Refuse rather than route on the hidden state.
        raise ValueError(
            "moe_ffn requires router_fn (a callable mapping a hidden vector to "
            "n_expert router logits); routing on the hidden state would be wrong")

    # Native path: routing + silu(gate)*up + the weighted expert accumulation
    # all run in C++ (src/moe.cpp). The expert-row reader still lives in Python
    # (it is I/O), and is handed to the block as a fetch callback.
    nm = _native_mod()
    if nm is not None:
        try:
            r = _moe_ffn_native(nm, h, cfg, expert_rows, shexp, block_rows, router_fn)
            if r is not None:
                return r
        except nm.SsmUnavailable:
            pass

    out = np.zeros((T, cfg.n_embd), dtype=np.float32)

    for t in range(T):
        # routing consumes the ROUTER's logits, which are n_expert wide -- not
        # the hidden state, which is n_embd wide. Conflating the two silently
        # routes on the wrong vector (and breaks whenever they differ in size).
        router_logits = router_fn(h[t])
        if router_logits.shape[-1] != cfg.n_expert:
            raise ValueError(
                f"router produced {router_logits.shape[-1]} logits for "
                f"{cfg.n_expert} experts")
        idx, w = route(router_logits, cfg)
        acc = np.zeros(cfg.n_embd, dtype=np.float32)
        for e, wt in zip(idx, w):
            # gate/up for the selected expert only
            g = expert_rows("ffn_gate_exps.weight", int(e), block_rows)
            u = expert_rows("ffn_up_exps.weight", int(e), block_rows)
            act = silu(h[t] @ g.T) * (h[t] @ u.T)
            d = expert_rows("ffn_down_exps.weight", int(e), block_rows)
            acc += wt * (act @ d.T)
        out[t] = acc

    if cfg.has_shared and shexp:
        # the shared expert always fires, gated by its own router
        sgate = 1.0 / (1.0 + np.exp(-(h @ shexp["gate_inp"].T)))
        g = h @ shexp["gate"].T
        u = h @ shexp["up"].T
        act = silu(g) * u
        out += sgate * (act @ shexp["down"].T)
    return out


def moe_weights_from(reader, layer: int, cfg: MoeConfig) -> dict:
    """Shared-expert weights + the router, for one layer.

    Routed experts are deliberately NOT returned: they are read per-expert on
    demand through the row reader, which is the whole memory argument.
    """
    p = f"blk.{layer}."
    out = {
        "gate_inp": reader.tensor(p + "ffn_gate_inp.weight"),
        "shexp_gate_inp": reader.tensor(p + "ffn_gate_inp_shexp.weight"),
    }
    if cfg.has_shared:
        out["shexp"] = {
            "gate_inp": out["shexp_gate_inp"],
            "gate": reader.tensor(p + "ffn_gate_shexp.weight"),
            "up": reader.tensor(p + "ffn_up_shexp.weight"),
            "down": reader.tensor(p + "ffn_down_shexp.weight"),
        }
    return out


def expert_plan(cfg: MoeConfig) -> dict:
    """Predictive footprint of ONE expert's three matrices in fp32.

    Used by the memory guard before any expert is read, so the plan is refused
    up front rather than discovered by OOM.
    """
    per = cfg.n_embd * cfg.ffn_len * 4
    return {
        "one_expert_fp32_bytes": 3 * per,
        "active_experts_fp32_bytes": 3 * per * cfg.n_expert_used,
        "all_experts_fp32_bytes": 3 * per * cfg.n_expert,
        "saving_factor": float(cfg.n_expert) / max(cfg.n_expert_used, 1),
    }
