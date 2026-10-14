"""gemma4 specifics, read off the real file (`gemma-4-12b-it`, 48 layers).

What the file actually says:

    attention.sliding_window         = 1024
    attention.sliding_window_pattern = [T,T,T,T,T,F, T,T,T,T,T,F, ...]  (48 entries)
    attention.head_count_kv          = [8,8,8,8,8,1, ...]  per-layer, not scalar
    final_logit_softcapping          = 30.0
    attention.key_length             = 512
    rope.dimension_count             = 512
    rope.freq_base                   = 1000000
    rope_freqs.weight                global tensor

    per layer: attn_norm, attn_q/k/v + q_norm/k_norm, attn_output,
               post_attention_norm, ffn_norm, ffn_gate/up/down,
               post_ffw_norm, layer_output_scale

Structural facts the assembly must honour, and which are DIFFERENT from qwen35:

  1. Every layer is a full transformer block. There is no SSM here.
  2. Two norms per block extra vs qwen: `ffn_norm` BEFORE the FFN and
     `post_ffw_norm` AFTER it. qwen35 has `post_attention_norm` before the FFN;
     gemma4 has both a pre-FFN and a post-FFN norm.
  3. `layer_output_scale` multiplies the whole block's residual contribution --
     a per-layer scalar, not a vector.
  4. Sliding window applies per the PATTERN, so 5 of every 6 attention layers
     attend only to the last 1024 positions, and every 6th attends globally.
  5. The KV head count is a LIST. Layer 5, 11, 17 ... have 1 KV head (the global
     layers) while the others have 8. A scalar read would silently be wrong.
  6. Logits are soft-capped at the end: logits = cap * tanh(logits / cap).

`per_layer_token_embd` etc. are NOT present in this file, so the per-layer
embedding path is only taken when the tensors actually exist -- declared by the
arch spec, and verified against the file rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np



@dataclass
class GemmaConfig:
    n_layer: int
    n_embd: int
    n_head: int
    head_dim: int
    value_dim: int
    ffn_len: int
    eps: float
    sliding_window: Optional[int]
    sliding_pattern: Optional[Sequence[bool]]
    softcap: Optional[float]
    kv_heads: Sequence[int]           # per layer -- may be a list in the file
    rope_base: float
    rope_dims: int
    has_per_layer_embd: bool = False
    n_layer_embd: int = 0
    # PER-LAYER ROPE. gemma4 gives its windowed and full layers different rotary
    # geometry entirely: on gemma-4-12b the SWA layers rotate 256 dims at base
    # 10000 while the global ones rotate 512 at base 1000000. Using one pair of
    # values for both mis-rotates every layer of one kind.
    rope_base_swa: Optional[float] = None
    rope_dims_swa: Optional[int] = None
    attn_scale: Optional[float] = None      # 1.0 for gemma4: no pre-attn scaling
    n_kv_shared: int = 0

    @staticmethod
    def kv_heads_for(raw, n_layer: int) -> List[int]:
        """A scalar widens to every layer; a list must match the layer count."""
        if raw is None:
            raise ValueError("gemma4 requires attention.head_count_kv")
        if isinstance(raw, (list, tuple)):
            if len(raw) != n_layer:
                raise ValueError(
                    f"head_count_kv has {len(raw)} entries for {n_layer} layers")
            return [int(x) for x in raw]
        return [int(raw)] * n_layer

    @staticmethod
    def pattern_for(raw, n_layer: int) -> Optional[List[bool]]:
        if raw is None:
            return None
        if isinstance(raw, (list, tuple)):
            if len(raw) != n_layer:
                raise ValueError(
                    f"sliding_window_pattern has {len(raw)} entries "
                    f"for {n_layer} layers")
            return [bool(x) for x in raw]
        return [bool(raw)] * n_layer

    def is_swa(self, layer: int) -> bool:
        if self.sliding_pattern is None:
            return False
        return bool(self.sliding_pattern[layer])

    def rope_base_at(self, layer: int) -> float:
        """SWA layers may use a different rotary base than the global ones."""
        if self.rope_base_swa is not None and self.is_swa(layer):
            return self.rope_base_swa
        return self.rope_base

    def rope_dims_at(self, layer: int) -> int:
        if self.rope_dims_swa is not None and self.is_swa(layer):
            return self.rope_dims_swa
        return self.rope_dims

    def window_at(self, layer: int) -> Optional[int]:
        """Window for this layer, or None for a global layer.

        A global layer MUST get None, not the window value -- passing the window
        to a global layer silently truncates its context to 1024.
        """
        if self.sliding_window is None or self.sliding_pattern is None:
            return None
        return self.sliding_window if self.sliding_pattern[layer] else None

    @property
    def window_layers(self) -> List[int]:
        if self.sliding_pattern is None:
            return []
        return [i for i, w in enumerate(self.sliding_pattern) if w]

    @property
    def global_layers(self) -> List[int]:
        if self.sliding_pattern is None:
            return list(range(self.n_layer))
        return [i for i, w in enumerate(self.sliding_pattern) if not w]


def softcap_logits(logits: np.ndarray, cap: Optional[float]) -> np.ndarray:
    """cap * tanh(logits / cap). Bounds the logit range, which bounds the
    softmax's sensitivity; without it the tail of a large vocab dominates."""
    if cap is None:
        return logits
    if cap <= 0:
        raise ValueError(f"softcap must be positive, got {cap}")
    return cap * np.tanh(logits.astype(np.float32) / cap)


def block(reader, layer: int, cfg: GemmaConfig):
    """The gemma4 block order, as named functions so it can be checked.

    h1 = x + layer_output_scale * attn(attn_norm(x))
    h2 = h1 + layer_output_scale * ffn(ffn_norm(h1))
    x  = post_ffw_norm(h2)
    """
    p = f"blk.{layer}."
    return {
        "attn_norm": reader.tensor(p + "attn_norm.weight"),
        "ffn_norm": reader.tensor(p + "ffn_norm.weight"),
        "post_attention_norm": reader.tensor(p + "post_attention_norm.weight"),
        "post_ffw_norm": reader.tensor(p + "post_ffw_norm.weight"),
        "layer_output_scale": reader.tensor(p + "layer_output_scale.weight"),
    }


def scale_of(scale_vec: np.ndarray) -> float:
    """layer_output_scale is stored as a 1-element tensor; broadcasting a vector
    of the wrong length would silently change the block's output scale."""
    s = np.asarray(scale_vec, dtype=np.float32).reshape(-1)
    if s.size != 1:
        raise ValueError(f"layer_output_scale has {s.size} elements, expected 1")
    return float(s[0])
