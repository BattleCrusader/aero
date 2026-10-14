"""Architecture registry.

Every architecture-specific difference lives here and nowhere else. The point is
that `Model` cannot silently build a model it does not actually understand: an
unknown arch raises, and each known arch declares exactly what it needs.

Why this matters: before this file existed, `ModelConfig.from_gguf` read
`r.arch()` and ignored it. Pointing it at a `qwen35moe` file would have built a
*dense* model with no error -- silently wrong output, which is the worst possible
failure mode for this project.

Adding an architecture = adding one ArchSpec below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ArchSpec:
    name: str
    # layer kind detection, in priority order; first tensor found wins
    layer_probe: tuple                  # ((tensor_suffix, kind), ...)
    has_ssm: bool = False
    has_moe: bool = False
    has_per_layer_embd: bool = False
    has_logit_softcap: bool = False
    has_sliding_window: bool = False
    note: str = ""


QWEN_COMMON = (
    ("ssm_a", "ssm"),
    ("attn_q.weight", "attn"),
)


def _gemma_probe() -> tuple:
    return (
        ("attn_q.weight", "attn"),
    )


ARCHES: Dict[str, ArchSpec] = {
    "qwen35": ArchSpec(
        name="qwen35",
        layer_probe=QWEN_COMMON,
        has_ssm=True,
        note="hybrid gated-delta-net + full attention; tied output head",
    ),
    "qwen35moe": ArchSpec(
        name="qwen35moe",
        layer_probe=QWEN_COMMON,
        has_ssm=True,
        has_moe=True,
        note="qwen35 plus MoE FFN (routed experts + shared expert)",
    ),
    "qwen3moe": ArchSpec(
        name="qwen3moe",
        layer_probe=(("attn_q.weight", "attn"),),
        has_moe=True,
        note="classic qwen3 MoE, dense attention throughout",
    ),
    "qwen3": ArchSpec(
        name="qwen3",
        layer_probe=(("attn_q.weight", "attn"),),
        note="classic dense qwen3",
    ),
    "gemma4": ArchSpec(
        name="gemma4",
        layer_probe=_gemma_probe(),
        has_per_layer_embd=True,
        has_logit_softcap=True,
        has_sliding_window=True,
        note="per-layer embeddings, sliding-window attention, logit soft-capping",
    ),
    "gemma3": ArchSpec(
        name="gemma3",
        layer_probe=_gemma_probe(),
        has_logit_softcap=True,
        has_sliding_window=True,
        note="sliding-window attention + logit soft-capping (no per-layer embd)",
    ),
    "llama": ArchSpec(
        name="llama",
        layer_probe=(("attn_q.weight", "attn"),),
        note="dense llama-family baseline",
    ),
    "qwen2": ArchSpec(
        name="qwen2",
        layer_probe=(("attn_q.weight", "attn"),),
    ),
}


class UnsupportedArch(ValueError):
    """Raised instead of silently building a model we do not understand."""


def spec_for(arch: str) -> ArchSpec:
    s = ARCHES.get(arch)
    if s is None:
        known = ", ".join(sorted(ARCHES))
        raise UnsupportedArch(
            f"architecture {arch!r} is not supported by this engine. "
            f"Known: {known}. Refusing to build a model from an architecture "
            f"whose forward pass has not been worked out -- a wrong-but-running "
            f"model is worse than an error."
        )
    return s


def detect_layer_kind(tensors, layer: int, spec: ArchSpec) -> str:
    """First probe that matches wins. Raises if the layer is unrecognisable."""
    for suffix, kind in spec.layer_probe:
        if f"blk.{layer}.{suffix}" in tensors:
            return kind
    raise UnsupportedArch(
        f"blk.{layer}: cannot determine layer kind for arch {spec.name!r}; "
        f"none of {[s for s, _ in spec.layer_probe]} are present"
    )


def capabilities(arch: str) -> str:
    """Human-readable capability line, for CLI/status output."""
    s = spec_for(arch)
    feats = [f for f, on in (("SSM", s.has_ssm), ("MoE", s.has_moe),
                             ("per-layer-embd", s.has_per_layer_embd),
                             ("softcap", s.has_logit_softcap),
                             ("sliding-window", s.has_sliding_window)) if on]
    return f"{s.name}: " + (", ".join(feats) if feats else "dense attention")
