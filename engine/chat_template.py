"""Chat-envelope rendering, transcribed from each model family's own GGUF template.

Why this is a module rather than three methods on the runtime: the interactive
runtime and the check tools MUST produce byte-identical input. If a tool renders
its own prompt it "verifies" something the engine never actually sends -- which is
the exact failure that hid the BOS bug (the engine dropped BOS, the tool didn't
test through the engine, and the model's garbage looked like an architecture bug).

Every envelope below is transcribed from the `tokenizer.chat_template` stored in the
GGUF, and every envelope is DETECTED from the file's own special tokens. An
unrecognised template raises rather than silently emitting unframed text.

The returned string already contains the leading BOS where the template emits one,
so it must be tokenized with add_bos=False. `Tokenizer.encode` additionally refuses
to double it.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

__all__ = ["EnvelopeError", "detect_envelope", "render_chat"]


class EnvelopeError(NotImplementedError):
    """The file's chat template is not one this build can render."""


def _gemma4(messages: List[Dict[str, str]], add_generation_prompt: bool,
            enable_thinking: bool) -> str:
    """gemma-4's no-tools path, from the file's Jinja template:

        {{- bos_token -}}
        {%- for m in messages -%}
            {%- set role = 'model' if m.role == 'assistant' else m.role -%}
            {{- '<|turn>' + role + '\\n' }}{{ content }}{{- '<turn|>\\n' }}
        {%- endfor -%}
        {%- if add_generation_prompt -%}
            {{- '<|turn>model\\n' -}}
            {%- if not enable_thinking -%}
                {{- '<|channel>thought\\n<channel|>' -}}
            {%- endif -%}
        {%- endif -%}

    Two details that are easy to get backwards:
      * the assistant role is spelled 'model', not 'assistant';
      * the thinking convention is INVERTED from intuition. Thinking DISABLED emits
        an immediately-closed thought channel, and that is what forces a direct
        answer. Thinking ENABLED emits nothing, leaving the model inside the
        channel so it thinks out loud.
    """
    out = ["<bos>"]                      # the template emits bos_token itself
    for m in messages:
        role = m.get("role", "user")
        role = "model" if role == "assistant" else role
        out.append(f"<|turn>{role}\n{m.get('content', '')}<turn|>\n")
    if add_generation_prompt:
        out.append("<|turn>model\n")
        if not enable_thinking:
            out.append("<|channel>thought\n<channel|>")
    return "".join(out)


def _chatml(messages: List[Dict[str, str]], add_generation_prompt: bool,
            enable_thinking: bool) -> str:
    """ChatML (qwen). After '<|im_start|>assistant\\n' this family expects a THINKING
    BLOCK -- ' thinking\\n' with thinking on, or ' thinking\\n\\n</think>\\n\\n' with it
    off. Omitting it makes the model emit newline-only continuations, which looks
    like a broken engine."""
    out = []
    for m in messages:
        role = m.get("role", "user")
        out.append(f"<|im_start|>{role}\n{m.get('content', '')}<|im_end|>\n")
    if add_generation_prompt:
        out.append("<|im_start|>assistant\n")
        out.append(" thinking\n" if enable_thinking
                   else " thinking\n\n</think>\n\n")
    return "".join(out)


# (envelope name, the special token whose presence identifies it, renderer)
ENVELOPES = (
    ("gemma4", "<|turn>", _gemma4),
    ("chatml", "<|im_start|>", _chatml),
)


def detect_envelope(tokenizer) -> Optional[str]:
    """Name of the envelope this file's tokenizer calls for, or None."""
    special = getattr(tokenizer, "_special", None) or {}
    for name, marker, _ in ENVELOPES:
        if marker in special:
            return name
    return None


def render_chat(messages: List[Dict[str, str]], tokenizer,
                add_generation_prompt: bool = True,
                enable_thinking: Optional[bool] = None) -> str:
    """Render `messages` the way this file's OWN chat template does.

    Raises EnvelopeError for a template that is not recognised, so a caller can
    never accidentally send unframed text to the model.
    """
    if tokenizer is None:
        raise EnvelopeError("this GGUF carries no tokenizer; cannot frame chat")
    if enable_thinking is None:
        enable_thinking = os.environ.get("AERO_THINKING", "0") == "1"
    name = detect_envelope(tokenizer)
    for env_name, _, renderer in ENVELOPES:
        if env_name == name:
            return renderer(messages, add_generation_prompt, enable_thinking)
    raise EnvelopeError(
        "unrecognised chat template: this file has neither '<|turn>' (gemma4) nor "
        "'<|im_start|>' (chatml). Refusing to emit unframed text.")
