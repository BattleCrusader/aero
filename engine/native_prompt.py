#!/usr/bin/env python3
"""The PROMPT PATH, native: C++ tokenizer + C++ chat renderer over ctypes.

WHY THIS FILE IS PYTHON AND STILL OK. The engine is C++ -- engine/tokenizer.py and
engine/chat_template.py are the test ORACLES now, not the implementation. What remains
here is the boundary: ctypes declarations, argument marshalling, and the buffer dance
the C ABI requires (ask for a size, allocate, render into it). No tokenizing, no BPE, no
template logic and no arch decisions happen in this file; if that ever changes, it is a
port regression, not a design choice.

WHY IT MATTERS THAT THIS IS NATIVE. Prompt construction is the one place where a wrong
answer is invisible: the model still answers something, just from a prompt it never saw
in training. This project already lost a day to exactly that -- the Python pre-tokeniser
split the leading space off every word, so every prompt was subtly wrong and every round
-trip test still passed. The native tokenizer is the one verified against llama-tokenize
(tests/test_tokenizer_cpp.py), and the native renderer against engine.chat_template
byte-for-byte (tests/test_chat_template_cpp.py), so both are pinned by an ORACLE rather
than by our own belief about our own code.

    AERO_NATIVE_PROMPT=0   fall back to the Python tokenizer/renderer (for the tests)
"""
from __future__ import annotations

import ctypes
import os
from ctypes import POINTER, c_char_p, c_int, c_int32, c_int64, c_void_p
from typing import Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_TOKLIB = os.path.join(_ROOT, "build", "libaero_tokenizer.dylib")
_CHATLIB = os.path.join(_ROOT, "build", "libaero_chat.dylib")


class PromptPathUnavailable(RuntimeError):
    """The native prompt path cannot be used (library missing or the file refused)."""


class EnvelopeError(ValueError):
    """The chat template is not a recognised envelope -- do not send unframed text."""


_tok = None
_chat = None


def available() -> bool:
    return os.path.exists(_TOKLIB) and os.path.exists(_CHATLIB)


def unavailable_reason() -> str:
    missing = [p for p in (_TOKLIB, _CHATLIB) if not os.path.exists(p)]
    return ("missing " + ", ".join(os.path.basename(p) for p in missing)) if missing else ""


def _load():
    global _tok, _chat
    if _tok is not None and _chat is not None:
        return _tok, _chat
    if not available():
        raise PromptPathUnavailable(unavailable_reason())
    _tok = ctypes.CDLL(_TOKLIB)
    _chat = ctypes.CDLL(_CHATLIB)

    _tok.aero_tok_open.argtypes = [c_char_p]
    _tok.aero_tok_open.restype = c_void_p
    _tok.aero_tok_free.argtypes = [c_void_p]
    _tok.aero_tok_last_error.restype = c_char_p
    for fn in ("aero_tok_n_vocab", "aero_tok_n_merges", "aero_tok_n_special",
               "aero_tok_bos_id", "aero_tok_eos_id", "aero_tok_pad_id"):
        getattr(_tok, fn).argtypes = [c_void_p]
        getattr(_tok, fn).restype = c_int64
    for fn in ("aero_tok_add_bos", "aero_tok_family", "aero_tok_pre"):
        getattr(_tok, fn).argtypes = [c_void_p]
        getattr(_tok, fn).restype = c_int
    _tok.aero_tok_is_special_id.argtypes = [c_void_p, c_int32]
    _tok.aero_tok_is_special_id.restype = c_int
    _tok.aero_tok_token.argtypes = [c_void_p, c_int32, c_char_p, c_int64]
    _tok.aero_tok_token.restype = c_int64
    _tok.aero_tok_encode.argtypes = [c_void_p, c_char_p, c_int64, POINTER(c_int32), c_int64, c_int]
    _tok.aero_tok_encode.restype = c_int64
    _tok.aero_tok_decode.argtypes = [c_void_p, POINTER(c_int32), c_int64, c_char_p, c_int64, c_int]
    _tok.aero_tok_decode.restype = c_int64

    _chat.aero_chat_detect.argtypes = [POINTER(c_char_p), c_int]
    _chat.aero_chat_detect.restype = c_int
    _chat.aero_chat_render_size.argtypes = [POINTER(c_char_p), POINTER(c_char_p), c_int,
                                            POINTER(c_char_p), c_int, c_int, c_int]
    _chat.aero_chat_render_size.restype = c_int
    _chat.aero_chat_render_into.argtypes = [POINTER(c_char_p), POINTER(c_char_p), c_int,
                                           POINTER(c_char_p), c_int, c_int, c_int,
                                           c_void_p, c_int]
    _chat.aero_chat_render_into.restype = c_int
    _chat.aero_chat_thinking_from_env.argtypes = []
    _chat.aero_chat_thinking_from_env.restype = c_int
    _chat.aero_chat_envelope_name.argtypes = [c_int]
    _chat.aero_chat_envelope_name.restype = c_char_p
    return _tok, _chat


class NativeTokenizer:
    """The C++ tokenizer, with the small surface the engine actually uses.

    Deliberately NOT a subclass of engine.tokenizer.Tokenizer: the Python one stays
    importable as the oracle, and nothing should be able to silently be both.
    """

    def __init__(self, path: str) -> None:
        lk, _ = _load()
        self._h = lk.aero_tok_open(os.fsencode(path))
        if not self._h:
            err = lk.aero_tok_last_error()
            raise PromptPathUnavailable(
                "native tokenizer refused %s: %s" % (path, err.decode() if err else "unknown"))
        self._lk = lk
        self._specials: Optional[List[str]] = None

    # ---- lifecycle ------------------------------------------------------
    def close(self) -> None:
        if getattr(self, "_h", None):
            self._lk.aero_tok_free(self._h)
            self._h = None

    def __del__(self):                      # pragma: no cover - best effort
        try:
            self.close()
        except Exception:
            pass

    # ---- metadata -------------------------------------------------------
    @property
    def n_vocab(self) -> int:
        return int(self._lk.aero_tok_n_vocab(self._h))

    @property
    def n_merges(self) -> int:
        return int(self._lk.aero_tok_n_merges(self._h))

    @property
    def bos_id(self) -> Optional[int]:
        v = int(self._lk.aero_tok_bos_id(self._h))
        return None if v < 0 else v

    @property
    def eos_id(self) -> Optional[int]:
        v = int(self._lk.aero_tok_eos_id(self._h))
        return None if v < 0 else v

    @property
    def pad_id(self) -> Optional[int]:
        v = int(self._lk.aero_tok_pad_id(self._h))
        return None if v < 0 else v

    @property
    def add_bos(self) -> bool:
        return bool(self._lk.aero_tok_add_bos(self._h))

    @property
    def family(self) -> str:
        return "gpt2" if int(self._lk.aero_tok_family(self._h)) == 0 else "spm"

    @property
    def n_special(self) -> int:
        return int(self._lk.aero_tok_n_special(self._h))

    @property
    def vocab_type(self) -> str:
        """Same name the Python Tokenizer uses, so callers need not care which is in use."""
        return self.family

    def specials(self) -> List[str]:
        """The file's control tokens, for envelope detection. Scanned once and cached.

        A scan of the whole vocab through ctypes is ~a third of a second on a 248k-token
        file; paying it once at first use is cheaper than maintaining a second source of
        truth about which strings are special.
        """
        if self._specials is None:
            out: List[str] = []
            n = self.n_vocab
            for i in range(n):
                if self._lk.aero_tok_is_special_id(self._h, i):
                    out.append(self.token_str(i))
            self._specials = out
        return self._specials

    # ---- the two operations the engine needs ----------------------------
    def token_str(self, i: int) -> str:
        need = int(self._lk.aero_tok_token(self._h, int(i), None, 0))
        if need <= 0:
            return ""
        buf = ctypes.create_string_buffer(need + 1)
        self._lk.aero_tok_token(self._h, int(i), buf, need + 1)
        return buf.value.decode("utf-8", "replace")

    def encode(self, text: str, add_bos: Optional[bool] = None) -> List[int]:
        b = text.encode("utf-8")
        want = int(self._lk.aero_tok_encode(self._h, b, len(b), None, 0,
                                            -1 if add_bos is None else int(add_bos)))
        if want < 0:
            raise RuntimeError("native tokenizer encode failed (rc=%d)" % want)
        if want == 0:
            return []
        buf = (c_int32 * want)()
        got = int(self._lk.aero_tok_encode(self._h, b, len(b), buf, want,
                                           -1 if add_bos is None else int(add_bos)))
        if got < 0:
            raise RuntimeError("native tokenizer encode failed on the sized pass")
        return [int(x) for x in buf[:got]]

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        arr = (c_int32 * len(ids))(*[int(i) for i in ids])
        want = int(self._lk.aero_tok_decode(self._h, arr, len(ids), None, 0,
                                           1 if skip_special else 0))
        if want <= 0:
            return ""
        buf = ctypes.create_string_buffer(want + 1)
        self._lk.aero_tok_decode(self._h, arr, len(ids), buf, want + 1,
                                 1 if skip_special else 0)
        return buf.value.decode("utf-8", "replace")


def render_chat_native(messages: List[Dict[str, str]], tokenizer: NativeTokenizer,
                       add_generation_prompt: bool = True,
                       enable_thinking: Optional[bool] = None) -> str:
    """Render `messages` with the C++ renderer. Raises EnvelopeError, never guesses."""
    _, lc = _load()
    roles = [str(m.get("role", "")) for m in messages]
    contents = [str(m.get("content", "")) for m in messages]

    def carr(items):
        if not items:
            return None, 0
        arr = (c_char_p * len(items))(*[s.encode("utf-8") for s in items])
        return arr, len(items)

    ra, rn = carr(roles)
    ca, cn = carr(contents)
    sa, sn = carr(tokenizer.specials())
    think = lc.aero_chat_thinking_from_env() if enable_thinking is None else int(enable_thinking)

    env = lc.aero_chat_detect(sa, sn)
    if env < 0:
        raise EnvelopeError(
            "no recognised chat envelope in this file's template (detect -> %d)" % env)

    want = int(lc.aero_chat_render_size(ra, ca, cn, sa, sn,
                                        1 if add_generation_prompt else 0, think))
    if want < 0:
        raise EnvelopeError("renderer refused the conversation (rc=%d)" % want)
    buf = ctypes.create_string_buffer(want + 1)
    got = int(lc.aero_chat_render_into(ra, ca, cn, sa, sn,
                                       1 if add_generation_prompt else 0, think,
                                       buf, want + 1))
    if got < 0:
        raise EnvelopeError("renderer refused the conversation on the sized pass")
    return buf.value.decode("utf-8", "replace")
