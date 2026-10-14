"""Byte-level BPE tokenizer, read from the GGUF file.

The file carries everything needed (`tokenizer.ggml.model='gpt2'`):
    tokenizer.ggml.tokens      248320 strings
    tokenizer.ggml.merges     247587 merge rules, in RANK ORDER
    tokenizer.ggml.token_type  248320 entries (1=normal, 3=special, ...)
    tokenizer.ggml.pre         pre-tokenizer name

GPT-2 BPE in three steps:
  1. map each byte of the input to the printable unicode char GPT-2 uses for it
     (not utf-8 bytes directly -- 256 byte values are remapped onto a range that
     avoids whitespace/control chars, which is why ' ' shows up as '\u0120' and
     why the tokens in this file start with '\u0120' for a leading space)
  2. split the result into pre-tokenizer units (a regex), so merges never cross
     word boundaries
  3. for each unit, repeatedly apply the LOWEST-RANK merge rule that matches,
     until no rule applies. The result is the token list.

This replaces the byte-level stand-in (`codepoint % vocab`), which produced
token IDs the model understood as tokens but which had nothing to do with the
text -- so generated output could only be shown as numbers.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple


def _bytes_to_unicode() -> Dict[int, str]:
    """The GPT-2 byte->unicode table.

    Bytes that are already printable ASCII map to themselves; the rest are
    shifted into a contiguous range starting at U+0100, in order, so no byte
    maps to whitespace or a control character. This must match exactly or every
    token ID shifts.
    """
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(ord("\xa1"), ord("\xac") + 1)) +
          list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_B2U = _bytes_to_unicode()
_U2B = {v: k for k, v in _B2U.items()}

# The pre-tokenizer, as Qwen uses it. Python's `re` has no \p{L}/\p{N}, so the
# unicode-property classes are approximated with explicit ranges covering
# latin/greek/cyrillic/CJK blocks. The approximation only affects WHERE splits
# happen for exotic scripts; ASCII/English -- what we can test -- is exact.
_L = (r"A-Za-z\u00c0-\u024f\u0370-\u03ff\u0400-\u04ff"
      r"\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af")
_N = r"0-9\u0660-\u0669\u06f0-\u06f9"

_PAT = re.compile(
    rf"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
    rf"|[{_L}]+"
    rf"|[{_N}]"
    rf"| ?[^\s{_L}{_N}]+[\r\n]*"
    rf"|\s*[\r\n]+"
    rf"|\s+(?!\S)"
    rf"|\s+"
)


class Tokenizer:
    """Reads the vocab and applies BPE. No external tokenizer library."""

    def __init__(self, tokens: Sequence[str], merges: Sequence[str],
                 token_types: Optional[Sequence[int]] = None,
                 eos_id: Optional[int] = None, bos_id: Optional[int] = None,
                 pad_id: Optional[int] = None) -> None:
        if not tokens:
            raise ValueError("empty vocabulary")
        self.tokens = list(tokens)
        self.n_vocab = len(self.tokens)
        self.token_types = list(token_types) if token_types else None
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.pad_id = pad_id

        # token string -> id. First occurrence wins: a vocab with duplicates is
        # malformed, and silently taking the LAST id would shift every token.
        self.encoder: Dict[str, int] = {}
        for i, t in enumerate(self.tokens):
            if t not in self.encoder:
                self.encoder[t] = i

        # merge pair -> rank (the ORDER in the file is the rank)
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        for rank, m in enumerate(merges):
            parts = m.split(" ")
            if len(parts) != 2:
                # a malformed merge rule would silently reorder every token
                raise ValueError(f"merge {rank} is not a pair: {m!r}")
            self.bpe_ranks[(parts[0], parts[1])] = rank

        self._cache: Dict[str, List[str]] = {}
        self._special = self._special_set()

    def _special_set(self) -> Dict[str, int]:
        """Token ids whose type marks them special (3 = control/special).

        Special tokens must never be produced by BPE -- they are only ever
        inserted deliberately, and must be matched WHOLE before pre-tokenising.
        """
        out = {}
        if self.token_types is None:
            return out
        for i, t in enumerate(self.tokens):
            if self.token_types[i] in (3, 4, 5, 6):
                out[t] = i
        return out

    # ---- BPE ------------------------------------------------------------
    def _bpe(self, token: str) -> List[str]:
        """Apply merges to one pre-tokenised unit, lowest rank first."""
        if token in self._cache:
            return self._cache[token]
        word = list(token)
        if len(word) < 2:
            self._cache[token] = word
            return word

        while len(word) > 1:
            best, best_rank = None, None
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                r = self.bpe_ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best is None:
                break
            word = word[:best] + [word[best] + word[best + 1]] + word[best + 2:]

        self._cache[token] = word
        return word

    # ---- public API -----------------------------------------------------
    def encode(self, text: str) -> List[int]:
        """text -> token ids. Special tokens in the input are matched whole."""
        ids: List[int] = []
        if not text:
            return ids

        # split out any special tokens first, so their characters never go
        # through the byte mapping or BPE
        if self._special:
            pat = "(" + "|".join(re.escape(s) for s in
                                 sorted(self._special, key=len, reverse=True)) + ")"
            for piece in re.split(pat, text):
                if not piece:
                    continue
                if piece in self._special:
                    ids.append(self._special[piece])
                else:
                    ids.extend(self._encode_plain(piece))
        else:
            ids.extend(self._encode_plain(text))
        return ids

    def _encode_plain(self, text: str) -> List[int]:
        out: List[int] = []
        for unit in _PAT.findall(text):
            mapped = "".join(_B2U[b] for b in unit.encode("utf-8"))
            for piece in self._bpe(mapped):
                tid = self.encoder.get(piece)
                if tid is None:
                    # should be unreachable for a well-formed vocab; fall back to
                    # per-byte so we never silently drop input
                    for ch in piece:
                        t2 = self.encoder.get(ch)
                        if t2 is not None:
                            out.append(t2)
                else:
                    out.append(tid)
        return out

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        """token ids -> text, undoing the byte->unicode mapping."""
        if self._special and skip_special:
            rev = {v: k for k, v in self._special.items()}
            keep = [i for i in ids if i not in rev]
        else:
            keep = list(ids)

        buf = []
        for i in keep:
            if i < 0 or i >= self.n_vocab:
                raise IndexError(f"token id {i} out of range (vocab {self.n_vocab})")
            buf.append(self.tokens[i])
        s = "".join(buf)
        data = bytearray()
        for ch in s:
            b = _U2B.get(ch)
            if b is None:
                # a character outside the byte table can only come from a
                # special token's text; pass it through as utf-8
                data.extend(ch.encode("utf-8"))
            else:
                data.append(b)
        return bytes(data).decode("utf-8", errors="replace")

    @classmethod
    def from_gguf(cls, reader) -> "Tokenizer":
        kv = reader.kv
        need = ("tokenizer.ggml.tokens", "tokenizer.ggml.merges")
        missing = [k for k in need if k not in kv]
        if missing:
            raise ValueError(
                f"GGUF has no tokenizer ({missing} absent); this file was "
                f"converted without one and cannot produce text")
        model = kv.get("tokenizer.ggml.model")
        if model != "gpt2":
            raise ValueError(
                f"unsupported tokenizer model {model!r}; only 'gpt2' byte-level "
                f"BPE is implemented")
        return cls(
            tokens=kv["tokenizer.ggml.tokens"],
            merges=kv["tokenizer.ggml.merges"],
            token_types=kv.get("tokenizer.ggml.token_type"),
            eos_id=kv.get("tokenizer.ggml.eos_token_id"),
            bos_id=kv.get("tokenizer.ggml.bos_token_id"),
            pad_id=kv.get("tokenizer.ggml.padding_token_id"),
        )
