#!/usr/bin/env python3
"""Tokenizer self-test: byte-level BPE read from the GGUF file.

The property that matters is round-trip exactness on REAL text -- a tokenizer
that silently maps bytes differently produces ids the model reads as the wrong
tokens, which looks exactly like a broken model.

    .venv-cml/bin/python tests/test_tokenizer.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from engine import gguf as G                # noqa: E402
from engine.tokenizer import Tokenizer, _B2U, _U2B   # noqa: E402

CHECKS = FAILS = 0
QWEN = ("/Volumes/Backup/lmstudio/models/empero-ai/"
        "Qwen3.8-4B-Distill-GGUF/Qwen3.8-4B-Q4_K_M.gguf")


def check(cond, what):
    global CHECKS, FAILS
    CHECKS += 1
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS += 1


def banner(t):
    print(f"\n{t}")


def test_byte_table():
    banner("[1] the GPT-2 byte<->unicode table (must be exactly right)")
    check(len(_B2U) == 256, f"covers all 256 bytes ({len(_B2U)})")
    check(len(set(_B2U.values())) == 256, "the mapping is injective (no collisions)")
    check(_B2U[ord(" ")] == "\u0120", "space maps to U+0120 (why tokens show a dot)")
    check(_B2U[ord("A")] == "A", "printable ASCII maps to itself")
    check(_B2U[0] == "\u0100", "byte 0 maps to U+0100 (first shifted value)")
    check(_B2U[10] == "\u010a", "newline maps into the shifted range")
    # no byte may map to real whitespace -- that would corrupt decoding
    bad = [b for b, c in _B2U.items() if c.isspace()]
    check(not bad, f"no byte maps to whitespace ({bad})")
    check(all(_U2B[c] == b for b, c in _B2U.items()),
          "the reverse table is consistent with the forward one")


def test_roundtrip(tk):
    banner("[2] round-trip on real text -- the property that matters")
    cases = [
        "Hello",
        "Hello, world!",
        "The capital of France is",
        "The quick brown fox jumps over the lazy dog.",
        "  leading spaces",
        "trailing spaces  ",
        "12345",
        "café résumé naïve",
        "import numpy as np\nx = 1\n",
        "def f(x):\n    return x * 2\n",
        "tabs\tand\nnewlines",
        "symbols: !@#$%^&*()[]{}<>",
        "mixed 123 and Words, yes!",
        "",
    ]
    bad = []
    for text in cases:
        ids = tk.encode(text)
        back = tk.decode(ids)
        if back != text:
            bad.append((text, ids, back))
        check(back == text, f"round-trips {text!r}")
    check(not bad, f"every case round-trips ({len(bad)} failed)")
    # multi-line and longer prose
    para = ("The model reads text with a byte-level BPE vocabulary. "
            "It must reproduce every byte, including unicode: \u00e9, \u4e2d\u6587.")
    check(tk.decode(tk.encode(para)) == para, "round-trips a longer paragraph")


def test_ids(tk):
    banner("[3] the ids are the real ones, not plausible-looking")
    # "Hello" is ONE token in this vocab; the id has to match the file
    ids = tk.encode("Hello")
    check(len(ids) == 1, f"'Hello' is a single token ({ids})")
    check(tk.tokens[ids[0]] == "Hello",
          f"the id resolves back to 'Hello' in the vocab ({tk.tokens[ids[0]]!r})")
    # a known multi-token string
    ids2 = tk.encode("Hello, world!")
    check(len(ids2) == 5, f"'Hello, world!' is 5 tokens ({ids2})")
    got = [tk.tokens[i] for i in ids2]
    check(all(t is not None for t in got), "every id resolves in the vocab")
    print(f"       'Hello, world!' -> {ids2}")
    print(f"       pieces          -> {[_p for _p in got]}")
    # ids must be in range
    check(all(0 <= i < tk.n_vocab for i in ids2),
          f"all ids within the {tk.n_vocab} vocab")
    # decode must reject out-of-range rather than silently wrap
    try:
        tk.decode([tk.n_vocab + 1])
        check(False, "an out-of-range id must raise")
    except IndexError:
        check(True, "an out-of-range id raises instead of wrapping")


def test_specials(tk):
    banner("[4] special tokens")
    n = len(tk._special)
    check(n > 0, f"the file declares special tokens ({n})")
    im_start = tk._special.get("<|im_start|>")
    check(im_start is not None, f"<|im_start|> is special (id {im_start})")
    # a special token must be ONE id, not encoded character-by-character
    ids = tk.encode("<|im_start|>")
    check(ids == [im_start],
          f"a special token encodes to its own single id ({ids})")
    # decode() has TWO documented behaviours and both must hold: the default
    # drops specials (so chat framing never leaks into user-visible text), and
    # skip_special=False returns the literal token.
    check(tk.decode([im_start]) == "",
          f"the default drops a special id ({tk.decode([im_start])!r})")
    check(tk.decode([im_start], skip_special=False) == "<|im_start|>",
          f"skip_special=False returns the literal "
          f"({tk.decode([im_start], skip_special=False)!r})")
    check(tk.decode(tk.encode("hi") + [im_start]) == "hi",
          "a special id mixed into real text is filtered out too")
    # decode can skip specials
    out = tk.decode(ids + tk.encode("hi"), skip_special=True)
    check("<|im_start|>" not in out,
          f"skip_special drops them from the text ({out!r})")
    # the chat template must be built from the file's tokens
    check(tk.eos_id is not None, f"eos id present ({tk.eos_id})")


def test_negatives(tk):
    banner("[5] negatives: it must refuse what it cannot do")
    # a vocab without merges is not a BPE tokenizer
    try:
        Tokenizer(tokens=["a", "b"], merges=None)
        check(False, "merges=None must raise")
    except (TypeError, ValueError):
        check(True, "merges=None raises")
    # a malformed merge rule must raise, not silently reorder every token
    try:
        Tokenizer(tokens=["a", "b", "ab"], merges=["a b", "broken"])
        check(False, "a malformed merge rule must raise")
    except ValueError:
        check(True, "a malformed merge rule raises instead of being skipped")
    # an empty vocab is unusable
    try:
        Tokenizer(tokens=[], merges=[])
        check(False, "an empty vocab must raise")
    except ValueError:
        check(True, "an empty vocab raises")
    # a non-gpt2 tokenizer model must be refused (only gpt2 is implemented)
    class _FakeKV(dict):
        arch = staticmethod(lambda: "qwen35")
    fake = _FakeKV({
        "tokenizer.ggml.tokens": ["a"],
        "tokenizer.ggml.merges": ["a a"],
        "tokenizer.ggml.model": "llama",
    })
    class _FakeR:
        kv = fake
    try:
        Tokenizer.from_gguf(_FakeR())
        check(False, "a non-gpt2 tokenizer model must be refused")
    except ValueError as e:
        check("gpt2" in str(e), "a non-gpt2 model raises with an explanation")
    # and a file with NO tokenizer must say so, not return garbage
    class _NoTok:
        kv = {"general.architecture": "qwen35"}
    try:
        Tokenizer.from_gguf(_NoTok())
        check(False, "a GGUF with no tokenizer must be refused")
    except ValueError as e:
        check("tokenizer" in str(e).lower(),
              "a missing tokenizer raises with an explanation")


def test_no_stand_in():
    banner("[6] the runtime no longer uses the byte stand-in")
    from engine.runtime import GgufRuntime
    import inspect
    src = inspect.getsource(GgufRuntime.tokenize)
    check("tokenizer.encode" in src,
          "GgufRuntime.tokenize calls the real tokenizer")
    check("% v" not in src and "%V" not in src,
          "...and no longer maps codepoints modulo the vocab")
    src2 = inspect.getsource(GgufRuntime.detokenize)
    check("tokenizer.decode" in src2, "detokenize decodes with the real vocab")


def main():
    print("=== tokenizer self-test ===")
    test_byte_table()
    if not os.path.exists(QWEN):
        print("\n(real model not present; id/special checks skipped)")
    else:
        with G.GGUFReader(QWEN) as r:
            kv = r.kv
            check(kv.get("tokenizer.ggml.model") == "gpt2",
                  f"the file declares a gpt2 tokenizer "
                  f"({kv.get('tokenizer.ggml.model')!r})")
            tk = Tokenizer.from_gguf(r)
            print(f"       vocab {tk.n_vocab}  merges {len(tk.bpe_ranks)}  "
                  f"specials {len(tk._special)}")
        test_roundtrip(tk)
        test_ids(tk)
        test_specials(tk)
    test_negatives(tk if os.path.exists(QWEN) else None)
    test_no_stand_in()
    print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
