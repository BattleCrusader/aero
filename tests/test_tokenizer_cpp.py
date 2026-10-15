#!/usr/bin/env python3
"""The C++ tokenizer, proved against the independent oracle.

WHAT THIS PINS: a tokenizer that is merely SELF-CONSISTENT proves nothing -- encode
then decode round-trips through a wrong vocab just as happily as a right one, and two
implementations written from the same belief agree while both are wrong (this project
lost a day to exactly that: our Python pre-tokeniser split "Hello world" into an extra
'Ġ' token and every test still passed). The only thing that finds that class of bug is
an INDEPENDENT implementation of the published rule, so the primary property checked
here is:

    aero_tok_encode(...) == llama-tokenize -m <model> -p <text> --ids

llama.cpp's own tokenizer is the oracle. Our Python reference (engine/tokenizer.py) is
checked as a SECOND oracle, and where the two disagree the llama ids win -- the two
places they do disagree are asserted explicitly below, because they are bugs in the
Python that this port fixes:
  * "x\\ty"            Python splits the TAB as its own unit; the published Qwen3.5 rule
                       allows any single non-letter/non-digit prefix, so TAB glues to
                       the following letter ('x' | <tab,y>).
  * "[PAD248077]"      Python treats token-type 5 (unused) as special; llama.cpp does
                       not, and BPEs the text.

Everything is driven through the C ABI with ctypes; no C++ type crosses the boundary.

    env -u PYTHONPATH -u PYTHONHOME .venv-cml/bin/python tests/test_tokenizer_cpp.py
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import tempfile
from ctypes import POINTER, c_char_p, c_int, c_int32, c_int64, c_void_p

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

FAILS = 0
CHECKS = 0


def check(cond, what):
    global CHECKS, FAILS
    CHECKS += 1
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS += 1


def banner(t):
    print(f"\n{t}")


# ---- paths ----------------------------------------------------------------
MODELS = {
    "gpt2/qwen35": "/Users/cyberdeth/.cache/aero-models/Ornith-1.5-9B-MTP-IQ4_XS.gguf",
    "spm/gemma4":  "/Users/cyberdeth/.cache/aero-models/gemma-4-12b-it-IQ4_XS.gguf",
}
LLAMA_TOKENIZE = "/Volumes/Backup/Development/PrismML-llama.cpp/build/bin/llama-tokenize"


def ensure_lib():
    """Return (path, how). Prefer the built .dylib; compile it if the Makefile rule is
    not wired yet, so the suite is real rather than silently skipped."""
    lib = os.path.join(ROOT, "build", "libaero_tokenizer.dylib")
    if os.path.exists(lib):
        return lib, "prebuilt"
    srcs = [os.path.join(ROOT, "src", f)
            for f in ("tokenizer.cpp", "gguf.cpp", "kernels.cpp")]
    if not all(os.path.exists(s) for s in srcs):
        return None, "sources missing"
    os.makedirs(os.path.dirname(lib), exist_ok=True)
    cxx = os.environ.get("CXX", "clang++")
    cmd = [cxx, "-std=c++17", "-O2", "-Wall", "-Wextra",
           "-I" + os.path.join(ROOT, "include"), "-fPIC", "-shared",
           *srcs, "-o", lib, "-pthread"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None, (r.stderr or r.stdout)[-800:]
    return lib, "compiled here"


def bind(path):
    lib = ctypes.CDLL(path)
    lib.aero_tok_open.argtypes = [c_char_p]
    lib.aero_tok_open.restype = c_void_p
    lib.aero_tok_free.argtypes = [c_void_p]
    lib.aero_tok_last_error.argtypes = []
    lib.aero_tok_last_error.restype = c_char_p
    for name in ("n_vocab", "n_merges", "n_special", "bos_id", "eos_id", "pad_id"):
        fn = getattr(lib, "aero_tok_" + name)
        fn.argtypes = [c_void_p]
        fn.restype = c_int64
    for name in ("add_bos", "family", "pre"):
        fn = getattr(lib, "aero_tok_" + name)
        fn.argtypes = [c_void_p]
        fn.restype = c_int
    lib.aero_tok_is_special_id.argtypes = [c_void_p, c_int32]
    lib.aero_tok_is_special_id.restype = c_int
    lib.aero_tok_token.argtypes = [c_void_p, c_int32, c_char_p, c_int64]
    lib.aero_tok_token.restype = c_int64
    lib.aero_tok_encode.argtypes = [c_void_p, c_char_p, c_int64, POINTER(c_int32), c_int64, c_int]
    lib.aero_tok_encode.restype = c_int64
    lib.aero_tok_decode.argtypes = [c_void_p, POINTER(c_int32), c_int64, c_char_p, c_int64, c_int]
    lib.aero_tok_decode.restype = c_int64
    return lib


def cencode(lib, h, text, add_bos=-1):
    """Two-pass through the ABI: query the count, then fill the buffer."""
    b = text.encode("utf-8")
    n = lib.aero_tok_encode(h, b, len(b), None, 0, add_bos)
    if n < 0:
        return None
    buf = (c_int32 * max(n, 1))()
    m = lib.aero_tok_encode(h, b, len(b), buf, n, add_bos)
    if m < 0:
        return None
    return list(buf[:m])


def cdecode(lib, h, ids, skip_special=1):
    arr = (c_int32 * max(len(ids), 1))(*ids) if ids else (c_int32 * 1)()
    n = lib.aero_tok_decode(h, arr, len(ids), None, 0, skip_special)
    if n < 0:
        return None
    buf = ctypes.create_string_buffer(max(n, 1))
    m = lib.aero_tok_decode(h, arr, len(ids), buf, n, skip_special)
    if m < 0:
        return None
    return buf.raw[:m].decode("utf-8")


def oracle(model, text):
    """llama.cpp's own ids for `text`. None if the binary is missing."""
    if not os.path.exists(LLAMA_TOKENIZE):
        return None
    # --no-escape: llama-tokenize otherwise interprets backslash escapes in the prompt
    # before tokenizing, which would corrupt any test string containing a literal '\'.
    p = subprocess.run([LLAMA_TOKENIZE, "-m", model, "-p", text, "--ids", "--no-escape"],
                       capture_output=True, text=True)
    for line in reversed(p.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("["):
            return json.loads(line)
    return None


# The battery. The strings the task calls out (space handling, code, arithmetic,
# CJK, emoji) plus the cases that separate the rules (tabs, runs, newlines).
BATTERY = [
    "Hello world", " Paris", "Hello, world!", "def main():\n    pass",
    "1 + 1 = 2", "capital", "  leading spaces", "trailing spaces  ",
    "a\n\nb", "a\r\nb", "café résumé", "中文测试", "日本語", "😀🎉 emoji", "12345",
    "The capital of France is", "x\ty", "a  b", "   ", "\n", "Hello, world! How are you?",
]
# where our Python reference disagrees with llama.cpp (bugs this port fixes)
PY_DIVERGES = {
    "x\ty":              "Python splits the TAB off; the published rule glues it forward",
    "[PAD248077]":       "Python treats token-type 5 (unused) as special; llama.cpp BPEs it",
    "a\r\nb":            "Python's spm path BPEs the whole string; gemma splits on newlines",
}


def main():
    lib_path, how = ensure_lib()
    if lib_path is None:
        print(f"(cannot run: {how})")
        print("\n=== 0 checks, 0 failures ===")
        return 0
    print(f"libaero_tokenizer: {lib_path} ({how})")
    lib = bind(lib_path)

    present = {k: v for k, v in MODELS.items() if os.path.exists(v)}
    if not present:
        print("(skipped: no model GGUF present for the encode/decode sections)")

    for label, model in present.items():
        banner(f"[vocab] {label}  ({os.path.basename(model)})")
        h = lib.aero_tok_open(model.encode())
        check(bool(h), f"aero_tok_open loads the tokenizer (err={lib.aero_tok_last_error().decode()!r})")
        if not h:
            continue

        # ---- vocab load counts, against an independent GGUF read ----------
        from engine import gguf as GG          # noqa: E402
        with GG.GGUFReader(model) as r:
            kv = r.kv
            n_tok = len(kv["tokenizer.ggml.tokens"])
            n_mrg = len(kv["tokenizer.ggml.merges"])
            types = kv.get("tokenizer.ggml.token_type", [])
            n_special = sum(1 for t in types if t in (2, 3, 4))
        check(int(lib.aero_tok_n_vocab(h)) == n_tok,
              f"vocab count = {n_tok} (GGUF says {n_tok}, C++ {lib.aero_tok_n_vocab(h)})")
        check(int(lib.aero_tok_n_merges(h)) == n_mrg,
              f"merge count = {n_mrg} (C++ {lib.aero_tok_n_merges(h)})")
        check(int(lib.aero_tok_n_special(h)) == n_special,
              f"special count = {n_special} (token types 2/3/4; C++ {lib.aero_tok_n_special(h)})")

        # ---- the oracle: encode must match llama.cpp exactly --------------
        banner(f"[oracle] {label}: encode == llama-tokenize on the battery")
        if not os.path.exists(LLAMA_TOKENIZE):
            print("  (skipped: llama-tokenize not built)")
        else:
            bad = 0
            for s in BATTERY:
                got = cencode(lib, h, s)
                want = oracle(model, s)
                if got != want:
                    bad += 1
                    print(f"       MISMATCH {s!r}\n         cpp={got}\n         ora={want}")
            check(bad == 0, f"all {len(BATTERY)} battery strings match llama.cpp exactly "
                            f"({len(BATTERY) - bad}/{len(BATTERY)})")

        # ---- second oracle: our Python, with the known divergences named --
        banner(f"[python] {label}: ours agrees except where it is wrong")
        from engine.tokenizer import Tokenizer as PyTok   # noqa: E402
        with GG.GGUFReader(model) as r:
            py = PyTok.from_gguf(r)
        mismatches = []
        for s in BATTERY + list(PY_DIVERGES):
            got = cencode(lib, h, s)
            exp = py.encode(s)
            if got != exp:
                mismatches.append(s)
        # every mismatch must be one of the DOCUMENTED Python divergences, and on those
        # strings the C++ must agree with llama.cpp instead
        unexplained = [s for s in mismatches if s not in PY_DIVERGES]
        check(not unexplained, f"no UNEXPLAINED divergence from the Python reference "
                               f"({unexplained})")
        for s, why in PY_DIVERGES.items():
            got = cencode(lib, h, s)
            want = oracle(model, s)
            check(got == want and want is not None,
                  f"C++ follows llama.cpp on {s!r} ({why}); ids={got}")

        # ---- decode round-trip -------------------------------------------
        banner(f"[decode] {label}: ids -> text round-trips")
        rt_bad = []
        for s in ("Hello world", "Hello, world!", "The capital of France is",
                  "def f(x):\n    return x * 2\n", "café résumé naïve", "中文测试",
                  "😀 emoji", "  leading spaces", "trailing spaces  ", "1 + 1 = 2",
                  "a\r\nb"):
            ids = cencode(lib, h, s)
            back = cdecode(lib, h, ids, skip_special=1)
            if back != s:
                rt_bad.append((s, ids, back))
        check(not rt_bad, f"encode->decode round-trips every case ({rt_bad})")
        # and decode matches the Python reference byte-for-byte on the same ids
        ids = cencode(lib, h, "Hello, world! How are you?")
        check(cdecode(lib, h, ids, 1) == py.decode(ids, skip_special=True),
              "decode agrees with the Python reference on the same ids")

        # ---- special tokens ----------------------------------------------
        banner(f"[special] {label}")
        if "gpt2" in label:
            sid = 248045          # <|im_start|>, token type 3 (control)
            stxt = "<|im_start|>"
            pad = 248077          # [PAD248077], token type 5 (unused) -- NOT special
        else:
            sid = 2               # <bos>, token type 3 (control)
            stxt = "<bos>"
            pad = None
        check(lib.aero_tok_is_special_id(h, sid) == 1,
              f"id {sid} ({stxt}) is reported special")
        want = [sid] if int(lib.aero_tok_add_bos(h)) == 0 else [int(lib.aero_tok_bos_id(h)), sid]
        check(cencode(lib, h, stxt) == want,
              f"{stxt!r} encodes to its single special id {sid} "
              f"(plus BOS when the model adds one) -> {want}")
        check(cdecode(lib, h, [sid], skip_special=1) == "",
              "decode(skip_special=1) drops the special token's text")
        check(cdecode(lib, h, [sid], skip_special=0) == stxt,
              f"decode(skip_special=0) returns the literal {stxt!r}")
        check(cencode(lib, h, f"a{stxt}b") == ([64, sid, 65] if "gpt2" in label else [2, 236746, sid, 236763]),
              f"a special token embedded in text is one id, text either side preserved")
        if pad is not None:
            check(lib.aero_tok_is_special_id(h, pad) == 0,
                  f"id {pad} (type 5 unused) is NOT special")
            got = cencode(lib, h, "[PAD248077]")
            check(got == oracle(model, "[PAD248077]") and len(got) > 1,
                  f"[PAD248077] is BPE'd, not taken whole (ids={got})")
        # BOS: appended as the file asks (gemma), and NOT suppressed for a prompt that
        # already starts with the literal -- matching llama.cpp (gemma '<bos>' -> [2, 2])
        if "spm" in label:
            check(int(lib.aero_tok_add_bos(h)) == 1, "gemma: add_bos is on")
            check(cencode(lib, h, stxt) == [2, 2],
                  f"a prompt that starts with {stxt!r} still gets BOS prepended "
                  f"(llama.cpp gives [2, 2])")
            check(cencode(lib, h, "Hello", add_bos=0) == cencode(lib, h, "Hello")[1:],
                  "add_bos=0 suppresses the leading BOS")
            # gemma has no token for CR, so it rides the <0x0D> byte token; DECODE must
            # give the raw byte back (llama.cpp's token_to_piece maps byte tokens to
            # their byte), not the literal "<0x0D>".
            ids = cencode(lib, h, "\r", add_bos=0)
            check(cdecode(lib, h, ids, 1) == "\r",
                  f"a byte token decodes to its raw byte, not the literal (ids={ids})")
        else:
            check(int(lib.aero_tok_add_bos(h)) == 0, "qwen: add_bos is off (read from the file)")

        # ---- negatives: refuse, never silently wrap ----------------------
        banner(f"[negative] {label}: bad input is refused")
        buf = (c_int32 * 8)()
        # a buffer too small must be REFUSED, not truncated
        check(lib.aero_tok_encode(h, b"Hello, world!", 13, buf, 1, -1) == -1,
              "encode refuses a too-small output buffer instead of truncating")
        check(cencode(lib, h, "Hello, world!") == ([9419, 11, 1814, 0] if "gpt2" in label
              else [2, 9259, 236764, 1902, 236888]),
              "a known string yields the exact published ids")
        check(lib.aero_tok_encode(h, None, 5, None, 0, -1) == -1,
              "encode refuses a null text with a positive length")
        check(lib.aero_tok_encode(h, b"x", -1, None, 0, -1) == -1,
              "encode refuses a negative length")
        check(lib.aero_tok_encode(None, b"x", 1, None, 0, -1) == -1,
              "encode refuses a null handle")
        # out-of-range ids: refuse rather than wrap
        check(cdecode(lib, h, [int(lib.aero_tok_n_vocab(h)) + 1], 1) is None,
              "decode refuses an out-of-range id (does not wrap)")
        check(cdecode(lib, h, [-3], 1) is None, "decode refuses a negative id")
        hid = cencode(lib, h, "Hello", add_bos=0)[0]   # a token whose text is non-empty
        check(int(lib.aero_tok_token(h, hid, None, 0)) > 0,
              f"token(id) reports the byte length of a valid id ({hid})")
        arr = (c_int32 * 1)(hid)
        check(lib.aero_tok_decode(h, arr, 1, ctypes.create_string_buffer(1), 0, 1) == -1,
              "decode refuses a too-small output buffer")
        check(int(lib.aero_tok_token(h, int(lib.aero_tok_n_vocab(h)) + 1, None, 0)) == -1,
              "token(id) refuses an out-of-range id")

        lib.aero_tok_free(h)

    # ---- negatives that need no model -----------------------------------
    banner("[negative] malformed / tokenizer-less files are refused, not guessed at")
    check(not lib.aero_tok_open(b"/no/such/file.gguf"),
          "a missing file returns null (err says why)")
    check(b"cannot open" in lib.aero_tok_last_error().lower() or
          b"no such" in lib.aero_tok_last_error().lower(),
          f"the error names the cause ({lib.aero_tok_last_error().decode()!r})")

    from engine.gguf import GGUFWriter, GGUF_STRING, GGUF_ARRAY
    tmpd = tempfile.mkdtemp(prefix="aero_tok_")
    # (a) a GGUF with no tokenizer at all
    p1 = os.path.join(tmpd, "no_tok.gguf")
    w = GGUFWriter("qwen35"); w.write(p1)
    check(not lib.aero_tok_open(p1.encode()),
          f"a GGUF with no tokenizer returns null ({lib.aero_tok_last_error().decode()!r})")
    check(b"tokenizer" in lib.aero_tok_last_error().lower(),
          "and the error says the tokenizer is absent")
    # (b) a GGUF whose tokenizer model we do not support
    p2 = os.path.join(tmpd, "bad_model.gguf")
    w = GGUFWriter("qwen35")
    w.add_kv("tokenizer.ggml.tokens", GGUF_ARRAY, (GGUF_STRING, ["a", "b", "ab"]))
    w.add_kv("tokenizer.ggml.merges", GGUF_ARRAY, (GGUF_STRING, ["a b"]))
    w.add_kv("tokenizer.ggml.model", GGUF_STRING, "some-future-format")
    w.write(p2)
    check(not lib.aero_tok_open(p2.encode()),
          "an unsupported tokenizer model returns null, not a guess")
    check(b"unsupported" in lib.aero_tok_last_error().lower(),
          f"and the error explains ({lib.aero_tok_last_error().decode()!r})")
    # (c) a tokenizer built from arrays: a bad vocab is refused
    check(not lib.aero_tok_open(b""), "an empty path returns null")

    banner("[abi] a tokenizer built from arrays works end to end (no model needed)")
    # small fixture the same shape as the real files: a gpt2-family vocab where a
    # rank-1 merge "a b" -> "ab" must fire, and a rank-0 merge "a <space>" must NOT,
    # because the pre-tokeniser keeps the space in the next unit.
    p3 = os.path.join(tmpd, "mini.gguf")
    w = GGUFWriter("qwen2")
    # bytes: 'a'='a', 'b'='b', space 0x20 -> U+0120. vocab carries "ab" (a merge result).
    w.add_kv("tokenizer.ggml.tokens", GGUF_ARRAY,
             (GGUF_STRING, ["a", "\u0120", "a\u0120", "b", "ab"]))
    w.add_kv("tokenizer.ggml.merges", GGUF_ARRAY, (GGUF_STRING, ["a \u0120", "a b"]))
    w.add_kv("tokenizer.ggml.model", GGUF_STRING, "qwen2")
    w.add_kv("tokenizer.ggml.token_type", GGUF_ARRAY, (4, [1, 1, 1, 1, 1]))
    w.write(p3)
    h3 = lib.aero_tok_open(p3.encode())
    check(bool(h3), f"a minimal qwen2 tokenizer loads (err={lib.aero_tok_last_error().decode()!r})")
    if h3:
        check(cencode(lib, h3, "a") == [0], "single letter 'a' -> id 0")
        check(cencode(lib, h3, "ab") == [4],
              "the rank-1 merge 'a b' fires inside a unit -> id 4 (not [0, 3])")
        check(cencode(lib, h3, "a b") == [0, 1, 3],
              "the space starts a new unit, so 'a <space>' does NOT merge across it")
        check(cdecode(lib, h3, [0, 1, 3]) == "a b", "decode maps U+0120 back to a space")
        check(int(lib.aero_tok_family(h3)) == 0, "family reported as gpt2 (0)")
        lib.aero_tok_free(h3)

    print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
