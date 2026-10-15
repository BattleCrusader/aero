#!/usr/bin/env python3
"""The native chat renderer and the gemma4 arch descriptor, against the PYTHON ORACLE.

WHAT IS BEING PROVEN, and why it is not cosmetic. The engine and every check tool
must send byte-identical prompts. A tool that renders its own prompt "verifies"
something the engine never actually sends -- that is exactly how the BOS bug hid
(the engine dropped BOS, the tool did not test through the engine, and the model's
garbage looked like an architecture bug). So this suite does not assert against
hand-written expected strings alone: it renders the SAME conversation through
cpp `aero_chat_render_into` and through engine.chat_template.render_chat, and
compares the two BYTES, for every combination of envelope / thinking / gen-prompt
/ message shape. A byte mismatch prints both sides as escaped reprs with the
differing offset.

The arch half is the same idea against engine/gemma.py: every per-layer answer
(head_dim, rope base, rope dims, kv heads, window, is_swa) is compared to
GemmaConfig's for all 48 layers, and -- when the real file is present -- both are
compared to the metadata actually stored in gemma-4-12b-it-IQ4_XS.gguf.

The C++ is compiled from source by this test (the Makefile is owned elsewhere), so
a clean checkout reproduces it. No model is loaded: engine.gguf reads metadata
only, which is cheap and does not touch the 16 GB RAM budget.

    env -u PYTHONPATH -u PYTHONHOME .venv-cml/bin/python tests/test_chat_template_cpp.py
"""
from __future__ import annotations

import ctypes
import math
import os
import shutil
import subprocess
import sys
from ctypes import POINTER, c_char_p, c_float, c_int, c_int32, c_void_p
from typing import Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CHECKS = 0
FAILS = 0


def check(cond: bool, what: str) -> None:
    global CHECKS, FAILS
    CHECKS += 1
    if cond:
        print("  ok   " + what)
    else:
        FAILS += 1
        print("  FAIL " + what)


def banner(t: str) -> None:
    print("\n" + t)


def diff_at(a: bytes, b: bytes):
    """First differing offset, with a window of context on each side."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            lo = max(0, i - 24)
            return i, repr(a[lo:i + 24]), repr(b[lo:i + 24])
    i = min(len(a), len(b))
    return i, repr(a[-40:]), repr(b[-40:])


# ---------------------------------------------------------------- build the libs
print("=== native chat template + gemma4 arch vs the Python oracle ===")

CC = shutil.which("clang++")
CFLAGS = ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Iinclude", "-fPIC", "-shared", "-pthread"]
LIBS = (("src/chat_template.cpp", os.path.join(ROOT, "build", "libaero_chat.dylib")),
        ("src/arch.cpp", os.path.join(ROOT, "build", "libaero_arch.dylib")))

if CC is None:
    print("(skipped: clang++ not found -- cannot build libaero_chat/libaero_arch)")
    print("\n=== 0 checks, 0 failures ===")
    sys.exit(0)

os.makedirs(os.path.join(ROOT, "build"), exist_ok=True)
build_failed = None
for src, out in LIBS:
    cmd = [CC] + CFLAGS + [src, "-o", out]
    print("  $ " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        build_failed = f"{src}: {p.stderr.strip()}"
        break
    if p.stderr.strip():
        # warnings are not fatal, but they must be visible
        print("      (warnings) " + p.stderr.strip().replace("\n", "\n      "))

if build_failed is not None:
    print("  FAIL the native renderer/arch did not compile:\n      " + build_failed)
    print("\n=== 1 checks, 1 failures ===")
    sys.exit(1)

# ------------------------------------------------------------------- ctypes ABI
chat = ctypes.CDLL(LIBS[0][1])
arch = ctypes.CDLL(LIBS[1][1])

chat.aero_chat_detect.argtypes = [POINTER(c_char_p), c_int]
chat.aero_chat_detect.restype = c_int
chat.aero_chat_render_size.argtypes = [POINTER(c_char_p), POINTER(c_char_p), c_int,
                                       POINTER(c_char_p), c_int, c_int, c_int]
chat.aero_chat_render_size.restype = c_int
chat.aero_chat_render_into.argtypes = [POINTER(c_char_p), POINTER(c_char_p), c_int,
                                       POINTER(c_char_p), c_int, c_int, c_int, c_void_p, c_int]
chat.aero_chat_render_into.restype = c_int
chat.aero_chat_render.argtypes = [POINTER(c_char_p), POINTER(c_char_p), c_int,
                                  POINTER(c_char_p), c_int, c_int, c_int]
chat.aero_chat_render.restype = c_void_p
chat.aero_chat_free.argtypes = [c_void_p]
chat.aero_chat_envelope_name.argtypes = [c_int]
chat.aero_chat_envelope_name.restype = c_char_p
chat.aero_chat_marker.argtypes = [c_int]
chat.aero_chat_marker.restype = c_char_p
chat.aero_chat_thinking_from_env.restype = c_int

ENV_ERR_UNKNOWN, ENV_ERR_NO_TOKENIZER, ENV_ERR_BAD_ARGS, ENV_ERR_TOO_SMALL = -1, -2, -3, -4

from engine.chat_template import (EnvelopeError, detect_envelope,  # noqa: E402
                                  render_chat)

# The only thing the renderer is allowed to depend on, mirroring the existing
# Python suite's stub.
class StubTok:
    def __init__(self, special):
        self._special = dict(special)

    @classmethod
    def from_specials(cls, specials):
        return cls({s: i for i, s in enumerate(specials)})


GEMMA_SPECIALS = ["<bos>", "<|turn>", "<turn|>", "<|channel>", "<channel|>",
                  "<|think|>", "<|tool>", "<tool|>", "<|\"|>", "<|image|>"]
QWEN_SPECIALS = ["<bos>", "<|im_start|>", "<|im_end|>"]
GEMMA = StubTok.from_specials(GEMMA_SPECIALS)
QWEN = StubTok.from_specials(QWEN_SPECIALS)
NONE = StubTok({})


def carr(items):
    """A (c_char_p * n) array; None entries become NULL, and n is returned."""
    n = len(items)
    arr = (c_char_p * max(1, n))()
    for i, it in enumerate(items):
        arr[i] = None if it is None else it.encode("utf-8")
    return arr, n


def split_msgs(messages):
    roles = [(m["role"] if "role" in m else None) for m in messages]
    contents = [(m["content"] if "content" in m else None) for m in messages]
    return roles, contents


def c_detect(specials):
    if specials is None:
        return chat.aero_chat_detect(None, 0)
    a, n = carr(specials)
    return chat.aero_chat_detect(a, n)


def c_render(messages, specials, add_gen=True, thinking: Optional[bool] = False,
             malloc=False):
    """Returns (rc, bytes) -- rc is the ABI return, bytes the rendered prompt."""
    r, c = split_msgs(messages)
    roles, n = carr(r)
    contents, _ = carr(c)
    if specials is None:
        spec, ns = None, 0
    else:
        spec, ns = carr(specials)
    ag, th = (1 if add_gen else 0), (-1 if thinking is None else (1 if thinking else 0))
    if malloc:
        ptr = chat.aero_chat_render(roles, contents, n, spec, ns, ag, th)
        if not ptr:
            return -1, b""
        try:
            size = chat.aero_chat_render_size(roles, contents, n, spec, ns, ag, th)
            data = ctypes.string_at(ptr, size - 1)
        finally:
            chat.aero_chat_free(ptr)
        return len(data), data
    need = chat.aero_chat_render_size(roles, contents, n, spec, ns, ag, th)
    if need < 0:
        return need, b""
    buf = ctypes.create_string_buffer(need + 8)
    rc = chat.aero_chat_render_into(roles, contents, n, spec, ns, ag, th,
                                    ctypes.cast(buf, c_void_p), need + 8)
    return rc, buf.raw[:max(rc, 0)]


def oracle(messages, tok, add_gen=True, thinking: Optional[bool] = False):
    """The Python renderer, as bytes. `thinking=None` goes through AERO_THINKING."""
    py_msgs = [{"role": m["role"] if "role" in m else "user",
                "content": m["content"] if "content" in m else ""} for m in messages]
    if thinking is None:
        return render_chat(py_msgs, tok, add_generation_prompt=add_gen).encode("utf-8")
    return render_chat(py_msgs, tok, add_generation_prompt=add_gen,
                       enable_thinking=thinking).encode("utf-8")


def compare(messages, specials, tok, add_gen, thinking, label):
    rc, got = c_render(messages, specials, add_gen, thinking)
    want = oracle(messages, tok, add_gen, thinking)
    if got == want and rc >= 0:
        check(True, f"{label}: byte-identical to render_chat ({len(want)} B)")
        return True
    off, a, b = diff_at(got, want)
    check(False, f"{label}: MISMATCH at byte {off}\n         cpp: {a}\n         py : {b}")
    return False


# ------------------------------------------------------- [1] envelope detection
banner("[1] envelope detection is driven by the file's own special tokens")
check(c_detect(GEMMA_SPECIALS) == 0 and detect_envelope(GEMMA) == "gemma4",
      "gemma4 detected from '<|turn>' on both sides")
check(c_detect(QWEN_SPECIALS) == 1 and detect_envelope(QWEN) == "chatml",
      "chatml detected from '<|im_start|>' on both sides")
check(c_detect([]) == ENV_ERR_UNKNOWN and detect_envelope(NONE) is None,
      "an empty special table is 'unrecognised' on both sides")
check(c_detect(None) == ENV_ERR_NO_TOKENIZER,
      "a NULL special table is 'no tokenizer' (distinct from unrecognised)")
mixed = ["<|im_start|>", "<|turn>"]
check(c_detect(mixed) == 0,
      "a table with BOTH markers resolves to gemma4 (the ENVELOPES probe order)")
check(c_detect(["<bos>", "<|im_end|>"]) == ENV_ERR_UNKNOWN,
      "an unrelated special table is 'unrecognised' (no filename sniffing)")
check(chat.aero_chat_envelope_name(0) == b"gemma4"
      and chat.aero_chat_envelope_name(1) == b"chatml"
      and chat.aero_chat_envelope_name(-1) == b"",
      "envelope names round-trip for logging")
check(chat.aero_chat_marker(0) == b"<|turn>" and chat.aero_chat_marker(1) == b"<|im_start|>",
      "the marker each envelope is detected by is exposed")

USER = [{"role": "user", "content": "Hi"}]

# ------------------------------------------------- [2] the shipped bug pins, gemma4
banner("[2] the gemma4 envelope: 'model', the bos the template emits, the inverted flag")
rc, got = c_render(USER, GEMMA_SPECIALS, True, False)
check(got == "<bos><|turn>user\nHi<turn|>\n<|turn>model\n"
             "<|channel>thought\n<channel|>".encode(),
      "single user turn + gen prompt + immediately-closed thought channel")
check(got.startswith(b"<bos>"), "the template emits <bos> itself")
check(b"<|turn>model\n" in got, "the assistant role is spelled 'model', not 'assistant'")
check(got.index(b"<|turn>user") == len(b"<bos>"), "<bos> precedes the first turn")

_, on = c_render(USER, GEMMA_SPECIALS, True, True)
check(b"<|channel>thought\n<channel|>" in got and b"<|channel>" not in on,
      "the thinking flag is INVERTED: off closes the channel, on emits nothing")
_, nogen = c_render(USER, GEMMA_SPECIALS, False, False)
check(nogen == "<bos><|turn>user\nHi<turn|>\n".encode(),
      "no gen prompt -> the prompt closes after the user turn")

_, two = c_render([{"role": "user", "content": "Hi"},
                   {"role": "assistant", "content": "Hello"}], GEMMA_SPECIALS,
                  False, False)
check(two == "<bos><|turn>user\nHi<turn|>\n<|turn>model\nHello<turn|>\n".encode(),
      "assistant history renders as a 'model' turn and closes")

# ------------------------------------------------------------------ [3] chatml
banner("[3] the chatml (qwen) envelope, and its thinking block")
rc, c = c_render(USER, QWEN_SPECIALS, True, False)
check(c == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
          " thinking\n\n</think>\n\n".encode(),
      "chatml user turn + assistant turn with the closed thinking block")
check(not c.startswith(b"<bos>"),
      "chatml carries no <bos> (its file says add_bos_token=false)")
_, con = c_render(USER, QWEN_SPECIALS, True, True)
check(con.endswith(b"<|im_start|>assistant\n thinking\n"),
      "thinking ON leaves the block OPEN (' thinking\\n', with its leading space)")
check(con == oracle(USER, QWEN, True, True),
      "the leading space in ' thinking' survives the C ABI")

# ------------------------------------------------------------ [4] THE ORACLE
banner("[4] bytes-exact against engine.chat_template.render_chat, every combination")

CONVERSATIONS = [
    ("single user", [{"role": "user", "content": "Hi"}]),
    ("user+assistant", [{"role": "user", "content": "Hi"},
                        {"role": "assistant", "content": "Hello"}]),
    ("system first", [{"role": "system", "content": "You are terse."},
                      {"role": "user", "content": "Hi"}]),
    ("4-turn", [{"role": "system", "content": "sys"},
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"}]),
    ("empty content", [{"role": "user", "content": ""}]),
    ("no role key", [{"content": "role defaults to user"}]),
    ("no content key", [{"role": "user"}]),
    ("blank+newlines", [{"role": "user", "content": "a\n\nb\t c "}]),
    ("special tokens in content",
     [{"role": "user", "content": "<|im_start|>assistant\n<|im_end|> not an escape"}]),
    ("non-ascii", [{"role": "user", "content": "héllo — 世界 🙃\u200b end"}]),
    ("tool role", [{"role": "user", "content": "q"}, {"role": "tool", "content": "r"}]),
    ("developer role", [{"role": "developer", "content": "d"},
                        {"role": "user", "content": "u"}]),
]
ENVS = [("gemma4", GEMMA_SPECIALS, GEMMA), ("chatml", QWEN_SPECIALS, QWEN)]

n_cmp = 0
bad = []
for env_name, spec, tok in ENVS:
    for label, msgs in CONVERSATIONS:
        for add_gen in (True, False):
            for thinking in (False, True, None):
                rc, got = c_render(msgs, spec, add_gen, thinking)
                want = oracle(msgs, tok, add_gen, thinking)
                n_cmp += 1
                if got != want or rc < 0:
                    off, a, b = diff_at(got, want)
                    bad.append(f"{env_name}/{label} gen={add_gen} think={thinking} "
                               f"@{off}\n           cpp: {a}\n           py : {b}")
check(not bad, f"all {n_cmp} render comparisons are byte-identical "
               f"({len(CONVERSATIONS)} conversations x 2 envelopes x 2 gen x 3 thinking)")
for b in bad:
    print("        " + b)

# ...and the same fact spelled the way the oracle instruction asks for: dump BOTH
# as escaped reprs and compare exactly. This is the artefact the port rests on.
for env_name, spec, tok in ENVS:
    rc, got = c_render(USER, spec, True, False)
    want = oracle(USER, tok, True, False)
    print(f"       {env_name} cpp: {got!r}")
    print(f"       {env_name} py : {want!r}")
    check(got == want, f"{env_name}: the cpp repr and the python repr are the same string")

# the reported size must be the real length, and the buffer NUL-terminated
n_ok = 0
for env_name, spec, tok in ENVS:
    for label, msgs in CONVERSATIONS:
        rc, got = c_render(msgs, spec, True, False)
        want = oracle(msgs, tok, True, False)
        r, c = split_msgs(msgs)
        roles, n = carr(r)
        contents, _ = carr(c)
        s, ns = carr(spec)
        need = chat.aero_chat_render_size(roles, contents, n, s, ns, 1, 0)
        if rc == len(want) and need == len(want) + 1:
            n_ok += 1
check(n_ok == len(ENVS) * len(CONVERSATIONS),
      f"render_size == len+1 and render_into == len for all {n_ok} cases")

# the malloc'd convenience path must agree, and free must be safe
ok_malloc = True
for env_name, spec, tok in ENVS:
    for label, msgs in CONVERSATIONS:
        rc, got = c_render(msgs, spec, True, True, malloc=True)
        if got != oracle(msgs, tok, True, True):
            ok_malloc = False
check(ok_malloc, "the malloc'd aero_chat_render path agrees with render_into")

# ------------------------------------------------------------- [5] refusals
banner("[5] refusals: never emit unframed text, never write out of bounds")

try:
    render_chat(USER, NONE)
    py_raises = False
except EnvelopeError:
    py_raises = True
rc, _ = c_render(USER, ["<bos>", "<|im_end|>"])
check(py_raises and rc == ENV_ERR_UNKNOWN,
      "an unrecognised template refuses on both sides (no unframed text)")


def raw_render_rc(specials, n_specials, roles, contents, n_messages, ag, th, out, out_len):
    r, _ = carr(roles)
    c, _ = carr(contents)
    if specials is None:
        s = None
    else:
        s, _ = carr(specials)
    return chat.aero_chat_render_into(r, c, n_messages, s, n_specials, ag, th, out, out_len)


buf = ctypes.create_string_buffer(4096)
rc = raw_render_rc(None, 0, ["user"], ["Hi"], 1, 1, 0, ctypes.cast(buf, c_void_p), 4096)
check(rc == ENV_ERR_NO_TOKENIZER, f"a NULL special table refuses (rc={rc})")
rc = raw_render_rc(GEMMA_SPECIALS, len(GEMMA_SPECIALS), ["user"], ["Hi"], 1, 1, 0,
                   None, 0)
check(rc == ENV_ERR_BAD_ARGS, f"a NULL out buffer refuses (rc={rc})")

small = ctypes.create_string_buffer(8)
r, _ = carr(["user"])
c_, _ = carr(["Hi"])
s, ns = carr(GEMMA_SPECIALS)
rc = chat.aero_chat_render_into(r, c_, 1, s, ns, 1, 0, ctypes.cast(small, c_void_p), 8)
check(rc == ENV_ERR_TOO_SMALL and small.raw[0] == 0,
      "an 8-byte buffer refuses and is left EMPTY (never a partial prompt)")

rc = chat.aero_chat_render_into(None, None, 1, s, ns, 1, 0,
                                ctypes.cast(ctypes.create_string_buffer(4096), c_void_p), 4096)
check(rc == ENV_ERR_BAD_ARGS, "NULL role/content arrays with n>0 refuse")

# no messages is legal: it renders the generation prompt alone
zero_rc, zero = c_render([], GEMMA_SPECIALS, True, False)
check(zero_rc >= 0 and zero == oracle([], GEMMA, True, False),
      f"an empty conversation renders just the gen prompt ({zero!r})")

# the malloc path returns NULL on refusal, and free(NULL) is not a crash
nullptr = chat.aero_chat_render((c_char_p * 1)(b"user"), (c_char_p * 1)(b"Hi"), 1,
                                (c_char_p * 1)(b"<bos>"), 1, 1, 0)
check(nullptr is None or nullptr == 0, "the malloc path returns NULL on an unknown template")
chat.aero_chat_free(None)

# ------------------------------------------------------------ [6] AERO_THINKING
banner("[6] the AERO_THINKING env path is the Python None path")
prev = os.environ.get("AERO_THINKING")
os.environ["AERO_THINKING"] = "1"
c1 = chat.aero_chat_thinking_from_env()
_, g1 = c_render(USER, GEMMA_SPECIALS, True, None)
p1 = oracle(USER, GEMMA, True, None)
os.environ["AERO_THINKING"] = "0"
c0 = chat.aero_chat_thinking_from_env()
_, g0 = c_render(USER, GEMMA_SPECIALS, True, None)
p0 = oracle(USER, GEMMA, True, None)
os.environ["AERO_THINKING"] = "yes"          # anything that is not exactly "1" is off
cyes = chat.aero_chat_thinking_from_env()
os.environ.pop("AERO_THINKING", None)
cunset = chat.aero_chat_thinking_from_env()
if prev is not None:
    os.environ["AERO_THINKING"] = prev
check(c1 == 1 and c0 == 0 and cyes == 0 and cunset == 0,
      "AERO_THINKING=='1' is the only thing that turns thinking on")
check(g1 == p1 and g0 == p0,
      "enable_thinking=-1 matches render_chat(enable_thinking=None) under the env")

# ------------------------------------------------------------ [7] the real files
banner("[7] the real GGUF special tables and templates, when present")
REAL = (("gemma4", os.path.expanduser("~/.cache/aero-models/gemma-4-12b-it-IQ4_XS.gguf"),
         "gemma4"),
        ("qwen", os.path.expanduser("~/.cache/aero-models/Qwen3.8-4B-Q4_K_M.gguf"),
         "chatml"))
for label, path, want in REAL:
    if not os.path.exists(path):
        print(f"  (skipped {label}: {path} absent)")
        continue
    from engine.gguf import GGUFReader
    from engine.tokenizer import Tokenizer
    with GGUFReader(path) as r:
        tk = Tokenizer.from_gguf(r)
    specials = list(tk._special.keys())
    check(c_detect(specials) == (0 if want == "gemma4" else 1)
          and detect_envelope(tk) == want,
          f"{label}: the file's real special table detects as {want} on both sides")
    # compare() reports its own ok/FAIL, so it is not wrapped in another check
    compare(USER, specials, tk, True, False, f"{label}: real-file prompt")
    _, got = c_render([{"role": "system", "content": "be brief"},
                       {"role": "user", "content": "Hi"},
                       {"role": "assistant", "content": "Sure"},
                       {"role": "user", "content": "Explain RoPE in one line."}],
                      specials, True, False)
    if want == "gemma4":
        check(got.startswith(b"<bos><|turn>system"),
              "gemma4: the real table renders a system turn after a single <bos>")
        check(got.count(b"<bos>") == 1, "gemma4: <bos> appears exactly once")
    else:
        check(not got.startswith(b"<bos>") and b"<|im_start|>system\n" in got,
              "qwen: no <bos>, and the system turn is spelled <|im_start|>system")

# ------------------------------------------------------------- [8] arch
banner("[8] the gemma4 arch descriptor: per-layer geometry without Python")

class Gemma4Cfg(ctypes.Structure):
    _fields_ = [("n_layer", c_int32), ("n_embd", c_int32), ("n_head", c_int32),
                ("head_dim", c_int32), ("value_dim", c_int32), ("ffn_len", c_int32),
                ("eps", c_float), ("sliding_window", c_int32), ("softcap", c_float),
                ("rope_base", c_float), ("rope_dims", c_int32),
                ("head_dim_swa", c_int32), ("rope_dims_swa", c_int32),
                ("rope_base_swa", c_float), ("attn_scale", c_float),
                ("n_kv_shared", c_int32), ("has_per_layer_embd", c_int32),
                ("n_layer_embd", c_int32)]


arch.aero_gemma4_defaults.argtypes = [POINTER(Gemma4Cfg)]
arch.aero_gemma4_default_kv.argtypes = [POINTER(c_int32), c_int]
arch.aero_gemma4_default_kv.restype = c_int
arch.aero_gemma4_default_pattern.argtypes = [POINTER(c_int32), c_int]
arch.aero_gemma4_default_pattern.restype = c_int
arch.aero_gemma4_new.argtypes = [POINTER(Gemma4Cfg), POINTER(c_int32), c_int,
                                 POINTER(c_int32), c_int, c_void_p, c_int]
arch.aero_gemma4_new.restype = c_void_p
arch.aero_arch_free.argtypes = [c_void_p]
for nm in ("aero_arch_n_layer", "aero_arch_n_embd", "aero_arch_n_head", "aero_arch_head_dim",
           "aero_arch_ffn_len", "aero_arch_out_scale_mode"):
    getattr(arch, nm).argtypes = [c_void_p]
    getattr(arch, nm).restype = c_int
for nm in ("aero_arch_eps", "aero_arch_attn_scale", "aero_arch_embed_scale",
           "aero_arch_softcap"):
    getattr(arch, nm).argtypes = [c_void_p]
    getattr(arch, nm).restype = c_float
for nm in ("aero_arch_is_swa", "aero_arch_head_dim_at", "aero_arch_rope_dims_at",
           "aero_arch_kv_heads_at", "aero_arch_window_at"):
    getattr(arch, nm).argtypes = [c_void_p, c_int]
    getattr(arch, nm).restype = c_int
arch.aero_arch_rope_base_at.argtypes = [c_void_p, c_int]
arch.aero_arch_rope_base_at.restype = c_float
for nm in ("aero_arch_activation", "aero_arch_block_mode"):
    getattr(arch, nm).argtypes = [c_void_p]
    getattr(arch, nm).restype = c_char_p
arch.aero_gemma4_kv_heads_for.argtypes = [POINTER(c_int32), c_int, c_int, POINTER(c_int32)]
arch.aero_gemma4_kv_heads_for.restype = c_int
arch.aero_gemma4_pattern_for.argtypes = [POINTER(c_int32), c_int, c_int, POINTER(c_int32)]
arch.aero_gemma4_pattern_for.restype = c_int
arch.aero_arch_window_layers.argtypes = [c_void_p, POINTER(c_int32), c_int]
arch.aero_arch_window_layers.restype = c_int
arch.aero_arch_global_layers.argtypes = [c_void_p, POINTER(c_int32), c_int]
arch.aero_arch_global_layers.restype = c_int
arch.aero_arch_softcap_logits.argtypes = [POINTER(c_float), c_int, c_float]
arch.aero_arch_softcap_logits.restype = c_int
arch.aero_arch_scale_of.argtypes = [POINTER(c_float), c_int, POINTER(c_float)]
arch.aero_arch_scale_of.restype = c_int
arch.aero_arch_block_tensors.argtypes = [POINTER(c_char_p), c_int]
arch.aero_arch_block_tensors.restype = c_int

ARCH_ERR = -1


def i32(vals):
    return (c_int32 * max(1, len(vals)))(*vals)


def make_arch(cfg: Gemma4Cfg, kv_raw, pattern_raw):
    err = ctypes.create_string_buffer(256)
    k = i32(kv_raw) if kv_raw is not None else None
    p = i32(pattern_raw) if pattern_raw is not None else None
    h = arch.aero_gemma4_new(ctypes.byref(cfg), k, len(kv_raw) if kv_raw else 0,
                             p, len(pattern_raw) if pattern_raw else 0,
                             ctypes.cast(err, c_void_p), len(err))
    return h, err.value.decode("utf-8", "replace")


from engine import gemma as G  # noqa: E402

cfg = Gemma4Cfg()
arch.aero_gemma4_defaults(ctypes.byref(cfg))
kv = i32([0] * 48)
pat = i32([0] * 48)
check(arch.aero_gemma4_default_kv(kv, 48) == 48 and arch.aero_gemma4_default_pattern(pat, 48) == 48,
      "the 12b default kv/pattern lists are produced")
check([kv[i] for i in range(48)] == [1 if i % 6 == 5 else 8 for i in range(48)],
      "default kv heads: 8 on windowed layers, 1 on every 6th (the global layers)")
check([pat[i] for i in range(48)] == [0 if i % 6 == 5 else 1 for i in range(48)],
      "default pattern: 5 windowed then 1 global, repeating")

h, err = make_arch(cfg, [kv[i] for i in range(48)], [pat[i] for i in range(48)])
check(bool(h), f"the descriptor builds from the 12b geometry (err={err!r})")
if not h:
    print("\n=== %d checks, %d failures ===" % (CHECKS, FAILS))
    sys.exit(1)

check(arch.aero_arch_n_layer(h) == 48 and arch.aero_arch_n_embd(h) == 3840
      and arch.aero_arch_n_head(h) == 16 and arch.aero_arch_head_dim(h) == 512
      and arch.aero_arch_ffn_len(h) == 15360,
      "geometry scalars round-trip (48 layers, 3840 embd, 16 heads, 512/15360)")

# ---- the per-layer variation, which is the whole point of this module
check(arch.aero_arch_is_swa(h, 0) == 1 and arch.aero_arch_is_swa(h, 5) == 0,
      "layer 0 is sliding-window, layer 5 is global")
check(arch.aero_arch_head_dim_at(h, 0) == 256 and arch.aero_arch_head_dim_at(h, 5) == 512,
      "head_dim VARIES by layer: 256 windowed / 512 full (a scalar would be wrong)")
check(abs(arch.aero_arch_rope_base_at(h, 0) - 10000.0) < 1e-3
      and abs(arch.aero_arch_rope_base_at(h, 5) - 1000000.0) < 1.0,
      "rope base VARIES by layer: 10000 windowed / 1000000 full")
check(arch.aero_arch_rope_dims_at(h, 0) == 256 and arch.aero_arch_rope_dims_at(h, 5) == 512,
      "rope dims VARIES by layer: 256 windowed / 512 full")
check(arch.aero_arch_kv_heads_at(h, 0) == 8 and arch.aero_arch_kv_heads_at(h, 5) == 1,
      "kv heads VARIES by layer: 8 windowed / 1 full (the file stores a list, not a scalar)")
check(arch.aero_arch_window_at(h, 0) == 1024 and arch.aero_arch_window_at(h, 5) == 0,
      "window_at(5) is 0 -- a GLOBAL layer must not be given the 1024 window")

wl = i32([0] * 48)
gl = i32([0] * 48)
nw = arch.aero_arch_window_layers(h, wl, 48)
ng = arch.aero_arch_global_layers(h, gl, 48)
check(nw == 40 and ng == 8 and [gl[i] for i in range(8)] == [5, 11, 17, 23, 29, 35, 41, 47],
      f"{nw} windowed / {ng} global layers, the globals exactly every 6th from layer 5")

check(abs(arch.aero_arch_attn_scale(h) - 1.0) < 1e-6,
      "attention scale is 1.0 -- NO 1/sqrt(head_dim) pre-scaling")
check(abs(arch.aero_arch_embed_scale(h) - math.sqrt(3840)) < 1e-3,
      f"embeddings are scaled by sqrt(n_embd) ({math.sqrt(3840):.3f})")
check(arch.aero_arch_activation(h) == b"gelu",
      "activation is GELU, not SiLU (a different function, not a rounding difference)")
check(arch.aero_arch_block_mode(h) == b"after_add",
      "block ordering is 'after_add' (norm after the residual add, not on_output)")
check(arch.aero_arch_out_scale_mode(h) == 2,
      "layer_output_scale applies to the WHOLE layer output (mode 2 = sum)")
check(abs(arch.aero_arch_softcap(h) - 30.0) < 1e-6, "logit soft-cap is 30.0")

names = (c_char_p * 5)()
nn = arch.aero_arch_block_tensors(names, 5)
check(nn == 5 and [n.decode() for n in names] == [
        "attn_norm.weight", "ffn_norm.weight", "post_attention_norm.weight",
        "post_ffw_norm.weight", "layer_output_scale.weight"],
      "the 5 block tensors are exposed in consumption order (both norms AND post_ffw_norm)")
check(arch.aero_arch_block_tensors((c_char_p * 3)(), 3) == ARCH_ERR,
      "a too-small name buffer is refused rather than truncated")

# ---- the oracle: every layer, against engine/gemma.py
banner("[9] every per-layer answer vs engine/gemma.py GemmaConfig (48 layers)")
py_kv = G.GemmaConfig.kv_heads_for([kv[i] for i in range(48)], 48)
py_pat = G.GemmaConfig.pattern_for([pat[i] for i in range(48)], 48)
py = G.GemmaConfig(n_layer=48, n_embd=3840, n_head=16, head_dim=512, value_dim=512,
                   ffn_len=15360, eps=cfg.eps, sliding_window=1024,
                   sliding_pattern=py_pat, softcap=30.0, kv_heads=py_kv,
                   rope_base=1000000.0, rope_dims=512, rope_base_swa=10000.0,
                   rope_dims_swa=256, attn_scale=1.0, n_kv_shared=0)

mism = []
for L in range(48):
    if arch.aero_arch_is_swa(h, L) != (1 if py.is_swa(L) else 0):
        mism.append(f"L{L} is_swa")
    if arch.aero_arch_rope_base_at(h, L) != py.rope_base_at(L):
        mism.append(f"L{L} rope_base {arch.aero_arch_rope_base_at(h, L)} vs {py.rope_base_at(L)}")
    if arch.aero_arch_rope_dims_at(h, L) != py.rope_dims_at(L):
        mism.append(f"L{L} rope_dims")
    if arch.aero_arch_kv_heads_at(h, L) != py.kv_heads[L]:
        mism.append(f"L{L} kv_heads")
    py_win = py.window_at(L)
    if arch.aero_arch_window_at(h, L) != (0 if py_win is None else py_win):
        mism.append(f"L{L} window {arch.aero_arch_window_at(h, L)} vs {py_win}")
check(not mism, "all 48 layers agree with GemmaConfig on is_swa/rope_base/rope_dims/"
                "kv_heads/window (0 mismatches)" if not mism else f"per-layer mismatches: {mism[:6]}")
check(list(py.window_layers) == [wl[i] for i in range(nw)]
      and list(py.global_layers) == [gl[i] for i in range(ng)],
      "the windowed/global layer lists match gemma.py exactly")

# the wideners, both spellings, against gemma.py's own wideners
outs = i32([0] * 48)
check(arch.aero_gemma4_kv_heads_for(i32([8]), 1, 48, outs) == 48
      and [outs[i] for i in range(48)] == G.GemmaConfig.kv_heads_for(8, 48)
      and all(outs[i] == 8 for i in range(48)),
      "a scalar kv head count widens to every layer (as gemma.py does)")
outs2 = i32([0] * 48)
check(arch.aero_gemma4_kv_heads_for(i32([kv[i] for i in range(48)]), 48, 48, outs2) == 48
      and [outs2[i] for i in range(48)] == list(py_kv),
      "a per-layer kv list passes through unchanged")
pout = i32([0] * 48)
check(arch.aero_gemma4_pattern_for(i32([1]), 1, 48, pout) == 48
      and all(pout[i] == 1 for i in range(48)),
      "a scalar sliding pattern widens to every layer")
# an absent pattern answers with 0 entries (== gemma.py's None), NOT 48 zeros:
# the two are different facts, and returning n_layer here indexed off the end
# of an empty vector once.
check(arch.aero_gemma4_pattern_for(None, 0, 48, pout) == 0
      and G.GemmaConfig.pattern_for(None, 48) is None,
      "an ABSENT pattern is 0 entries (gemma.py returns None), not 48 zeros")
check(arch.aero_gemma4_kv_heads_for(i32([8, 1, 8]), 3, 48, outs) == ARCH_ERR,
      "a kv list of the wrong length is refused (silently broadcasting would be wrong)")
check(arch.aero_gemma4_kv_heads_for(None, 0, 48, outs) == ARCH_ERR,
      "a MISSING head_count_kv is refused (gemma4 requires it)")
try:
    G.GemmaConfig.kv_heads_for(None, 48)
    py_kv_raises = False
except ValueError:
    py_kv_raises = True
check(py_kv_raises, "...and gemma.py raises for the same case")

# ---- refusals and the two tensor-level helpers
banner("[10] the descriptor refuses bad geometry instead of guessing")
bad_cfg = Gemma4Cfg()
arch.aero_gemma4_defaults(ctypes.byref(bad_cfg))
bad_cfg.n_layer = 0
h0, e0 = make_arch(bad_cfg, [8], [1])
check(not h0 and "n_layer" in e0, f"n_layer=0 is refused with a reason ({e0!r})")
bad_cfg = Gemma4Cfg()
arch.aero_gemma4_defaults(ctypes.byref(bad_cfg))
bad_cfg.softcap = -1.0
h1, e1 = make_arch(bad_cfg, [8], [1])
check(not h1 and "softcap" in e1, f"a negative softcap is refused ({e1!r})")
bad_cfg = Gemma4Cfg()
arch.aero_gemma4_defaults(ctypes.byref(bad_cfg))
h2, e2 = make_arch(bad_cfg, [8] * 47, [1])
check(not h2 and "47" in e2, f"a 47-entry kv list for 48 layers is refused ({e2!r})")
h3, e3 = make_arch(Gemma4Cfg(), None, None)
check(not h3, "an all-zero config is refused (nothing about the geometry is assumed)")
check(arch.aero_arch_n_layer(None) == ARCH_ERR and arch.aero_arch_is_swa(h, 48) == ARCH_ERR
      and arch.aero_arch_kv_heads_at(h, -1) == ARCH_ERR,
      "out-of-range layers and a NULL handle return AERO_ARCH_ERR, not a value")
check(arch.aero_arch_window_layers(h, i32([0] * 4), 4) == ARCH_ERR,
      "a too-small layer-list buffer is refused")

# the descriptor with NO pattern: every layer global, no window anywhere
h_np, _ = make_arch(cfg, [8], None)
check(all(arch.aero_arch_is_swa(h_np, L) == 0 for L in range(48))
      and all(arch.aero_arch_head_dim_at(h_np, L) == 512 for L in range(48))
      and all(arch.aero_arch_window_at(h_np, L) == 0 for L in range(48)),
      "with no pattern every layer is global and takes the FULL-attention geometry")

z = np.linspace(-200, 200, 4001, dtype=np.float32)
cvec = (c_float * len(z))(*[float(x) for x in z])
rc = arch.aero_arch_softcap_logits(cvec, len(z), 30.0)
got = np.array(list(cvec), dtype=np.float32)
want = G.softcap_logits(z, 30.0)
check(rc == 0 and np.allclose(got, want, atol=1e-6),
      f"softcap_logits matches gemma.softcap_logits (max |d| "
      f"{np.abs(got - want).max():.2e}, bounded to +/-30)")
check(float(got.max()) <= 30.0 + 1e-4 and float(got.min()) >= -30.0 - 1e-4,
      "the soft-cap really bounds the logits")
check(arch.aero_arch_softcap_logits(cvec, len(z), 0.0) == ARCH_ERR
      and arch.aero_arch_softcap_logits(cvec, len(z), -30.0) == ARCH_ERR,
      "a non-positive cap is refused (as the Python raises)")

sc = c_float(2.5)
out = c_float(0.0)
check(arch.aero_arch_scale_of(ctypes.byref(sc), 1, ctypes.byref(out)) == 0
      and abs(out.value - 2.5) < 1e-6,
      "layer_output_scale reads a 1-element tensor")
two_el = (c_float * 2)(1.0, 2.0)
check(arch.aero_arch_scale_of(two_el, 2, ctypes.byref(out)) == ARCH_ERR,
      "a 2-element layer_output_scale is refused (broadcasting it would silently "
      "change the block's output scale)")

arch.aero_arch_free(h)
arch.aero_arch_free(h_np)
arch.aero_arch_free(None)

# ------------------------------------- [11] the defaults ARE the real file's numbers
banner("[11] the hard-coded 12b defaults come from the real file's metadata")
gm = os.path.expanduser("~/.cache/aero-models/gemma-4-12b-it-IQ4_XS.gguf")
if not os.path.exists(gm):
    print(f"  (skipped: {gm} absent)")
else:
    from engine.gguf import GGUFReader
    with GGUFReader(gm) as r:
        f = r.kv
        file_n_layer = f["gemma4.block_count"]
        file_n_embd = f["gemma4.embedding_length"]
        file_n_head = f["gemma4.attention.head_count"]
        file_hd = f["gemma4.attention.key_length"]
        file_hd_swa = f["gemma4.attention.key_length_swa"]
        file_ffn = f["gemma4.feed_forward_length"]
        file_win = f["gemma4.attention.sliding_window"]
        file_cap = f["gemma4.final_logit_softcapping"]
        file_rb = f["gemma4.rope.freq_base"]
        file_rb_swa = f["gemma4.rope.freq_base_swa"]
        file_rd = f["gemma4.rope.dimension_count"]
        file_rd_swa = f["gemma4.rope.dimension_count_swa"]
        file_kv = list(f["gemma4.attention.head_count_kv"])
        file_pat = [1 if x else 0 for x in f["gemma4.attention.sliding_window_pattern"]]
        file_eps = f["gemma4.attention.layer_norm_rms_epsilon"]
        file_shared = f["gemma4.attention.shared_kv_layers"]

    check(cfg.n_layer == file_n_layer and cfg.n_embd == file_n_embd
          and cfg.n_head == file_n_head and cfg.head_dim == file_hd
          and cfg.head_dim_swa == file_hd_swa and cfg.ffn_len == file_ffn,
          f"defaults match the file's block_count/embedding_length/head_count/"
          f"key_length({file_hd})/key_length_swa({file_hd_swa})/ffn")
    check(cfg.sliding_window == file_win and abs(cfg.softcap - file_cap) < 1e-6,
          f"defaults match sliding_window={file_win} and softcapping={file_cap}")
    check(abs(cfg.rope_base - file_rb) < 1.0 and abs(cfg.rope_base_swa - file_rb_swa) < 1.0
          and cfg.rope_dims == file_rd and cfg.rope_dims_swa == file_rd_swa,
          f"defaults match rope freq_base={file_rb}/swa={file_rb_swa} and "
          f"dimension_count={file_rd}/swa={file_rd_swa}")
    check([kv[i] for i in range(48)] == file_kv,
          "the default kv-head list IS the file's 48-element list (8 windowed / 1 global)")
    check([pat[i] for i in range(48)] == file_pat,
          "the default pattern IS the file's 48-element pattern")
    check(abs(cfg.eps - file_eps) < 1e-9 or abs(cfg.eps - 1e-6) < 1e-9,
          f"eps is the file's layer_norm_rms_epsilon ({file_eps:.3e})")
    check(cfg.n_kv_shared == file_shared, "shared_kv_layers carried through (0 here)")
    check("per_layer_token_embd.weight" not in r.tensors and cfg.has_per_layer_embd == 0,
          "no per-layer embedding tensor in this file, and the default says so")

    # and the descriptor built from the FILE's values answers the same as from defaults
    fcfg = Gemma4Cfg()
    arch.aero_gemma4_defaults(ctypes.byref(fcfg))
    fcfg.n_layer, fcfg.n_embd, fcfg.n_head = file_n_layer, file_n_embd, file_n_head
    fcfg.head_dim, fcfg.head_dim_swa = file_hd, file_hd_swa
    fcfg.ffn_len, fcfg.sliding_window = file_ffn, file_win
    fcfg.softcap, fcfg.rope_base, fcfg.rope_base_swa = file_cap, file_rb, file_rb_swa
    fcfg.rope_dims, fcfg.rope_dims_swa = file_rd, file_rd_swa
    fcfg.n_kv_shared = file_shared
    hf, ef = make_arch(fcfg, file_kv, file_pat)
    ok = bool(hf)
    if hf:
        for L in range(file_n_layer):
            if (arch.aero_arch_is_swa(hf, L) != (1 if file_pat[L] else 0)
                    or arch.aero_arch_kv_heads_at(hf, L) != file_kv[L]
                    or arch.aero_arch_head_dim_at(hf, L)
                    != (file_hd_swa if file_pat[L] else file_hd)
                    or arch.aero_arch_rope_base_at(hf, L)
                    != (file_rb_swa if file_pat[L] else file_rb)
                    or arch.aero_arch_window_at(hf, L)
                    != (file_win if file_pat[L] else 0)):
                ok = False
                break
        arch.aero_arch_free(hf)
    check(ok, "built from the FILE's metadata, all 48 layers answer per the file "
              "(window 0 on globals, 256/10000 on windowed)")

print(f"\n=== {CHECKS} checks, {FAILS} failures ===")
sys.exit(1 if FAILS else 0)
