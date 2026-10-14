"""Chat-envelope rendering, checked against each family's own GGUF template.

The renderer is pure string assembly given a tokenizer's special-token table, so
most of this runs against a stub with no model file at all. The real-file section
at the end is skipped when the model is absent, so the gate stays honest on a
machine that does not have it.

What is being pinned down is not cosmetics. Each of these was a real bug:
  * the old renderer sniffed only for '<|im_start|>', so on gemma4 it emitted
    'user\\n...\\n' with NO turn markers -- silently unframed input;
  * gemma4 spells the assistant role 'model';
  * gemma4's template emits <bos> itself, so the tokenizer must not add a second;
  * gemma4's thinking flag is inverted relative to intuition (thinking DISABLED
    emits an immediately-closed thought channel, which is what forces an answer).
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.chat_template import EnvelopeError, detect_envelope, render_chat

_fail = 0
_n = 0


def check(cond, desc):
    global _fail, _n
    _n += 1
    if cond:
        print("  ok   " + desc)
    else:
        _fail += 1
        print("  FAIL " + desc)


class StubTok:
    """The only thing the renderer is allowed to depend on."""

    def __init__(self, special):
        self._special = dict(special)


GEMMA = StubTok({"<bos>": 2, "<|turn>": 105, "<turn|>": 106,
                 "<|channel>": 100, "<channel|>": 101, "<|think|>": 98})
QWEN = StubTok({"<bos>": 1, "<|im_start|>": 248045, "<|im_end|>": 248046})
NONE = StubTok({})

USER = [{"role": "user", "content": "Hi"}]

print("\n[1] envelope detection is driven by the file's own tokens")
check(detect_envelope(GEMMA) == "gemma4", "gemma4 detected from '<|turn>'")
check(detect_envelope(QWEN) == "chatml", "chatml detected from '<|im_start|>'")
check(detect_envelope(NONE) is None, "unknown tokenizer detects as None")

print("\n[2] gemma4 envelope is transcribed from the file's template")
got = render_chat(USER, GEMMA, enable_thinking=False)
check(got == "<bos><|turn>user\nHi<turn|>\n<|turn>model\n"
             "<|channel>thought\n<channel|>",
      "single user turn + gen prompt + closed thought channel")
check(got.startswith("<bos>"), "template emits <bos> itself")
check("<|turn>model\n" in got, "assistant role is spelled 'model', not 'assistant'")
check(got.index("<|turn>user") == len("<bos>"), "<bos> precedes the first turn")

print("\n[3] the thinking flag is inverted, and that is deliberate")
off = render_chat(USER, GEMMA, enable_thinking=False)
on = render_chat(USER, GEMMA, enable_thinking=True)
check("<|channel>thought\n<channel|>" in off,
      "thinking OFF emits an immediately-closed channel (forces a direct answer)")
check("<|channel>" not in on,
      "thinking ON emits no channel, leaving the model inside it")

print("\n[4] multi-turn, and add_generation_prompt=False")
two = render_chat(USER + [{"role": "assistant", "content": "Hello"}], GEMMA,
                  add_generation_prompt=False)
check(two == "<bos><|turn>user\nHi<turn|>\n<|turn>model\nHello<turn|>\n",
      "assistant history renders as a 'model' turn, no gen prompt")
check(render_chat(USER, GEMMA, add_generation_prompt=False)
      == "<bos><|turn>user\nHi<turn|>\n", "no gen prompt -> prompt closes after user")

print("\n[5] chatml envelope (qwen) still renders as before")
c = render_chat(USER, QWEN, enable_thinking=False)
check(c == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
          " thinking\n\n</think>\n\n",
      "chatml user turn + assistant turn with closed thinking block")
check(not c.startswith("<bos>"),
      "chatml template carries no <bos> (its file says add_bos_token=false)")

print("\n[6] an unrecognised template refuses rather than emitting unframed text")
try:
    render_chat(USER, NONE)
    check(False, "unknown template raises")
except EnvelopeError as e:
    check("unframed" in str(e) or "unrecognised" in str(e),
          "unknown template raises EnvelopeError with a reason")
try:
    render_chat(USER, None)
    check(False, "no tokenizer raises")
except EnvelopeError:
    check(True, "no tokenizer raises EnvelopeError")

print("\n[7] against the real files, when present")
for label, path, want in (
        ("gemma4", "/Users/cyberdeth/.cache/aero-models/gemma-4-12b-it-IQ4_XS.gguf",
         "gemma4"),
        ("qwen35", "/Users/cyberdeth/.cache/aero-models/Qwen3.8-4B-Q4_K_M.gguf",
         "chatml")):
    if not os.path.exists(path):
        print(f"  (skipped {label}: {path} absent)")
        continue
    from engine.gguf import GGUFReader
    from engine.tokenizer import Tokenizer
    with GGUFReader(path) as r:
        tk = Tokenizer.from_gguf(r)
    check(detect_envelope(tk) == want, f"{label}: real file detects as {want}")
    ids = tk.encode(render_chat(USER, tk, enable_thinking=False))
    run = 0
    while run < len(ids) and ids[run] == tk.bos_id:
        run += 1
    check(run <= 1, f"{label}: rendered prompt has at most one leading BOS "
                    f"(got {run})")
    check(ids.count(tk.bos_id) <= 1 if tk.bos_id is not None else True,
          f"{label}: <bos> is not doubled anywhere (encode() guards it)")
    if want == "gemma4":
        check(ids[0] == tk.bos_id, "gemma4: rendered prompt starts with <bos>")
        check(tk.encode("<bos>Hi", add_bos=True)[0] == tk.bos_id
              and tk.encode("<bos>Hi", add_bos=True)[1] != tk.bos_id,
              "gemma4: an explicit leading <bos> is not doubled by encode()")

print(f"\n=== {_n} checks, {_fail} failures ===")
sys.exit(1 if _fail else 0)
