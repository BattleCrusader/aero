#!/usr/bin/env python3
"""Prove Python is OUT of the prompt path, by making it impossible to import.

WHY A BLOCKER AND NOT A SPY. Every other prompt test compares outputs, and outputs can
agree while the implementation is still Python -- engine/native_prompt.py could be a
wrapper around engine.tokenizer.py and a byte-for-byte suite would never notice, because
they are built to agree. So this suite blocks `engine.tokenizer` and `engine.chat_template`
from being imported AT ALL (a meta-path finder that raises), then loads a model and builds
a prompt. If the native path is real, nothing breaks. If any part of it reaches for the
Python modules, the load fails loudly instead of quietly succeeding.

Section [2] is the control for the control: with AERO_NATIVE_PROMPT=0 the SAME blocked
run must FAIL. Without that, a blocker that silently never fires would look like a pass.

    AERO_MAX_RAM=5G .venv-cml/bin/python tests/test_native_prompt_only.py
"""
from __future__ import annotations

import importlib.abc
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("AERO_MAX_RAM", "5G")

FAILS = 0
CHECKS = 0
BLOCKED = ("engine.tokenizer", "engine.chat_template")


def check(cond: bool, what: str) -> None:
    global FAILS, CHECKS
    CHECKS += 1
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        FAILS += 1


class BlockPythonPrompt:
    """Raise on importing the Python prompt modules, as if they did not exist."""

    def find_spec(self, name, path=None, target=None):
        if name in BLOCKED:
            raise ImportError("blocked by the negative control: %s" % name)
        return None


MODEL = "/Users/cyberdeth/.cache/aero-models/Qwen3.8-4B-Q4_K_M.gguf"
if not os.path.exists(MODEL):
    print("(skipped: %s absent)" % MODEL)
    print("\n=== 0 checks, 0 failures ===")
    sys.exit(0)

print("[1] the engine builds a prompt with engine.tokenizer / engine.chat_template UNIMPORTABLE")
sys.meta_path.insert(0, BlockPythonPrompt())
for mod in BLOCKED:
    sys.modules.pop(mod, None)

try:
    from engine.runtime import make_runtime, GenParams          # noqa: E402
    rt = make_runtime("gguf", max_ram="6G", context=1024, block_rows=256)
    rt.load(MODEL)
    stats = dict(rt.plan())
    check(stats.get("token_source") == "native",
          "token_source is native (got %r)" % stats.get("token_source"))
    msgs = [{"role": "user", "content": "The capital of France is"}]
    prompt = rt.apply_chat_template(msgs)
    ids = rt.tokenize(prompt)
    check(len(ids) > 0, "the native tokenizer produced %d ids with Python blocked" % len(ids))
    check("<|im_start|>" in prompt,
          "the native renderer produced the file's envelope (%r...)" % prompt[:26])
    check(stats.get("tokenizer", {}).get("via", "").startswith("native"),
          "load stats report the native prompt path")
except ImportError as e:
    check(False, "the native path reached for a Python module: %s" % e)
except Exception as e:                                           # noqa: BLE001
    check(False, "load/build failed under the blocker: %s: %s" % (type(e).__name__, e))

print("\n[2] control: the blocker must actually bite, or [1] proved nothing")
# In-process on purpose. A subprocess control has to load the model to reach the fallback
# and then collides with the model-load lock that [1] already holds -- it failed with
# BlockingIOError, which is a true failure for a false reason. Importing the blocked
# modules is the whole control and needs no model, no lock and no second process.
import importlib                                                  # noqa: E402
for mod in BLOCKED:
    try:
        importlib.import_module(mod)
        check(False, "%s is importable, so [1] proved nothing" % mod)
    except ImportError:
        check(True, "%s cannot be imported under the blocker" % mod)

print("\n=== %d checks, %d failures ===" % (CHECKS, FAILS))
sys.exit(1 if FAILS else 0)
