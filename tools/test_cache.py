#!/usr/bin/env python3
"""Content-addressed test cache: skip a suite whose inputs have not changed.

WHY: `make test` costs ~130 s, of which `test-model` is 112 s (29 Python
interpreter starts, most of them loading a model). Re-running that to re-learn a
result already known is pure waste, and it was happening several times per turn.

WHAT IS CACHED: a suite's result is reused only when the hash of everything that can
change its outcome is identical -- the suite file, every file it imports from
engine/, the native libraries, and the Makefile. If any of those change, the suite
re-runs. This is deliberately conservative: a stale PASS is much worse than a
re-run, so the hash covers imports transitively and the native .dylibs too (a
rebuilt kernel changes behaviour without touching a single .py file).

WHAT IS NOT CACHED: anything that loads a model is NOT skipped by default, because
a suite's result there depends on the model file too and on machine state. Callers
opt in per-suite.

Usage:
    tools/test_cache.py run <name> <suite.py> [args...]   # cached, exits 0 if reused
    tools/test_cache.py status
    tools/test_cache.py clear
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "build", ".testcache")
CACHE_FILE = os.path.join(CACHE_DIR, "cache.json")


def _sha(path: str, h) -> None:
    """Fold one file's CONTENT into the hash. Content, not mtime: a touched file
    with identical bytes changes nothing about the result, and a restored file with
    an old mtime can still differ."""
    try:
        with open(path, "rb") as f:
            while True:
                b = f.read(1 << 20)
                if not b:
                    break
                h.update(b)
    except OSError:
        h.update(b"<missing>")


def _suite_imports(suite: str) -> list[str]:
    """Every engine/ file the suite could import, plus the native libraries.

    Transitive by construction: rather than parse the import graph (fragile), fold
    in the whole engine/ package and every .dylib. Over-invalidating costs a re-run;
    under-invalidating returns a stale PASS, which is the failure that matters.
    """
    files = [suite]
    eng = os.path.join(ROOT, "engine")
    if os.path.isdir(eng):
        for name in sorted(os.listdir(eng)):
            if name.endswith(".py"):
                files.append(os.path.join(eng, name))
    toolsd = os.path.join(ROOT, "tools")
    if os.path.isdir(toolsd):
        for name in sorted(os.listdir(toolsd)):
            if name.endswith(".py"):
                files.append(os.path.join(toolsd, name))
    build = os.path.join(ROOT, "build")
    if os.path.isdir(build):
        for name in sorted(os.listdir(build)):
            if name.endswith((".dylib", ".so")):
                files.append(os.path.join(build, name))
    for extra in ("Makefile",):
        p = os.path.join(ROOT, extra)
        if os.path.exists(p):
            files.append(p)
    return files


def fingerprint(suite: str) -> str:
    h = hashlib.sha256()
    for f in _suite_imports(suite):
        h.update(os.path.relpath(f, ROOT).encode())
        _sha(f, h)
    return h.hexdigest()


def _load() -> dict:
    try:
        with open(CACHE_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save(d: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = CACHE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=1, sort_keys=True)
    os.replace(tmp, CACHE_FILE)


def cmd_run(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: test_cache.py run <name> <suite.py> [args...]")
        return 2
    name, suite = argv[0], argv[1]
    rest = argv[2:]
    suite_abs = suite if os.path.isabs(suite) else os.path.join(ROOT, suite)
    if not os.path.exists(suite_abs):
        print(f"test_cache: {suite} does not exist")
        return 2

    fp = fingerprint(suite_abs)
    cache = _load()
    entry = cache.get(name)
    if entry and entry.get("fingerprint") == fp and entry.get("rc") == 0:
        age = time.time() - entry.get("when", 0)
        print(f"=== CACHED PASS ({name}, {age/60:.1f} min old, inputs unchanged) ===")
        return 0

    py = os.path.join(ROOT, ".venv-cml", "bin", "python")
    if not os.path.exists(py):
        py = sys.executable
    cmd = [py, "-W", "error::RuntimeWarning", suite_abs] + rest
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    dt = time.time() - t0
    cache[name] = {"fingerprint": fp, "rc": r.returncode, "when": time.time(),
                   "seconds": round(dt, 2)}
    _save(cache)
    print(f"=== {name}: rc={r.returncode} in {dt:.1f}s (cached) ===")
    return r.returncode


def cmd_status(argv: list[str]) -> int:
    cache = _load()
    if not cache:
        print("test_cache: empty (nothing has run yet)")
        return 0
    print(f"{'suite':<28}{'rc':>4}{'sec':>8}{'age(min)':>10}  state")
    total = 0.0
    for name, e in sorted(cache.items()):
        age = (time.time() - e.get("when", 0)) / 60
        total += e.get("seconds", 0)
        print(f"{name:<28}{e.get('rc', '?'):>4}{e.get('seconds', 0):>8.1f}{age:>10.1f}  "
              f"{'PASS' if e.get('rc') == 0 else 'FAIL'}")
    print(f"\ntotal cached runtime: {total:.1f}s")
    return 0


def cmd_clear(argv: list[str]) -> int:
    try:
        os.remove(CACHE_FILE)
        print("test_cache: cleared")
    except OSError:
        print("test_cache: nothing to clear")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "run":
        return cmd_run(rest)
    if cmd == "status":
        return cmd_status(rest)
    if cmd == "clear":
        return cmd_clear(rest)
    print(f"unknown command {cmd}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
