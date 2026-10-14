"""The bulk-read path must be one syscall, and must verify its own length.

Two separate bugs are pinned here.

(1) PERFORMANCE, measured not assumed: reading a tensor through the buffered
    handle's seek+read ran at ~854 MB/s on the gemma-4 file, while os.pread of the
    identical bytes at the identical offset ran at ~5480 MB/s -- 6.4x, same disk,
    same page cache. A decode reads ~5.9 GB/token, so this was ~6.8 s of every
    token. This test asserts the reader actually routes bulk reads through pread,
    because a silent regression here costs seconds per token and looks like nothing
    at all.

(2) CORRECTNESS: a short read must raise, not return a short buffer. If pread
    returns fewer bytes than asked (truncated file, or a bad offset) and the caller
    does not check, the dequantiser reads past the buffer and silently produces
    wrong numbers.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.gguf import GGUFReader

REAL = "/Users/cyberdeth/.cache/aero-models/gemma-4-12b-it-IQ4_XS.gguf"

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


print("\n[1] the reader exposes a one-syscall bulk read")
if not os.path.exists(REAL):
    print(f"  (skipped: {REAL} absent)")
else:
    with GGUFReader(REAL) as r:
        check(hasattr(r, "_pread"), "GGUFReader has _pread")
        check(getattr(r, "_fd", None) is not None,
              "a raw descriptor is open for pread (not just the buffered handle)")
        name = next(n for n, i in r.tensors.items() if i.n_elements > 1_000_000)
        info = r.tensors[name]
        nbytes = r.row_bytes(info) * info.shape[0]
        via_pread = r._pread(nbytes, r.data_offset + info.offset)
        via_buffered = r.tensor_bytes(info)
        check(len(via_pread) == nbytes, f"_pread returned the full {nbytes} bytes")
        check(via_pread == via_buffered,
              "pread and the buffered path return IDENTICAL bytes")

        print("\n[2] tensor_bytes routes through pread (not the buffered handle)")
        calls = {"n": 0}
        real_fd = r._fd
        real_pread = os.pread
        def counting_pread(fd, n, off):
            calls["n"] += 1
            return real_pread(fd, n, off)
        os.pread = counting_pread
        try:
            r.tensor_bytes(info)
        finally:
            os.pread = real_pread
        check(calls["n"] == 1,
              f"one pread syscall per tensor read (saw {calls['n']})")

        print("\n[3] a short read raises instead of returning a short buffer")
        try:
            r._pread(nbytes, os.path.getsize(r.path) - 4)
            check(False, "reading past EOF raises")
        except EOFError:
            check(True, "reading past EOF raises EOFError")
        try:
            r._pread(10_000_000_000, 0)
            check(False, "absurd length raises")
        except (EOFError, OSError, OverflowError):
            check(True, "absurd length raises")

        print("\n[4] tensor_rows uses pread too (the token-embedding path)")
        rows_info = r.tensors["token_embd.weight"] if "token_embd.weight" in r.tensors else info
        calls["n"] = 0
        os.pread = counting_pread
        try:
            r.tensor_rows(rows_info, 0, 4)
        finally:
            os.pread = real_pread
        check(calls["n"] == 1, f"one pread for a row slice (saw {calls['n']})")

        print("\n[5] the descriptor is closed with the reader")
        fd = r._fd
    try:
        os.fstat(fd)
        check(False, "descriptor closed on __exit__")
    except OSError:
        check(True, "descriptor closed on __exit__")

print(f"\n=== {_n} checks, {_fail} failures ===")
sys.exit(1 if _fail else 0)
