"""Shared test helpers.

ONE place that knows how to react to the model-load lock. Patching each load site
individually is fragile -- test_kv alone has eleven -- and a suite that crashes on
the lock is indistinguishable from one that failed, while a suite that silently
does not run looks exactly like one that passed. Both are worse than a skip.
"""

from __future__ import annotations

import os
import tempfile

_LOCK_PATH = os.path.join(tempfile.gettempdir(), "aero-model-load.lock")


def load_lock_free() -> bool:
    """True when no other aero process holds the model-load lock.

    Probes by taking and immediately releasing the lock. There is an unavoidable
    race between this probe and the actual load, so callers must ALSO handle the
    refusal at the load site -- `skip_if_locked()` below does both.
    """
    try:
        import fcntl
        fd = os.open(_LOCK_PATH, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            return True
        except OSError:
            return False
        finally:
            os.close(fd)
    except Exception:
        return True


def is_lock_refusal(exc: BaseException) -> bool:
    """True when `exc` is the guard refusing a second concurrent model load."""
    return "already holds the model-load lock" in str(exc)


class LockedOut(Exception):
    """Raised by `load_model()` when another process holds the lock."""


def load_model(reader, *, skip_banner: bool = True, **kw):
    """`Model(reader, **kw)` but a lock refusal raises LockedOut instead.

    Every test that loads a model should go through this so the whole suite can
    report one clear skip rather than a traceback per call site.
    """
    from engine.model import Model
    try:
        return Model(reader, **kw)
    except RuntimeError as e:
        if is_lock_refusal(e):
            raise LockedOut() from None
        raise


def skip_note(where: str = "") -> str:
    tail = f" ({where})" if where else ""
    return (f"\n=== SKIPPED{tail}: another aero process holds the model-load "
            f"lock. One load at a time is deliberate -- two concurrent loads "
            f"OOMed a 16 GB host. ===\n")
