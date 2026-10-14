"""Device core limits for Aero.

Each --use-<device> flag takes an integer 0..LIMIT. This module discovers the
limits on the running machine.

HONESTY NOTE -- what a count can and cannot mean:

  CPU : the count is REAL. N streams run on N cores; the thread pool bounds it.
        Detection: sysctl hw.perflevel0.logicalcpu (performance cores).

  GPU : Apple exposes NO API to use "N of M GPU cores". Metal and CoreML present
        the GPU as a single execution resource. So the count is implemented as
        N parallel worker streams on that one device plus a proportional share
        of the work -- NOT a hardware core pin. Measured reality (see
        PERFORMANCE_BUDGET.md): N>1 streams on one unit contend rather than add.

  ANE : same limitation, and worse -- no public API even reports the core count.
        ANE_CORE_LIMIT comes from Apple's published per-chip figures, is
        labelled `detected=False`, and defaults to 16 when the chip is unknown.
"""

from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import dataclass


#: macOS thread QoS classes -- the real mechanism for P-core vs E-core preference.
#: Threads marked USER_INITIATED are scheduled on performance cores; UTILITY is
#: biased to efficiency cores. This is how the P-then-E allocation is expressed.
QOS_CLASS_USER_INITIATED = 0x19
QOS_CLASS_UTILITY = 0x11


def set_thread_qos(qos_class: int) -> bool:
    """Apply a QoS class to the CALLING thread. Returns True on success.

    This is the documented way to express 'prefer a performance core' vs
    'prefer an efficiency core' on Apple Silicon; there is no CPU-affinity API.
    """
    try:
        import ctypes

        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        fn = libc.pthread_set_qos_class_self_np
        fn.argtypes = [ctypes.c_uint, ctypes.c_int]
        fn.restype = ctypes.c_int
        return fn(ctypes.c_uint(qos_class), 0) == 0
    except Exception:  # noqa: BLE001
        return False


@dataclass(frozen=True)
class CpuTopology:
    performance: int
    efficiency: int

    @property
    def total(self) -> int:
        return self.performance + self.efficiency

    def split(self, n: int) -> Tuple[int, int]:
        """Allocate n cores PERFORMANCE-FIRST, then append efficiency cores.

        n <= performance      -> (n, 0)             all performance
        performance < n < all -> (performance, n-p) perf full, remainder on efficiency
        n >= total            -> (performance, efficiency)
        """
        n = max(0, min(int(n), self.total))
        n_p = min(n, self.performance)
        n_e = max(0, n - n_p)
        return n_p, n_e

    def describe(self, n: int) -> str:
        n_p, n_e = self.split(n)
        parts = []
        if n_p:
            parts.append(f"{n_p} performance")
        if n_e:
            parts.append(f"{n_e} efficiency")
        return " + ".join(parts) if parts else "none"


def cpu_topology() -> CpuTopology:
    p = _sysctl("hw.perflevel0.logicalcpu")
    e = _sysctl("hw.perflevel1.logicalcpu")
    total = _sysctl("hw.ncpu")
    if p > 0 and (e > 0 or p == total):
        return CpuTopology(p, e)
    if total > 0:
        return CpuTopology(total, 0)   # no perflevel split available
    return CpuTopology(1, 0)


@dataclass(frozen=True)
class CoreLimit:
    device: str          # "cpu" | "gpu" | "ane"
    limit: int           # inclusive upper bound for --use-<device>
    detected: bool       # True = measured on this machine, False = documented constant
    detail: str          # human-readable provenance

    def clamp(self, n: int) -> int:
        return max(0, min(int(n), self.limit))


def _sysctl(name: str) -> int:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True,
                             text=True, timeout=5, check=False)
        return int(out.stdout.strip())
    except Exception:  # noqa: BLE001
        return 0


def cpu_limit() -> CoreLimit:
    """TOTAL logical cores -- the allocation order (P first, then E) is handled
    by CpuTopology.split(), not by shrinking the limit to the P-core count."""
    topo = cpu_topology()
    if topo.total > 0:
        return CoreLimit("cpu", topo.total, True,
                         f"sysctl: {topo.performance} performance + "
                         f"{topo.efficiency} efficiency = {topo.total} logical cores "
                         f"(allocated performance-first)")
    return CoreLimit("cpu", 1, False, "detection failed; assuming 1")


def gpu_limit() -> CoreLimit:
    """GPU core count, from the system profiler (e.g. 'Total Number of Cores: 8')."""
    try:
        out = subprocess.run(["system_profiler", "SPDisplaysDataType"],
                             capture_output=True, text=True, timeout=20, check=False)
        m = re.search(r"Total Number of Cores:\s*(\d+)", out.stdout)
        if m:
            n = int(m.group(1))
            return CoreLimit("gpu", n, True,
                             f"system_profiler SPDisplaysDataType: {n} GPU cores")
    except Exception:  # noqa: BLE001
        pass
    return CoreLimit("gpu", 1, False, "GPU core count not reported; assuming 1")


#: Apple's published ANE core counts. There is no API for this.
_ANE_CORES = {
    "Apple M1": 16, "Apple M1 Pro": 16, "Apple M1 Max": 16, "Apple M1 Ultra": 32,
    "Apple M2": 16, "Apple M2 Pro": 16, "Apple M2 Max": 16, "Apple M2 Ultra": 32,
    "Apple M3": 16, "Apple M3 Pro": 16, "Apple M3 Max": 16, "Apple M3 Ultra": 32,
    "Apple M4": 16, "Apple M4 Pro": 16, "Apple M4 Max": 16,
}
_ANE_DEFAULT = 16


def _chip_name() -> str:
    """Correct chip name. On macOS platform.processor() returns just 'arm',
    so machdep.cpu.brand_string ('Apple M1') is the reliable source."""
    try:
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True, timeout=5, check=False)
        name = out.stdout.strip()
        if name:
            return name
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or ""


def ane_limit() -> CoreLimit:
    """ANE core count. NOT detectable -- CoreML exposes no such query."""
    chip = _chip_name()
    n = _ANE_CORES.get(chip.strip())
    if n is not None:
        return CoreLimit("ane", n, False,
                         f"documented constant for {chip.strip()} (not API-detected)")
    return CoreLimit("ane", _ANE_DEFAULT, False,
                     f"unrecognised chip {chip.strip()!r}; defaulting to {_ANE_DEFAULT} "
                     f"(not API-detected)")


def all_limits() -> dict:
    """Every device's limit, freshly detected."""
    return {"cpu": cpu_limit(), "gpu": gpu_limit(), "ane": ane_limit()}


def format_limits(limits: dict | None = None) -> str:
    limits = limits or all_limits()
    lines = []
    for name in ("cpu", "gpu", "ane"):
        L = limits[name]
        tag = "detected" if L.detected else "DOCUMENTED (not detected)"
        lines.append(f"  --use-{name}=0..{L.limit:<3d}  [{tag}] {L.detail}")
    return "\n".join(lines)
