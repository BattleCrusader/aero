"""Device allocation for Aero.

Flags take a COUNT, bounded by each device's limit:

    --use-cpu=0..CPU_CORE_LIMIT     default 0
    --use-gpu=0..GPU_CORE_LIMIT     default 1   <-- the project default
    --use-ane=0..ANE_CORE_LIMIT     default 0

A flag you name overrides only itself, so all combinations are expressible:

    (nothing)                        -> gpu=1
    --use-cpu=4                      -> cpu=4 + gpu=1
    --use-gpu=0 --use-cpu=8          -> cpu=8
    --use-ane=16 --use-gpu=0         -> ane=16
    --use-cpu=4 --use-gpu=8 --use-ane=16 -> all three
    --use-cpu=0 --use-gpu=0 --use-ane=0  -> ERROR (nothing to run on)

MEANING OF THE COUNT (see limits.py for the full honesty note):
  * CPU  -- real: N worker streams run on N cores.
  * GPU  -- N parallel streams on the single GPU execution resource, with a
            proportional share of the work. NOT a hardware core pin; Apple
            exposes no such control.
  * ANE  -- as GPU, and its core count is a documented constant, not detected.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterator, Optional, Tuple

from .limits import CoreLimit, all_limits

ALL_DEVICES: Tuple["Device", ...] = ()


class Device(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    ANE = "ane"

    def coreml_units(self):
        """The CoreML compute-unit selector for this device."""
        import coremltools as ct

        return {
            Device.CPU: ct.ComputeUnit.CPU_ONLY,
            Device.GPU: ct.ComputeUnit.CPU_AND_GPU,
            Device.ANE: ct.ComputeUnit.CPU_AND_NE,
        }[self]

    @property
    def flag(self) -> str:
        return f"--use-{self.value}"

    @property
    def label(self) -> str:
        return {"cpu": "CPU", "gpu": "GPU (Metal)", "ane": "ANE (Neural Engine)"}[self.value]


ALL_DEVICES = (Device.CPU, Device.GPU, Device.ANE)

#: Per-flag defaults. gpu=1 is the project default; the others are opt-in.
DEFAULT_COUNTS: Dict[str, int] = {"cpu": 0, "gpu": 1, "ane": 0}

#: Devices the blocked matmul can ACTUALLY compute on. The C++ block loop
#: (aero::matmul) is CPU-only -- there is no Metal or ANE kernel behind it -- so a GPU
#: or ANE count is a RECORDED request, never a silent compute stream. Splitting layers
#: across units was measured SLOWER on this machine (CPU+GPU 0.56x, GPU+ANE 0.89x;
#: bus-limited, not unit-limited), so wiring a split would be a regression dressed as
#: a feature.
COMPUTE_CAPABLE: Tuple[Device, ...] = (Device.CPU,)


class NoDevicesSelected(ValueError):
    """Every device count was zero -- nothing to run on."""


class DeviceCountOutOfRange(ValueError):
    """A count fell outside 0..limit for its device."""


@dataclass(frozen=True)
class DevicePlan:
    """Per-device stream counts. Only devices with count > 0 are retained."""

    counts: Tuple[Tuple[Device, int], ...]

    @property
    def devices(self) -> Tuple[Device, ...]:
        return tuple(d for d, _ in self.counts)

    def count(self, device: Device) -> int:
        for d, n in self.counts:
            if d is device:
                return n
        return 0

    def __contains__(self, device: object) -> bool:
        return device in self.devices

    def __iter__(self) -> Iterator[Tuple[Device, int]]:
        return iter(self.counts)

    def __len__(self) -> int:
        return len(self.counts)

    @property
    def total_streams(self) -> int:
        return sum(n for _, n in self.counts)

    def share(self, device: Device) -> float:
        """Fraction of the work this device should be given, by stream count."""
        t = self.total_streams
        return (self.count(device) / t) if t else 0.0

    @property
    def compute_streams(self) -> int:
        """Streams that can actually compute -- see COMPUTE_CAPABLE."""
        return sum(n for d, n in self.counts if d in COMPUTE_CAPABLE)

    def recorded_only(self) -> Tuple[Tuple[Device, int], ...]:
        """Plan entries with no compute kernel behind them, in plan order."""
        return tuple((d, n) for d, n in self.counts if d not in COMPUTE_CAPABLE)

    def notice(self) -> Optional[str]:
        """One line naming what is recorded but cannot compute; None when clean.

        A flag that is accepted and ignored is worse than one that errors: it makes
        the user believe they changed something. So the truth is stated, not implied.
        """
        ro = self.recorded_only()
        if not ro:
            return None
        asked = ", ".join(f"--use-{d.value}={n}" for d, n in ro)
        names = "/".join(d.value.upper() for d, _ in ro)
        # The CPU clause must never read as zero: with the default plan (cpu=0) there
        # are no CPU *streams* but the block loop still computes, on the hardware
        # thread count. "Compute runs on 0 CPU stream(s)" was the same class of
        # misleading statement this notice exists to prevent.
        where = (f"compute runs on {self.compute_streams} CPU stream(s)"
                 if self.compute_streams else
                 "compute runs on the CPU block loop")
        return (f"{asked} is recorded but not driving compute: the block loop has no "
                f"{names} kernel (cross-unit splitting measured 0.56x-0.89x here). "
                f"{where}.")

    def thread_budget(self, hardware: int) -> int:
        """Threads the C++ block loop should get.

        An explicit --use-cpu=N is honoured exactly. Left at the default of 0, the
        loop gets the machine's hardware thread count: the CPU count is the only flag
        the compute path can act on, so collapsing it to 1 (which `rt.threads or 1`
        did) made the flag meaningless and left every core but one idle.
        """
        n = self.count(Device.CPU)
        return n if n > 0 else max(1, int(hardware))

    def __str__(self) -> str:
        return "+".join(f"{d.value}={n}" for d, n in self.counts)


def parse_device_alloc(use_cpu: Optional[int] = None,
                       use_gpu: Optional[int] = None,
                       use_ane: Optional[int] = None,
                       limits: Optional[Dict[str, CoreLimit]] = None) -> DevicePlan:
    """Resolve the three count flags into a DevicePlan.

    `None` means "not specified" -> that flag keeps its default.
    A specified count is validated against 0..limit for that device.
    All three resolving to 0 raises NoDevicesSelected.
    """
    lim = limits or all_limits()
    given = {"cpu": use_cpu, "gpu": use_gpu, "ane": use_ane}

    resolved: Dict[str, int] = {}
    for name, value in given.items():
        n = DEFAULT_COUNTS[name] if value is None else int(value)
        upper = lim[name].limit
        if n < 0 or n > upper:
            raise DeviceCountOutOfRange(
                f"--use-{name}={n} is outside the valid range 0..{upper} "
                f"({lim[name].detail})"
            )
        resolved[name] = n

    if sum(resolved.values()) == 0:
        raise NoDevicesSelected(
            "every device count is 0 -- nothing to run on. "
            "Enable at least one of --use-cpu=N --use-gpu=N --use-ane=N"
        )

    counts = tuple((d, resolved[d.value]) for d in ALL_DEVICES if resolved[d.value] > 0)
    return DevicePlan(counts)


def all_combinations(limits: Optional[Dict[str, CoreLimit]] = None):
    """Every non-empty device subset at its maximum count -- 'all combinations'."""
    lim = limits or all_limits()
    out = []
    for mask in range(1, 1 << len(ALL_DEVICES)):
        kw = {}
        for i, d in enumerate(ALL_DEVICES):
            if mask & (1 << i):
                kw[f"use_{d.value}"] = lim[d.value].limit
        out.append(parse_device_alloc(**kw, limits=lim))
    return out
