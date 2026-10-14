"""Aero engine -- executes a weight matvec across the allocated devices.

The allocation is a DevicePlan (per-device stream counts). Work is partitioned
across the streams in proportion to the counts, every stream runs concurrently,
and each backend records how many inferences it actually performed -- so "was
device X really used?" is measured, not assumed.

CPU streams are allocated PERFORMANCE-FIRST: the first `performance` streams are
tagged QOS_CLASS_USER_INITIATED (P-core preference) and any remainder
QOS_CLASS_UTILITY (E-core preference), which is the only mechanism macOS offers
for expressing that preference.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .backends import BaseBackend, CoreMLBackend, NumpyBackend
from .devices import Device, DevicePlan, parse_device_alloc
from .limits import (QOS_CLASS_USER_INITIATED, QOS_CLASS_UTILITY, cpu_topology,
                     set_thread_qos)


@dataclass
class Stream:
    device: Device
    index: int          # which stream on that device (0-based)
    backend: BaseBackend
    row_start: int = 0
    row_end: int = 0
    core_class: str = ""   # "performance" | "efficiency" | "" (non-CPU)

    @property
    def n_rows(self) -> int:
        return self.row_end - self.row_start

    def __str__(self) -> str:
        extra = f" ({self.core_class})" if self.core_class else ""
        return f"{self.device.value}[{self.index}]{extra}"


@dataclass
class RunStats:
    seconds: float = 0.0
    calls_by_device: Dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.seconds * 1000:.1f} ms, calls={self.calls_by_device}"


class AeroEngine:
    """Weight matvec spread across the allocated devices."""

    def __init__(self, plan: DevicePlan, use_coreml: bool = True) -> None:
        self.plan = plan
        self.use_coreml = use_coreml
        self.streams: List[Stream] = []
        self._shapes: Optional[Tuple[int, int]] = None

    # ---- construction ----------------------------------------------------
    def _make_backend(self, device: Device, stream_index: int,
                      core_class: str) -> BaseBackend:
        if device is Device.CPU or not self.use_coreml:
            b = NumpyBackend(device)
            b.core_class = core_class  # type: ignore[attr-defined]
            return b
        return CoreMLBackend(device)

    def build(self, weights: np.ndarray) -> None:
        """Partition `weights` rows across the streams and load each backend."""
        w = np.asarray(weights)
        if w.ndim != 2:
            raise ValueError(f"weights must be 2-D, got shape {w.shape}")
        self._shapes = w.shape
        n_out = w.shape[0]

        topo = cpu_topology()
        n_cpu = self.plan.count(Device.CPU)
        n_perf, _n_eff = topo.split(n_cpu)

        # one stream per unit of count, per device
        self.streams = []
        for device, count in self.plan:
            for i in range(count):
                if device is Device.CPU:
                    core_class = "performance" if i < n_perf else "efficiency"
                else:
                    core_class = ""
                backend = self._make_backend(device, i, core_class)
                self.streams.append(Stream(device, i, backend, core_class=core_class))

        # partition rows proportionally to stream counts, in stream order
        counts = [1] * len(self.streams)
        splits = np.cumsum([0] + counts)
        bounds = np.linspace(0, n_out, len(self.streams) + 1).round().astype(int)
        for s, lo, hi in zip(self.streams, bounds[:-1], bounds[1:]):
            s.row_start, s.row_end = int(lo), int(hi)

        for s in self.streams:
            if s.n_rows > 0:
                s.backend.load(w[s.row_start:s.row_end])

    # ---- execution -------------------------------------------------------
    def _run_stream(self, s: Stream, x: np.ndarray) -> np.ndarray:
        if s.core_class == "performance":
            set_thread_qos(QOS_CLASS_USER_INITIATED)
        elif s.core_class == "efficiency":
            set_thread_qos(QOS_CLASS_UTILITY)
        return s.backend.run(x)

    def matvec(self, x: np.ndarray, concurrent: bool = True,
               repeat: int = 1) -> Tuple[np.ndarray, RunStats]:
        """y = W x, executed across every allocated stream."""
        if not self.streams:
            raise RuntimeError("engine not built")

        calls_before = {d.value: 0 for d in self.plan.devices}
        t0 = time.perf_counter()
        for _ in range(repeat):
            if concurrent and len(self.streams) > 1:
                with ThreadPoolExecutor(max_workers=len(self.streams)) as ex:
                    futs = [ex.submit(self._run_stream, s, x) for s in self.streams]
                    parts = [f.result() for f in futs]
            else:
                parts = [self._run_stream(s, x) for s in self.streams]
        seconds = time.perf_counter() - t0

        y = np.concatenate(parts) if len(parts) > 1 else parts[0]

        stats = RunStats()
        stats.seconds = seconds
        for d in self.plan.devices:
            calls_before[d.value] = sum(s.backend.calls for s in self.streams
                                        if s.device is d)
        stats.calls_by_device = calls_before
        return y, stats

    # ---- introspection ---------------------------------------------------
    def device_streams(self, device: Device) -> List[Stream]:
        return [s for s in self.streams if s.device is device]

    def devices_used(self) -> Tuple[str, ...]:
        """Devices with at least one stream that actually executed."""
        return tuple(d.value for d in self.plan.devices
                     if any(s.backend.calls > 0 for s in self.device_streams(d)))

    def describe(self) -> str:
        lines = [f"plan: {self.plan}  ({self.plan.total_streams} stream(s))"]
        topo = cpu_topology()
        for d, n in self.plan:
            if d is Device.CPU:
                lines.append(f"  cpu={n} -> {topo.describe(n)}")
            else:
                lines.append(f"  {d.value}={n} -> {n} stream(s) on one execution resource")
        if not self.streams:
            lines.append("  (no weights loaded yet: streams and row spans are created by build())")
            return "\n".join(lines)
        for s in self.streams:
            lines.append(f"    {s}  rows {s.row_start}..{s.row_end} ({s.n_rows})")
        return "\n".join(lines)


def build_engine(use_cpu=None, use_gpu=None, use_ane=None,
                 use_coreml: bool = True, limits=None) -> AeroEngine:
    """Convenience: parse flags then construct an engine (not yet loaded)."""
    plan = parse_device_alloc(use_cpu=use_cpu, use_gpu=use_gpu, use_ane=use_ane,
                              limits=limits)
    return AeroEngine(plan, use_coreml=use_coreml)
