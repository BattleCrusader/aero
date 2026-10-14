"""Aero engine package.

Quiet CoreML on import: the MIL pipeline prints per-pass tqdm progress bars and
repeats I/O dtype warnings, which drowns the test/gate output without adding
information. Set here so it applies before coremltools is first imported.
"""

import logging
import os

os.environ.setdefault("TQDM_DISABLE", "1")          # no MIL progress bars
logging.getLogger("coremltools").setLevel(logging.ERROR)

from .devices import (ALL_DEVICES, DEFAULT_COUNTS, Device, DeviceCountOutOfRange,  # noqa: E402
                      DevicePlan, NoDevicesSelected, all_combinations, parse_device_alloc)
from .limits import (CoreLimit, CpuTopology, all_limits, ane_limit, cpu_limit,  # noqa: E402
                     cpu_topology, format_limits, gpu_limit)
from .backends import BaseBackend, CoreMLBackend, NumpyBackend  # noqa: E402
from .engine import AeroEngine, Stream, build_engine  # noqa: E402

__all__ = [
    "ALL_DEVICES", "DEFAULT_COUNTS", "Device", "DeviceCountOutOfRange", "DevicePlan",
    "NoDevicesSelected", "all_combinations", "parse_device_alloc",
    "CoreLimit", "CpuTopology", "all_limits", "ane_limit", "cpu_limit",
    "cpu_topology", "format_limits", "gpu_limit",
    "BaseBackend", "CoreMLBackend", "NumpyBackend",
    "AeroEngine", "Stream", "build_engine",
]
