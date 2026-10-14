"""Per-device compute backends.

Each backend owns a compiled CoreML graph for the weight slice assigned to it,
and records how many inferences it has actually run -- so "was this device
really used?" is answerable rather than assumed.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .devices import Device


class BaseBackend:
    """Common surface: load a weight slice, run y = W x, count calls."""

    def __init__(self, device: Device) -> None:
        self.device = device
        self.calls = 0
        self.rows = 0
        self.load_seconds = 0.0
        self.last_seconds = 0.0
        self.core_class = ""   # "performance" | "efficiency" | "" (non-CPU)

    def load(self, weights: np.ndarray) -> None:
        raise NotImplementedError

    def run(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @property
    def n_rows(self) -> int:
        return self.rows


class CoreMLBackend(BaseBackend):
    """A CoreML graph compiled for one device (cpu -> CPU_ONLY, gpu -> CPU_AND_GPU,
    ane -> CPU_AND_NE). CoreML owns placement; we own the weight slice.

    I/O is fp32 with an in-graph cast to fp16: CoreML's I/O layer only accepts
    fp32/int32 at this deployment target, so declaring fp16 I/O would emit a
    warning per conversion. Casting inside the graph keeps the weights fp16
    (the bandwidth-relevant part) and keeps the log clean.
    """

    def __init__(self, device: Device) -> None:
        super().__init__(device)
        self._model = None

    def load(self, weights: np.ndarray) -> None:
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import types as miltypes

        w = np.ascontiguousarray(weights, dtype=np.float16)
        n_out, n_in = w.shape
        self.rows = n_out

        @mb.program(input_specs=[mb.TensorSpec(shape=(n_in,), dtype=miltypes.fp32)])
        def prog(x):
            x16 = mb.cast(x=x, dtype="fp16", name="x_fp16")
            c = mb.const(val=w, name="W")
            y16 = mb.linear(x=x16, weight=c, name="y_fp16")
            return mb.cast(x=y16, dtype="fp32", name="y")

        t0 = time.perf_counter()
        self._model = ct.convert(
            prog,
            convert_to="mlprogram",
            compute_units=self.device.coreml_units(),
            minimum_deployment_target=ct.target.macOS15,
        )
        self.load_seconds = time.perf_counter() - t0

    def run(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError(f"{self.device.value} backend not loaded")
        t0 = time.perf_counter()
        y = self._model.predict({"x": np.asarray(x, dtype=np.float32)})["y"]
        self.last_seconds = time.perf_counter() - t0
        self.calls += 1
        return np.asarray(y, dtype=np.float32).reshape(-1)


class NumpyBackend(BaseBackend):
    """Plain NumPy/BLAS backend -- real CPU execution on a real thread.

    Used for CPU streams, when CoreML is unavailable, and as the correctness
    reference the CoreML backends are compared against.
    """

    def __init__(self, device: Device = Device.CPU) -> None:
        super().__init__(device)
        self._w: Optional[np.ndarray] = None

    def load(self, weights: np.ndarray) -> None:
        self._w = np.ascontiguousarray(weights, dtype=np.float32)
        self.rows = self._w.shape[0]
        self.load_seconds = 0.0

    def run(self, x: np.ndarray) -> np.ndarray:
        if self._w is None:
            raise RuntimeError("numpy backend not loaded")
        t0 = time.perf_counter()
        y = self._w @ np.asarray(x, dtype=np.float32)
        self.last_seconds = time.perf_counter() - t0
        self.calls += 1
        return np.asarray(y, dtype=np.float32).reshape(-1)
