"""GGUF reader and writer.

Reader: parses the header, metadata and tensor index of a .gguf file, and gives
byte-range access to each tensor without loading the whole file.

Writer: used only to build tiny fixtures with known weights, so the transformer
and the dequantisers can be verified against independent reference maths.

Format (little-endian throughout):

    magic "GGUF" | u32 version | u64 n_tensors | u64 n_kv
    n_kv    x (string key, u32 type, value)
    n_tensors x (string name, u32 n_dims, u64 dims[n_dims], u32 type, u64 offset)
    pad to `general.alignment`
    tensor data

A string is u64 length + UTF-8 bytes. Array = u32 elem_type, u64 count, elements.
Tensor `offset` is relative to the START OF THE DATA SECTION.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

try:
    from . import native
except Exception:                                   # pragma: no cover
    native = None

MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32

# ggml value types
GGUF_UINT8, GGUF_INT8, GGUF_UINT16, GGUF_INT16 = 0, 1, 2, 3
GGUF_UINT32, GGUF_INT32, GGUF_FLOAT32, GGUF_BOOL = 4, 5, 6, 7
GGUF_STRING, GGUF_ARRAY, GGUF_UINT64, GGUF_INT64, GGUF_FLOAT64 = 8, 9, 10, 11, 12

# ggml tensor types (subset we support)
T_F32, T_F16 = 0, 1
T_Q4_0, T_Q4_1 = 2, 3
T_Q5_0, T_Q5_1 = 6, 7
T_Q8_0 = 8
T_Q4_K, T_Q5_K, T_Q6_K = 12, 13, 14
T_IQ4_XS = 23
T_BF16 = 30

TYPE_NAMES: Dict[int, str] = {
    T_F32: "F32", T_F16: "F16", T_Q4_0: "Q4_0", T_Q4_1: "Q4_1",
    T_Q5_0: "Q5_0", T_Q5_1: "Q5_1", T_Q8_0: "Q8_0",
    T_Q4_K: "Q4_K", T_Q5_K: "Q5_K", T_Q6_K: "Q6_K", T_IQ4_XS: "IQ4_XS",
    T_BF16: "BF16",
}

_SCALAR_FMT = {
    GGUF_UINT8: "<B", GGUF_INT8: "<b", GGUF_UINT16: "<H", GGUF_INT16: "<h",
    GGUF_UINT32: "<I", GGUF_INT32: "<i", GGUF_FLOAT32: "<f", GGUF_BOOL: "<?",
    GGUF_UINT64: "<Q", GGUF_INT64: "<q", GGUF_FLOAT64: "<d",
}


@dataclass
class TensorInfo:
    name: str
    dims: Tuple[int, ...]      # GGUF order: fastest-varying first
    type_id: int
    offset: int                # relative to the data section
    n_elements: int = 0

    @property
    def shape(self) -> Tuple[int, ...]:
        """Row-major shape (GGUF stores dims reversed)."""
        return tuple(reversed(self.dims))

    @property
    def type_name(self) -> str:
        return TYPE_NAMES.get(self.type_id, f"type{self.type_id}")

    def __str__(self) -> str:
        return f"{self.name} {self.shape} {self.type_name}"


def _run_spans(fn, spans, nthreads: int):
    """Run fn(span) over spans, across nthreads. Returns a peak hint or None.

    Threads (not processes) because numpy releases the GIL for the dense ops;
    the dequantise step is elementwise numpy and releases it too.

    CAUTION: callers must ensure `fn` does not touch shared mutable state. In
    particular GGUFReader.tensor_rows seeks a SHARED file handle, so the
    seek-based path must stay single-threaded.
    """
    if nthreads <= 1 or len(spans) <= 1:
        for s in spans:
            fn(s)
        return None
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=nthreads) as ex:
        list(ex.map(fn, spans))
    return None


class GGUFReader:
    """Random-access reader over a .gguf file."""

    def __init__(self, path: str) -> None:
        # Small-tensor cache. Position-independent constants (layer norms,
        # ssm_a, dt_bias, ssm_norm, layer_output_scale) are re-read EVERY token
        # -- 329 calls per decode step for this model -- and each call pays a
        # seek on the backing file. Measured on USB: ~30 ms per seek regardless
        # of size, i.e. 10.1 s of a 24.6 s token (41%) spent fetching ~1 MiB of
        # constants. Caching anything below this threshold removes essentially
        # all of it for well under a megabyte of RAM.
        self._small_cache: Dict[str, np.ndarray] = {}
        self._small_bytes = 0
        self.path = path
        self.kv: Dict[str, Any] = {}
        self.tensors: Dict[str, TensorInfo] = {}
        self.version = 0
        self.alignment = DEFAULT_ALIGNMENT
        self.data_offset = 0
        self._fh = open(path, "rb")
        self._parse_index()

    # ---- parsing ---------------------------------------------------------
    def _read(self, n: int) -> bytes:
        b = self._fh.read(n)
        if len(b) != n:
            raise EOFError(f"{self.path}: truncated (wanted {n} bytes, got {len(b)})")
        return b

    def _u32(self) -> int:
        return struct.unpack("<I", self._read(4))[0]

    def _u64(self) -> int:
        return struct.unpack("<Q", self._read(8))[0]

    def _str(self) -> str:
        n = self._u64()
        return self._read(n).decode("utf-8", "replace")

    def _value(self, t: int) -> Any:
        if t == GGUF_STRING:
            return self._str()
        if t == GGUF_ARRAY:
            et = self._u32()
            n = self._u64()
            return [self._value(et) for _ in range(n)]
        fmt = _SCALAR_FMT.get(t)
        if fmt is None:
            raise ValueError(f"unknown GGUF value type {t}")
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self._read(size))[0]

    def _parse_index(self) -> None:
        if self._read(4) != MAGIC:
            raise ValueError(f"{self.path}: not a GGUF file (bad magic)")
        self.version = self._u32()
        if self.version not in (2, 3):
            raise ValueError(f"{self.path}: unsupported GGUF version {self.version}")
        n_tensors = self._u64()
        n_kv = self._u64()

        for _ in range(n_kv):
            key = self._str()
            self.kv[key] = self._value(self._u32())

        self.alignment = int(self.kv.get("general.alignment", DEFAULT_ALIGNMENT))
        if self.alignment <= 0 or (self.alignment & (self.alignment - 1)):
            raise ValueError(f"bad alignment {self.alignment} (must be a power of two)")

        for _ in range(n_tensors):
            name = self._str()
            n_dims = self._u32()
            dims = tuple(self._u64() for _ in range(n_dims))
            type_id = self._u32()
            offset = self._u64()
            n_el = 1
            for d in dims:
                n_el *= d
            self.tensors[name] = TensorInfo(name, dims, type_id, offset, n_el)

        # data section starts after alignment padding
        pos = self._fh.tell()
        pad = (-pos) % self.alignment
        self._fh.seek(pad, 1)
        self.data_offset = self._fh.tell()

    # ---- metadata helpers ------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        return self.kv.get(key, default)

    def arch(self) -> str:
        return str(self.kv.get("general.architecture", "unknown"))

    def hp(self, suffix: str, default: Any = None) -> Any:
        """Architecture-scoped hyperparameter, e.g. hp('block_count')."""
        return self.kv.get(f"{self.arch()}.{suffix}", default)

    # ---- tensor access ---------------------------------------------------
    def tensor_bytes(self, info: TensorInfo) -> bytes:
        """Raw bytes of one tensor, read from disk on demand."""
        from .quant import ggml_type_size

        nbytes = ggml_type_size(info.type_id, info.n_elements)
        self._fh.seek(self.data_offset + info.offset)
        return self._read(nbytes)

    def tensor(self, name: str):
        """Dequantise one tensor to a float32 numpy array of its row-major shape.

        Small tensors are memoised -- see _small_cache for why that is worth
        41% of a decode step.
        """
        from .quant import dequantize, ggml_type_size

        hit = self._small_cache.get(name)
        if hit is not None:
            return hit

        info = self.tensors[name]
        raw = self.tensor_bytes(info)

        # NATIVE DEQUANTISE. The Python dequantiser was 71% of a decode step once
        # the matmul moved to C++: the SSM and attention weight matrices are all
        # loaded through here (201 calls/token), and each one paid the Python
        # loop. Falls back to the Python path when the library or the type is
        # unsupported, so a missing toolchain degrades rather than breaks.
        out = None
        if native is not None and native.available() and info.type_id in native.NATIVE_TYPES:
            out = native.dequant(info.type_id, raw, info.shape)
        if out is None:
            out = dequantize(raw, info.type_id, info.shape)

        # cache only the small constants; a 12 MB weight matrix must not be held
        if out.nbytes <= self.SMALL_TENSOR_BYTES:
            self._small_cache[name] = out
            self._small_bytes += out.nbytes
        return out

    # tensors at or below this size are treated as constants and memoised
    SMALL_TENSOR_BYTES = 1 << 20          # 1 MiB

    @property
    def small_cache_bytes(self) -> int:
        return self._small_bytes

    def row_bytes(self, info: TensorInfo) -> int:
        """Bytes per row (last dim) -- needed for slice reads."""
        from .quant import ggml_type_size

        per_row = self.n_row_elements(info)
        return ggml_type_size(info.type_id, per_row)

    @staticmethod
    def n_row_elements(info: TensorInfo) -> int:
        n_rows, _ = GGUFReader.row_shape(info)
        return info.n_elements // n_rows

    @staticmethod
    def row_shape(info: TensorInfo) -> Tuple[int, int]:
        """(n_rows, n_per_row). A 1-D tensor is one row."""
        if len(info.shape) <= 1:
            return 1, int(info.shape[0]) if info.shape else 0
        return int(info.shape[0]), int(info.shape[1])

    def tensor_rows(self, info: TensorInfo, r0: int, r1: int) -> np.ndarray:
        """Dequantise rows [r0, r1) only.

        This is the primitive that makes a blocked dequant-matvec possible: read
        and expand a bounded slice instead of the whole tensor, so fp32 weights
        are never materialised in full.
        """
        import numpy as np

        from .quant import dequantize, ggml_type_size

        n_rows, per_row = self.row_shape(info)
        if not (0 <= r0 <= r1 <= n_rows):
            raise ValueError(f"row range [{r0},{r1}) outside [0,{n_rows})")
        rb = ggml_type_size(info.type_id, per_row)
        self._fh.seek(self.data_offset + info.offset + r0 * rb)
        raw = self._read((r1 - r0) * rb)
        shape = (r1 - r0, per_row) if len(info.shape) > 1 else (r1 - r0,)
        return dequantize(raw, info.type_id, shape)

    def matmul(self, name: str, X: np.ndarray, block_rows: int = 128,
               read_once: bool = True, threads: int = 1):
        """X @ W^T for X of shape (T, in). Blocked over the OUTPUT dim.

        Returns (Y (T, out), peak_scratch_bytes).

        BLOCKING: the weight is never materialised in fp32, so a 2.6 GB quantized
        model runs without a ~18 GB fp32 copy. `read_once` reads the tensor's
        bytes in one sequential read and dequantises blocks from memory.

        THREADS: default 1, i.e. NO parallelism, because parallelism was measured
        to be slower. Splitting the block range across 1/2/4/8 threads on
        Qwen3.8-4B gives 25.8 / 28.7 / 37.2 / 50.4 s per decode token -- monotonic
        degradation. The loop is bound by per-call interpreter and numpy dispatch
        overhead (a blocked matmul is ~140 ms against 4.17 ms for the same fp32
        matvec), so extra threads only add contention. Pass threads>1 to
        re-measure; do not assume it helps.
        """
        import numpy as np

        from .quant import dequantize

        info = self.tensors[name]
        n_rows, per_row = self.row_shape(info)
        Xm = np.asarray(X, dtype=np.float32)
        if Xm.ndim == 1:
            Xm = Xm[None, :]
        if Xm.shape[1] != per_row:
            raise ValueError(f"{name}: expects {per_row} inputs, got {Xm.shape[1]}")

        T = Xm.shape[0]
        Y = np.empty((T, n_rows), dtype=np.float32)
        spans = [(r0, min(r0 + block_rows, n_rows))
                 for r0 in range(0, n_rows, block_rows)]

        # NATIVE FAST PATH. The C++ kernel fuses dequantise into the inner loop
        # and measured 7.3x faster than this Python loop on real weights
        # (5.9 ms vs 43.2 ms for a 9216x2560 Q4_K matrix), agreeing to 4.9e-07
        # relative -- fp32 accumulation order, not algorithmic error. Falls back
        # to the pure-Python path when the library is absent or the type has no
        # kernel, so the engine still works without a C++ toolchain.
        if native is not None and native.available() and info.type_id in native.NATIVE_TYPES:
            raw = self.tensor_bytes(info)
            nthreads = threads if threads > 1 else 0    # 0 = let C++ pick
            if nthreads == 0:
                nthreads = native.hardware_threads()
            Y = native.matmul(info.type_id, raw, Xm, nthreads)
            return Y, len(raw)

        if read_once:
            # one sequential read, then dequantise blocks from memory
            raw = self.tensor_bytes(info)
            row_bytes = self.row_bytes(info)
            # peak = the read buffer + one fp32 block (what the guard must allow)
            peak = len(raw) + spans[0][1] * per_row * 4 if spans else len(raw)

            def do(span):
                r0, r1 = span
                chunk = raw[r0 * row_bytes:r1 * row_bytes]
                w = dequantize(chunk, info.type_id, (r1 - r0, per_row))
                Y[:, r0:r1] = Xm @ w.T

            _run_spans(do, spans, threads)
            return Y, peak

        def do_seek(span):
            r0, r1 = span
            w = self.tensor_rows(info, r0, r1)
            Y[:, r0:r1] = Xm @ w.T

        # single-threaded regardless: tensor_rows seeks a SHARED file handle, so
        # concurrent spans would race on the seek position and read each other's
        # bytes
        _run_spans(do_seek, spans, 1)
        return Y, block_rows * per_row * 4

    def matvec(self, name: str, x: np.ndarray, block_rows: int = 64):
        """y = W @ x WITHOUT materialising W as fp32.

        Dequantises in row blocks, so peak float scratch is
        `block_rows * n_per_row * 4` bytes instead of the whole matrix. The
        resident cost of a weight is therefore its QUANTIZED size, which for
        Q4_K is 7.1x smaller than the fp32 form.

        Returns (y, peak_scratch_bytes).
        """
        import numpy as np

        info = self.tensors[name]
        n_rows, per_row = self.row_shape(info)
        xv = np.asarray(x, dtype=np.float32).reshape(-1)
        if xv.shape[0] != per_row:
            raise ValueError(f"{name}: expects {per_row} inputs, got {xv.shape[0]}")

        y = np.empty(n_rows, dtype=np.float32)
        peak = 0
        for r0 in range(0, n_rows, block_rows):
            r1 = min(r0 + block_rows, n_rows)
            w = self.tensor_rows(info, r0, r1)
            peak = max(peak, w.nbytes)
            y[r0:r1] = w @ xv
        return y, peak

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "GGUFReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def summary(self) -> str:
        lines = [f"{self.path}",
                 f"  gguf v{self.version}, arch={self.arch()}, "
                 f"{len(self.tensors)} tensors, {len(self.kv)} metadata keys"]
        for k in ("block_count", "embedding_length", "attention.head_count",
                  "attention.head_count_kv", "attention.key_length",
                  "feed_forward_length", "context_length"):
            v = self.hp(k)
            if v is not None:
                lines.append(f"    {k} = {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Writer -- fixtures only
# ---------------------------------------------------------------------------

class GGUFWriter:
    """Minimal writer. Used to build tiny fixtures with known weights so the
    transformer and dequantisers can be checked against reference maths."""

    def __init__(self, arch: str, alignment: int = DEFAULT_ALIGNMENT) -> None:
        self.arch = arch
        self.alignment = alignment
        self.kv: List[Tuple[str, int, Any]] = [
            ("general.architecture", GGUF_STRING, arch),
            ("general.alignment", GGUF_UINT32, alignment),
        ]
        self.tensors: List[Tuple[str, Tuple[int, ...], int, bytes]] = []

    def add_kv(self, key: str, type_id: int, value: Any) -> None:
        self.kv.append((key, type_id, value))

    def add_scalar(self, key: str, value: Any) -> None:
        t = GGUF_UINT32 if isinstance(value, int) else GGUF_FLOAT32
        self.add_kv(key, t, value)

    def add_tensor(self, name: str, shape: Tuple[int, ...], raw: bytes,
                   type_id: int) -> None:
        """`shape` is row-major; it is reversed to GGUF's fastest-first order."""
        self.tensors.append((name, tuple(reversed(shape)), type_id, raw))

    # ---- serialisation ---------------------------------------------------
    @staticmethod
    def _w_str(s: str) -> bytes:
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    @classmethod
    def _w_value(cls, t: int, v: Any) -> bytes:
        if t == GGUF_STRING:
            return cls._w_str(v)
        if t == GGUF_ARRAY:
            et, items = v
            out = struct.pack("<IQ", et, len(items))
            for it in items:
                out += cls._w_value(et, it)
            return out
        return struct.pack(_SCALAR_FMT[t], v)

    def write(self, path: str) -> None:
        out = bytearray()
        out += MAGIC
        out += struct.pack("<I", 3)
        out += struct.pack("<Q", len(self.tensors))
        out += struct.pack("<Q", len(self.kv))
        for key, t, v in self.kv:
            out += self._w_str(key) + struct.pack("<I", t) + self._w_value(t, v)

        # lay out tensor data first so offsets are known
        offsets = []
        cursor = 0
        for _name, _dims, _t, raw in self.tensors:
            offsets.append(cursor)
            cursor += len(raw)
            cursor += (-cursor) % self.alignment

        for (name, dims, t, raw), off in zip(self.tensors, offsets):
            out += self._w_str(name)
            out += struct.pack("<I", len(dims))
            for d in dims:
                out += struct.pack("<Q", d)
            out += struct.pack("<I", t)
            out += struct.pack("<Q", off)

        out += b"\x00" * ((-len(out)) % self.alignment)
        data_start = len(out)
        for (_name, _dims, _t, raw), off in zip(self.tensors, offsets):
            target = data_start + off
            if len(out) < target:
                out += b"\x00" * (target - len(out))
            out += raw
        out += b"\x00" * ((-len(out)) % self.alignment)

        with open(path, "wb") as f:
            f.write(out)
