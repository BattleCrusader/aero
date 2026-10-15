"""Model assembly: embed -> N x [norm -> (attn | ssm) -> FFN] -> norm -> head.

This is the loop that turns the verified pieces (gguf, quant, transformer, ssm)
into something that can actually produce a token.

MEMORY STRATEGY: weights are never held in fp32 in full. Big matrices go through
`GGUFReader.matmul` (row-blocked dequant, scratch bounded by block_rows), and
per-layer weights are loaded and released one layer at a time. A 2.6 GB quantized
model therefore runs without an ~18 GB fp32 copy -- which is the property the
whole design is built around.

VERIFIED FROM THE FILE (see tests/test_model.py):
  * layer kinds, all tensor shapes, tied output head (no `output.weight`)

ASSUMPTIONS still to settle against a real forward pass:
  * attention q projection emits [q | gate] concatenated; output = sigmoid(gate) * attn
  * partial RoPE over `rope.dimension_count` dims; MRoPE `dimension_sections`
    degenerates to standard RoPE for text-only input (sections share one position)
  * the four SSM conventions documented in ssm.py
"""

from __future__ import annotations

import os

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import arch as ARCH
from . import transformer as T
from . import gemma as GEMMA
from . import moe as MOE
from . import ssm as SSM
from .gguf import GGUFReader
from .transformer import (AttnConfig, AttnWeights, KVCache, attention, gelu,
                          rms_norm, swiglu)


# ---------------------------------------------------------------- gemma4 knobs
# Every one of these is a point where the tensor LIST does not say what the
# reference GRAPH does. They are switches rather than baked-in choices so the
# alternatives stay measurable side by side on the real file -- the same reason
# transformer.py exposes GATE_MODE / ROPE_MODE / BLOCK_MODE.
#   GEMMA_EMBED_SCALE : "sqrt_embd" | "none"
#   GEMMA_ROPE_FREQS  : "full_only" | "all" | "none"   (proportional RoPE)
#   GEMMA_OUT_SCALE   : "sum" | "deltas" | "none"      (layer_output_scale)
GEMMA_EMBED_SCALE = "sqrt_embd"
GEMMA_ROPE_FREQS = "full_only"
GEMMA_OUT_SCALE = "sum"
# Diagnostics: when a list is bound here, _gemma_block appends
# (layer, rms_in, rms_attn_out, rms_ffn_out, out_scale) per layer.
GEMMA_TRACE = None


@dataclass
class LayerSpec:
    index: int
    kind: str            # "attn" | "ssm"


@dataclass
class ModelConfig:
    arch: str
    n_layer: int
    n_embd: int
    n_head: int
    n_head_kv: int
    head_dim: int
    value_dim: int
    ffn_len: int
    rope_base: float
    rope_dims: Optional[int]
    eps: float
    vocab: int
    tied_output: bool
    layers: List[LayerSpec] = field(default_factory=list)
    ssm_cfg: Optional[SSM.SsmConfig] = None
    moe_cfg: Optional[MOE.MoeConfig] = None
    gemma_cfg: Optional[GEMMA.GemmaConfig] = None
    spec: Optional[ARCH.ArchSpec] = None
    n_head_kv_per_layer: Optional[List[int]] = None
    softcap: Optional[float] = None
    sliding_pattern: Optional[List[bool]] = None
    sliding_window: Optional[int] = None
    n_layer_total: Optional[int] = None      # incl. any MTP heads
    n_nextn: int = 0                         # trailing MTP heads, not decoded
    # INPUT EMBEDDING SCALE. gemma multiplies the token embedding by
    # sqrt(n_embd) before layer 0 (llama.cpp gemma4.cpp: `inpL = ggml_scale(inpL,
    # sqrtf(n_embd))`). Without it the residual stream enters layer 0 ~62x too
    # small while every block delta keeps its full magnitude, so the embedding
    # carries no weight against the deltas -- an architecture-level error, not a
    # rounding one.
    embed_scale: Optional[float] = None
    # PROPORTIONAL RoPE table for the full-attention layers (global tensor
    # `rope_freqs.weight`, length n_embd_head/2). Only the layers whose window is
    # None use it; windowed layers pass freq_factors=None.
    rope_freqs: Optional[np.ndarray] = None
    # PER-LAYER ATTENTION GEOMETRY. gemma4 narrows its windowed layers: blk.0 has
    # head_dim 256 (q 4096, 8 kv heads) while the global blk.5 has 512 (q 8192,
    # 1 kv head). `attention.key_length` is the MAXIMUM, so using it globally is
    # wrong for every windowed layer. Derived from the tensors, which cannot lie.
    head_dim_per_layer: Optional[List[int]] = None
    value_dim_per_layer: Optional[List[int]] = None

    @property
    def attn_layers(self) -> List[int]:
        return [l.index for l in self.layers if l.kind == "attn"]

    @property
    def ssm_layers(self) -> List[int]:
        return [l.index for l in self.layers if l.kind == "ssm"]

    def kv_heads_at(self, layer: int) -> int:
        if self.n_head_kv_per_layer is None:
            return self.n_head_kv
        if layer >= len(self.n_head_kv_per_layer):
            raise IndexError(
                f"layer {layer} is outside the {len(self.n_head_kv_per_layer)} "
                f"decode layers (is it an MTP head?)")
        return self.n_head_kv_per_layer[layer]

    def head_dim_at(self, layer: int) -> int:
        if self.head_dim_per_layer is None or layer >= len(self.head_dim_per_layer):
            return self.head_dim
        return self.head_dim_per_layer[layer]

    def value_dim_at(self, layer: int) -> int:
        if self.value_dim_per_layer is None or layer >= len(self.value_dim_per_layer):
            return self.value_dim
        return self.value_dim_per_layer[layer]

    def window_at(self, layer: int) -> Optional[int]:
        if self.sliding_window is None or self.sliding_pattern is None:
            return None
        return self.sliding_window if self.sliding_pattern[layer] else None

    @classmethod
    def from_gguf(cls, r: GGUFReader) -> "ModelConfig":
        name = r.arch()
        spec = ARCH.spec_for(name)          # raises on an unknown architecture

        n_layer = int(r.hp("block_count"))
        n_embd = int(r.hp("embedding_length"))
        n_head = int(r.hp("attention.head_count"))
        head_dim = int(r.hp("attention.key_length"))
        value_dim = int(r.hp("attention.value_length") or head_dim)
        ffn_len = int(r.hp("feed_forward_length") or 0)
        rope_dims = int(r.hp("rope.dimension_count") or head_dim)

        # MTP / next-token-prediction heads sit at the END of the block list and
        # are NOT part of the main decoder path. Running one as a decoder layer
        # corrupts the whole forward pass -- verified: blk.32 of Qwen3.8-4B
        # carries `nextn.enorm/hnorm/eh_proj/shared_head_norm` (15 tensors vs 11
        # for a real layer) and `nextn_predict_layers=1`.
        n_nextn = int(r.hp("nextn_predict_layers") or 0)
        if n_nextn < 0 or n_nextn >= n_layer:
            raise ARCH.UnsupportedArch(
                f"nextn_predict_layers={n_nextn} is out of range for {n_layer} layers")
        n_decode = n_layer - n_nextn
        if n_nextn:
            for j in range(n_decode, n_layer):
                if not any(k.startswith(f"blk.{j}.nextn.") for k in r.tensors):
                    raise ARCH.UnsupportedArch(
                        f"nextn_predict_layers={n_nextn} implies blk.{j} is an MTP "
                        f"head, but it carries no nextn.* tensors")

        # layer kinds come from the arch spec, not a per-call guess
        layers = [LayerSpec(i, ARCH.detect_layer_kind(r.tensors, i, spec))
                  for i in range(n_decode)]

        vocab, tok_dim = r.tensors["token_embd.weight"].shape
        if tok_dim != n_embd:
            raise ValueError(
                f"token_embd width {tok_dim} != embedding_length {n_embd}")

        ssm_cfg = None
        if spec.has_ssm and any(l.kind == "ssm" for l in layers):
            ssm_cfg = SSM.SsmConfig(
                n_embd=n_embd,
                inner_size=int(r.hp("ssm.inner_size")),
                state_size=int(r.hp("ssm.state_size")),
                n_group=int(r.hp("ssm.group_count")),
                conv_kernel=int(r.hp("ssm.conv_kernel")),
                time_step_rank=int(r.hp("ssm.time_step_rank")),
                eps=float(r.hp("attention.layer_norm_rms_epsilon") or 1e-6),
            )
            if not any(l.kind == "ssm" for l in layers):
                raise ARCH.UnsupportedArch(
                    f"{name} declares SSM support but no layer matched")

        # ---- MoE -------------------------------------------------------
        moe_cfg = None
        if spec.has_moe:
            n_expert = r.hp("expert_count")
            n_used = r.hp("expert_used_count")
            if n_expert is None or n_used is None:
                raise ARCH.UnsupportedArch(
                    f"{name} is a MoE architecture but the file does not report "
                    f"expert_count/expert_used_count; refusing to guess routing")
            # the routed expert width must actually be present
            kt = f"blk.{layers[0].index}.ffn_gate_exps.weight"
            if kt not in r.tensors:
                raise ARCH.UnsupportedArch(
                    f"{name}: expert tensors absent ({kt} missing) -- MoE arch "
                    f"declared but file is dense?")
            has_shared = f"blk.{layers[0].index}.ffn_gate_shexp.weight" in r.tensors
            # ffn length per expert must come from the TENSOR, not from
            # feed_forward_length: this file has no feed_forward_length at all,
            # and a routed expert is narrower than the dense FFN in other arches.
            gt = r.tensors[kt]
            if len(gt.shape) != 3:
                raise ARCH.UnsupportedArch(
                    f"{kt} has shape {gt.shape}; expected (n_expert, ffn, n_embd)")
            expert_ffn = int(gt.shape[1])
            if gt.shape[2] != n_embd:
                raise ARCH.UnsupportedArch(
                    f"{kt} input width {gt.shape[2]} != n_embd {n_embd}")
            moe_cfg = MOE.MoeConfig(
                n_expert=int(n_expert), n_expert_used=int(n_used),
                n_embd=n_embd, ffn_len=expert_ffn, has_shared=has_shared,
                norm_topk_prob=True,
            )

        # ---- gemma4 ----------------------------------------------------
        raw_kv_hp = r.hp("attention.head_count_kv")
        n_head_kv_local = 1 if isinstance(raw_kv_hp, (list, tuple)) else int(raw_kv_hp)
        hd_pl = vd_pl = None
        gemma_cfg = None
        kv_per_layer = None
        softcap = None
        sliding_pattern = None
        sliding_window = None
        embed_scale = None
        rope_freqs = None
        if spec.has_sliding_window or spec.has_logit_softcap \
                or spec.has_per_layer_embd:
            raw_kv = r.hp("attention.head_count_kv")
            kv_per_layer = GEMMA.GemmaConfig.kv_heads_for(raw_kv, n_decode)
            sliding_window = r.hp("attention.sliding_window")
            sliding_pattern = GEMMA.GemmaConfig.pattern_for(
                r.hp("attention.sliding_window_pattern"), n_decode)
            sc = r.hp("final_logit_softcapping")
            softcap = float(sc) if sc else None
            # per-layer embeddings only if the tensors are genuinely there
            has_ple = "per_layer_token_embd.weight" in r.tensors
            ple_dim = 0
            if has_ple:
                ple_dim = int(r.tensors["per_layer_token_embd.weight"].shape[-1])
            elif spec.has_per_layer_embd:
                # declared by the arch but absent from this file: that is fine,
                # the file wins. Recorded rather than silently assumed.
                has_ple = False
            # derive head/value dim per layer from the attention tensors, indexed
            # BY LAYER NUMBER so head_dim_at(i) indexes directly
            hd_pl = [0] * n_decode
            vd_pl = [0] * n_decode
            # NB: loop variable must NOT be named `spec` -- the outer `spec` is the
            # ArchSpec that ends up on the returned config, and shadowing it left
            # ModelConfig.spec holding a LayerSpec.
            for lspec in layers:
                i = lspec.index
                nkv = kv_per_layer[i] if kv_per_layer else n_head_kv_local
                kw = r.tensors[f"blk.{i}.attn_k.weight"].shape[0]
                # gemma4's global layers carry no attn_v -- V IS K there, so the
                # value width is the key width.
                vname = f"blk.{i}.attn_v.weight"
                vw = r.tensors[vname].shape[0] if vname in r.tensors else kw
                if kw % nkv or vw % nkv:
                    raise ARCH.UnsupportedArch(
                        f"blk.{i}: k/v width {kw}/{vw} not divisible by {nkv} kv heads")
                hd_pl[i] = kw // nkv
                vd_pl[i] = vw // nkv

            gemma_cfg = GEMMA.GemmaConfig(
                n_layer=n_decode, n_embd=n_embd, n_head=n_head, head_dim=head_dim,
                value_dim=value_dim, ffn_len=ffn_len,
                eps=float(r.hp("attention.layer_norm_rms_epsilon") or 1e-6),
                sliding_window=int(sliding_window) if sliding_window else None,
                sliding_pattern=sliding_pattern, softcap=softcap,
                kv_heads=kv_per_layer,
                rope_base=float(r.hp("rope.freq_base") or 10000.0),
                rope_dims=rope_dims, has_per_layer_embd=has_ple,
                n_layer_embd=ple_dim,
                # per-layer rotary geometry (gemma4: swa 256@1e4, full 512@1e6)
                rope_base_swa=(float(r.hp("rope.freq_base_swa"))
                               if r.hp("rope.freq_base_swa") else None),
                rope_dims_swa=(int(r.hp("rope.dimension_count_swa"))
                               if r.hp("rope.dimension_count_swa") else None),
                # gemma4 sets self.scaling = 1.0: there is NO pre-attention scaling
                attn_scale=1.0,
                n_kv_shared=int(r.hp("attention.shared_kv_layers") or 0),
            )
            # gemma scales the token embedding by sqrt(n_embd) before layer 0.
            embed_scale = float(np.sqrt(n_embd))
            # proportional RoPE table: a GLOBAL tensor used by the full-attention
            # layers only (llama.cpp loads it per non-SWA layer, duplicated).
            if "rope_freqs.weight" in r.tensors:
                rope_freqs = np.asarray(r.tensor("rope_freqs.weight"),
                                        dtype=np.float32).reshape(-1)

        return cls(
            arch=name, n_layer=n_decode, n_embd=n_embd, n_head=n_head,
            n_head_kv=(kv_per_layer[0] if kv_per_layer else
                       int(r.hp("attention.head_count_kv"))),
            head_dim=head_dim, value_dim=value_dim,
            ffn_len=ffn_len, rope_base=float(r.hp("rope.freq_base") or 10000.0),
            rope_dims=rope_dims,
            eps=float(r.hp("attention.layer_norm_rms_epsilon") or 1e-6),
            vocab=vocab, tied_output="output.weight" not in r.tensors,
            layers=layers, ssm_cfg=ssm_cfg, moe_cfg=moe_cfg, gemma_cfg=gemma_cfg,
            n_layer_total=n_layer, n_nextn=n_nextn,
            head_dim_per_layer=hd_pl, value_dim_per_layer=vd_pl,
            spec=spec, n_head_kv_per_layer=kv_per_layer, softcap=softcap,
            sliding_pattern=sliding_pattern,
            sliding_window=int(sliding_window) if sliding_window else None,
            embed_scale=embed_scale, rope_freqs=rope_freqs,
        )


@dataclass
class ModelState:
    """Everything that persists between tokens."""
    kv: Optional[KVCache]
    ssm_states: dict          # layer -> SsmState
    gdn: dict                 # layer -> (n_v_heads, d, d)
    moe_cache: dict = field(default_factory=dict)   # layer -> {(tensor, expert): fp32}
    # layer -> conv tail for the NATIVE SSM path. The Python path keeps its tail
    # inside SsmState/ConvState; the native path needs the same persistence, and it
    # must survive across calls or the depthwise conv sees a stale (or zero) history
    # and the SSM produces plausible-but-wrong output.
    ssm_tails: dict = field(default_factory=dict)
    pos: int = 0

    @property
    def nbytes(self) -> int:
        n = self.kv.nbytes if self.kv else 0
        n += sum(s.nbytes for s in self.ssm_states.values())
        n += sum(g.nbytes for g in self.gdn.values())
        return n


class Model:
    """A quantized decoder-only model read from a GGUF file."""

    # ---- the memory ceiling ------------------------------------------------
    _guard_armed = False                  # class-level: one ceiling per process
    _lock_fd = None                       # holds the cross-process load lock

    def _take_load_lock(self) -> None:
        """Refuse to load a model if another process is already loading one.

        A per-process ceiling does NOT prevent the failure that actually matters:
        two 6 GiB loads on a 16 GB host. The cap is per-process, so N processes
        each stay under it while the HOST goes into swap and OOM. Measured: two
        concurrent model loads pushed this box to 69 GB of footprint.

        So the ceiling is paired with a cross-process lock. It is an OFD lock
        (fcntl on an open file), which the kernel releases automatically when the
        holder dies -- unlike a pid file, which survives a kill -9 and would wedge
        every future run.
        """
        if Model._lock_fd is not None:
            return
        import fcntl
        import tempfile
        path = os.path.join(tempfile.gettempdir(), "aero-model-load.lock")
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise RuntimeError(
                "refusing to load a second model: another aero process already "
                "holds the model-load lock. Two concurrent loads are what OOMs a "
                "16 GB host (a per-process cap does not prevent it). Wait for the "
                "other process to exit, or kill it, then retry.")
        os.write(fd, str(os.getpid()).encode())
        Model._lock_fd = fd

    def _arm_guard(self, max_ram_bytes: Optional[int]) -> None:
        """Arm the C++ memory ceiling for this process, once.

        Loads the compiled aero::MemoryGuard (build/libaero_guard.dylib) and
        starts its monitor thread, which polls phys_footprint and hard-exits on
        breach. Idempotent: the first Model in a process arms it; later ones are
        no-ops, so a second model cannot silently raise the ceiling.

        If the library is missing the ceiling is NOT silently skipped -- it raises,
        because "ran without a guard" is the exact condition that OOMs the host.
        """
        if Model._guard_armed:
            return
        cap = max_ram_bytes
        if cap is None:
            env = os.environ.get("AERO_MAX_RAM")
            if env:
                cap = int(env) if env.isdigit() else None
                if cap is None:
                    from .cli import parse_max_ram
                    cap = parse_max_ram(env)
        if not cap:
            raise RuntimeError(
                "refusing to load a model with no memory ceiling: set "
                "AERO_MAX_RAM (e.g. '6G') or pass max_ram_bytes=N. An unguarded "
                "model load is how the host gets OOMed.")

        try:
            from .native import MemoryGuard
        except Exception as e:                    # pragma: no cover
            raise RuntimeError(
                f"memory guard unavailable ({e}); build it with `make guard` and "
                f"refusing to load unguarded") from e
        g = MemoryGuard.instance()
        g.set_budget(int(cap))
        g.start_monitor(50)                       # 50 ms poll, matches the tests
        Model._guard_armed = True
        self._guard = g

    def __init__(self, reader: GGUFReader, block_rows: int = 64,
                 progress=None, kv_dtype: str = "fp32",
                 matmul_threads: int = 1,
                 max_ram_bytes: Optional[int] = None) -> None:
        # PROCESS-WIDE MEMORY CEILING, armed by construction.
        #
        # This exists because a guard that callers must opt into is a guard that
        # gets bypassed: six diagnostic probes constructed Model directly and
        # never consulted MemoryGuard, and two of them running at once pushed a
        # 16 GB host into OOM. So the ceiling is armed HERE, at the single point
        # every path (runtime, CLI, probes, tests) must pass through.
        #
        # The enforcement itself is C++ (aero::MemoryGuard): a monitor thread
        # polls phys_footprint -- which, unlike `ps -o rss`, counts GPU/Metal
        # memory -- and _exit(87)s the process on breach. Set AERO_MAX_RAM to
        # override (bytes, or a --max-ram string like "6G"); 0/unset leaves the
        # previous ceiling in place rather than disarming it.
        # The cross-process lock comes FIRST: taking it before anything is read
        # means a second process fails fast instead of loading and then discovering
        # the conflict after the memory is already committed.
        self._take_load_lock()
        self._arm_guard(max_ram_bytes)
        self.kv_dtype = kv_dtype
        # Threads for the blocked matmul. Default 1 = OFF, because parallelism
        # was MEASURED slower: 1/2/4/8 threads give 25.8/28.7/37.2/50.4 s per
        # decode token. The block loop is bound by interpreter and numpy
        # dispatch overhead, not by CPU, so threads only add contention.
        self.matmul_threads = matmul_threads
        self.r = reader
        self.cfg = ModelConfig.from_gguf(reader)
        self.block_rows = block_rows
        self.progress = progress
        self.step = 0


    def _mm(self, name: str, X):
        """matmul with this model's block size and thread setting."""
        return self.r.matmul(name, X, self.block_rows,
                             threads=self.matmul_threads)

    # ---- state -----------------------------------------------------------
    def new_state(self, capacity: int = 4096, kv_dtype: str = "fp32",
                  k_dtype: Optional[str] = None,
                  v_dtype: Optional[str] = None) -> ModelState:
        cfg = self.cfg
        kv = None
        if cfg.attn_layers:
            # ONLY the attention layers. A hybrid model's SSM layers keep O(1)
            # state and never append; giving them a KV cache wastes 3.67x of the
            # largest allocation in the engine (measured, Qwen3.8-4B: 66 GiB
            # vs 18 GiB at 262144 context, identical output).
            #
            # Windowed layers get a ring of window+slack instead of the whole
            # context: a sliding-window layer structurally cannot read past its
            # window, so a full buffer is capacity it can never use.
            win = {}
            if cfg.sliding_pattern is not None:
                for i in cfg.attn_layers:
                    w = cfg.window_at(i)
                    if w:
                        win[i] = int(w)
            heads = None
            if cfg.n_head_kv_per_layer is not None:
                heads = {i: cfg.kv_heads_at(i) for i in cfg.attn_layers}
            kdims = vdims = None
            if cfg.head_dim_per_layer is not None:
                kdims = {i: cfg.head_dim_at(i) for i in cfg.attn_layers}
                vdims = {i: cfg.value_dim_at(i) for i in cfg.attn_layers}
            kv = KVCache(n_layer=cfg.n_layer, n_head_kv=cfg.n_head_kv,
                         head_dim=cfg.head_dim, value_dim=cfg.value_dim,
                         capacity=capacity, dtype=self.kv_dtype,
                         layer_indices=cfg.attn_layers, windowed=win,
                         heads_per_layer=heads,
                         head_dims=kdims, value_dims=vdims,
                         k_dtype=k_dtype or self.kv_dtype,
                         v_dtype=v_dtype or self.kv_dtype)
        ssm_states, gdn = {}, {}
        if cfg.ssm_cfg is not None:
            for i in cfg.ssm_layers:
                st = SSM.SsmState.create(cfg.ssm_cfg)
                ssm_states[i] = st
                gdn[i] = st.zeros()
        return ModelState(kv=kv, ssm_states=ssm_states, gdn=gdn)

    # ---- weights, one layer at a time ------------------------------------
    def _attn_weights(self, layer: int) -> AttnWeights:
        p = f"blk.{layer}."
        has = self.r.tensors.__contains__
        return AttnWeights(
            wq=self.r.tensor(p + "attn_q.weight"),
            wk=self.r.tensor(p + "attn_k.weight"),
            # gemma4's global layers carry no attn_v: V is the raw K projection.
            wv=self.r.tensor(p + "attn_v.weight") if (p + "attn_v.weight") in self.r.tensors
               else None,
            wo=self.r.tensor(p + "attn_output.weight"),
            q_norm=self.r.tensor(p + "attn_q_norm.weight")
                   if (p + "attn_q_norm.weight") in self.r.tensors else None,
            k_norm=self.r.tensor(p + "attn_k_norm.weight")
                   if (p + "attn_k_norm.weight") in self.r.tensors else None,
            v_norm_weightless=True,
        )

    def _expert_rows_fn(self, layer: int):
        """Row reader for ONE expert of a routed matrix.

        Expert matrices are stored (n_expert, out, in) -- the expert axis is the
        OUTERMOST, and the row axis is the middle one. `tensor_rows` slices the
        outermost axis, so asking for [e, e+1) yields exactly one expert's full
        (out, in) matrix, which is what the caller multiplies against.

        Reading one expert instead of all 90 is the reason this engine exists.
        """
        r = self.r

        def read(name: str, expert: int, block_rows: int) -> np.ndarray:
            info = r.tensors[f"blk.{layer}.{name}"]
            if len(info.shape) != 3:
                raise ValueError(
                    f"blk.{layer}.{name} has shape {info.shape}; expected "
                    f"(n_expert, out, in) for a routed expert matrix")
            if expert < 0 or expert >= info.shape[0]:
                raise IndexError(
                    f"{name} has {info.shape[0]} experts, asked for {expert}")
            mat = r.tensor_rows(info, expert, expert + 1)
            return mat.reshape(info.shape[1], info.shape[2])

        return read

    # ---- the forward pass ------------------------------------------------
    def forward(self, tokens: List[int], state: ModelState,
                positions: Optional[np.ndarray] = None) -> np.ndarray:
        """Returns logits for the LAST position only: (vocab,)."""
        cfg = self.cfg
        T = len(tokens)
        if positions is None:
            positions = np.arange(state.pos, state.pos + T, dtype=np.int64)

        emb = self.r.tensors["token_embd.weight"]
        x = np.empty((T, cfg.n_embd), dtype=np.float32)
        for i, tok in enumerate(tokens):
            x[i] = self.r.tensor_rows(emb, int(tok), int(tok) + 1).reshape(-1)
        # gemma multiplies the token embedding by sqrt(n_embd) before layer 0.
        # The reference does this only for REAL tokens (multimodal embeddings are
        # fed unscaled, `ubatch.token ? sqrtf(n_embd) : 1.0f`), and every decode we
        # do is real tokens, so the ternary is not exercised here.
        if cfg.embed_scale is not None and GEMMA_EMBED_SCALE == "sqrt_embd":
            x = x * np.float32(cfg.embed_scale)

        gemma = cfg.gemma_cfg

        for spec in cfg.layers:
            p = f"blk.{spec.index}."
            if gemma is not None:
                x = self._gemma_block(x, spec.index, positions, state, gemma)
            else:
                x = self._qwen_block(x, spec, positions, state)

            self.step += 1
            if self.progress:
                self.progress(spec.index, cfg.n_layer)

        if state.kv is not None:
            state.kv.commit(T)
        state.pos += T

        h = rms_norm(x, self.r.tensor("output_norm.weight"), cfg.eps)
        head = ("output.weight" if not cfg.tied_output else "token_embd.weight")
        logits, _ = self._mm(head, h[-1:])
        logits = logits[0]
        # gemma4 clamps the logit range; qwen does not. Applying it when the
        # file does not ask for it would silently flatten the distribution.
        if cfg.softcap is not None:
            logits = GEMMA.softcap_logits(logits, cfg.softcap)
        return logits

    def _qwen_block(self, x, spec, positions, state):
        """qwen35 / qwen35moe block: norm -> (attn | ssm) -> residual -> FFN."""
        cfg = self.cfg
        p = f"blk.{spec.index}."
        h = rms_norm(x, self.r.tensor(p + "attn_norm.weight"), cfg.eps)

        if spec.kind == "ssm":
            w = SSM.ssm_weights_from(self.r, spec.index)
            o, gdn_new = SSM.ssm_forward(h, w, cfg.ssm_cfg,
                                         state.ssm_states[spec.index],
                                         state.gdn[spec.index])
            state.gdn[spec.index] = gdn_new
            del w
        else:
            w = self._attn_weights(spec.index)
            acfg = AttnConfig(n_head=cfg.n_head, n_head_kv=cfg.kv_heads_at(spec.index),
                              head_dim=cfg.head_dim, value_dim=cfg.value_dim,
                              rope_base=cfg.rope_base, rope_dims=cfg.rope_dims,
                              eps=cfg.eps)
            o = attention(h, positions, w, acfg, cache=state.kv,
                          layer=spec.index, commit=False)
            del w

        if T.BLOCK_MODE == "on_output":
            # post_attention_norm normalises the ATTENTION OUTPUT, and the MLP
            # runs on that; the single residual add comes after. Qwen3-Next's
            # layout. (The alternative normed the SUM instead, which leaves
            # shapes valid and values finite while destroying the learned
            # function -- visible only as "predicts the frequency prior".)
            h2 = rms_norm(o, self.r.tensor(p + "post_attention_norm.weight"),
                          cfg.eps)
            if cfg.moe_cfg is not None:
                x = x + self._moe_ffn(h2, spec.index, cfg.moe_cfg, state)
            else:
                g, _ = self._mm(p + "ffn_gate.weight", h2)
                u, _ = self._mm(p + "ffn_up.weight", h2)
                d, _ = self._mm(p + "ffn_down.weight", swiglu(g, u))
                x = x + d
                state.moe_cache.pop(spec.index, None)
            return x

        x = x + o
        h2 = rms_norm(x, self.r.tensor(p + "post_attention_norm.weight"), cfg.eps)

        if cfg.moe_cfg is not None:
            x = x + self._moe_ffn(h2, spec.index, cfg.moe_cfg, state)
        else:
            g, _ = self._mm(p + "ffn_gate.weight", h2)
            u, _ = self._mm(p + "ffn_up.weight", h2)
            act = swiglu(g, u)
            d, _ = self._mm(p + "ffn_down.weight", act)
            x = x + d
            state.moe_cache.pop(spec.index, None)
        return x

    def _moe_ffn(self, h2, layer, mcfg, state):
        """Routed+shared expert FFN, with a per-layer expert cache.

        Experts are cached so a token that routes to the same expert twice (a
        different token position in the same batch) does not re-read the disk.
        The cache is cleared when the layer is left, so at most n_expert_used
        experts are resident at once rather than all 90.
        """
        cache = state.moe_cache.setdefault(layer, {})
        read = self._expert_rows_fn(layer)
        w = MOE.moe_weights_from(self.r, layer, mcfg)
        shexp = w.get("shexp")
        gate_inp = w["gate_inp"]          # (n_expert, n_embd)

        def router_fn(hv):
            return (hv @ gate_inp.T).reshape(-1)

        def cached(name: str, expert: int, block_rows: int) -> np.ndarray:
            key = (name, expert)
            if key not in cache:
                cache[key] = read(name, expert, block_rows)
            return cache[key]

        out = MOE.moe_ffn(h2, mcfg, cached, shexp, self.block_rows,
                          router_fn=router_fn)
        # release experts now that this layer is done with them
        cache.clear()
        return out

    def _gemma_block(self, x, layer, positions, state, g):
        """gemma4 block, transcribed from the reference graph.

            h        = attn_norm(x)
            a        = post_attention_norm(attn(h))      <- norm the ATTN OUTPUT
            attn_out = x + a                             <- residual AFTER the norm
            h2       = ffn_norm(attn_out)
            d        = post_ffw_norm(ffn(h2))            <- norm the FFN OUTPUT
            x_out    = (attn_out + d) * layer_output_scale

        The last line is the part that is easy to get subtly wrong: the reference
        multiplies the WHOLE layer output (residual included) by the per-layer
        scalar, not just the two delta terms. It is a real multiplier, not a
        rounding detail -- on gemma-4-12b the 48 values run 0.0045 .. 0.8867, so
        scaling the deltas only leaves the residual stream damped 48 times while
        scaling the sum compounds the damping into the input of every later
        layer. Those are different functions and only one of them is the model.
        """
        cfg = self.cfg
        p = f"blk.{layer}."
        raw_scale = GEMMA.scale_of(self.r.tensor(p + "layer_output_scale.weight"))
        if GEMMA_OUT_SCALE == "none":
            ds = ss = 1.0
        elif GEMMA_OUT_SCALE == "deltas":
            ds, ss = raw_scale, 1.0
        else:
            ds, ss = 1.0, raw_scale

        h = rms_norm(x, self.r.tensor(p + "attn_norm.weight"), cfg.eps)
        w = self._attn_weights(layer)
        # PER-LAYER geometry: a windowed gemma layer has a narrower head than a
        # global one (256 vs 512 on gemma-4-12b), so the dims come from the layer,
        # not from the file-wide maximum.
        # per-layer rotary geometry: the SWA and full layers of gemma4 rotate
        # different widths at different bases (256@1e4 vs 512@1e6 on 12b).
        # PROPORTIONAL RoPE applies to the FULL layers ONLY -- windowed layers pass
        # freq_factors=None (`if (!hparams.is_swa(il)) freq_factors = ...`).
        if GEMMA_ROPE_FREQS == "none":
            ff = None
        elif GEMMA_ROPE_FREQS == "all":
            ff = cfg.rope_freqs
        else:
            ff = cfg.rope_freqs if cfg.window_at(layer) is None else None
        acfg = AttnConfig(
            n_head=cfg.n_head, n_head_kv=cfg.kv_heads_at(layer),
            head_dim=cfg.head_dim_at(layer), value_dim=cfg.value_dim_at(layer),
            rope_base=cfg.gemma_cfg.rope_base_at(layer),
            rope_dims=min(cfg.gemma_cfg.rope_dims_at(layer),
                          cfg.head_dim_at(layer)),
            eps=cfg.eps,
            # gemma4 sets scaling = 1.0: no pre-attention scale at all
            scale=cfg.gemma_cfg.attn_scale,
            freq_factors=ff,
        )
        o = attention(h, positions, w, acfg, cache=state.kv, layer=layer,
                      window=cfg.window_at(layer), commit=False)
        del w

        a = rms_norm(o, self.r.tensor(p + "post_attention_norm.weight"), cfg.eps)
        attn_out = x + ds * a

        h = rms_norm(attn_out, self.r.tensor(p + "ffn_norm.weight"), cfg.eps)
        gt, _ = self._mm(p + "ffn_gate.weight", h)
        up, _ = self._mm(p + "ffn_up.weight", h)
        # GELU, not SiLU: gemma4's FFN is gelu(gate) * up. Using swiglu here is a
        # different non-linearity on every FFN in the model.
        d, _ = self._mm(p + "ffn_down.weight", gelu(gt) * up)
        d = rms_norm(d, self.r.tensor(p + "post_ffw_norm.weight"), cfg.eps)

        out = (attn_out + ds * d) * np.float32(ss)
        if GEMMA_TRACE is not None:
            r = lambda v: float(np.sqrt(np.mean(v.astype(np.float32) ** 2)))
            GEMMA_TRACE.append((layer, r(x), r(o), r(attn_out), r(d),
                                raw_scale, r(out)))
        return out

    # ---- generation ------------------------------------------------------
    def generate(self, prompt_tokens: List[int], state: ModelState,
                 max_new: int = 32, temperature: float = 0.0,
                 top_k: int = 0, top_p: float = 1.0,
                 seed: Optional[int] = None):
        from .transformer import SampleParams, sample

        rng = np.random.default_rng(seed)
        params = SampleParams(temperature=temperature, top_k=top_k,
                              top_p=top_p, seed=seed)
        toks = list(prompt_tokens)
        logits = self.forward(toks, state)
        for _ in range(max_new):
            nxt = sample(logits, params, rng)
            yield nxt
            logits = self.forward([nxt], state)


# ------------------------------------------------------------- the plan

def plan_footprint(reader, context: int, block_rows: int = 64,
                   kv_dtype: str = "fp32") -> dict:
    """Predict the peak footprint BEFORE any weight data is read.

    The tensor index gives every tensor's size, so the plan needs only the
    header -- no weight bytes, no dequantisation. That is what makes the refusal
    cheap and safe: a model that cannot fit is rejected in milliseconds, not
    discovered by an OOM.

    The prediction is deliberately per-LAYER rather than whole-model: weights are
    read one layer at a time and released, so the resident weight cost is the
    LARGEST single layer plus the embeddings, not the sum of all layers.
    """
    cfg = ModelConfig.from_gguf(reader)

    # embeddings (always resident: needed for the first and last matmul)
    emb = reader.tensors["token_embd.weight"]
    emb_bytes = _tensor_bytes(reader, emb)

    # largest single layer, counting only its own tensors
    per_layer: dict = {}
    for name, info in reader.tensors.items():
        if name.startswith("blk."):
            blk = name.split(".")[1]
            per_layer[blk] = per_layer.get(blk, 0) + _tensor_bytes(reader, info)
    largest = max(per_layer.values()) if per_layer else 0

    # final norm + the output head (tied -> the embedding matrix is reused)
    head_bytes = 0
    if "output.weight" in reader.tensors:
        head_bytes = _tensor_bytes(reader, reader.tensors["output.weight"])

    # KV cache: ONLY the attention layers hold keys/values. Allocating for every
    # layer over-predicts by n_layer/len(attn_layers) (3.67x on qwen35), which
    # would make the guard refuse models that actually fit.
    # per-layer head counts: a file may narrow the global layers (gemma4: 1
    # head vs 8). Summing a uniform count over-predicts by up to 8x there.
    n_attn = len(cfg.attn_layers)
    kv_elem = 1.0 if kv_dtype == "int8" else 4.0
    kv_bytes = int(sum(cfg.kv_heads_at(i) for i in cfg.attn_layers) *
                   2 * cfg.head_dim * context * kv_elem)
    if kv_dtype == "int8":
        # one fp32 scale per (position, head)
        kv_bytes += int(sum(cfg.kv_heads_at(i) for i in cfg.attn_layers) *
                         2 * context * 4)

    # SSM state: constant per SSM layer, independent of context (the point)
    ssm_bytes = 0
    if cfg.ssm_cfg is not None:
        ssm_bytes = len(cfg.ssm_layers) * (
            cfg.ssm_cfg.n_v_heads * cfg.ssm_cfg.state_size ** 2 * 4)

    # MoE: at most n_expert_used experts resident at once, one layer at a time
    moe_bytes = 0
    if cfg.moe_cfg is not None:
        m = cfg.moe_cfg
        moe_bytes = m.n_expert_used * 3 * m.n_embd * m.ffn_len * 4

    # scratch: the blocked matmul bound, plus the widest activation
    scratch = block_rows * max(cfg.n_embd, cfg.ffn_len or cfg.n_embd) * 4
    scratch += cfg.n_embd * cfg.n_embd * 4

    weights = emb_bytes + largest + head_bytes
    state = kv_bytes + ssm_bytes + moe_bytes
    total = weights + state + scratch

    return {
        "arch": cfg.arch,
        "n_layer": cfg.n_layer,
        "embeddings_bytes": emb_bytes,
        "largest_layer_bytes": largest,
        "head_bytes": head_bytes,
        "weights_bytes": weights,
        "kv_bytes": kv_bytes,
        "ssm_bytes": ssm_bytes,
        "moe_bytes": moe_bytes,
        "state_bytes": state,
        "scratch_bytes": scratch,
        "total_bytes": total,
        "context": context,
    }


def _tensor_bytes(reader, info) -> int:
    from .quant import ggml_type_size
    return ggml_type_size(info.type_id, info.n_elements)
