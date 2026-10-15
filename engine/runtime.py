"""Runtime layer -- the pluggable model behind the server and the CLI.

The serving layer (server.py, chat.py) never talks to a model directly; it talks
to a ModelRuntime. That keeps the OpenAI surface and the chat REPL independent of
which compute path is underneath.

HONEST STATUS: only the dev runtime is implemented. It is deterministic and
useful for exercising the API and the CLI end to end, but it is NOT a language
model. A real runtime (GGUF load + attention/SSM/MoE + KV cache + decode loop) is
the remaining work; see `GgufRuntime` for the contract it must satisfy.
"""

from __future__ import annotations

import os
import sys

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

from .devices import DevicePlan

try:                        # imported as engine.runtime (usual)
    from .chat_template import render_chat
except ImportError:        # pragma: no cover - run as a plain script
    from engine.chat_template import render_chat


@dataclass
class GenParams:
    """Sampling parameters, mapped 1:1 from the OpenAI request body."""

    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0          # 0 = disabled. Not an OpenAI field, but the engine
                            # samples with it, so the request can carry it.
    stop: Tuple[str, ...] = ()
    seed: Optional[int] = None

    @classmethod
    def from_openai(cls, body: dict) -> "GenParams":
        stop = body.get("stop") or ()
        if isinstance(stop, str):
            stop = (stop,)
        return cls(
            max_tokens=int(body.get("max_tokens") or body.get("max_completion_tokens") or 256),
            temperature=float(body.get("temperature", 0.7)),
            top_p=float(body.get("top_p", 1.0)),
            top_k=int(body.get("top_k") or 0),
            stop=tuple(str(s) for s in stop),
            seed=body.get("seed"),
        )


class ModelRuntime(ABC):
    """What the serving layer needs from a model."""

    #: reported by /v1/models
    name: str = "unknown"

    @abstractmethod
    def load(self, spec: str) -> None:
        """Prepare the model. `spec` is a path or model id."""

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], params: GenParams) -> Iterator[str]:
        """Yield response TEXT chunks. Must honour params.stop."""

    def tokenize(self, text: str) -> List[str]:
        return text.split()

    def detokenize(self, tokens: List[str]) -> str:
        return " ".join(tokens)

    def apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Default flat rendering; a real runtime should use the model's own template."""
        out = []
        for m in messages:
            out.append(f"{m.get('role', 'user')}: {m.get('content', '')}")
        return "\n".join(out) + "\nassistant:"

    @property
    def stats(self) -> dict:
        return {"runtime": type(self).__name__, "model": self.name}

    def close(self) -> None:
        pass


class EchoRuntime(ModelRuntime):
    """Deterministic DEV runtime. NOT a language model.

    Exists so the OpenAI surface and the CLI can be tested and demonstrated
    without a model. It echoes the conversation deterministically, so responses
    are reproducible and assertable. Never present it as inference.
    """

    name = "aero-echo-dev"

    def __init__(self) -> None:
        self.loaded = False
        self.calls = 0

    def load(self, spec: str) -> None:
        self.name = spec or self.name
        self.loaded = True

    def generate(self, messages: List[Dict[str, str]],
                 params: GenParams) -> Iterator[str]:
        if not self.loaded:
            raise RuntimeError("runtime not loaded")
        self.calls += 1

        last = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last = str(m.get("content", ""))
                break
        # deterministic "reply": mirrors the prompt with a marker, then counts
        words = last.split()
        reply = (f"[echo:{self.name}] you said {len(words)} word(s). "
                 f"first={words[0] if words else '(nothing)'}. "
                 f"turns={len(messages)}.")
        if params.seed is not None:
            reply += f" seed={params.seed}."

        # emit in small chunks so streaming paths are genuinely exercised
        for i in range(0, len(reply), 8):
            yield reply[i:i + 8]
        for i in range(min(params.max_tokens, 5)):
            yield f" tok{i}"


class GgufRuntime(ModelRuntime):
    """The real runtime: GGUF in, tokens out, under a --max-ram guard.

    This is the wiring that turns the verified pieces into a working model:

        MemoryGuard          refuses a model that cannot fit, BEFORE the read
        GGUFReader.matmul    row-blocked dequant, so no fp32 weight is ever whole
        Model.forward        the assembly (attn / SSM / MoE / gemma4)
        expert cache         at most n_expert_used experts resident, not all 90

    Honest status: this runs and produces tokens. It is SLOW -- the matmuls go
    through numpy on the CPU, and a 14B MoE is minutes per token. Correctness
    first; the AeroEngine device split is the next step, not this one.
    """

    name = "aero-gguf"

    def __init__(self, max_ram: Optional[str] = None, block_rows: int = 64,
                 threads: Optional[int] = None, context: int = 4096,
                 kv_dtype: str = "fp32", k_dtype: Optional[str] = None,
                 v_dtype: Optional[str] = None,
                 device_plan: Optional[DevicePlan] = None) -> None:
        self.max_ram = max_ram
        self.block_rows = block_rows
        self.device_plan = device_plan
        # THE DEVICE FLAGS LAND HERE. --use-cpu is the only flag the C++ block loop
        # can act on, and it arrives as this thread count:
        #   runtime.threads -> aero_geom.threads -> aero::matmul(W, Xm, Ym, g_.threads)
        # An explicit count is honoured exactly; left at the default (cpu=0) the loop
        # gets the machine's hardware thread count. It used to collapse to 1 through
        # `rt.threads or 1`, so the flag changed nothing and 7 of 8 cores sat idle.
        # Without a plan (existing callers) it stays None -- unchanged behaviour.
        self.threads = threads
        self.device_notice = None
        if self.threads is None and device_plan is not None:
            from .native import hardware_threads
            self.threads = device_plan.thread_budget(hardware_threads())
        if device_plan is not None:
            self.device_notice = device_plan.notice()
        self.context = context
        self.kv_dtype = kv_dtype
        self.k_dtype = k_dtype or kv_dtype
        self.v_dtype = v_dtype or kv_dtype
        self.model = None
        self.state = None
        self.guard = None
        self._reader = None
        self.tokenizer = None
        self._load_stats: dict = {}

    # ---- the guard ------------------------------------------------------
    def _budget_bytes(self) -> Optional[int]:
        if self.max_ram is None:
            return None
        # the parser lives in cli.py as the Python mirror of the C++ guard's
        # parser; reusing it keeps the two definitions from drifting.
        from .cli import parse_max_ram
        return parse_max_ram(self.max_ram)

    def load(self, spec: str) -> None:
        from . import arch as ARCH
        from .cli import parse_max_ram
        from .gguf import GGUFReader
        from .model import Model, plan_footprint

        path = spec
        budget = self._budget_bytes()

        # (1) PREDICTIVE GATE -- decide before touching the disk. This is the
        # whole reason the guard exists: a model that cannot fit must be refused
        # with an explanation, not discovered by an OOM that takes the box down.
        probe = GGUFReader(path)
        arch = probe.arch()
        ARCH.spec_for(arch)                    # refuse unknown arches here, early
        # native=: the C++ engine mmaps the file, so its footprint can approach the
        # file size, while the Python reference streams per layer. Planning for the
        # wrong one under-predicts by (file - one layer) -- 6.4 GB on gemma-4-12b.
        plan = plan_footprint(probe, self.context, self.block_rows,
                              kv_dtype=self.kv_dtype,
                              native=os.environ.get("AERO_NATIVE_LOOP", "1") == "1")
        probe.close()

        if budget is not None and plan["total_bytes"] > budget:
            raise MemoryError(
                f"refusing to load: planned footprint "
                f"{plan['total_bytes']/2**30:.2f} GiB exceeds --max-ram "
                f"{budget/2**30:.2f} GiB\n"
                f"  model weights : {plan['weights_bytes']/2**30:.2f} GiB\n"
                f"  kv + state    : {plan['state_bytes']/2**30:.2f} GiB\n"
                f"  scratch       : {plan['scratch_bytes']/2**30:.2f} GiB\n"
                f"  (largest layer {plan['largest_layer_bytes']/2**30:.2f} GiB)"
            )

        # (2) the guard itself. The authoritative footprint reader is the C++
        # one (phys_footprint, which sees Metal; ps -o rss does not). If the
        # shared library is not built we fall back to the C++ guard when present,
        # else to NO measured monitor -- and say so, rather than pretending a
        # cap is enforced when nothing is measuring.
        self.guard = _make_guard(budget)

        self._reader = GGUFReader(path)
        self._arch = arch
        self.model = Model(self._reader, block_rows=self.block_rows,
                           kv_dtype=self.kv_dtype)
        self.state = self.model.new_state(capacity=self.context,
                                          k_dtype=self.k_dtype,
                                          v_dtype=self.v_dtype)

        # ---- the native block loop (AERO_NATIVE_LOOP, default 1) -------------
        # The C++ block loop owns the per-layer wiring; attention/SSM/MoE arrive
        # through hooks backed by the same Python primitives, so behaviour is
        # identical and each hook can move to its native implementation
        # independently afterwards. Falls back to the Python loop when the dylib is
        # not built, and SAYS SO in the load stats rather than silently choosing.
        self._native_loop = None
        self._native_error = None
        self._native_vocab = 0
        if os.environ.get("AERO_NATIVE_LOOP", "1") == "1":
            try:
                from . import native_model as NM
                self._native_loop, self._native_vocab = self._build_native_loop(NM)
            except Exception as e:            # missing lib, refused geometry...
                self._native_loop = None
                self._native_error = type(e).__name__ + ": " + str(e)
            self._load_stats["native_block_loop"] = self._native_loop is not None
            if self._native_loop is None:
                self._load_stats["native_block_loop_error"] = self._native_error
                # THE ENGINE IS C++. Falling back to the Python loop silently would
                # make the Python reference the thing that generates tokens -- a dev
                # stub that becomes the production path is how a port dies.
                #
                # Enforced as a LOUD WARNING rather than a refusal for now: making it a
                # hard error broke tests/test_runtime_gguf.py, which deliberately loads a
                # runtime whose model has no tokenizer (so the native loop cannot be
                # built) and expects to proceed. Enforcement lands with the attention
                # hook, once the engine can actually run without Python. Until then the
                # fact is printed AND recorded in load stats -- never silent.
                msg = ("[aero] WARNING: inference is running on the PYTHON reference "
                       "loop, not the C++ engine: " + str(self._native_error))
                print(msg, file=sys.stderr)
                self._load_stats["native_block_loop_warning"] = msg
        else:
            self._load_stats["native_block_loop"] = False
            self._load_stats["native_block_loop_error"] = "disabled by AERO_NATIVE_LOOP=0"

        # the tokenizer is what turns ids into text; without it the runtime can
        # only report numbers, so load it and say clearly if it is absent.
        tok_err = None
        try:
            from .tokenizer import Tokenizer
            self.tokenizer = Tokenizer.from_gguf(self._reader)
        except Exception as e:
            self.tokenizer = None
            tok_err = f"{type(e).__name__}: {e}"

        self._load_stats = {
            "arch": arch,
            "threads": self.threads,
            "device_notice": self.device_notice,
            "layers": self.model.cfg.n_layer,
            "budget_bytes": budget,
            "planned_bytes": plan["total_bytes"],
            "state_bytes": self.state.nbytes,
            "moe": self.model.cfg.moe_cfg is not None,
            "ssm": self.model.cfg.ssm_cfg is not None,
            "kv_dtype": self.kv_dtype,
            "kv_bytes": self.state.kv.nbytes if self.state.kv else 0,
            "kv_layers": self.state.kv.n_layers_cached if self.state.kv else 0,
            "tokenizer": self.token_info(),
        }
        if tok_err:
            self._load_stats["tokenizer_error"] = tok_err

    def plan(self) -> dict:
        return dict(self._load_stats)

    # ---- tokenization ---------------------------------------------------
    def apply_chat_template(self, messages: List[Dict[str, str]],
                            add_generation_prompt: bool = True,
                            enable_thinking: Optional[bool] = None) -> str:
        """Render the conversation the way the file's OWN chat template does.

        Delegates to engine.chat_template so the runtime and the check tools cannot
        drift apart -- a tool that renders its own prompt "verifies" something the
        engine never actually sends, which is precisely how the BOS bug stayed
        hidden.

        The returned text already contains the leading BOS where the template emits
        one, so it is tokenized with add_bos=False. (Tokenizer.encode also refuses
        to double it.)
        """
        return render_chat(messages, self.tokenizer,
                           add_generation_prompt=add_generation_prompt,
                           enable_thinking=enable_thinking)


    def tokenize(self, text: str) -> List[int]:
        """Real BPE tokenization, from the vocab in the file."""
        if self.model is None:
            raise RuntimeError("load() first")
        if self.tokenizer is None:
            raise RuntimeError(
                "this GGUF carries no tokenizer; text cannot be tokenized")
        return self.tokenizer.encode(text)

    def detokenize(self, tokens: List[int]) -> str:
        """Real BPE decoding -- this is what makes generated output readable."""
        if self.tokenizer is None:
            return " ".join(str(t) for t in tokens)
        return self.tokenizer.decode(tokens)

    def token_info(self) -> dict:
        if self.tokenizer is None:
            return {"available": False}
        t = self.tokenizer
        return {"available": True, "vocab": t.n_vocab,
                "merges": len(t.bpe_ranks), "specials": len(t._special),
                "eos_id": t.eos_id}

    # ---- the native block loop -------------------------------------------
    def _build_native_loop(self, native_model_module):
        """Open the C++ block loop with hook-backed layers.

        Returns (loop, vocab). Raises on refusal so load() records WHY; this
        runtime never pretends the native path is active.
        """
        from .native_loop import build_native_loop
        loop = build_native_loop(self)
        return loop, loop.vocab

    def generate(self, messages: List[Dict[str, str]],
                 params: GenParams) -> Iterator[str]:
        if self.model is None:
            raise RuntimeError("load() first")
        import numpy as np

        from .transformer import SampleParams, sample

        text = self.apply_chat_template(messages)
        prompt = self.tokenize(text)
        if not prompt:
            prompt = [0]

        rng = np.random.default_rng(params.seed)
        sp = SampleParams(temperature=params.temperature,
                          top_k=params.top_k, top_p=params.top_p,
                          seed=params.seed)

        # the forward: through the C++ block loop when it is available (hooks
        # supply attention/SSM/MoE from the verified Python primitives), else
        # the verified Python loop. The choice is recorded in load stats.
        if self._native_loop is not None:
            logits = self._forward_via_native(prompt)
        else:
            logits = self.model.forward(prompt, self.state)
        for _ in range(params.max_tokens):
            # (3) measured monitor between tokens -- if the footprint has grown
            # past the cap mid-generation, stop cleanly instead of dying.
            if self.guard is not None and self.guard.exceeded():
                yield f"\n[stopped: measured footprint exceeded --max-ram]"
                return
            nxt = int(sample(logits, sp, rng))
            yield self.detokenize([nxt]) + " "
            if self._native_loop is not None:
                logits = self._forward_via_native([nxt])
            else:
                logits = self.model.forward([nxt], self.state)

    def _forward_via_native(self, tokens):
        """One pass through the C++ block loop. Returns (vocab,) logits."""
        import numpy as np
        logits = np.zeros(self._native_vocab, dtype=np.float32)
        positions = np.arange(self.state.pos, self.state.pos + len(tokens),
                              dtype=np.int64)
        self._native_loop.forward(tokens, positions, logits)
        self.state.kv.commit(len(tokens))
        self.state.pos += len(tokens)
        return logits

    def stats(self) -> dict:
        d = dict(self._load_stats)
        if self.guard is not None:
            d["measured_bytes"] = self.guard.measured_bytes()
            d["peak_bytes"] = self.guard.peak_bytes()
        if self.model is not None:
            d["positions"] = self.state.pos
        return d

    def close(self) -> None:
        self.model = None
        self.state = None
        if self._reader is not None:
            self._reader.close()
            self._reader = None


def _own_footprint() -> Optional[int]:
    """phys_footprint of this process.

    Delegates to the existing reader in cli.py, which uses `/usr/bin/footprint`.
    phys_footprint is the right quantity because it includes GPU/Metal memory,
    which `ps -o rss` misses entirely (measured: a 512 MiB GPU allocation moves
    phys_footprint by 513 MiB and resident_size by 0).

    Returns None rather than 0 on failure, so a caller can tell the difference
    between "nothing allocated" and "nothing is measuring" -- claiming a cap is
    enforced when no measurement is happening would be the worst answer here.
    """
    try:
        from .cli import own_footprint_bytes
        return int(own_footprint_bytes())
    except Exception:
        return None


class _MeasuredCap:
    """A cap enforced by MEASUREMENT, not accounting.

    Deliberately minimal: the authoritative guard is the C++ MemoryGuard, whose
    monitor thread polls phys_footprint (which includes GPU memory -- `ps -o
    rss` does not). This wrapper exists so the runtime can ask "am I over the
    cap?" between tokens without threading the C++ class through Python.
    """

    def __init__(self, cap_bytes: Optional[int]) -> None:
        self.cap_bytes = cap_bytes
        self.peak = 0

    @property
    def measurable(self) -> bool:
        # probed live, not cached: a subclass that cannot measure (or a machine
        # where the reader fails) must report False, and that must not be frozen
        # at construction time.
        return self.measured_bytes() is not None

    def measured_bytes(self) -> Optional[int]:
        v = _own_footprint()
        if v is not None:
            self.peak = max(self.peak, v)
        return v

    def peak_bytes(self) -> int:
        self.measured_bytes()
        return self.peak

    def exceeded(self) -> bool:
        if self.cap_bytes is None:
            return False
        v = self.measured_bytes()
        if v is None:
            # nothing is measuring -- do NOT claim the cap is enforced
            return False
        return v > self.cap_bytes


def _make_guard(budget: Optional[int]) -> Optional[_MeasuredCap]:
    return _MeasuredCap(budget) if budget is not None else None


RUNTIMES: Dict[str, type] = {
    "echo": EchoRuntime,
    "gguf": GgufRuntime,
}


def accepts_kwarg(name: str, key: str) -> bool:
    """Whether runtime `name` accepts constructor kwarg `key`.

    The stubs have bare constructors, so a caller that has a device plan to hand over
    must ask first -- make_runtime() refuses unknown kwargs rather than dropping them
    silently, which is right, but it would turn `aero-chat --runtime echo` into a
    Traceback if callers just passed the plan unconditionally.
    """
    if name not in RUNTIMES:
        raise ValueError(f"unknown runtime {name!r}; available: {sorted(RUNTIMES)}")
    import inspect
    return key in inspect.signature(RUNTIMES[name].__init__).parameters


def make_runtime(name: str, **kwargs) -> ModelRuntime:
    """Build a runtime. Unknown names raise; kwargs are forwarded only when the
    target accepts them, so a stub with a bare constructor still works."""
    if name not in RUNTIMES:
        raise ValueError(f"unknown runtime {name!r}; available: {sorted(RUNTIMES)}")
    cls = RUNTIMES[name]
    if not kwargs:
        return cls()
    import inspect
    sig = inspect.signature(cls.__init__)
    accepts = {k: v for k, v in kwargs.items() if k in sig.parameters}
    rejected = set(kwargs) - set(accepts)
    if rejected:
        raise TypeError(
            f"runtime {name!r} does not accept {sorted(rejected)}; "
            f"it takes {sorted(sig.parameters)}")
    return cls(**accepts)
