# Aero — memory guard

`--max-ram <size>` hard-caps the (V)RAM the engine may use, checked continuously, with a load that
stops the instant the cap would be or is exceeded.

```
make            # builds the test and the demo
make test       # 62 checks
./build/aero-load-demo --max-ram 8 --simulate      # refuses a naive 8.47 GiB load
```

## Usage

```
aero --max-ram 8            # 8 GiB  (bare numbers are GiB — the colloquial "8GB of RAM")
aero --max-ram 8G           # 8 GiB
aero --max-ram 8GB          # 8.0e9 bytes (decimal)
aero --max-ram 512M         # 512 MiB
aero --max-ram 8589934592   # raw bytes (bare values >= 2^20 are read as bytes)
```

The resolved byte count is always logged, so the convention in force is never a guess.

## How it enforces the cap — four gates, in order

| # | Gate | When | What it does |
|---|---|---|---|
| 1 | **Predictive plan** | before any disk read | `admit_plan()` compares the whole load plan to the cap and **refuses up front** |
| 2 | **Pre-flight** | before any allocation | refuses to start if the host lacks `cap + headroom` reclaimable memory |
| 3 | **Exact accounting** | every allocation | `try_reserve()` refuses any allocation that would exceed the cap; consults the evictor first. Runs on *every* reservation, so it is continuous by construction, not by polling |
| 4 | **Measured monitor** | continuously, ~50 ms | polls the real footprint and fires a breach if reality exceeds the cap for any reason the accounting cannot see |

A load in flight is stopped by **`LoadGate`**, which reserves one chunk at a time and re-checks.
With `set_chunk_pause_ms(n)` the loader **sleeps between chunks**, which is what gives the monitor a
window to sample the true footprint *while the memory is held* — this is how a breach that the
accounting cannot see (baseline process memory, allocator slack, GPU driver allocations) is caught.

`LoadGate` distinguishes two outcomes that must never be conflated:

- `ok() == true && done()` → the plan was fully loaded — **success**
- `ok() == false` → the guard stopped the load — **failure**

## Why it is measured this way (macOS specifics)

- **`ulimit -v` is not used.** It is a no-op for malloc on macOS.
- **RSS is not used.** `ps -o rss` does not include Metal/GPU allocations, so a process holding a
  large GPU mapping can look tiny and still OOM the host.
- **`phys_footprint` is used** (`task_info(TASK_VM_INFO)`), which *does* include GPU memory and is
  the same number `/usr/bin/footprint` reports. It is read in microseconds via Mach, so a 50 ms
  poll costs nothing — versus spawning `footprint` as a process every tick.
  The test asserts the two agree (observed: 901760 vs 902144 bytes).

## Failure modes it defends against

| Failure | Caught by |
|---|---|
| a plan that cannot fit, discovered halfway through loading | gate 1 (predictive) |
| the machine already being under memory pressure | gate 2 (pre-flight) |
| an allocation pushing past the cap | gate 3 (accounting) — *before* the allocation |
| memory the accounting never tracked (GPU driver, allocator slack) | gate 4 (measured) |
| the budget being consumed by another component mid-load | gate 3 via `LoadGate` |

## Evidence

- `make test` → **118 checks, 0 failures** (62 guard + 13 Metal/GPU + 43 engine)
- the real model plan (8.47 GiB) against `--max-ram 8` → **refused before loading**, with the reason
- a paced 136 MiB load inside a 256 MiB cap → **completed**, ledger `200.00 MiB / 256.00 MiB`

### The GPU-memory proof (`tests/test_metal_guard.mm`)

The guard's central design choice is measuring `phys_footprint` instead of RSS. That is now proven
against a real GPU allocation, not assumed:

```
allocating and filling 512 MiB of GPU-ONLY memory (MTLResourceStorageModePrivate)

  phys_footprint   5.0 MiB -> 517.9 MiB   (delta +512 MiB)   <- sees it
  resident_size    9.2 MiB ->  10.1 MiB   (delta  +0 MiB)    <- blind to it
```

**A guard built on `rss` would have missed 512 MiB of real GPU memory.** That is exactly the failure
mode behind a process looking small and still OOM-ing the host. The test then proves the guard *acts*
on it: the continuous monitor fires, admission refuses a further 512 MiB plan, and the load gate
closes on the measured footprint.

### Two design contracts the test pins down

1. **The two layers have different scopes, deliberately.** `admit_plan()` is the *accounting* gate —
   it reasons only about bytes the engine declared, so a GPU allocation it never tracked is invisible
   to it (`admit_plan(1)` legitimately returns true even over budget). The *measured* layer
   (`check_measured()`, the monitor) is what catches untracked memory. **Both are required**; either
   alone has a hole.
2. **Metal does not always return GPU memory immediately.** After releasing the buffer the footprint
   stayed at 518 MiB (driver cache). That is a recorded property, not a bug: a reading that lags
   **high** is the safe direction — it makes the guard refuse more, never less. The test asserts the
   footprint is *bounded* after release rather than that it drops.

---


## Serving layer — OpenAI API + CLI chat

```sh
./aero-serve --model my-model --port 8000          # OpenAI-compatible API
./aero-chat                                        # CLI chat
./aero-chat --server http://127.0.0.1:8000         # CLI against a running server
```

Stdlib only — no web framework, no dependencies.

| method | path | notes |
|---|---|---|
| GET | `/v1/models` | OpenAI `list` shape |
| POST | `/v1/chat/completions` | `stream=true` (SSE) and `false` |
| POST | `/v1/completions` | legacy text completion |
| GET | `/health` | uptime, devices, runtime |

Streaming is real SSE — `data: {...}\n\n` terminated by `data: [DONE]`, with an initial
`role: assistant` chunk and a final `finish_reason: stop` chunk — so existing OpenAI clients work
unchanged. `stop`, `max_tokens`, `temperature`, `top_p`, `seed` are honoured; errors use the OpenAI
error shape. The CLI chat has multi-turn history, streamed output and
`/help /reset /stats /devices /exit`, and accepts the engine's device flags.

### ⚠️ What is NOT done — read before expecting answers

**The runtime is a dev stub.** `--runtime echo` is deterministic and exists so the API and CLI could
be built and tested end to end. **It is not a language model.**

`--runtime gguf` is declared but **deliberately raises `NotImplementedError`** and exits 3 with a
clear message — no silent fallback, no plausible-looking fake text. Remaining work:

- GGUF parse (hparams, tokenizer, tensor index)
- attention kernels, the SSM/gated-delta-net scan, MoE routing
- KV cache, sampling, decode loop
- wiring those onto `AeroEngine`'s device split under the `--max-ram` guard

**So: the serving layer and the device layer are complete and verified; the transformer is not
built.** Aero today is an API + device-allocation skeleton with a working memory guard.

---

## Transformer — core built and verified

| module | contents |
|---|---|
| `engine/gguf.py` | GGUF v2/v3 reader (metadata, tensor index, byte-range reads) **+ writer** for fixtures |
| `engine/quant.py` | dequantisers: F32, F16, BF16, Q4_0/1, Q5_0/1, Q8_0, Q2_K, Q4_K, Q5_K, Q6_K |
| `engine/transformer.py` | RMSNorm, RoPE (full + **partial**), GQA attention with QK-norm, SwiGLU, KV cache, sampling, causal masking |

**Verification** (`make test` → 230 checks, 0 failures):

- **Structural:** parsing Qwen3.8-4B, the summed tensor bytes end at
  `2783446304` and the file is `2783446304` — **slack = 0**. That single check
  validates the parser and *every* block size at once.
- **Bit layouts:** K-quant super-blocks built field-by-field with `d=1, dmin=0`
  and all sub-scales forced to 1, so the correct output is known analytically
  (`Q4_K` → output *is* the quant; `Q6_K` → `quant-32`).
- **RoPE:** the defining property — the attention score depends only on *relative*
  position (identical at (3,1) and (103,101)), and rotation preserves the norm.
- **Attention:** matches an independently written loop-based reference to 9.5e-07;
  editing token 5 leaves outputs 0–4 bit-identical (no future leakage, and the
  test is non-vacuous).
- **KV cache:** token-by-token decoding equals one batched forward pass to 1.1e-06
  — the classic decoder invariant.
- **Sampling:** temperature verified *quantitatively* — P(dominant token) = 0.814
  measured vs 0.809 predicted at T=2.

### Bugs the tests caught (all real)

1. **`_f16` returned shape `(nb,1)` instead of `(nb,)`.** With one block it
   broadcasts fine; with many, `d * scales` broadcasts to `(nb,nb)`. The
   hand-built single-block test passed while **real files crashed** — which is
   why the real-file check earns its place.
2. **GQA head expansion was missing** — the einsum could not broadcast 4 query
   heads against 2 KV heads.
3. Two of my own test expectations were wrong (Q6_K high-nibble paths, and
   assuming T=2 is deterministic).

### ⚠️ Still not built — no real model runs yet

- **gated-delta-net SSM layer** — **DONE**, see below.
- **MoE** — `ffn_*_exps` / `ffn_*_shexp` / `ffn_gate_inp`, top-k routing + shared
  experts. Blocks `qwen35moe` only.
- **gemma4 extras** — sliding-window pattern, per-layer embeddings, logit
  soft-capping.
- **Model assembly** — embedding → per-layer (norm → attn *or* ssm → residual) →
  output norm → LM head. The pieces exist; the loop that ties them together does not.

---

## SSM layer — gated delta net (`engine/ssm.py`)

24 of Qwen3.8-4B's 33 layers are SSM, so nothing runs without this. The
recurrence is derived from the published formulation (Yang/Kautz/Hatamizadeh):

```
g     = -exp(A_log) * softplus(alpha + dt_bias)     log-space decay
beta  = sigmoid(beta_proj)
S     = S * exp(g)                                  decay the state
v_new = (v - k @ S) * beta                          delta rule
S     = S + outer(k, v_new)                         rank-1 write
o     = scale * (q @ S)                             read
```

### Verified

| check | result |
|---|---|
| `SsmConfig` from the real hparams vs **all 9 SSM tensor shapes in the file** | **exact match** |
| layout closes | q 2048 + k 2048 + v 4096 = 8192 = `attn_qkv` width |
| recurrence vs a naive triple-loop reference | **1.3e-08** |
| causality | editing token 5 leaves outputs 0–4 identical |
| state size | **constant** (128 B here) across 1 / 64 / 256 tokens |
| gate limits | beta→0 writes nothing; full decay erases the past (two different initial states converge) |
| real weights | `exp(g)` ∈ (0,1] for every sampled decay — a valid gate |

### ⚠️ Four assumptions that remain UNVERIFIED

The shapes are read from the real file and the arithmetic closes exactly. What the
file does **not** specify — and which would silently produce garbage if wrong — is
isolated in one place each so it can be flipped once a real-model check exists:

1. **q and k are L2-normalised per head**, scale applied to q. (Standard for GDN,
   and consistent with SSM layers having no `q_norm`/`k_norm` weights — a
   parameter-free normalisation leaves no trace in the file.)
2. **Output gate is `silu(gate) * rms_norm(o)`**, applied before `ssm_out`.
3. **Conv is causal and depthwise** over the fused qkv channels.
4. **v-head h reads k-head `h // (n_v/n_k)`** — grouping direction is convention.

**This layer is structurally verified, not numerically verified against a real
model's output.** The test that would settle it is a real forward pass compared
against a known-good logits vector, which needs the model assembly first.


---


```sh
./aero --limits                                  # detected device limits
./aero --plan                                    # default allocation (gpu=1)
./aero --use-cpu=6 --use-gpu=0 --plan            # cpu=6 -> 4 performance + 2 efficiency
./aero --use-cpu=2 --use-gpu=1 --use-ane=4 --bench 5 --rows 4096 --cols 1024
```

| flag | range | default |
|---|---|---|
| `--use-cpu=N` | 0..**8** (detected) | 0 |
| `--use-gpu=N` | 0..**8** (detected) | **1** |
| `--use-ane=N` | 0..**16** (documented) | 0 |

Counts, not booleans — **0 disables a device**, and every combination works. No flag at all means
`--use-gpu=1`. All-zero is a hard error, never a silent fallback.

### CPU allocation is performance-first

`--use-cpu=6` on this 4P+4E machine → **4 performance + 2 efficiency**. The first `performance`
streams are tagged `QOS_CLASS_USER_INITIATED` and the remainder `QOS_CLASS_UTILITY` — on macOS that
QoS class is the only mechanism that expresses a P-core vs E-core preference (there is no CPU-affinity
API).

### What a count can and cannot mean — read this

- **CPU — the count is real.** N streams run on N cores; CPU work is genuinely parallel.
- **GPU / ANE — the count is streams, NOT a hardware core pin.** Apple exposes no API to use "N of 8
  GPU cores"; Metal and CoreML present each device as a single execution resource. So `--use-gpu=8`
  means 8 parallel work streams on the one GPU, with a proportional share of the work.
- **`--use-ane`'s limit is documented, not detected** — no public API reports the ANE core count. It
  is labelled `[DOCUMENTED (not detected)]` everywhere it appears, including `--limits`.

Measured behaviour backs this up: **more streams on one unit contend rather than add.** GPU+ANE
side-by-side measured **0.89×** of either alone, and CPU+GPU concurrent **0.56×** of GPU alone
(see `PERFORMANCE_BUDGET.md` §12–13). The flags are implemented faithfully and all combinations run
correctly — they are just not a speed lever on this hardware.

### Verification

`tests/test_engine.py` (43 checks) proves, for **every one of the 7 non-empty combinations**:

- it parses, and counts are range-checked (`limit+1` and `-1` both rejected)
- it **runs** and matches a NumPy reference (rel. err ~1e-4, fp16 CoreML vs fp32)
- **every allocated device actually executed** — asserted via per-backend call counters, so "must be
  used" is measured, not assumed
- row spans partition the matrix exactly: no gap, no overlap, sums to the row count
- CPU streams are tagged performance-first

`check-py` compiles all Python sources, so no Python file can silently fall out of the gate.

