# Engine design — from first principles (own engine, no fork)

No code. Design only. To be reconciled with the four in-flight investigator reports.

## 1. Restate the problem correctly

It is **not** a memory problem. It is a **bytes-read-per-token** problem under a hard memory cap.

For a 25 GB model / 64 layers:
- one layer ≈ 390 MB → trivially fits in a 6 GB arena.
- but a naive stream reads **25 GB per token** → at 3 GB/s ≈ **8.3 s/token ≈ 0.12 tok/s**.

So "can it fit" is easy; "is it usable" is the whole game. Any engine that only solves fitting
(single-layer streaming) is a toy. **The design objective is to minimise bytes read per token
while keeping resident bytes ≤ B.**

Formally: minimise `bytes_read / tokens_produced` subject to `resident ≤ B`.

## 2. The only three levers that reduce bytes/token

Everything reduces to these; an engine should exploit all three simultaneously.

1. **Sparsity — don't read what isn't used.**
   - MoE: read only active experts (8/90 → 9% of expert bytes).
   - **Dense: read only active FFN neurons.** "LLM in a flash" (arXiv 2312.11514) exploits
     ReLU/SwiGLU activation sparsity (~90% of FFN activations are zero) and loads only the
     neurons that fire. This is the dense-model equivalent of expert skipping — and it is the
     lever that makes a *dense* 25 GB model tractable at all. Needs a *predictor* (previous
     token's activation pattern predicts the next token's, temporally correlated).
2. **Reuse — amortise one read over many tokens.**
   - **Speculative decoding**: draft k tokens, verify them in ONE forward pass. Weights are read
     once for k tokens → bytes/token ÷ k. This is the single biggest lever available on a
     bandwidth-bound single machine, and it is *independent of sparsity* (multiplies with it).
   - Prefix/KV reuse for prompts; expert working-set pinning across tokens.
3. **Reclamation — never hold what you don't need.**
   - **Hybrid SSM layers need no KV** (confirmed: `qwen35moe` has `full_attention_interval=4`).
     3 of every 4 layers are SSM → constant-size recurrent state, no growing KV. KV is NOT the
     binding constraint on this model — weights are. Budget accordingly.
   - Evict by utility, not recency alone (see §4).

**Combined estimate** (25 GB dense, 3 GB/s SSD): sparsity 40% retained weights × speculative
k=4 → ~2.5 GB read per token → **~0.8 s/token (~1.2 tok/s)**. That is the target. Without
levers 1+2 it is 0.12 tok/s — an 10x difference.

**CORRECTION (measured — see §9.4):** an earlier draft of this doc claimed prefetch/IO-overlap
hides the streaming cost behind compute. **That is false on the SSD path.** Measured hiding is
only **~5 %** (SSD-LLaMA, GLM5.2) to **~10 %** (airllm) because at batch-1 the SSD read
*dwarfs* compute. Do not budget overlap as a lever — budget **fewer bytes read**. Overlap is a
minor efficiency win, not the mechanism.

## 3. Engine architecture

**A reuse-maximising weight streamer with an admission controller.** Five subsystems:

```
        ┌──────────────────────────────────────────────────────────┐
        │  ADMISSION CONTROLLER  (hard cap B; refuse > evict)      │
        │   - projected resident = weights + state + arena slack   │
        │   - mirrors exo's hard-fit gate + user's RSS watchdog    │
        └───────────────┬──────────────────────────────────────────┘
                        │ owns
        ┌───────────────▼──────────────────────────────────────────┐
        │  ARENA  — fixed byte budget, no growth                   │
        │   [ pinned trunk ][ hot blocks LRU/utility ][ scratch ]  │
        └───────┬──────────────────────┬───────────────────────────┘
                │                      │
   ┌────────────▼────────┐   ┌─────────▼───────────────────────────┐
   │ BLOCK CATALOGUE     │   │ PREFETCH SCHEDULER                  │
   │ dense: attn | ffn   │   │  double-buffer; issue read(L+1)     │
   │        neuron-tile  │   │  during compute(L);                  │
   │ moe:   attn |router │   │  MoE: router-ahead expert predict    │
   │        |expert e    │   │  dense: neuron-pattern predict       │
   │ ssm:   recursive st │   └──────────────────────────────────────┘
   └─────────────────────┘
                        │
        ┌───────────────▼──────────────────────────────────────────┐
        │  REUSE ENGINE — speculative verify pass, batching,        │
        │  expert/neuron working-set pinning                       │
        └──────────────────────────────────────────────────────────┘
```

### 3.1 Unified block abstraction (this is what makes it work for dense AND MoE)
The streamable unit is a **block**: `(layer, kind, index)` where kind ∈ {attn, ffn-tile, expert,
router, ssm-state}. Dense models yield `ffn-tile` blocks; MoE yields `expert` blocks; hybrid
yields `ssm-state` blocks. **The scheduler never knows the architecture** — a per-arch *adapter*
just enumerates blocks and their byte ranges. Same engine, three model families. This is the
clean generalisation of the MLX eager expert loader, extended to neuron-tiles so dense models
get sparsity too.

### 3.2 Arena with pinned trunk
Small, always-resident set that is *never* evicted: embeddings, all norms, router weights, and
SSM recurrent state. For the Qwen3.6 MoE this is small (tiny FFN=512 experts). Size the trunk
first, hand the remainder to the block cache. Trunk pinning is what stops latency thrash.

### 3.3 Prefetch scheduler (a minor efficiency, NOT the lever)
Strictly double-buffered: compute block i while block i+1 streams. Two prediction strategies:
- **Dense:** next block is deterministic (layer order) → trivially prefetchable. No prediction
  needed.
- **MoE:** the router for layer L is *already computed* before layer L's experts are needed, so
  read the router output → prefetch layer L's selected experts immediately. "Router-ahead" is
  exact, not speculative. For L+1, predict from L's expert pattern (temporal correlation).

**Expectation-setting (§9.4):** measured overlap is only ~5–10 % on the SSD path. Build it
because it is nearly free, not because it is the answer. Expert *prediction* (next-layer) has
real implementations (MoE-Infinity, Pre-Attention 2511.10676) but needs >60 % accuracy to pay
off, and one practitioner measurement saw only 49.5 % recall. Router-ahead (same-layer, exact)
is strictly better than cross-layer prediction — prefer it.

### 3.4 Reuse engine — speculative verify pass
Treat speculative decoding as a **first-class IO-amortisation device**, not a speed feature:
- draft `k` tokens with a small resident draft model;
- run the big model over all `k` in one pass → one weight stream per `k` tokens.
The draft model must itself be cheap and **fully resident** (it is the one thing we do not
stream). Choose `k` adaptively to the measured per-layer IO time.

### 3.5 Admission controller (the safety guarantee)
Before every materialisation: `projected = trunk + hot + incoming`. If `> B`, either evict a
lower-utility block or **refuse and fall back to a smaller batch/bigger reuse**. This is exo's
placement gate moved to runtime. Combined with an external watchdog (SIGKILL at the hard cap),
the machine physically cannot be OOMed — the two layers are complementary:
- engine refuses to over-admit (graceful degradation),
- watchdog kills if the engine is ever wrong (backstop).

**SAFETY CORRECTION (must not be got wrong — see §8.1):** the watchdog must poll
`/usr/bin/footprint -p <pid>`, **not** `ps -o rss=` and **not** `ulimit -v`. On macOS,
`ps -o rss=` does **not** see Metal/MTLBuffer/IOSurface allocations, so a 13 GB Metal mapping
reads as ~3 GB rss and OOMs the host. `ulimit -v` is a no-op for malloc. Poll at ~50 ms and
SIGKILL at the cap. An `rss`-based or `ulimit -v`-based guard must never ship.

## 4. Eviction policy — utility, not recency

Plain LRU is wrong for this workload. Score a resident block by
`utility = P(needed next token) × bytes_saved_if_resident − size`:
- MoE experts recur across a conversation → high `P`; pin them.
- FFN neuron tiles in dense models recur (sparsity pattern is temporally stable) → pin.
- Attention blocks are read once per layer per token → near-zero reuse → never pin, always stream.
This is a *knapsack over reuse probability*, which is why the predictor matters even for eviction.

## 5. Required adapters (per-arch, thin)

| Arch | Streamable blocks | State that must stay resident |
|---|---|---|
| `qwen35` dense ternary | attn, ffn-tile (neuron-tiled) | none beyond norms/embed |
| `qwen35moe` hybrid MoE+SSM | attn, router, expert, ssm-state | SSM recurrent state (fixed size) |
| `gemma4` hybrid SWA | attn (per-layer head_kv differs!), ffn-tile | none beyond norms/embed |

Note `gemma4` has a per-layer `head_count_kv` **list** (48 entries) — blocks must carry
per-layer shape, not assume a uniform layer.

## 6. Open items the investigators are resolving

1. Is there a real ternary compute path, or does `qwen35` dequant to fp16 on load?
   → decides whether the ternary target is bandwidth-cheap (packed, 2-bit) or secretly fp16.
2. Is `MADV_PAGEOUT` usable on macOS? → decides whether a passive page-cache tier is a valid
   *fallback* beneath the engine, or whether the engine must own all IO explicitly.
3. Upstream `-n-cpu-moe` semantics (experts to CPU RAM vs true disk paging) → decides whether
   part of the block catalogue can be delegated to llama.cpp instead of hand-rolled.
4. Measured IO/compute overlap fraction on a single machine → sets the achievable `k` and
   therefore the realistic tok/s target.

## 7. What is genuinely new here (vs the surveyed prior art)

- Existing systems optimise *fitting* (airllm, llama.cpp mmap) or *cluster capacity* (exo,
  Petals, prima.cpp). **This engine optimises bytes-per-token under a hard cap** — a different
  objective, and the only one that makes a single 8 GB box interactive.
- **Neuron-tile blocks give dense models the same sparsity lever MoE gets for free** — this is
  the piece the surveyed systems lack for dense models.
- **Speculative decoding treated as IO amortisation** rather than a latency trick.
- **Utility-based (knapsack) eviction** instead of LRU.

## 8. Prior-art reconciliation (grounded in the user's own accumulated work)

Recovered from archived + active skills. Several parts of this design are **already solved**, and
two of my assumptions were **wrong**. Do not rebuild what is verified.

### 8.1 Corrections to this document
1. **Watchdog metric.** Must be `/usr/bin/footprint -p PID` (top-line "Footprint:" includes GPU),
   ~50 ms poll. `ps -o rss=` misses Metal memory (13 GB Metal map → ~3 GB rss → host OOM);
   `ulimit -v` is a no-op for malloc on macOS. (Corrected in §3.5.)
2. **`lazy=True` is mandatory, and it is NOT "disk paging".** In MLX,
   `load_model(path, lazy=True, strict=False)` mmaps so tensors fault in on first use.
   `lazy=False` materialises **all** experts of every MoE layer *before* any name filter can drop
   them → ~20 GB vs a ~4.5 GB full load (a 4.4× regression). Never `lazy=False`.
3. **The verified byte numbers are in the prior art, not estimates.** Use them, don't derive:

| Quantity (Qwen3.6-**35B**-A3B, Q4_K_M, 21 GB) | Verified value |
|---|---|
| Experts / active | 256 / 8 (note: one doc also says "top-2" — reconcile before trusting) |
| Layers | 40 |
| Per-expert payload | gate 576 KB + up 576 KB + down 840 KB = **1,992,192 B** |
| Per-expert **stride** (incl. padding) | gate 589,824 · up 589,824 · down 860,160 |
| **Peak active expert working set** | 40 × 8 × 1.99 MB ≈ **638 MB** |
| Compute buffers, remap approach | 16 MB Metal + 498 MB CPU ≈ **514 MB** (vs ~20 GB full) |

   ⚠️ Use the **stride** for `pread` arithmetic, not the payload — they differ by internal padding.
4. **Experts are stored contiguously** in these GGUFs → a per-expert `pread` needs only
   `offset + e*stride`. This is why the block catalogue's byte-range addressing works.

### 8.2 The MoE half is essentially solved (verified)
The "**expert remap**" approach ran **Qwen3.6-35B-A3B Q4_K_M (21 GB) on a 16 GB box** with
~514 MB of compute buffers and 16 expert slots/layer (~1.2 GB), 1 expert preloaded. That is your
25 GB class, on your hardware class. The engine's MoE path should *reuse* this, not reinvent it.

### 8.3 Bugs already paid for — inherit, don't rediscover
- **CPU_REPACK format mismatch (critical).** The `vec_dot` kernel expects repacked
  (`block_q4_Kx8`), GGUF stores `block_q4_K`. Handing the mmap pointer to a REPACK-backed tensor
  → **silent garbage**. Correct fix: in `buft_for_tensor()`, when `expert_no_alloc` is set and the
  name matches `ffn_gate_exps|ffn_up_exps|ffn_down_exps`, return `ggml_backend_cpu_buffer_type()`.
  Do **not** set `t->data` to the mmap pointer, do **not** repack only selected experts, do **not**
  remove CPU_REPACK from `ctx_map`, do **not** pre-assign `t->data` before allocation (SIGSEGV).
  *(The user has flagged this file area as untouchable twice — treat CPU_REPACK as load-bearing.)*
- **`gather_qmm` (MLX) shape/bias conventions**: expand `x` to `[B,T,1,1,H]` then `squeeze(-2)`;
  mxfp4 rejects biases (`Biases must be null`) → add bias after the call via
  `mx.expand_dims(B[local], -2)`; `mx.compile(..., shapeless=True)` fails ("GatherQMM cannot infer
  output shapes") → use plain `mx.compile`.
- **Packing factor**: `in_dims = packed_dim * (32 // BITS)`, not `8 // BITS`.
- **`local` map semantics**: `page_layer` returns a `[B,T,K]` position→slot map; write manual
  ground truth at router **position**, not slot (classic 30 %-wrong scatter bug).
- **bf16**: `(u16.astype(uint32) << 16).view(float32)`, never `uint16 → float32 → bfloat16`.
- **Down-proj transpose asymmetry**: gate/up dequant to `[out,in]` → `h @ W.T`; down is the
  opposite → `act @ dd.T`. Verify each projection's shape.
- **Testing**: LFM2.5 layers 0–9 are **dense** FFN — point tests at a MoE layer (L=2 or 10), never L=0.
- **Chat template is mandatory** for gpt-oss (raw tokens → coherent first token, then garbage).

### 8.4 What is still genuinely new in this design
The prior art is **MoE-only expert paging**. Unchanged novelty:
- **dense neuron-tile sparsity** (the "LLM in a flash" lever) — gets dense models the sparsity
  MoE has natively; nothing in the prior art does this;
- **speculative decoding as IO amortisation** (multiplies with sparsity);
- **utility/knapsack eviction** rather than pure LRU;
- **the unified block abstraction** that lets one scheduler run dense, MoE, and SSM-hybrid.

### 8.5 Upstream pointers worth reading before building
- llama.cpp **issue #25257** — SSD streaming of MoE expert weights, async prefetch design.
- llama.cpp **PR #14753** — explicit expert aggregation (avoids page faults during warmup).
- `--n-cpu-moe N` — keeps N MoE layers on **CPU**, i.e. still RAM-resident. Solves VRAM, **not**
  a RAM cap. Not a substitute for disk paging.
- MLX path pitfalls + launch plumbing: skill `mlx-lazy-moe-loading` (active) and its
  `references/watchdog-launch.md`, `references/lfm2-lazy-correctness.md`.

## 9. Investigator verdicts (four parallel agents, evidence-cited)

### 9.1 Ternary — REAL, but only in the PrismML fork
**Decode is genuinely packed-domain, no fp16 materialisation** — so the ternary model keeps its
bandwidth advantage where it matters (batch-1 decode is the binding constraint).
- `GGML_TYPE_PQ2_0 = 142`, `GGML_TYPE_PTQ1_0 = 143` — `PrismML-llama.cpp/ggml/include/ggml.h:435-437`.
- Metal: `mul_mv.metal:3-4` ("dot = d * (2*Σ(y where bit=1) - sumy)"), PQ2_0 `:81-82`;
  PTQ1_0 `ptq1_0_dot_reg` `:886` decodes trits in registers — "never leaves the float pipe".
- CPU: `ggml-cpu.c:228,234,240,246`; body `quants.c:164-173` is `(mask&0x01) ? qy[0] : -qy[0]`.
- Loader does **not** upcast: `llama-model-loader.cpp:778-779` maps the types to an ftype label only.
- **Caveat:** PTQ1_0 *prefill* (batch>1) dequantizes **inside** the kernel (`mul_mm.metal:747,775`).
  Decode keeps the fused path.
- **Correction to my own doc:** "ggml type 141" does **not exist** — 43..141 are deliberately
  unused (`ggml.h:433-434`). `141` is the *file type* `LLAMA_FTYPE_MOSTLY_PQ2_0` (`llama.h:159`),
  which resolves to ggml type **142**. Type **30** = `GGML_TYPE_BF16` (`ggml.h:420`), while
  *file type* 30 = IQ4_XS.
- **Consequence:** this advantage is **contingent on running `PrismML-llama.cpp`**. The
  `MoE_llama` and `ggml-llama.cpp` forks stop at `COUNT=43` — no PQ2_0/PTQ1_0.
- **UNKNOWN:** MLX has no equivalent ternary kernel found.

### 9.2 macOS madvise — prefetch VIABLE, eviction NOT VIABLE
OS observed: macOS 15.7.4 / Darwin 24.6.0 arm64.
- `MADV_WILLNEED` — defined (`sys/mman.h:214`), **honoured**, probe ret=0.
- `MADV_PAGEOUT` — defined (`:221`, "internal only") but **ENOTSUP(45)** on RO file, RW file and
  anon. XNU `bsd/kern/kern_mman.c:1461` → `return ENOTSUP`; the real path is behind
  `#if MACH_ASSERT` (debug kernels only). Even then `vm_map.c:17418` skips non-internal objects.
- `MADV_COLD` **absent** from every SDK. `MADV_FREE_REUSABLE`/`MADV_ZERO` → **EPERM** on RO.
- `MADV_DONTNEED` is only an LRU demote (`vm_map.c:16750`), weaker than Linux.
- **macOS eviction is automatic and pressure-driven** (pageout daemon / jetsam) — never
  caller-controlled. Metal wraps the mapping via `newBufferWithBytesNoCopy`, so pages are dropped
  under pressure and **refaulted on the next prefill** (llama.cpp discussion #29347;
  `--mlock`/`--no-mmap` stops it).
- **Verdict: MARGINAL.** Read-side works, eviction-side does not. Fallback: `MAP_SHARED` mmap +
  `MADV_WILLNEED` to prefetch + **`munmap` slices when done** to release refs so the pageout
  daemon can evict; `mlock` a hot subset. Never rely on explicit pageout.

### 9.3 Engine choice — upstream is NOT sufficient for a RAM cap
- `-cmoe/--cpu-moe` and `-ncmoe/--n-cpu-moe N` move experts to the **CPU buffer** (system RAM).
  On unified memory that is the **same physical RAM** → relieves the ~10.9 GB Metal ceiling,
  **not** the RAM budget. (`common/arg.cpp:2472-2492`, PR #15077; discussion #19163: "llama.cpp
  will only load it to system memory".)
- `--no-mmap` is **worse** for RAM. Upstream even warns about combining overrides with mmap
  (`llama-model-loader.cpp:1176`).
- True upstream disk paging is **UNMERGED** RFC+PoC **#23324** (`--moe-n-slots`/`--moe-n-layers`,
  Metal `kernel_moe_interceptor` + pread + LRU) — and it ran **Qwen3-30B-A3B-Q6_K on 16 GB at
  13 tok/s**.
- **Real header, `Qwen3.6-14B-A3B` Q4_K_M (8,465,553,920 B):** experts pageable **6.386 GiB
  (81.1 %)**, non-expert resident **1.488 GiB (18.9 %)**, avg **1.816 MiB/expert**; decode
  working set 8×40 = 320 experts ≈ **0.58 GiB**. **KV exists only on the 10 full-attention
  layers** ≈ 20 KiB/token (~640 MiB @32k); the other 30 layers are gated-delta-net (constant
  state). `full_attention_interval=4`.
- **16 GB:** upstream with no fork is plausibly sufficient (`-ngl 999 --n-cpu-moe 40` + mmap).
  **8 GB:** not sufficient. MLX eager remains the right carrier for a deterministic cap.
- **Gap to close:** the MLX prototype reads **safetensors**, the on-disk models are **GGUF** →
  needs conversion. `mlx_lazy_lfm2.py` (attention+conv hybrid) is the closer template for this
  arch family than the gpt-oss file.
- **Housekeeping found:** `test_expert_store.py` is stale (calls removed API:
  `store.index`/`weight_map`/`get_expert`); `scripts/moe_run.sh` hard-locks a **6 GiB** cap.

### 9.4 Frontier — the lever is FEWER BYTES, not overlap
**This is the finding that rewrote §2/§3.3.** Measured IO/compute overlap on the SSD path:
- SSD-LLaMA (2609.18110): only **~5 %** of transfer latency hideable — "the time required to
  transfer the expert is significantly longer than the computation time."
- airllm prefetch: **~10 %**.
⇒ At batch-1 the SSD read dominates; overlap cannot hide it.

**What actually works (measured):**
| System | Mechanism | Result |
|---|---|---|
| MoE-Infinity (2401.14361) | sparsity-aware expert cache + activation tracing + prefetch | **3.1–16.7×** TPOT |
| **NunSpark / vates** (MLX, Apple Silicon) | MLX MoE expert streaming + expert-aware cache + temporal prefetch | **16 GB M4: 0.5 → 7–8 tok/s**; ~**18 MB/token vs ~1 GB**; prefetch 2.4× |
| LLM in a Flash (2312.11514) | activation windowing + row-column bundling | I/O **2196 → 87 ms**; **1.25 → 2.25 GB/s** |
| Pre-Attention expert pred. (2511.10676) | predict experts before attention | 93–97 % accuracy; disk load 33.5–49.8 ms vs 4.0–9.5 ms in-memory |
| prima.cpp PRP | per-round prefetch↔compute to avoid cache over-fill | −50 % TPOT (multi-device) |

**THEORETICAL only — do not present as measured:** "overlap hides most of the streaming cost"
(**disproven**, ~5–10 %); deep prefetch beyond RAM (prima.cpp shows the prefetch-release
conflict makes it *harmful*); Petals' network-beats-SSD break-even (swarm only, never measured
single-machine); SSD-LLaMA's 3-tier design on unified memory (untested — no H2D tier exists).

**Expert prediction is NOT a research gap** (MoE-Infinity, Pre-Attention, SP-MoE 2510.10302,
PreScope 2509.23638, DuoServe-MoE 2509.07379, FineMoE 2502.05370, vates). But it needs **>60 %
accuracy** to pay off and a practitioner measurement saw only **49.5 %** recall. On the SSD path
even a correct prediction hides only ~5 %.

**⚠️ Prior-art overlap to resolve:** **NunSpark/vates already implements much of §3 on MLX** and
reports 7–8 tok/s on a 16 GB M4. Study it before building — it may subsume the MoE half outright.
