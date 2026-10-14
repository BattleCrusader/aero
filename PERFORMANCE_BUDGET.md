# Speed budget — can we hit 10–20 tok/s?

New requirement (user, 2026-10-14): memory alone is not the goal — **make the engine usable at
10–20 tok/s.** This document works the budget from measured hardware numbers. It changes the
design objective from *stream-first* to *cache-first*.

## 1. Measured tiers (this machine: Apple M1, 8 cores, 16 GB)

| Tier | Measured | How |
|---|---|---|
| USB drive (`/Volumes/Backup`, models live here) | **0.70 GB/s** | `dd` 1 GB from a GGUF |
| Internal SSD (cold) | **2.26 GB/s** | `dd` 1 GB from an app binary |
| Internal SSD (warm / page cache) | 7.79 GB/s | second pass |
| M1 peak memory bandwidth | **68.25 GB/s** | spec (base M1, 8-channel LPDDR4X) |

⚠️ **The models are on a USB drive.** That is a **3.2× penalty** vs the internal SSD before any
engine work begins.

⚠️ Free space: internal has only **28 GiB** available; the USB has 272 GiB.

## 2. The governing equation

At batch-1 decode (no reuse), every token must be read through once:

```
tok/s  =  BW_effective  /  bytes_active_per_token
```

`bytes_active_per_token` is the whole model for **dense**, and only the **active experts** for MoE.
That single difference decides everything.

Take `BW_effective = η · 68.25` with `η ∈ [0.4, 0.6]` (Metal kernel efficiency — an assumption to be
validated by measurement, §5).

## 3. Dense: the target is unreachable, at any memory configuration

| Model size | η = 0.4 | η = 0.6 | peak (η = 1.0) |
|---|---|---|---|
| 6.7 GB (Bonsai PQ2) | 4.1 t/s | 6.1 t/s | 10.2 t/s |
| 7.9 GB (Qwen 14B-A3B as dense) | 3.5 t/s | 5.2 t/s | 8.6 t/s |
| 21 GB | 1.3 t/s | 2.0 t/s | 3.2 t/s |
| **25 GB** | **1.1 t/s** | **1.6 t/s** | **2.7 t/s** |

**A 25 GB dense model tops out at 2.7 tok/s on an M1 even with 100 % efficient kernels and infinite
RAM.** It is a *memory-bandwidth* ceiling, not a memory-capacity problem — no engine, cache, or
streaming trick can move it. Dense models above ~10 GB are simply off the table for 10 tok/s.

**Consequence: the 10–20 tok/s target requires a MoE model.** That is not a preference, it is the
only architecture whose `bytes_active_per_token` is small enough.

## 4. MoE: the target IS reachable — but only near-fully-resident

Both candidate MoEs are **A3B (3B active params)**:

| Quant | active bytes/token | η = 0.4 | η = 0.6 |
|---|---|---|---|
| 0.55 B/param (Q4-class) | **1.65 GB** | **16.5 t/s** | **24.8 t/s** |
| 0.25 B/param (Q2-class) | 0.75 GB | 36.4 t/s | 54.6 t/s |

**At realistic Q4, both the 14B-A3B and the 35B-A3B sit exactly in the target band — but only if the
active weights are served at memory speed.**

### The catch: cache hit rate must be near-perfect

Let `h` = fraction of the active set served from RAM, `1−h` from disk:

| Disk tier | Target | Required `h` |
|---|---|---|
| Internal SSD (2.26 GB/s) | 10 tok/s | **94.1 %** |
| Internal SSD (2.26 GB/s) | 20 tok/s | **101.6 % — impossible** |
| USB (0.70 GB/s) | 10 tok/s | **98.3 %** |
| USB (0.70 GB/s) | 20 tok/s | **100.5 % — impossible** |

**Read that carefully:** 20 tok/s is unreachable *with disk in the loop at all* — it requires the
active set ~100 % RAM-resident. 10 tok/s is reachable but demands 94 % (internal SSD) or 98 % (USB).

**So the design goal is not "stream efficiently" — it is "never touch the disk."**

## 5. What this means for the engine (the objective inverts)

My earlier design (§2 of `NOVEL_ENGINE.md`) optimised *bytes read per token* and treated the stream
as the mechanism. That is now the wrong emphasis. The revised objective:

> **Maximise RAM residency of the active-expert set.** Disk streaming is the cold-miss fallback,
> not the engine.

Concretely, the levers re-rank:

1. **Hot-expert cache in RAM (dominant).** A ~6–8 GB expert cache must cover ≥94 % of routing
   traffic. This is entirely determined by **routing skew** — the single most important unknown
   (§6).
2. **Coalescing / union across a speculative batch.** Reduces *distinct* experts per token, so a
   given cache covers more traffic. Multiplies with (1).
3. **Speculative decoding.** Amortises one read over k verified tokens.
4. **Storage tier placement.** Move the model to the internal SSD (2.26 GB/s vs 0.70) — worth 3.2×,
   and it lowers the required `h` from 98 % to 94 %. Costs 21 GB of the 28 GiB free.
5. Compression: **closed**, measured 1.01× ceiling (see `NOVEL_ENGINE.md` §11).

### Two immediate consequences

- **Qwen3.6-14B-A3B (7.88 GiB) already fits in 16 GB.** It needs *no engine at all* — load it fully
  and it should land in the 16–25 tok/s band at η 0.4–0.6. **This is the baseline that must be
  measured first**, and it also calibrates η.
- **Only models larger than RAM need the engine** — i.e. the 21–25 GB class.

## 6. The decisive unknown: routing skew

Everything in §5(1) hinges on one number: **how concentrated is expert selection?** If routing is
Zipfian (a few experts dominate), a 6–8 GB cache can cover 94 %+ of traffic with a lossless design.
If routing is near-uniform, `h` can never reach 94 % and **10 tok/s is unachievable for any model
that doesn't fit in RAM.**

This is measurable and cheap: run an MoE **small enough to be safe** (gpt-oss-20b, 3.2 GB, already
verified working under the watchdog) with routing visible, over a real corpus, and compute the
cache-hit-rate curve `h(cache_size)`. That curve is the go/no-go for the whole design.

**Gate:** if `h(8 GB) < 0.94`, then for models larger than RAM the honest answer is *"10 tok/s is not
reachable; use a smaller quantisation so the model fits in RAM instead."*

## 7. The pragmatic alternative worth stating plainly

Because 20 tok/s needs ~100 % residency, and because `h` must beat 94 % at 10 tok/s, there is a much
simpler route to the target for the 21–25 GB class:

> **Quantise until it fits in RAM.** A 35B-A3B at Q2-class (~0.25 B/param) is ~10 GB total with
> 0.75 GB active/token → **36–55 tok/s** at η 0.4–0.6, with *no disk in the loop at all*, and no
> exotic engine.

That trades model quality for speed and simplicity. Given that lossless compression is provably
dead (§11 of `NOVEL_ENGINE.md`), **lossy quantisation is the only remaining lever that removes disk
traffic entirely** — and disk traffic is what the 10–20 tok/s target forbids.

So the decision the user actually faces is:

| Route | Mechanism | Speed | Cost |
|---|---|---|---|
| **A. Fit-in-RAM** | lower quant (Q2/IQ2) until resident | 36–55 t/s | model quality |
| **B. Cache-first engine** | 6–8 GB hot-expert cache, 94 %+ hits | 10–16 t/s | complexity; **gated on skew** |
| **C. Dense anything** | — | ≤2.7 t/s | **impossible** |

## 9. Concrete target: Qwen3.6-14B-A3B in 8 GB at ≥10 tok/s

**Spec (user, 2026-10-14): this model, ≤8 GB, ≥10 tok/s.**

Measured inputs (GGUF header + prior verification):

| Quantity | Value |
|---|---|
| file | **8.466 GB** (7.884 GiB) |
| non-expert (must be resident) | 1.598 GB |
| expert pool (pageable) | 6.857 GB |
| ACTIVE per token (320 experts × 1.816 MiB) | **0.609 GB** |

### 9.1 Why the engine is required — but only just

The weights alone are **8.466 GB**, against an 8 GB cap (8.000 GB) or even 8 GiB (8.590 GB).
So it **cannot** be fully resident; but the shortfall is only **~0.6–1.2 GB of the expert pool
(8–22 %)**. This is a *far* easier problem than the 25 GB class — the engine must shed a slice of
the experts, not stream a model.

### 9.2 Resulting speed (uniform-routing worst case)

| Cap | overhead | cache | coverage | **internal SSD** | USB |
|---|---|---|---|---|---|
| 8.590 GiB | 0.40 GB | 6.59 GB | 96.1 % | **31.4 t/s** | 18.2 t/s |
| 8.590 GiB | 0.70 GB | 6.29 GB | 91.8 % | **23.4 t/s** | 10.9 t/s |
| 8.000 GB | 0.40 GB | 6.00 GB | 87.5 % | **18.8 t/s** | 7.8 t/s |
| **8.000 GB** | **0.70 GB** | **5.70 GB** | **83.2 %** | **15.6 t/s** | **6.1 t/s ✗** |
| 8.000 GB | 1.00 GB | 5.40 GB | 78.8 % | 13.4 t/s | 4.9 t/s ✗ |

Hit rate required for ≥10 tok/s: **68.6 %** (internal SSD) vs **90.8 %** (USB).

### 9.3 Verdict: YES — achievable, with two hard conditions

**✅ Target met on the internal SSD: 13–31 tok/s** (all configs clear 10 tok/s with margin).
**❌ Target fails on the USB drive at an 8 GB cap** (6.1 tok/s) — only the looser 8 GiB cap + low
overhead scrapes 10.9 t/s.

**Condition 1 — move the model to the internal SSD.** This is the single highest-leverage action:
2.26 GB/s vs 0.70 GB/s, and it drops the required hit rate from 90.8 % → 68.6 %. Cost: 8.47 GB of
the 28 GiB free.

**Condition 2 — keep runtime overhead ≤0.7 GB.** Overhead moves the result more than anything else
in the table (18.8 → 13.4 t/s across 0.4–1.0 GB). **This must be measured, not assumed** — it is the
one number in this analysis that is not yet observed.

All figures are **uniform-routing worst case**. Real MoE routing is skewed, so effective coverage
exceeds the size fraction and the real numbers should be *better*, never worse.

### 9.4 The design, now concrete

Not a streaming engine — an **expert cache**:

```
resident   : non-expert weights 1.598 GB  (attention, SSM state, embeddings, routers, norms)
cache      : expert pool, budget ~5.4–6.6 GB, LRU/utility eviction
overhead   : ≤0.7 GB  (must be measured)
total      : ≤ 8.0 GB  (hard cap, footprint-watched)
miss path  : pread from INTERNAL SSD (2.26 GB/s), not USB
```

The whole engine reduces to: *hold as much of the 6.86 GB expert pool as fits, and make the misses
cheap.* No speculation, no coalescing, and no compression are needed to clear 10 tok/s — those are
upside for the harder 25 GB class, not prerequisites here.

### 9.5 Simpler alternative worth stating

If the goal is only "this model, 8 GB, ≥10 tok/s", a **slightly smaller quant solves it with no
engine at all**: a Q3_K_M-class file (~6.5 GB) fits entirely in 8 GB, so nothing is paged and the
speed is memory-bandwidth-bound only — comfortably above 10 tok/s. The engine is needed to keep
*this* Q4_K_M file, not to make the model usable in general.

## 10. Compute-buffer overhead — derived from source (no execution)

The last unknown was the runtime overhead. Derived from `PrismML-llama.cpp/src`:

| Finding | Location |
|---|---|
| Compute buffer is sized to **`n_tokens = min(n_ctx, n_ubatch)`** | `llama-context.cpp:699` |
| `qwen35moe` has a **dedicated node budget**: `max(n_tokens*40, 32*n_tensors)` → `max(20480, 23456)` = **23,456 nodes** | `llama-context.cpp:2413` |
| Node metadata buffer = `tensor_overhead()*max_nodes + graph_overhead_custom(...)` → **~10 MB** (metadata only, not data) | `llama-graph.cpp:1333` |
| `flash_attn_type` defaults to **AUTO → ENABLED** on Metal ⇒ **no quadratic attention matrix materialised** | `llama-context.cpp:3637, 3691` |
| default `n_ubatch = 512` | `llama-context.cpp:3626` |

**Three consequences that matter:**

1. **The overhead does NOT scale with the expert pool.** It is graph-sized, so a 6.9 GB expert pool
   and a 200 MB one have the same compute-buffer cost. Our cache budget is therefore not competing
   with a growing overhead.
2. **It is controllable.** Because the reserve graph is `min(n_ctx, n_ubatch)` and the default
   `n_ubatch` is 512, a large `n_ctx` does **not** inflate the reserve — only `n_ubatch` does.
   **Lowering `n_ubatch` (e.g. 128–256) directly shrinks the overhead**, at the cost of prefill
   speed (irrelevant for decode-heavy interactive use).
3. **Flash attention being on removes the largest classic term**, so the realistic overhead is
   activation-scale, not attention-matrix-scale.

**Revised estimate: ~0.15–0.30 GB**, versus the 0.70 GB I previously assumed (which is retained below
as a conservative bound).

### 10.1 Resulting speed at the 8 GB target (ctx 4096, cache = 8.000 − 1.598 − 0.084 − ovh)

| overhead | cache | coverage | **internal SSD** | USB |
|---|---|---|---|---|
| 0.15 GB | 6.17 GB | 90.0 % | **21.2 t/s** | 9.3 |
| 0.20 GB | 6.12 GB | 89.2 % | **20.4 t/s** | 8.8 |
| 0.30 GB | 6.02 GB | 87.8 % | **19.0 t/s** | 7.9 |
| 0.50 GB | 5.82 GB | 84.8 % | 16.7 t/s | 6.6 |
| 0.70 GB (conservative) | 5.62 GB | 81.9 % | 14.9 t/s | 5.7 |

At the looser 8.590 GiB cap: **32–38.6 t/s** (internal SSD), 18.8–28.9 (USB).

### 10.2 Verdict — target met with margin, on one condition

**✅ At 8.000 GB with realistic overhead: ~19–21 tok/s on the internal SSD** — roughly double the
10 tok/s requirement, and near the top of the 10–20 band.
**❌ Still 7.9–9.3 t/s on the USB drive** — just below target.

So the **internal-SSD relocation remains the single required action**, and everything else clears.

## 11. Context ≥ 65535 — KV is now a real term

**Spec (user): context ≥ 65535, "more = better".**

### 11.1 The hybrid architecture is doing a lot of work here

`qwen35moe` has `full_attention_interval=4` ⇒ **only 10 of 40 layers have a growing KV**; the other
30 are gated-delta-net with **constant** state.

| | bytes/token | at 65536 |
|---|---|---|
| this model (10 attn + 30 SSM) | 20.0 KiB | 1.342 GB |
| *if all 40 layers were attention* | 80.0 KiB | 5.369 GB |

**The hybrid saves 4.03 GB at 65536.** Without it, 64k context would be impossible in 8 GB outright.
Long context is where this architecture earns its keep.

### 11.2 KV cost by dtype

| context | f16 | q8_0 | q4_0 |
|---|---|---|---|
| 16384 | 0.336 | 0.168 | 0.084 |
| 32768 | 0.671 | 0.336 | 0.168 |
| **65536** | **1.342** | **0.671** | **0.336** |
| 131072 | 2.684 | 1.342 | 0.671 |
| 262144 (model max) | 5.369 | 2.684 | 1.342 |

### 11.3 Ledger at ctx 65536 (overhead 0.20 GB)

| cap | KV dtype | KV | cache | coverage | **intSSD t/s** |
|---|---|---|---|---|---|
| 8.000 GB | f16 | 1.342 | 4.86 | 70.9 % | 10.6 ✓ *(barely)* |
| 8.000 GB | **q8_0** | 0.671 | 5.53 | 80.7 % | **14.3 ✓** |
| 8.000 GB | q4_0 | 0.336 | 5.87 | 85.6 % | 17.2 ✓ |
| 8.590 GiB | f16 | 1.342 | 5.45 | 79.5 % | 13.7 ✓ |
| 8.590 GiB | **q8_0** | 0.671 | 6.12 | 89.3 % | **20.5 ✓** |
| 8.590 GiB | q4_0 | 0.336 | 6.46 | 94.2 % | 27.2 ✓ |

**✅ 65536 clears the ≥10 tok/s target at every KV dtype** — but f16 + 8.000 GB is a *bare* 10.6 t/s,
i.e. no margin. **q8_0 KV is the recommendation**: it doubles the context headroom and buys ~4 t/s.

### 11.4 "More = better" — how far context can go

Freeing the budget below the 68.6 % hit-rate floor (which is what ≥10 tok/s requires):

| KV dtype | KV budget available | **max context at ≥10 tok/s, 8.000 GB** |
|---|---|---|
| f16 | 1.498 GB | **73,149 tokens** |
| **q8_0** | 1.498 GB | **146,298 tokens** |
| q4_0 | 1.498 GB | 292,597 tokens |

So with **q8_0 KV the ceiling is ~146k** — more than double the 65535 requirement, and still at
≥10 tok/s. At the looser 8.590 GiB cap it is higher again.

**Recommendation: `--cache-type-k q8_0 --cache-type-v q8_0`.** q8 KV is generally regarded as
near-lossless, and here it doubles the usable context while *increasing* speed. That is the single
best quality-preserving trade available.

### 11.5 Long context also shrinks the compute buffer — if `n_ubatch` stays small

The reserve graph uses `n_tokens = min(n_ctx, n_ubatch)` (§10), so a large `n_ctx` does **not**
inflate the compute buffer while `n_ubatch` stays at 128–256. **But the attention mask scales with
`n_ctx × n_ubatch`** (~67 MB at 65536 × 512, ~134 MB at 131072 × 512), so **keep `n_ubatch` modest
at long context** — it is the knob that keeps the mask and the compute buffer both small.

### 11.6 Verdict

**The full spec is satisfied: ≤8 GB, ≥10 tok/s, ≥65535 context.**
Best configuration: **internal SSD + q8_0 KV + ctx 65536 + n_ubatch 128–256**
→ ~14.3 t/s at an 8.000 GB cap, ~20.5 t/s at 8.590 GiB; context can extend to ~146k without
dropping below 10 t/s.

This is a *scaling law plus a bound*, not a measured number: the exact buffer size requires the
graph allocator to run (`ggml_backend_sched_reserve`, `llama-context.cpp:2553`). What is **certain**
from source is the *relationship* — overhead tracks `min(n_ctx, n_ubatch)`, not the expert pool —
which is the part the budget depends on. The 0.70 GB column is retained precisely so the plan holds
even if the estimate is wrong by 2×.

**Do not run a full load of this model.** User: *"no. do not run it, it WILL OOM my machine."*
This is consistent with the arithmetic: full load = 8.466 GB weights + ~0.7 GB runtime ≈ **9.2 GB**,
against ~8 GB usable headroom. **The full-load baseline is off the table permanently.** η and the
runtime overhead cannot be obtained by loading the model.

#### Consequence: the design must be **safe by construction**, not measured-then-capped

This is a real constraint, and it actually simplifies the architecture. The engine must never hold
the full model at any instant — not even transiently during load. That rules out:

- `lazy=False` / eager materialisation (the known 20 GB regression);
- `buffer_from_host_ptr` over the whole mmap (known machine-crash);
- any baseline that loads all weights resident.

It mandates:

- **incremental load**: bring in the non-expert set, then fill the expert cache *up to its budget
  and stop* — the cache never exceeds its cap, so peak footprint is bounded by design;
- **a hard footprint watchdog** as the backstop (already verified footprint-based);
- **refuse-to-admit** on the cache: if a block would exceed the budget, evict first, and never
  temporarily hold both.

#### How to obtain η and overhead without a full load

1. **Through the engine's own bounded path** (preferred): run the cache-bounded engine and read
   tok/s + footprint directly. The engine is capped by construction, so this is safe, and it
   measures the real system rather than a baseline that will never be used in production.
2. **A model that is already proven safe on this box** (e.g. LFM2.5-8B-A1B, which ran verified at
   ~4908 MiB under an 8 GB cap) as an η proxy — a hardware-efficiency figure generalises better
   than a memory figure. *Only with explicit user consent.*
3. **Static derivation** from `llama.cpp` source for the overhead term (context + compute buffer
   allocation), which needs no execution.

Until one of these is chosen, all tok/s figures in §9.2 remain **modelled, not observed** — and
should be reported as such.

---

## 12. MEASURED: unit bandwidth, and the multi-unit question settled

Ran a real streaming-read probe on this machine (`tools/aero-bwprobe.mm`, 256 MB buffer, run under
the Aero memory guard, cap 768 MB). The access shape is a weight matvec: every weight read once,
sequentially.

| configuration | best | mean | vs spec peak (68.25 GB/s) |
|---|---|---|---|
| **GPU (Metal) alone** | **54.20 GB/s** | 46.00 | **79 %** |
| CPU (P-cores) alone | 15.23 GB/s | 14.56 | 22 % |
| **CPU + GPU together** | **30.11 GB/s** | 29.04 | 44 % |

### 12.1 Verdict: multi-unit is a PESSIMISATION, not a lever

Running the CPU alongside the GPU **more than halves** aggregate throughput — 30.11 vs 54.20 =
**0.56×**. They contend for the same unified memory bus, and the CPU's presence actively degrades
the GPU's rate. A third contender (ANE) would contend again.

So *"use every lever at once"* is answered by the hardware: **no.** Notably this is **not** the
argument I made earlier. I argued "we have spare FLOPs, so more compute is unnecessary" — true, but
not decisive, since the real question was whether one unit can saturate the bus. It can, to 79 %,
and sharing the bus costs more than it gains.

**The parallelism that does exist is not across units.** Transformer layers are strictly sequential
within a token — layer L+1 needs layer L's output — so units cannot work on different layers. The
only intra-token parallelism is between *independent tensors of the same layer* (routed experts,
gate vs up projections), and §12.1 shows splitting those across units would slow them down.

### 12.2 Correction: η was wrong, and every speed estimate improves

All earlier figures here carried `η ∈ [0.4, 0.6]`, flagged throughout as *assumed, never
measured*. The probe measures **η ≈ 0.79** for streaming reads. Rebasing the per-token budget
(2.207 GB naive) at 54.2 GB/s:

| configuration | GB/token | ms/token | **tok/s** |
|---|---|---|---|
| naive (uniform 4.5 bpw) | 2.207 | 40.7 | **24.6** |
| + row-addressed embedding | 1.934 | 35.7 | **28.1** |
| + always-read set @ 3.0 bpw | 1.492 | 27.5 | **36.3** |
| + always-read set @ 2.125 bpw | 1.235 | 22.8 | **43.9** |

Materially better than the η ≈ 0.4 column — the earlier numbers were pessimistic, and even the
naive uniform-precision config clears 10 tok/s on bandwidth alone.

**Caveat, honestly:** this measures a *pure streaming read* (grid-stride reduction). That is a
reasonable proxy for a matvec but not identical — real kernels have activation reuse, different
tile shapes and tail effects, so production will land below 54.2 GB/s. Treat 54.2 as an **upper
bound on the streaming component** and 0.79 as a *ceiling* on η, not an achieved value. The
contention result in §12.1 is the robust finding; the absolute number is the optimistic one.

### 12.3 What this changes

- **Drop multi-unit from the design.** One Metal command stream, no CPU/ANE co-scheduling. This
  simplifies the scheduler: no cross-unit synchronisation, no work splitting, no third residency
  domain.
- **The bit-allocation dial remains the only lever** — now more clearly so, with unit count settled
  by measurement rather than opinion.
- **Speed targets gain firmer footing**: η's uncertainty is reduced from a 1.5× band to a measured
  ceiling, and the design has more margin than previously documented.

---

## 13. MEASURED: real model loads across CPU / Metal (and why ANE is unreachable)

Ran two real models through `llama-bench` under the footprint watchdog, escalating from safest to
riskiest. All runs exited cleanly (`rc=0`, no OOM).

### 13.1 Results

| model | layers on Metal | prompt t/s | **generation t/s** |
|---|---|---|---|
| Qwen3.8-4B Q4_K (2.58 GiB) | 0 (CPU only) | 24.24 | **8.06** |
| Qwen3.8-4B | 16 of 40 (**Metal + CPU together**) | 73.08 | **8.21** |
| Qwen3.8-4B | 99 (full Metal) | 152.88 | **14.26** |
| Gemma-4-E2B Q4_K (3.20 GiB) | 99 (full Metal) | 289.61 | **24.17** |

### 13.2 "Metal + CPU at the same time" is 1.74× SLOWER than Metal alone

| configuration | ms/token |
|---|---|
| full Metal | **70.1** |
| CPU only | 124.1 |
| **split 16 GPU + 24 CPU** | **121.8** |
| *split if purely additive* | *102.5* |

Using both units measured **121.8 ms/token vs 70.1 ms for the GPU alone — 74 % slower.** It is even
**19.3 ms worse than purely additive**, so there is a cross-boundary cost on top.

**Two independent reasons it must fail, both now measured:**

1. **The layers are sequential.** Layer L+1 needs layer L's output. In a chain of dependent stages,
   assigning some stages to a fast unit and the rest to a slow one yields the **sum** of the times,
   never the max. Concurrency would only help if the units worked on the *same* stage.
2. **Even where independence exists** (the 8 routed experts of a layer are independent matvecs),
   §12.1 shows the units contend for one bus: CPU+GPU concurrent measured **0.56×** of GPU alone.

The instinct — *"use every lever"* — is reasonable, and on a compute-bound workload it would be
right. On this workload both the dependency chain and the shared memory bus veto it, and the vetoes
are now measurements rather than arguments.

### 13.3 ANE — MEASURED (and I was too dismissive)

**Correction first:** I said the ANE "is not usable". That was wrong, and this measurement shows it.

Installed `coremltools` 9.0 (own venv, `tools/ane_probe.py`) and drove the ANE through CoreML —
the only public door. Op shape is the same streaming matvec as the Metal probe (134 MB fp16
weight matrix, every weight read once), so the numbers are directly comparable.

**Test A — one large op** (a single op is placed on *one* device, so this measures per-unit speed,
not concurrency):

| compute unit | GB/s |
|---|---|
| CPU_ONLY | 8.31 |
| CPU_AND_GPU | 51.26 |
| **CPU_AND_NE (ANE)** | **54.41** |
| ALL | 50.59 |

**The ANE is a fully capable streaming engine — 54.4 GB/s, matching the GPU (51.3) and my Metal
probe (54.20).** Far better than I implied.

**Test B — 8 independent ops.** This is the *real* concurrency test: a single op can only sit on one
device, so Test A could never show what parallelism buys. With independent branches the scheduler
is free to spread them:

| compute unit | GB/s |
|---|---|
| CPU_ONLY | 8.12 |
| CPU_AND_GPU | 51.75 |
| CPU_AND_NE (ANE) | 51.10 |
| **ALL (every unit)** | **51.32** |

**Concurrency test: GPU-only 51.75 vs ALL 51.32 → 0.99×.**

So even with maximum scheduling freedom, **enabling every unit is worth nothing** — it is marginally
*negative*. The units do not add; they contend.

**Why, precisely:** each unit individually reaches ~51–54 GB/s against a 68.25 GB/s bus. They are
**bus-limited, not unit-limited**. Two bus-limited engines sharing one bus finish no sooner than one
using it fully — which is what §12.1 predicted and what is now measured three independent ways
(Metal probe 0.56×, llama.cpp `-ngl` split 1.74× slower, CoreML ALL 0.99×).

**Test C — GPU + ANE running CONCURRENTLY.** CoreML exposes **no GPU+ANE mode** at all (the enum
is only `ALL`, `CPU_AND_GPU`, `CPU_AND_NE`, `CPU_ONLY`), and `ALL` drags in the slow CPU (~8–10
GB/s), so `ALL` cannot isolate the GPU+ANE pair. Tested it directly instead: two models side by
side, one pinned to the GPU and one to the ANE, measuring **aggregate** throughput.

| | GB/s |
|---|---|
| GPU alone | 52.61 |
| ANE alone | **53.78** |
| **GPU + ANE together** | **48.02 → 0.89× of the best single unit** |

**Running GPU and ANE concurrently is SLOWER than using either one alone.** Not merely "no gain" —
a 11 % *loss*. The two units contend for the same bus, and co-scheduling them adds overhead on top.

*(Apple not offering a GPU+ANE compute-unit mode is consistent with this measurement.)*

**Complete tally — four independent methods, all agreeing:**

| method | result |
|---|---|
| Metal probe, CPU+GPU concurrent | **0.56×** of GPU alone |
| llama.cpp `-ngl` split (metal+CPU per token) | **1.74× slower** than full Metal |
| CoreML `ALL` vs `CPU_AND_GPU` | **0.93–0.99×** |
| CoreML GPU+ANE side-by-side | **0.89×** |

**Why, precisely:** each unit alone reaches ~52–54 GB/s against a 68.25 GB/s bus — they are
**bus-limited, not unit-limited**. Two bus-limited engines sharing one bus cannot sum; the bus is
the cap. This is now measured four ways rather than argued once.

**Caveat on absolutes:** run-to-run variance is large. The ANE measured 40.68 / 51.10 / 53.59 /
53.78 / 54.41 GB/s across runs; the GPU 51.26–52.61. Single-shot samples, so treat per-unit absolutes
as indicative. **The concurrency ratios (≤1.0× in every trial) are the robust finding.**

**Revised position on the ANE:** it is a genuinely capable engine — on this op shape it *matched or
beat* the GPU in several runs — so it is a legitimate **alternative carrier**, not a complement.
Combining it buys nothing because it shares the only scarce resource.



### 13.4 Good news: the target is already met for small models

Both models **already exceed 10 tok/s at full Metal with no engine at all** — Qwen3.8-4B at
**14.26**, Gemma-4-E2B at **24.17**. Both fit entirely in RAM (2.58 / 3.20 GiB), so nothing needs
paging.

**This sharpens the scope of the whole project.** The engine earns its keep only for models that
*do not* fit — the 8.47 GB class. For ≤4 GiB models, full Metal already delivers the target and
paging would only add risk. Worth stating plainly rather than building machinery that isn't needed.

### 13.5 Practical note recorded

Metal's reported `recommendedMaxWorkingSetSize` on this box is **11453 MB** — consistent with the
~10.9 GB ceiling that previously caused a crash when the whole file was mapped into a Metal buffer.

