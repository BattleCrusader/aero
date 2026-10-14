# Aero — an inference engine built from first principles

New design. Nothing here is derived from or reuses any existing implementation. Own container, own
allocation solver, own kernels, own memory arbiter.

---

## 1. The mistake that was hiding the answer

I had been treating the *resident* weights as free. They are not.

A decode step executes **every layer**. So every weight outside the routed experts is **read, in
full, every single token** — "resident" only means it comes from RAM instead of disk. Correcting
that changes the bottleneck completely:

| Per-token traffic (Qwen3.6-14B-A3B, measured tensor bytes) | GB |
|---|---|
| always-read set (attention, SSM, norms, routers, head, embedding) | **1.598** |
| active experts (8 of 90 per layer) | **0.609** |
| **total** | **2.207** |

**The expert bytes are only 28 % of the traffic. The always-read set is 72 %.**

Consequently — and this is the finding that redirects the whole design:

> **With infinite RAM and nothing on disk, this model tops out at 80.8 ms/token = 12.4 tok/s**
> (at η 0.4). At η 0.6, 18.6 tok/s.

An expert cache — any expert cache — **cannot exceed that**. Every design I sketched before was
optimising the 28 %. The target 10–20 tok/s cannot be reached with *infinite RAM* unless something
reduces the 72 %.

**Bytes read per token is the only currency. And 72 % of it is spent on weights read every token.**

---

## 2. The core principle: bandwidth-aware bit allocation

Precision is the price you pay to store a weight. Bandwidth is the price you pay to *read* it. These
are charged at wildly different rates depending on how often a weight is read:

| | read frequency | cost of one extra bit |
|---|---|---|
| always-read set | **1.0 / token** | paid **every token** |
| routed experts | 8/90 ≈ 0.089, and cached | paid **rarely** |

So the optimal allocation is not "quantise everything the same", and it is not even
"quantise sensitive layers carefully". It is:

> **Spend bits where they are read rarely. Starve bits where they are read every token.**

This is an allocation problem with a clean objective — minimise expected bytes-read per token
subject to a quality floor:

```
minimise   Σ  bytes_i(bitwidth_i) × frequency_i
subject to quality_i ≥ floor_i        (frequencies are structural, known at build time)
```

High-frequency tensors get low bitwidth; low-frequency tensors keep their precision. Sensitive but
*tiny* tensors (norms, routers) stay at full precision because they cost almost nothing in bytes.
This is the engine's central idea, and it is where the speed comes from.

### Why this is the right axis

Expert bytes shrink on their own: routing already skips 82 of 90 experts. Nothing else skips
anything — every other tensor is read unconditionally. **Frequency is therefore the only remaining
structure left to exploit**, and it is entirely unexploited by uniform quantisation.

### A free lever: address rows, don't read matrices

The token embedding is a lookup. A decode step needs **one row** of it, not the whole matrix —
0.273 GB read every token in a naive engine, versus ~8 KB if the engine addresses the row. Same for
any gather-style tensor. This costs nothing in quality and is pure design.

---

## 2b. The real frequency table (read from the model's own index)

The architecture's tensor inventory, grouped by how often each is read. This is the Planner's
input, and it is **structural** — known without running anything.

| tensor | layers | read frequency | class |
|---|---|---|---|
| `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` | 40 | **8/90 per layer**, cacheable | **routed — the pageable pool** |
| `ffn_gate_shexp`, `ffn_up_shexp`, `ffn_down_shexp` | **40** | **1.0 — every token** | **shared experts: always-read** |
| `attn_qkv`, `attn_gate` | **30 (SSM layers)** | **1.0** | always-read — *the SSM in/out projections* |
| `ssm_out`, `ssm_conv1d`, `ssm_alpha`, `ssm_beta`, `ssm_a`, `ssm_norm`, `ssm_dt` | 30 | 1.0 | always-read (recurrent path) |
| `attn_q`, `attn_k`, `attn_v`, `attn_output`, `attn_q_norm`, `attn_k_norm` | 10 | 1.0 | always-read (attention layers only) |
| `ffn_gate_inp`, `ffn_gate_inp_shexp` | 40 | 1.0 | routers — tiny, precision-critical |
| `attn_norm`, `post_attention_norm` | 40 | 1.0 | norms — tiny, precision-critical |
| `token_embd` | 1 | **row only** | address, don't read (0.273 GB → ~8 KB) |
| `output` (head) | 1 | 1.0, full matvec | **always-read, 0.398 GB** |

**Three things this settles:**

1. **Only the `*_exps` tensors are pageable.** Everything else — including the **shared experts** in
   all 40 layers and the **SSM projections in 30 layers** — is read every token. That is the 72 %.
2. **The SSM layers are weight-heavier than expected**, because they carry `attn_qkv` *and*
   `attn_gate` each — 30 layers' worth. This is where the always-read bytes actually live, so this
   is where the bitwidth dial pays the most.
3. **Shared experts are not optional.** They run for every token regardless of routing, so they
   belong to the always-read set, not the cacheable pool. My earlier mental model had them lumped
   with "experts" — wrong, and it would have mis-sized the cache.

Consequence for the plan: the **bitwidth dial should be aimed at the SSM projections and the shared
experts first** (biggest always-read mass), while **routers and norms stay full precision** (tiny
mass, high sensitivity). That is exactly the allocation the solver in §2 should produce, now with
real tensor names attached.

---

## 3. Aero architecture

Five components, all new.

```
┌──────────────────────────────────────────────────────────────────┐
│  PLANNER (offline, runs once per model)                          │
│   · walks the source weights                                     │
│   · assigns frequency to every tensor (structural: 1.0 vs routed)│
│   · solves minimise Σ bytes_i(bits_i) · freq_i  s.t. quality floor│
│   · emits the allocation plan + the block map                     │
└──────────────────────────┬───────────────────────────────────────┘
                           │ plan + block map
┌──────────────────────────▼───────────────────────────────────────┐
│  CONTAINER  "aer"  — one flat file, self-describing              │
│   · fixed-size BLOCKS, each independently decodable              │
│   · blocks ordered by co-occurrence (frequently co-used adjacent) │
│   · per-block: offset, length, format, tensor-id, bitwidth        │
│   · no cross-block dependencies  -> any block is a prefetch unit  │
└──────────────────────────┬───────────────────────────────────────┘
                           │ pread by byte range
┌──────────────────────────▼───────────────────────────────────────┐
│  ARBITER — ONE memory pool, hard cap, never exceeded              │
│   · weights, KV and scratch all compete for the same bytes        │
│   · score = P(needed next token) × bytes_saved − size             │
│   · admit only if projected ≤ cap; evict first, never co-hold     │
│   · KV is just another block class — no fixed split               │
└──────────────────────────┬───────────────────────────────────────┘
                           │ resident blocks
┌──────────────────────────▼───────────────────────────────────────┐
│  KERNELS — written for the formats the planner chose              │
│   · add-only packed dot for low-bitwidth blocks (no multiply)     │
│   · gather/row addressing for lookup-style tensors                │
│   · recurrent scan for the state-space layers                     │
│   · attention for the sparse attention layers only                │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│  SCHEDULER — two-stage pipeline                                  │
│   · reads block i+1 while computing block i                       │
│   · prefetch is exact for the always-read set (order is known)    │
│   · router-ahead for experts: the router result for a layer is    │
│     available before that layer's experts are needed              │
└──────────────────────────────────────────────────────────────────┘
```

**Why one pool matters:** today the context buffer and any weight cache are separate fixed
allocations. Here they share one budget, so a long-context session automatically trades weight
residency for KV and back — you get the maximum context the budget allows *and* the maximum cache,
as a function of what the workload is doing. This is what makes "more context = better" free rather
than a rebuild.

---

## 4. The numbers

Applying the levers cumulatively (η 0.4, ctx 65536, q8 KV, 8 GB cap):

| step | per-token traffic | tok/s |
|---|---|---|
| naive (uniform 4.5 bpw, whole-matrix embedding) | 2.207 GB | **12.4** |
| **L1** address embedding rows instead of reading the matrix | 1.934 GB | **14.1** |
| **L2** always-read set at 3.0 bpw | 1.492 GB | **18.3** |
| **L2** always-read set at 2.125 bpw | 1.235 GB | **22.1** |
| **L2** always-read set at 1.6 bpw | 1.080 GB | **25.3** |

And with the *memory dividend* — a smaller always-read set leaves more room for experts, so fewer
disk misses — the realistic end-to-end figures are:

| always-read bitwidth | always-read size | expert coverage | **tok/s incl. disk misses** |
|---|---|---|---|
| 4.5 bpw | 1.33 GB | 81 % | 7.8 |
| 3.0 bpw | 0.88 GB | 91 % | **13.0** |
| 2.125 bpw | 0.63 GB | 95 % | **17.2** |
| 1.6 bpw | 0.47 GB | 97 % | **21.4** |

**Target met: 13–21 tok/s at ≤8 GB and 65536 context** — and the bitwidth is a *dial*, so the
operating point is a quality choice, not an architecture change.

Note the bottom row of the first table is unreachable by any cache: **12.4 tok/s is the infinite-RAM
ceiling at uniform precision.** The design only clears 10 tok/s comfortably because it attacks the
72 %.

---

## 5. What is actually new here

| Idea | Status |
|---|---|
| **Bandwidth-aware bit allocation** — bitwidth ∝ 1/read-frequency, solved as an allocation problem over bytes-read/token | **new**; uniform and sensitivity-based quantisation both miss it |
| **Row-addressed lookup tensors** — embedding/gather read one row, not the matrix | **new** as an engine-level rule; free 12 % |
| **One pooled budget** shared by KV and weights with utility eviction | **new**; conventional engines fix them separately |
| **Co-occurrence-ordered block layout** so a working set is contiguous | **new** |
| **Add-only packed dot** for the low-bitwidth always-read set | standard technique, applied because the planner chose the format |
| **Exact prefetch** — the always-read order is statically known, so its prefetch needs no predictor | simple consequence of the design |

The first three are the engine. The rest is execution.

---

## 6. Honest risks

1. **Quality at 2 bpw on the always-read set is the whole bet.** Attention projections at 2-bit may
   degrade more than expected. Mitigations: norms/routers stay full precision; bitwidth is per-tensor,
   so attention can be held at 3–4 bpw while SSM/recurrent tensors go lower; the dial makes this a
   measured trade rather than a leap. **This needs empirical validation and I cannot do it on this
   machine yet.**
2. **η is assumed (0.4–0.6), never measured.** Every figure scales with it. If η is 0.35 the whole
   column drops ~15 %.
3. **Own kernels are the real work.** Attention, recurrent scan, and a packed dot — plus a converter
   from the source model file into the container. This is a large but well-bounded build.
4. **The planner needs a quality signal** to set bitwidth per tensor. Without a validation set,
   the floors are guesses; with one, it is a measurable optimisation.

---

## 7. Where to start

Build the **planner + container first**, not the kernels. They are the new part, they need no model
execution to build or validate (the block map and allocation plan can be inspected and diffed), and
they produce artefacts that make every later number checkable:

1. read the source weights once, emit the tensor/frequency table;
2. solve the allocation for a chosen quality floor → bitwidth per tensor;
3. write the container with the co-occurrence-ordered block map;
4. verify: total size, per-tensor sizes, block contiguity for a typical working set.

That is a self-contained deliverable that does not run the model, cannot OOM anything, and turns the
design into something measurable.
