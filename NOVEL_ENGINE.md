# A novel engine — union-amortised weight streaming ("Coalesce")

A new design, not an assembly of existing systems. No code. Nothing executed.

I am deliberately **not** reusing: MLX lazy-expert paging, the llama.cpp expert-remap fork,
NunSpark/vates, MoE-Infinity, airllm, or llama.cpp mmap+`-ncmoe`. Those are surveyed in
`ENGINE_DESIGN.md`; this file supersedes them as the design of record.

---

## 1. Start from the floor, not from prior art

Two facts fix the whole problem. Everything else follows.

**Fact 1 — IO dominates, and overlap cannot fix it.** Measured: only ~5 % (SSD-LLaMA) to ~10 %
(airllm) of a weight transfer can be hidden behind compute, because at batch-1 the read *is*
longer than the math. So "prefetch harder" is capped at ~10 %. Any design whose main idea is
overlap is dead on arrival.

**Fact 2 — the only quantity that matters is bytes-read per *distinct* token.**

```
cost(tok/s) ≈  effective_read_bandwidth / bytes_read_per_token
```

For a 25 GB dense model at 3 GB/s: 8.3 s/token. That is the number to attack.

So the entire design space is: **reduce bytes read per token.** There are exactly three ways:

| # | Way | Ceiling |
|---|---|---|
| A | fewer bytes per weight (quantise harder) | fixed at model-prep time |
| B | don't read weights that aren't needed (sparsity) | MoE: free. Dense: needs zeros |
| C | **read once, use for many tokens** (amortise) | **unbounded in principle** |

A is done (these models are already 2–4 bit). B is architecture-given for MoE and needs an
exploit for dense. **C is where the unexploited headroom is, and it is the axis my design
attacks.** C is unbounded because it depends on *how many tokens you can process per read* —
not on the model at all.

The prior art treats C as "speculative decoding, k=4". I claim that is a severe under-use, for
a reason nobody exploits: **sparse reads across a batch are highly redundant.** That redundancy
is the engine.

---

## 2. The core idea: read the **union**, not the sum

Consider processing a batch of `B` tokens in one pass (B = speculative drafts × concurrent
requests). Each token `t` needs a sparse set `S_t` of weight blocks at layer `L` (its routed
experts, or its fired neuron-tiles).

Every existing system reads per token:

```
naive bytes = Σ_{t∈B} |S_t|          (per layer, summed over layers)
```

But the blocks **overlap**. Two tokens in the same batch are usually about the same thing —
same prompt, adjacent positions, same conversation, same task. Their routers pick largely the
**same** experts. So the correct thing to read is:

```
coalesced bytes = Σ_L |∪_{t∈B} S_{t,L}|
```

Define the **coalescing factor**:

```
ρ  =  |∪ S| / ( |B| · E|S| )        (0 < ρ ≤ 1)
speedup vs per-token reads = 1/ρ
```

`ρ` is the entire engine. Per-token bytes fall as `ρ`, and ρ is *not* a constant — it is a
measurable property of the model, the text, and the batch composition. Nobody schedules on it.

**Why this is more than batching.** With B independent requests the sets are independent and
`ρ → 1` (no win — and this is why naive continuous batching does nothing for RAM-bound decode).
The win comes from choosing the batch to *maximise correlation*:
- speculative tokens are adjacent positions of the same text → very high correlation;
- requests sharing a system prompt / task type → correlated routing;
- so **batch composition is a scheduling decision, not a queue-order accident.**

**Conjecture (must be measured first — §7):** for MoE with 8/90 routing, `ρ` for a speculative
batch of adjacent positions is far below 1 — plausibly 0.2–0.4. At ρ=0.3 with B=8, per-token
bytes drop ~2.7× *on top of* the 8× sparsity MoE already gives.

---

## 3. The second idea: make the union **physically contiguous**

Reading `|∪S|` blocks is only half the win. Scattered blocks pay a **seek + latency cost per
block**, not just a byte cost. A 2 MB expert read from 8 random positions costs far more than
16 MB read sequentially — LLM-in-a-flash measured `row-column bundling` lifting throughput
1.25 → 2.25 GB/s purely from contiguity.

So: **order the weight file so that blocks which co-occur are adjacent.**

This is a genuinely new step. Take the routing co-occurrence matrix `C[i][j]` = how often
expert `i` and expert `j` are selected together (measured once offline, per model, on a
representative corpus). Find a permutation of blocks minimising expected union span — i.e. a
**Hamiltonian-path / TSP-style ordering over the co-occurrence graph**, greedily or with a
spectral/2-opt heuristic. Write that permutation into the file layout (a permuted copy of the
GGUF, or an index of ranges). At runtime a typical union becomes **one near-contiguous extent**.

Two properties make this attractive:
- It is computed **offline from routing statistics** — no change to the model's math, no
  retraining, no accuracy effect whatsoever (it is a pure file permutation).
- The co-occurrence graph is a *new artifact the engine owns*: it also powers prefetch
  ordering, eviction, and admission.

For **dense** models the same technique applies with `ffn` **neuron-tiles** as the blocks:
which FFN neurons fire together is a stable, co-occurring set (this is exactly the structure
LLM-in-a-flash exploits for density, but it never orders the file by it).

---

## 4. The third idea: one stream, many sessions

C is unbounded, so push it further: **decouple the weight stream from the request.** A single
streaming pass should serve *every* in-flight generation at once — one read feeds all of them.

This is a *different* engine shape from a per-request loader: the stream is a scheduler over a
global "needed-blocks-by-layer" frontier, and each token of each session attaches to the
passes it needs. Two consequences:
- per-token IO divides by the number of sessions multiplexed onto the stream;
- sessions naturally **desynchronise** (different text, different layers in flight), so they
  fill each other's compute gaps — the one place overlap actually pays.

The user already runs many concurrent agents/sessions, so this is not hypothetical. It also
means the engine is a **server**, not a library — which is the right shape for a shared 25 GB
model on one box.

---

## 5. Why this sidesteps macOS's broken memory story

Finding (measured): macOS has **no caller-controlled eviction** — `MADV_PAGEOUT` is ENOTSUP on
shipping kernels, `MADV_COLD` does not exist, and the pageout daemon may drop
`newBufferWithBytesNoCopy` pages so they **refault on the next prefill**. Any design built on
"the OS will evict what I'm done with" is broken on macOS by construction.

The stream design **does not rely on eviction at all.** It has:
- **no resident weight set** (nothing to evict) — weights pass through a **fixed-size ring
  buffer** and are released the moment they are consumed;
- **no page-cache dependence** — one explicit `pread` per coalesced extent, into the ring;
- **`MADV_WILLNEED` only** (the one madvise that genuinely works here) to warm the *next*
  extent.

So the macOS limitation that kills the cache-based designs is simply **not on the critical path**
for this one. That is a design-level fit, not a workaround.

The resident set is then tiny and *bounded by construction*:
```
resident = trunk (norms, routers, SSM state)  +  ring buffer (fixed)  +  KV  +  OS
```
with the ring being the only tunable — a **knapsack in one variable**, so the admission
controller is trivially safe: `ring ≤ B − trunk − KV − OS`. No eviction policy to get wrong.

---

## 6. Quality is exact, not approximate

Everything above is **lossless by construction**, which is unusual for a low-memory trick:

- Union reading is not an approximation — it reads a *superset* of what is needed and computes
  exactly the same result. Skipping a block never happens; coalescing only changes *when* bytes
  arrive.
- Speculative decoding is verifiably output-equivalent to the target model: rejected drafts are
  discarded and the target's own token is emitted. So amortisation cannot change the
  distribution.
- The file permutation changes no arithmetic.

Contrast with the tempting-but-unsafe alternatives I am deliberately avoiding: activation-delta
reuse and output memoisation both *approximate* the matmul and need error bounds; the stream
design needs none. **The only "dial" is speed, never correctness.** For a user who has been
burned by unverified loaders, that property is worth more than any single speedup.

---

## 7. The one experiment that decides everything

The design lives or dies on `ρ`. It must be measured **before** any engine is built, and it is
cheap and safe to measure:

> Run a *small* model with routing visible, over a short corpus, and record the per-layer
> selected-expert sets. Compute, as a function of batch composition:
> (a) `ρ` for adjacent-position batches (speculative-like),
> (b) `ρ` for same-system-prompt batches (session-like),
> (c) the co-occurrence matrix `C` and the union-span reduction from a TSP ordering.

Do it under the `footprint` watchdog at a low cap. If `ρ` is not comfortably below ~0.5, the
core idea is dead and I should say so rather than build it — that is the honest gate. A
secondary cheap check: does `C` have real structure (a few dominant off-diagonal clusters), or
is routing near-uniform? TSP ordering only helps if the graph is clustered.

**Gate:** build only if `ρ_adjacent ≲ 0.5` **and** `C` is clustered. Otherwise the engine
degenerates to plain batching, and plain batching already lost (see §2).

---

## 8. What is genuinely new here

| Mechanism | Status |
|---|---|
| Read the **union** of sparse sets across a batch, and *schedule batch composition to maximise correlation* | **new** — prior art reads per-token and batches by queue order |
| **Co-occurrence-ordered file layout** (TSP over the routing graph) so a union is physically contiguous | **new** — a pure file permutation, offline-computed, zero accuracy cost |
| **Cross-session stream multiplexing** as the primary amortiser (stream-per-request → stream-per-machine) | **new** for this constraint |
| Dense models get the same treatment via **neuron-tile** co-occurrence | **new** — prior art uses density but never orders by it |
| No-resident-set streaming → immune to macOS's missing eviction API | **new** framing |
| Exactness as a design invariant | new *as a stated priority* |

**Not new** (irreducible, and I will not pretend otherwise): sparsity as a lever, speculative
decoding as amortisation, quantisation, prefetch, the fact that batch-1 decode is IO-bound.

---

## 9. Honest risks

1. **ρ may be high.** If routers are near-uniform and uncorrelated, there is nothing to coalesce.
   §7 gates on this; I will not build past a failed gate.
2. **TSP ordering needs a good co-occurrence estimate**, which needs representative routing data
   — a model-specific offline step. If `C` is flat, contiguity gains vanish.
3. **Cross-session multiplexing needs concurrency to exist.** With one interactive session and
   no speculation, the win collapses to the speculative factor alone.
4. **The ring-buffer stream is strictly serial per extent.** No caching means a block needed
   again next token is re-read. So this engine *requires* high per-pass reuse to beat a cache —
   it trades the cache's reuse for the stream's contiguity and boundedness. That trade is the
   whole bet, and it is why `ρ` must be measured first.
5. **Dense remains the hard case.** For dense the sparsity set is large, so `ρ` is the only
   lever, and `ρ`'s ceiling with adjacent-position speculation is unknown until measured.

---

## 11. Inline compression / decompression — MEASURED, and rejected

Suggested by the user. I analysed it (§11.1–11.8 below, kept as reasoning) and then **measured it on
the actual GGUF files**. The measurement kills the generic version of the idea. This is the honest
outcome, not the expected one — I went in expecting ~1.5×.

### 11.0 MEASURED RESULT (64/32 MB slices from 4 regions of each file; zstd/lz4 on this M1)

| File | @64 MB | @512 MB | @1024 MB | @3072 MB | verdict |
|---|---|---|---|---|---|
| Ternary-Bonsai-2-27B **PQ2_0** | 1.202 | 1.204 | 1.209 | 1.212 | ~1.20× only |
| Ternary-Bonsai-2-27B **PTQ1_0** | 1.000 | 1.000 | 1.000 | 1.000 | **incompressible** |
| Qwen3.6-14B-A3B **Q4_K_M** | 1.000 | 1.022 | 1.054 | 1.025 | **incompressible** |
| gemma-4-12b **IQ4_XS** | 1.022 | 1.043 | 1.013 | 1.011 | **incompressible** |

- **zstd −1 vs −19 made no difference** (1.202 vs 1.201) — a hallmark of already-random data.
- **lz4 = 1.000** on everything.
- Consistent across four different regions per file, so this is **not** slice-luck. The
  PQ2-vs-PTQ1 split (1.20 vs 1.00) is real and reproduces at every offset.

**Verdict: quantised weights are entropy-saturated.** Quantisation has *already* removed the
redundancy — that is what quantisation is. There is essentially nothing left for a generic lossless
compressor to remove.

**Decode cost is also fatal.** Measured zstd −19 decode = **0.62 GB/s** on this M1 (single frame).
§11.3 requires `D > 3 GB/s` just to stay off the critical path. At 0.62 GB/s, decoding a 25 GB
stream costs **~40 s/token** — so even PQ2's real 1.20× is a heavy net loss.

### 11.0b A SECOND, BIGGER CORRECTION: the disk is 755 MB/s, not 3 GB/s

Measured raw read on this volume: **755,507,116 B/s ≈ 0.755 GB/s**. Every cost figure in §1 and
§11.3 assumed 3 GB/s and is therefore **~4× optimistic**. Rebasing with the measured bandwidth:

| Case | Bytes read/token | Time/token @ 0.755 GB/s |
|---|---|---|
| 25 GB dense, naive full stream | 25 GB | **33.1 s** |
| Ternary-Bonsai 27B (6.7 GB), full stream | 6.7 GB | **8.9 s** |
| Qwen3.6-14B-A3B MoE, sparse (resident 1.49 GiB + active experts ≈0.57 GiB) | ~2.06 GiB | **~2.7 s** |

The MoE row is the important one: **sparsity alone already gets the 14B MoE to ~2.7 s/token** on
this disk. Dense has no such lever, which is exactly why compression was attractive — and why its
measured failure matters.

### 11.0c WHY: PQ2_0 is not ternary, and PTQ1_0 is already entropy-optimal

Read from the fork's own block definitions
(`PrismML-llama.cpp/ggml/src/ggml-common.h:199-220`):

```c
// PQ2_0: Prism-private Q2_0 at group size 128. Same 2-bit codec as Q2_0
// (group 64) but one fp16 scale per 128 weights (~5% smaller).      [type 142]
typedef struct { ggml_half d; uint8_t qs[128/4]; } block_pq2_0;   // 2.125 bpw

// PTQ1_0: Prism-private ternary at group size 128. Same base-3 trit packing as
// upstream TQ1_0 (type 34) but one fp16 scale per 128 weights.      [type 143]
// 1.75 bpw vs PQ2_0's 2.125
typedef struct { uint8_t qs[24]; uint8_t qh[2]; ggml_half d; } block_ptq1_0;
//   qs = 5 trits/byte -> 120 values;  qh = 4 trits/byte -> 8 values;  total 128
```

Two consequences that explain the measurements exactly:

1. **PQ2_0 is a 4-level 2-bit quant, NOT ternary.** Its 1.20× is ordinary skew in a 4-symbol
   alphabet (2.125 bpw stored vs a lower true entropy). Nothing special — but a PQ2-aware entropy
   coder could plausibly reach ~1.2–1.4×.

2. **PTQ1_0 is the real ternary, and its packing is already at the information-theoretic limit:**

   ```
   5 trits per byte  =  8/5 = 1.600 bits/trit
   log2(3)           =        1.585 bits/trit   <- the hard floor
   efficiency        =  1.585/1.600 = 99.1 %
   ```

   The base-3 packing sits **within 1 % of the floor for a uniform ternary distribution**. That is
   precisely why it measured **1.000×** — there is nothing left to remove.

**The exact ceiling for ternary compression** is therefore not a guess; it is `1.600 / H(p₀)`:

| zero-fraction p₀ | H(p₀) bits/trit | best possible gain |
|---|---|---|
| 1/3 (uniform) | 1.585 | **1.01× — nothing** |
| 0.50 | 1.500 | 1.07× |
| 0.70 | 1.181 | 1.36× |
| 0.80 | 0.922 | 1.74× |

So the entire ternary-compression question reduces to **one number: the trit zero-fraction `p₀`**.
If the model's trits are near-uniform (`p₀ ≈ 1/3`) — which the 1.000× measurement strongly
suggests — the idea is closed. Only a high `p₀` would reopen it, and even then the gain caps at
1.74× for a *perfect* coder with a fast in-register LUT decode.

**→ MEASURED. See §11.0d: `p₀ = 0.346`. It is uniform. The thread is CLOSED.**

### 11.0d MEASURED: the trit statistics — compression formally CLOSED

Measured by byte-histogram probe (needs no parser): all-zero trits pack to byte `0x00`, so
`P(byte=0) = p₀⁵` for 5-trit packing and `= p₀⁴` for 2-bit packing. 8 MB sample at offset 512 MB.

| Model | P(byte=0x00) | implied symbol prob | symbol entropy | stored | **max gain** |
|---|---|---|---|---|---|
| PTQ1_0 ternary | 0.4944 % | `p₀ = 0.3458` (uniform 0.3333) | 1.5845 bits/trit | 1.600 | **1.0098×** |
| PQ2_0 2-bit | 1.2066 % | `p = 0.3314` (uniform 0.2500) | 1.9760 bits/sym | 2.000 | **1.0121×** |

(reference: uniform trits ⇒ 0.4115 %; uniform 2-bit ⇒ 0.3906 %)

**Verdict: the ternary payload is at its entropy floor.** `p₀ = 0.346` is essentially uniform, so
trit entropy is 1.5845 bits against a 1.600-bit packing — a *perfect* entropy coder could recover
**1.01×**. Measured zstd recovered 1.000×. Closed for a **mechanistic** reason: the format is
already optimal, not merely unlucky.

**The PQ2_0 1.20× is explained too — it is not the payload.** Its symbol skew permits only
**1.01×**, so the observed 1.20× must come from *non-quantised* tensors mixed into the sampled
region (fp16 scales, embeddings, norms). There is no PQ2 payload headroom either.

**Inline compression of these formats is definitively dead.** Not "it didn't work" — the quantisers
already sit at the information-theoretic floor, and the 0.62 GB/s decode wall would forfeit any
residual gain anyway. The only routes to fewer bytes are **lossy** (lower precision → a quality
trade) or **skipping** (sparsity / coalescing). The ternary LUT-codec idea is dead on measurement:
1.01× does not justify a kernel change.

**Caveat, stated honestly:** zstd is byte-oriented and cannot properly model base-3 packing, so
1.000× is *evidence* for uniform trits but not proof. A trit-aware histogram of the `qs[]` bytes
(a ~15-line parse of the 5-trit packing) would settle it definitively. That is the one remaining
measurement — and it is the only reason the ternary compression thread is not yet formally closed.

| Format | Stored code | Entropy-coded? |
|---|---|---|
| Q4_K_M | fixed 4-bit nibbles + 6-bit block scales/min | no |
| IQ4_XS | LUT indices, fixed width | no (codebook, not entropy) |
| ternary PQ2/PTQ1 | trits at a fixed 2.0 / ~1.58 bits | no |

**Measured evidence (best available, same problem shape):** SSD-LLaMA (2609.18110) applies
**lossless rANS to SSD-resident MoE experts** and reports SSD traffic **8.89 → 5.93 GB/token =
−33.2 %**, i.e. a real ratio of **≈1.50×** on quantised expert weights. That is the honest
expectation for generic lossless compression of quantised weights: **~1.5×, not 3×.**

### 11.2 Ternary is the standout case (and Bonsai is ternary)

For a ternary alphabet the entropy ceiling is `log2(3) = 1.585` bits/trit — reached only at a
perfectly balanced `p0 = 1/3`. Real ternary LLMs skew toward **zero**, and the ratio is set
purely by the zero fraction:

| zero fraction p₀ | entropy (bits/trit) | ratio vs 1.585 stored |
|---|---|---|
| 0.33 | 1.585 | 1.00 |
| 0.50 | 1.500 | 1.06 |
| 0.70 | 1.181 | **1.34** |
| 0.80 | 0.922 | **1.72** |

So ternary is worth **~1.1–1.7×**, and *the zero fraction is the whole knob*. That is directly
measurable — see §11.7.

### 11.3 The cost model, honestly

```
no compression :  T = W / BW
with compression: T = max( W / (c·BW),  W / D )      D = decode throughput
```
(W = bytes, BW = disk bandwidth, c = ratio, D = decode speed.)

For the 25 GB target at 3 GB/s: `W/BW = 8.33 s`. At c = 1.5 the read drops to 5.56 s, and the
pipeline is `max(5.56, 25/D)`. Decode must therefore clear **D > 3 GB/s** merely to avoid becoming
the new bottleneck. On M1 (8 cores) that is plausible only with parallel independent frames.

**The insight that makes this better than it first looks:** my §1 statement that overlap hides only
~5–10 % was about hiding a *read* behind *compute*. Compression introduces a **third stage that
consumes an otherwise-idle resource** — the CPU is doing nothing while blocked on the disk read,
so it can decode the previous block *during* the next read. The pipeline becomes
`max(read, decode, compute)` instead of `read + compute-hidden`. Decompression is therefore close
to **free in wall-clock terms** as long as it is not itself the longest stage. That is the real
argument for inline compression — not the ratio alone, but that it converts dead waiting time into
work.

### 11.4 Three variants, in increasing order of novelty

1. **Generic lossless over the existing container** (zstd/lz4/rANS over GGUF bytes). Cheapest to
   build. ~1.5× (§11.1). Random access handled by compressing **independent frames**.
2. **Entropy-coding the quantiser's own codes** — trits/nibbles/scales rather than raw bytes.
   Better ratio than byte-level (it works in the code alphabet, not the byte stream), and the
   decoder output is exactly what the matmul wants. This is a **container re-design**, not a
   compressor bolted on.
3. **In-kernel fused decode** — never materialise a decompressed buffer. Read codes → decode in
   registers → multiply-accumulate. This is the elegant version: it removes the intermediate
   buffer entirely, so RAM and memory bandwidth stay flat.

**Ternary makes variant 3 nearly free.** The PrismML ternary kernel *already* consumes trits as
bits and never materialises floats — `dot = d * (2·Σ(y where bit=1) − sumy)`. Replacing "read
fixed-width trits" with "read an entropy-coded stream, LUT-decode to trits in registers" is a
small delta to a kernel that is *already* doing bit-domain arithmetic. Highest fit, lowest risk,
and it is only possible on the fork that has ternary kernels.

Variant 3 for Q4/dense is much harder: variable-length codes (FSE/RANS tables) inside a simdgroup
matmul kernel. Doable, but this is where the engineering cost actually lives.

### 11.5 The engineering tension: random access

A compressed stream **is not seekable**. You cannot decompress "expert 47" from a monolithic
archive. Therefore compression must be applied in **independently decodable extents aligned to the
streaming read unit** — and this *coincides exactly* with the coalesced extent of §3. One
mechanism serves both: the file is a sequence of self-contained frames; a coalesced union read
becomes "decode frames i..j".

The tension to tune: **small frames compress worse, large frames are less flexible.** Frame size is
a real design parameter with an optimum, and §3's co-occurrence ordering reduces the number of
frames a typical union touches — so the two ideas reinforce each other.

### 11.6 Why this is the dense lever specifically

- **MoE:** sparsity already removes ~90 % of expert bytes; compression then applies to the
  remaining active set (same ratio, smaller absolute).
- **Dense:** there is nothing to skip — *every* weight must be read. So a 1.5× on the read is a
  1.5× on the **dominant term**, with no other lever available except speculative amortisation.

So compression and sparsity are **complementary, not competing**: sparsity is the MoE lever,
compression is the dense lever, and both multiply the §2 coalescing win.

### 11.7 It only pays when genuinely disk-bound — and this is a trap

If a block is already in the page cache, compression **adds** decode cost to a read that was
already fast. So the engine must apply compression selectively to **cold, disk-resident** extents
only. Combined with §5 (macOS refaults `newBufferWithBytesNoCopy` pages under pressure), the safe
rule is: **the stream owns its reads explicitly; never assume a block was cached.**

### 11.8 Honest ceiling — SUPERSEDED BY MEASUREMENT (§11.0d)

**Measured ceiling is 1.01×, i.e. nothing.** An earlier draft of this section predicted a
"~1.3–1.7× multiplier, measured at ~1.5× on quantised MoE experts" based on the published
SSD-LLaMA figure. **That does not reproduce on these formats** and the section is retained only to
record the wrong prediction. The measured reality:

- ternary PTQ1_0 payload: **1.01× ceiling**, format already at its entropy floor;
- 2-bit PQ2_0 payload: **1.01× ceiling**;
- Q4_K_M / IQ4_XS: **1.00–1.05×** measured with zstd;
- decode wall: 0.62 GB/s — would forfeit even a real gain.

What remains true from this section: the only routes to fewer bytes are **lossy** (lower precision
→ a quality trade-off, not a free multiplier) or **skipping** (sparsity / coalescing). Compression
as a *lossless generic* mechanism is closed.

It also *shifts* the bottleneck: once reads shrink, decode throughput or memory bandwidth becomes
binding (on M1, writing 25 GB of decompressed weights costs ~0.37 s at 68 GB/s — not binding;
CPU decode is).

### 11.9 The measurement that decides it (blocked pending your approval)

I attempted to measure the **actual compressibility of your four GGUFs** (96 MB slice from the
tensor region of each, compressed with zstd −1/−19 and lz4, plus a decompression-throughput
timing). Safe by construction: pure file IO + compression, **no model load, no large allocation**.
Both attempts were blocked by the approval prompt, so I stopped rather than retry.

What it would answer, and why each matters:
- **real ratio** on *your* quantised weights (is it 1.5×, or 1.2×?),
- **ternary zero-fraction** → which row of §11.2 applies to Bonsai,
- **zstd decode throughput on this M1** → whether `D > 3 GB/s` (does decode stay off the critical
  path?).

Until that runs, §11.1–11.2 are grounded in external measurement (SSD-LLaMA) and arithmetic, not in
your files. I will not claim a ratio for your models without it.

