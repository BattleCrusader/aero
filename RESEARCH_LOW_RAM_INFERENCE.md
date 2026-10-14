# Running Large Models on 8GB-Class Machines — Research Survey

No code in this document. Research + design options only. Nothing here was executed
against the target models.

## 0. The three targets (actual GGUF metadata, read from disk)

| Model | File | Size | arch | layers | experts | active/tok | embed | head_kv | notes |
|---|---|---|---|---|---|---|---|---|---|
| Ternary-Bonsai-2-27B PQ2_0 | `PrismLM/...PQ2_0.gguf` | 6.7 GB | `qwen35` | 64 | dense | — | 5120 | 4 | **ternary/2-bit** (file_type 141), ffn 17408 |
| Ternary-Bonsai-2-27B PTQ1_0 | `PrismLM/...PTQ1_0.gguf` | 5.5 GB | `qwen35` | 64 | dense | — | 5120 | 4 | ternary 1-bit class |
| Qwen3.6-14B-A3B FableVibes | `tvall43/...Q4_K_M.gguf` | 7.9 GB | `qwen35moe` | 40 | **90** | **8** | 2048 | 2 | MoE, expert_ffn **512** (tiny experts) |
| gemma-4-12b-it | `unsloth/...IQ4_XS.gguf` | 5.9 GB | `gemma4` | 48 | dense | — | 3840 | 16 | IQ4_XS (file_type 30), ffn 15360, SWA layers |

Observations that drive the whole design:

1. **These are all small-ish files (5.5–7.9 GB).** On an 8 GB machine the bottleneck is not
   "the model doesn't fit on disk-to-RAM" — it is that **the model + KV cache + OS do not
   fit in RAM at the same time**. 8 GB RAM, ~2 GB OS/browser headroom → ~6 GB usable. A
   6.7 GB model leaves zero room for KV. So the problem reduces to: *shrink what must be
   resident, and page the rest.*
2. **The ternary model is the interesting one.** Ternary (PQ2/PTQ1) weights mean the model
   is already ~2 bits/weight; the *disk* cost is low but **dequantization cost is high** —
   you reconstruct a full-precision matmul input from packed ternary codes every use. That
   changes the offload economics completely (compute-bound, not IO-bound, on reload).
3. **The MoE target has unusually small experts** (expert_ffn=512 vs dense ffn 17408 for the
   dense sibling). 90 experts x 40 layers x 512 ffn. This is a *fine-grained* MoE: routing is
   spread thin, so an LRU expert cache will thrash less than a coarse 8-expert MoE, but the
   per-expert tensors are so small that disk IO granularity becomes the problem (thousands of
   tiny reads).
4. **gemma4 has per-layer head_count_kv as a 48-length list** — i.e. **hybrid attention**
   (some full-attention, some sliding-window-attention layers, note `key_length_swa`/
   `value_length_swa`). That means a uniform "offload N layers" scheme is wrong for it; the
   KV cost is non-uniform per layer.

## 1. What the field actually does (taxonomy)

There are five distinct families. They solve *different* problems and are often confused.

### A. Layer/parameter streaming (offload whole layers)
* **airllm** — the canonical "run 70B on 4GB" recipe. Splits the model by layer, keeps exactly
  one (or two, with prefetch) layer resident, streams the next layer from disk/CPU each step.
* **Apple MLX / PyTorch `device_map="auto"` + `offload_folder`** — same idea, framework-level.
* **llama.cpp `--n-gpu-layers` / `-ot` / `--no-mmap`** — partial offload; with mmap, the OS
  page cache *is* the tier manager.
* **Reality check (Petals paper, §3.1):** for a single token through 175B at PCIe 4.0 ×16
  (31.5 GB/s), best-case *hardware*, offloading costs **11 s/forward pass**. On a consumer
  SSD (2–7 GB/s) it's far worse. So per-layer streaming gives maybe **0.05–1 tok/s** on the
  models you named — and only because they're small. This is the classic approach and it
  **works but is slow**; the win is purely "it fits", never "it's fast".

### B. OS page-cache offload via mmap (no custom code)
* Standard llama.cpp **mmaps the GGUF**: pages not touched stay on disk; the kernel keeps the
  hot "resident trunk" cached. On Linux this is genuinely effective; on macOS, `MAP_SHARED`
  eviction is less aggressive and the machine will swap/thrash first.
* **Justine Tunney's `madvise(MADV_WILLNEED)` / `MADV_PAGEOUT` work** on llama.cpp is the
  refinement: explicitly tell the kernel which pages are coming next (prefetch) and which are
  done (evict). This is the *cheapest* possible implementation — no runtime, just hints.
* **Caveat for you:** page cache is still RAM. Your memory note is exactly right — "it's page
  cache, not RSS" is a violation pattern. On 8 GB macOS you must budget page cache *inside*
  the 6 GB usable, not on top of it.

### C. Expert-level lazy loading (MoE only) — *your prior art*
* Load the model, but keep **expert tensors out of the resident set**; page in only the
  experts the router selects this step (8 of 90 here), LRU-evict the rest.
* llama.cpp now has native **`-ot`/`--override-tensor` with `-n-cpu-moe`/`--n-cpu-moe`**, which
  pushes MoE expert blocks to CPU/disk while keeping attention on GPU. This is the upstream
  version of the fork you abandoned.
* **Your documented wall (LAZY_EXPERT_LOADING.md):** correct routing needs the *in-graph*
  expert ids produced inside `llama_decode`, which llama.cpp doesn't expose → offline router
  predicts wrong experts → garbage. **MLX solved this** because it executes eagerly, so you can
  read ids in Python between router and matmul. Your `mlx_lazy_moe.py` works (verified 3221 MB,
  coherent) on gpt-oss-20b.
* **Key insight for the new targets:** the Qwen3.6 MoE has *very small* experts (ffn 512), so
  the working set is tiny — this is the *ideal* case for (C). But the dense models
  (Bonsai, gemma) get **nothing** from (C) — no experts to skip. For dense, you're back to (A).

### D. Distributed / pooled inference (multi-device)
* **Petals** (arXiv 2312.08361) — pool machines over the internet; for >50B, sending
  *activations over a slow network* beats streaming *weights from local SSD*. Fault-tolerant,
  uneven-hardware load balancing. This is the "offload layers to other devices" line.
* **prima.cpp** (arXiv 2504.08791) — the most relevant recent paper. Distributed on-device
  inference for 30–70B on **home clusters** with mixed CPU/GPU, **insufficient RAM/VRAM, slow
  disks, Wi-Fi**. Two ideas matter:
  - **Pipelined-Ring Parallelism (PRP):** layers are cut into striped chunks rotating across
    devices, *overlapping disk IO with compute and communication*; explicitly fixes the
    "prefetch-release conflict" in mmap offloading.
  - **Halda:** heterogeneity-aware scheduler that co-optimises per-device CPU/GPU work under
    RAM/VRAM caps. Results: **70B at 674 ms/token (≈1.5 tok/s) on 4 consumer devices**,
    **32B at 26 tok/s** with speculative decoding, **<6% memory pressure, OOM-free**.
  - Crucially it stores weights in **OS page cache** so the OS can reclaim under pressure —
    "preserves user experience". This is the design philosophy you want on a 16 GB Mac.
* **The IoT / botnet result** (Anthropic, self-replicating-AI writeup) is the
  "layers offloaded to IoT devices" paper you mentioned: **Gemma4-26B-A4B across 15 × $55
  Android TV boxes → 0.39 tok/s**, two layers per box, Petals-style. It proves the *mechanism*
  (distributed layer sharding) works on memory-starved nodes — but 0.39 tok/s and a botnet
  threat model. Useful as evidence, not as a product.

### E. Compression-first (make the model smaller, not the machine bigger)
* This is the family that actually **fits 8 GB without a runtime**: quantise harder. Your
  Bonsai PQ2 is already here (ternary). For the others: Q4_K_M → IQ3/IQ2, or AQLM/GPTQ-2bit.
* **Ternary/BitNet angle:** if the ternary model is BitNet-style, a dedicated **ternary kernel**
  (ADD-only matmul) is dramatically faster and smaller than dequant→fp16→matmul. On 8 GB, a
  purpose-built ternary path may beat any offloading scheme outright.

## 2. Why the naive plans fail on *your* hardware

| Trap | Why it fails on 8–16 GB macOS |
|---|---|
| `ulimit -v` as a RAM cap | No-op on macOS for malloc (your note). Never a guard. |
| "It's mmap, so it's on disk" | macOS keeps `MAP_SHARED` pages resident and swaps first. Page cache ∈ RAM. |
| Full mmap into a Metal buffer | `buffer_from_host_ptr` maps the *whole* file; overflows Metal's ~10.9 GB limit → machine crash. **This already bit you.** |
| Uniform layer offload | gemma4's KV is per-layer non-uniform (SWA vs full) → wrong budget, thrash. |
| Offline router preload | Wrong experts → garbage (proven, documented). |
| Streaming dense 27B per-token | ~0.1 tok/s. Fits, but unusable interactively. |

## 3. Recommended architecture (hybrid, per-model)

The honest conclusion: **there is no single method**. The right design is a dispatcher that
picks the family per architecture. Concretely:

```
                 ┌─────────────────────────────────────────────┐
                 │  Model classifier (from GGUF kv metadata)   │
                 │  dense vs moe · ternary vs affine · hybrid  │
                 └───────────────┬─────────────────────────────┘
        ┌────────────────────────┼────────────────────────────┐
        ▼                        ▼                            ▼
  MoE (small experts)      Dense affine              Ternary dense
   → expert-LRU lazy        → page-cache + madvise    → ternary kernel,
     (MLX eager, your         (mmap + WILLNEED/         no runtime offload
      existing code)           PAGEOUT) + partial GPU    (compression-first)
```

**Tier order (cheapest → most invasive), apply in order, stop when it fits:**

1. **Compression-first.** Try IQ3/IQ2 quant of the dense models; for ternary, use a
   BitNet-style ADD kernel. If the quantised file + KV fits in ~5.5 GB, *you need no runtime at
   all*. This is the highest-ROI step and you're already halfway there with Bonsai.
2. **mmap + explicit prefetch/evict hints.** No custom runtime. `madvise(MADV_WILLNEED)` for
   the next N layers, `MADV_PAGEOUT` for completed ones (Tunney's technique). Works for dense.
   Budget page cache inside the RAM cap.
3. **Expert-LRU lazy loading (MoE only).** Reuse your verified `mlx_lazy_moe.py`. The Qwen3.6
   fine-grained MoE is the *best-case* candidate (tiny experts, 8/90 active).
4. **Partial GPU + CPU split (`-n-cpu-moe` / `-ot`).** Keep attention + router on GPU, expert
   blocks on CPU/disk. Upstream equivalent of your abandoned llama.cpp fork — no graph surgery.
5. **Pipelined layer streaming (last resort, dense + too big).** Per-layer prefetch overlapping
   compute, prima.cpp-style. Accept 0.1–1 tok/s.

**Two things to borrow from prima.cpp regardless of tier:**
- **Overlap IO with compute** (prefetch layer L+1 while computing L) — the single biggest
  latency win. Without it, streaming is serial and unusable.
- **Page-cache-resident weights + OS reclaim** — so the machine degrades gracefully instead of
  OOM-rebooting. This directly matches your "physically cannot crash the machine" requirement.

**Two things to keep from your own prior art:**
- The **MLX eager-execution trick** for reading router ids — it's the only clean way to do
  correct MoE lazy loading, and you already proved it works.
- The **per-tensor allocation gate** (`!expert_no_alloc`) — never `buffer_from_host_ptr` on the
  full mmap, or Metal overflows and the box dies.

### F. exo (how it actually works — read from source)

Cloned `exo-explore/exo@main` and read `src/exo/master/placement*.py`,
`src/exo/worker/engines/mlx/auto_parallel.py`, `utils_mlx.py`, `shared/types/worker/shards.py`.
It is **not** a low-RAM single-machine technique — it's a **cluster** technique, and it's
worth understanding precisely *why* it doesn't answer your question.

**What it does**

1. **Two sharding modes** (`Sharding` enum): `Pipeline` and `Tensor`.
   - `Pipeline` = layers split by count across nodes. Each node holds a contiguous half-open
     layer interval `[start_layer, end_layer)` (`PipelineShardMetadata`). Activation is passed
     node→node like a relay: `PipelineFirstLayer` does `mx.distributed.recv_like(x, r-1)`,
     `PipelineLastLayer` sends to `r+1`. Ring topology, one token in flight through the chain.
   - `Tensor` = every node holds **all** layers, but each tensor is sliced: attention heads and
     `n_kv_heads` are divided by world size (`n_kv_heads //= self.N`), MoE experts are split via
     `ShardedMoE`, and each shard does an all-sum to combine (`_AllSumLinear`). This is where the
     "1.8x on 2 devices, 3.2x on 4" comes from — it's *speed*, not capacity.
2. **Placement is memory-aware and proportional.** `allocate_layers_proportionally()` gives each
   node a layer count proportional to its `ram_available / total_ram`, then
   `_allocate_and_validate_layers()` **raises if any node can't hold its share**:
   `required = total_storage * node_layers / total_layers`, error if `> ram_available`.
   → **A cycle is only chosen if the *sum* of node RAM can hold the whole model**
   (`filter_cycles_by_memory`: `sum(ram_available) >= required_memory`). Nodes are picked by
   *total* free RAM, then `get_smallest_cycles` prefers the fewest nodes that fit.
3. **Memory model is crude and weight-only.** `ModelCard.storage_size` (from HF config) is the
   only input. **There is no KV-cache term in the placement math** — KV is allocated at runtime,
   so exo sizes placement on weights alone and can still OOM/swap on long contexts.
4. **Weight loading uses MLX `lazy=True`** (`load_model(model_path, lazy=True, strict=False)`).
   That's MLX's *lazy evaluation* (build graph, defer materialisation), **not** disk paging: the
   shard's weights still get evaluated into unified memory, layer by layer, for progress
   reporting. **There is no disk-offload/mmap-of-weights path in the MLX engine** — no `MADV_*`,
   no `pread`, no eviction. `mx.clear_cache()` is called after load to drop transient buffers.
5. **KV/state**: `_patch_hybrid_cache` + `KVCache`/`ArraysCache` per shard — each node keeps the
   KV for *its own* layers only. So pipeline sharding does reduce per-node KV proportionally.

**Why it does not solve your 8 GB problem**

| Question | exo's answer |
|---|---|
| Can a model bigger than *one* node's RAM run? | Yes — but only if there are **other nodes** to put layers on. |
| Does it make a single 8 GB box run a 27B model? | **No.** `_allocate_and_validate_layers` explicitly raises when a node can't hold its layer share. One node = you must fit the whole model. |
| Is there any disk/mmap offload? | **No.** Weights are `lazy=True`-loaded into unified memory. No page-cache tier, no eviction. |
| Does it handle MoE experts specially? | Only in *tensor* mode: `ShardedMoE` splits *which experts* across nodes (capacity + speed). No per-token expert paging from disk. |
| Is it an option for your three models? | Only as a **multi-machine** answer. And at 5.5–7.9 GB the network hop (even 99%-lower-latency RDMA) costs more than it saves. |

**Where exo *is* genuinely instructive for your design**

- **Pipeline-as-relay is the cleanest layer-offload contract**: layer interval + `recv_like`/
  `send` at the boundaries. If you ever add a cheap second box, this is the shape to copy —
  and `PipelineFirstLayer`/`PipelineLastLayer` are ~20 lines each. You could implement the same
  contract with **disk instead of network**: the "previous rank" is a `pread` from the GGUF.
- **Proportional-by-RAM allocation with a hard validation gate.** Copy the *gate*: refuse to
  place a shard that doesn't fit rather than silently swapping. This is exactly the
  "physically cannot crash the machine" discipline you want — exo enforces it at placement time,
  your watchdog enforces it at runtime. Combining both is the right move.
- **Their memory model is weight-only — that's the bug to avoid.** Any 8 GB design must budget
  **weights + KV + page cache + OS** together. exo doesn't, which is why it can still thrash.
- **`set_wired_limit_for_model(get_weights_size(shard))`** — they pin the weights region. Note
  the *inverse* of your lesson: exo pins only the shard it owns; you must pin only the working
  set, never the full mmap (your `buffer_from_host_ptr` crash).

**Verdict:** exo is family (D) done well — cluster capacity + tensor-parallel *speed*. It is
orthogonal to your single-machine low-RAM problem, and it explicitly refuses the case you care
about (one small node, oversized model). Take its **layer-interval shard contract** and its
**hard-fit validation gate**; ignore everything else for the 8 GB case.

## 4. Multi-device option (your "IoT offload" reference)

If single-machine is not enough, the proven pattern is **Petals/prima.cpp-style layer sharding
across a small home cluster** (e.g. the Mac + a couple of cheap boxes/Raspberry Pis). Numbers
to set expectations honestly:
- 70B on 4 consumer devices: ~1.5 tok/s (prima.cpp, <6% mem pressure).
- 26B-A4B over 15 TV boxes: 0.39 tok/s (Anthropic experiment).
- Petals: 70B/176B over the internet, "up to faster than offloading" for interactive
  generation when network ≫ SSD bandwidth.

For *your* 5.5–7.9 GB models the multi-device overhead (network latency, Wi-Fi jitter) would
likely make it *slower* than single-machine mmap+prefetch. Multi-device pays off only above
~30–50B. **Recommendation: keep the three named models single-machine; reserve distributed for
a future 70B+ target.**

## 5. Open questions before any implementation

1. **Ternary kernel availability:** does llama.cpp/MLX have a ternary (PQ2) fast path for
   `qwen35`, or does it dequant to fp16? If the latter, the ternary model's whole advantage
   evaporates on reload — this changes the tier choice. (Check `PrismML-llama.cpp` fork — you
   have it locally.)
2. **KV size at 8 GB:** 64 layers × kv4 × key_len 256 for Bonsai, and gemma4's hybrid layers —
   compute exact KV bytes per 1k context *before* budgeting. At 262144 context they're huge;
   you'll need a small context anyway.
3. **Expert tensor granularity (Qwen3.6):** with ffn=512, are expert tensors contiguous per
   expert in the GGUF? If not, `pread` per expert needs byte-range math from the GGUF tensor
   index (same trick as the safetensors `data_offsets` in your DESIGN.md).
4. **macOS page-out behaviour:** whether `MADV_PAGEOUT` is honoured as aggressively as on Linux
   (it is a no-op on some macOS versions). If not, tier 2 degrades and you fall to tier 3/4.
5. **Which engine:** MLX (eager, easy router hooks, Metal) vs llama.cpp (upstream `-n-cpu-moe`,
   no fork needed, GGUF-native). Your prior art chose MLX for control; llama.cpp upstream may
   now make tier 4 free.

## 6. Sources

- Petals — *Distributed Inference and Fine-tuning of LLMs Over The Internet*, arXiv 2312.08361
- prima.cpp — *Fast 30–70B LLM Inference on Heterogeneous and Low-Resource Home Clusters*, arXiv 2504.08791
- Strata — *Hierarchical Context Caching for Long Context LLM Serving*, arXiv 2508.18572 (KV, not weights; relevant if context is the binding constraint)
- NEO — *Saving GPU Memory Crisis with CPU Offloading*, MLSys 2025 (KV + attention compute offload)
- VeriSplit — *Secure and Practical Offloading of ML Inferences across IoT Devices*, arXiv 2406.00586 (layer-by-layer offload, privacy/verification angle)
- airllm (GitHub) — per-layer streaming reference implementation
- Justine Tunney — *Edge AI Just Got Faster* (llama.cpp `madvise` prefetch/evict)
- Anthropic — self-replicating-AI / IoT experiment (Gemma4-26B-A4B over 15 TV boxes, 0.39 tok/s)
- llama.cpp issues #5207 (low-RAM gguf) and `-n-cpu-moe` / `--override-tensor` docs
- Your own: `MoE_llama/LAZY_EXPERT_LOADING.md`, `MoE_llama/DESIGN.md`
