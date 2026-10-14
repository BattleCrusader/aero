# Prefill vs decode — optimisation notes

Prompt processing and generation are **different problems** and must not be
optimised with the same instinct. Numbers below are for `qwen35` (Qwen3.8-4B):
embed 2560, 16 heads / 4 KV heads, head_dim 256, ffn 9216, fp32.

| | prefill (T tokens at once) | decode (1 token) |
|---|---|---|
| matmul shape | `[T,in] @ [in,out]` — a real GEMM | `[1,in] @ [in,out]` — a GEMV |
| weights read | **once per T tokens** | **once per token** |
| bottleneck | compute + activation memory | **memory bandwidth** |
| scales with | prompt length | model size |

Consequence: prefill is relatively cheap per token and gets *cheaper* as the
prompt grows; decode is where a bandwidth-bound engine lives or dies. This is
why the earlier finding — 2.207 GB read per generated token — is the number that
matters, and why nothing about decode gets better by adding compute units.

---

## 1. The current attention is a memory bomb in prefill — a real defect

`transformer.attention` builds the full score matrix:

```python
scores = np.einsum("thd,shd->ths", q, k_all)     #  (T, H, S) fp32
```

| prompt T | score matrix |
|---|---|
| 512 | 16.8 MB |
| 2048 | 268 MB |
| 4096 | 1.07 GB |
| 16384 | 17.2 GB |
| **65536** | **256 GiB** |

**At a 64k prompt this allocates 256 GiB and OOMs the host.** Not a slowdown — a
crash. Decode is unaffected (1×16×S is trivial), so this is prefill-only, which
is exactly why it is easy to miss: short prompts look fine.

**Fix: tile over the key axis (flash-attention style).** Keep a running softmax
(online max + rescaled accumulator) and process the keys in blocks of, say, 1024.
Peak score memory becomes `T × H × block` instead of `T × H × S` — for
T=65536, H=16, block=1024 that is 4.3 GB, and with a smaller block or chunked
prefill it drops further. Numerically identical, because online softmax is exact.

This is also the single most important thing for the memory guard to watch: a
256 GiB allocation is precisely the "untracked allocation" the measured layer
exists to catch.

## 2. Fused dequant-matvec — 7.11× less traffic on the hot path

The current path calls `GGUFReader.tensor()`, which dequantises a whole tensor to
fp32 and then does an fp32 matvec. Two problems:

**a. It inflates the always-read set.** Per layer for qwen35:

| form | bytes |
|---|---|
| resident as fp32 (after dequant) | 123.2 MB |
| still quantized (Q4_K) | **17.3 MB** |
| ratio | **7.11×** |

Dequantising to fp32 makes the *always-read* weights — the 72 % of per-token
traffic identified earlier — **seven times larger**. That single choice undoes
every bit-allocation gain the design was built around.

**b. It re-reads from disk on every access.** `tensor()` seeks and dequantises
each call, so a decode step re-dequantises the entire model, per token.

**Fix: a fused dequant-matvec that consumes the quantized block format directly**
— read each block once, unpack in registers, accumulate — never materialising an
fp32 weight array. Reads drop from 2 passes (quantized + fp32) to 1, and the
resident set stays quantized.

## 3. KV path is O(n²)

```python
self.k[layer] = np.concatenate([self.k[layer], k], axis=0)
```

`np.concatenate` reallocates and copies the whole history every token: total
work over n tokens is O(n²) copying. **Fix: preallocate a capacity-sized buffer
and write at an offset.** The KV dtype should also be q8 (halves both memory and
traffic) — the earlier measurement showed KV is only 20 KiB/token for these
hybrid models, so this is a minor win here but essential for attention-heavy
models like gemma4.

## 4. Chunked prefill (bounded scratch)

Prefill in fixed-size chunks (512–2048 tokens) rather than one pass. Bounds the
activation scratch to a known maximum *independent of prompt length*, which is
what lets `--max-ram` be enforced by construction rather than by hope. Costs a
little efficiency; buys a hard guarantee.

---

## Priority

1. **Tiled attention** — without it long prompts crash. Correctness-adjacent.
2. **Fused dequant-matvec** — 7.11× on the dominant term of the dominant phase.
3. **Preallocated KV** — removes O(n²) copying.
4. **Chunked prefill** — makes the cap enforceable.

Items 2 and 4 are the ones that interact with `AeroEngine`'s device split: a
fused dequant-matvec is the natural unit to hand to a device backend, and chunked
prefill is what keeps the per-chunk plan inside the guard's admission check.

---

## STATUS: all four implemented and verified

`tests/test_optimizations.py` (32 checks, run in the gate under
`-W error::RuntimeWarning` so a nan in the hot path fails rather than warns).

### 1. Tiled attention — `attention_tiled()`, now the default path

Online softmax over key blocks; algebraically identical to the batch form.

| check | result |
|---|---|
| vs the naive full-matrix form | max diff **5.4e-07** |
| block sizes 1 / 7 / 128 / 1024 | identical results |
| peak score memory at T=512 | **8.0× smaller** than (T,H,S) |
| **at T=65536** | **256 GiB → 4.00 GiB** peak |

`stats={"peak_scores_bytes": ...}` reports the actual peak, so the bound is
measured rather than asserted. GQA head expansion now happens **per block**, so
the expanded copy is `(block,H,D)` not `(S,H,D)` — expansion no longer undoes
GQA's memory advantage.

### 2. Blocked dequant-matvec — `GGUFReader.matvec(name, x, block_rows=64)`

| check | result |
|---|---|
| vs dequantise-then-matmul | max diff **0.00e+00** (exact) |
| block_rows 1 / 7 / 128 / 512 | identical results |
| peak fp32 scratch | `block_rows × cols × 4`, **16× smaller** than the full matrix |
| quantized vs fp32 resident | **7.11×** (measurement confirms the earlier estimate) |

The weight is never materialised in full, so its resident cost is its *quantized*
size — which is the property the whole bit-allocation design depends on.

### 3. Preallocated KV cache

| check | result |
|---|---|
| buffer reallocated while appending | **never** (identity-checked) |
| appending past capacity | raises `MemoryError` instead of growing silently |
| token-by-token vs batched forward | still equal (1.1e-06, from the core suite) |

Capacity is now a constructor argument, so the KV footprint is known up front
and can be charged to `--max-ram` before a single token is generated.

### 4. Chunked prefill — `prefill_chunks(total, chunk)`

Verified to cover `[0, total)` exactly with no gaps or overlaps across
`total/chunk` combinations including degenerate ones (chunk > total, total=1).

### 5. SIGPIPE (found while running the gate, not planned)

`aero-chat | head` printed `Exception ignored ... BrokenPipeError`. Fixed by
resetting `SIGPIPE` to `SIG_DFL` so the CLI behaves like a normal Unix tool. The
gate now asserts stderr is free of `BrokenPipe` when the pipe closes early —
previously this passed only because the exit code happened to stay 0.


## Not worth doing (measured, for the record)

- **More compute units.** GPU+ANE measured 0.89×, CPU+GPU 0.56×. Decode is
  bandwidth-bound; extra units contend for one bus.
- **Lossless weight compression.** 1.01× ceiling — the quantizers already sit at
  the entropy floor.
- **Dequantising to fp32 and caching it.** It looks like a speedup and is a 7.11×
  inflation of the hot set.
