# Aero engine — C++ design

Status: the engine is C++. Python remains only as a POC reference and test oracle.
Nothing in the inference path is Python.

## What "the engine" means

    GGUF on disk
      -> parse header + tensor index                 (no weights read)
      -> mmap the file                               (OS page cache is the loader)
      -> per token:
           embed -> N x [ norm -> (attention | SSM) -> FFN ] -> norm -> head
             -> sample -> detokenise

That is the whole engine. Everything below exists to make those three lines
fast and bounded.

## Modules

| file | responsibility | replaces (Python) |
|---|---|---|
| `gguf.cpp`     | header/index parse, mmap, tensor bytes view | `gguf.py` reader |
| `quant.cpp`    | dequant + fused quantised matvec/matmul | `quant.py`, `kernels.cpp` |
| `tensor.cpp`   | 2-D fp32 buffer, matmul/matvec/row-slice | numpy |
| `attention.cpp`| per-layer attn, GQA, partial/NEOX rope, KV cache | `transformer.py` |
| `ssm.cpp`      | gated delta net: conv, decay, delta rule | `ssm.py` |
| `moe.cpp`      | top-k routing, expert row reader, shared expert | `moe.py` |
| `arch.cpp`     | architecture registry + per-layer geometry | `arch.py`, `gemma.py` |
| `model.cpp`    | config from the file, layer loop, sampling | `model.py` |
| `tokenizer.cpp`| byte-level BPE and sentencepiece | `tokenizer.py` |
| `guard.cpp`    | phys_footprint ceiling + load lock | (already C++) |

**Allocation rule.** One arena, sized up front and checked against the ceiling
BEFORE it is allocated. Per-token work reuses buffers; nothing allocates in the
decode loop. A malloc inside the layer loop is a bug, not a slow path.

**Dequantisation rule.** The fp32 weight never exists. A `Q4_K` block is 144
bytes for 256 weights, and the dot product folds the scale and min into the
accumulation:

    sum_i w[i]*x[i] = d*sc * sum_i q[i]*x[i]  -  dmin*m * sum_i x[i]

so only the 4-bit `q` and one x partial per sub-block are ever needed. Measured:
7.1x less memory touched, and the fused path beat pure-fp32-BLAS while also
doing the dequantisation.

**Geometry rule.** Every attention parameter that can vary per layer is read
from the tensors, never assumed. gemma4 proves why: SWA layers have head_dim 256
/ 8 kv heads / rope base 1e4, while global layers have 512 / 1 / 1e6, and the
global layers carry no `attn_v` at all. A single global shape silently
mis-computes most of the model.

## What the memory ceiling binds on

`mmap` changes what the guard must measure. With explicit reads, the guard
counted what we malloc'd. With mmap the page cache holds the weights, so the
number that matters is the process's `phys_footprint` -- which the guard already
polls. The ceiling therefore stays the same and gets MORE honest: it now bounds
what is actually resident rather than what we asked for.

## Testing

Every module gets a test that compares against an independently derived
reference, not against itself:
  * quant   -> hand-built blocks with analytically known values, AND the Python
               reference on real model weights (two implementations, one answer)
  * rope    -> the relative-position property across offsets, plus per-layer base
  * attention -> a scalar loop reference, plus causality
  * ssm     -> a triple-loop reference, plus causality and state boundedness
  * model   -> coherence: the model must ANSWER. A finite, correctly-shaped logit
               vector proves nothing about whether it is right.

## Gate

`make test` runs every suite. It must stay green, and the coherence check must
pass with a real generation, because a fast wrong engine is worth nothing.
