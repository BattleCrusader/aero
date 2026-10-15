// Aero compute kernels — native dequantise + matmul, no Python in the loop.
//
// WHY THIS EXISTS (measured, Qwen3.8-4B on M1, one decode token):
//
//     blocked numpy matmul of a 9216x2560 Q4_K matrix   140 ms
//     the same matrix as pure fp32 BLAS matvec            4.17 ms
//     1 / 2 / 4 / 8 threads in Python                 25.8 / 28.7 / 37.2 / 50.4 s
//
// So ~97% of the time was interpreter + numpy dispatch overhead, and adding
// threads made it worse. The arithmetic is not the problem; the language is.
//
// This implements the same maths in C++ with the dequantisation FUSED into the
// inner loop. The key point: the fp32 weight is never materialised. Q4_K stores
// 256 weights in 144 bytes as
//
//     w[i] = d * sc[j] * q[i] - dmin * m[j]
//
// for sub-block j, so a dot product becomes
//
//     sum_i w[i]*x[i] = d*sc[j] * sum_i q[i]*x[i]  -  dmin*m[j] * sum_i x[i]
//
// Both sums are over the same 32 x-values. Only the 4-bit q values and one x
// partial sum per sub-block are needed, so the inner loop reads 144 bytes and
// does 256 FMAs without ever expanding to 1024 bytes of fp32.
//
// The memory target is the M1's ~68 GB/s: 144 bytes per 256 weights gives a
// 1.78 FLOP/byte ratio, so a bandwidth-bound matvec over the ~2.1 GiB of block
// tensors should take tens of milliseconds per token, not tens of seconds.
#pragma once

#include <cstddef>
#include <cstdint>

namespace aero {

// ------------------------------------------------------------------ formats
// Mirrors the ggml type ids used in the GGUF index. Only the ones the models on
// disk actually use are implemented; anything else must be refused by the
// caller rather than silently misread.
enum class DType : int {
    F32    = 0,
    F16    = 1,
    Q8_0   = 8,
    Q4_K   = 12,
    Q5_K   = 13,
    Q6_K   = 14,
    IQ4_XS = 23,
    BF16   = 30,
};

const char* dtype_name(DType t);

// Bytes a block type occupies and how many values it holds, so the caller can
// size buffers from the tensor index alone.
int64_t dtype_block_bytes(DType t);
int64_t dtype_block_values(DType t);

// ------------------------------------------------------------------ reference
// Straightforward, obviously-correct dequantisation. Used as the test oracle
// and as the fallback for types without a fused kernel. `raw` must hold
// n_values worth of whole blocks; `out` receives n_values floats.
//
// `raw_len` is how many bytes the caller ACTUALLY has, and is checked against
// what `n_values` demands before a single block is touched. The kernel cannot
// otherwise know: it is handed a bare pointer, so a caller that passes a short
// buffer (truncated read, wrong offset, a slice) makes it read past the end.
// That is not hypothetical -- it segfaulted the test suite. Pass the real length;
// the check costs one comparison per call and refuses instead of reading OOB.
// Returns false when the buffer is too short or the type has no dequantiser, so
// a caller can fall back rather than compute something wrong.
bool dequant_ref(DType t, const uint8_t* raw, int64_t raw_len,
                 float* out, int64_t n_values);

// Threaded dequantisation, block-parallel: every block is independent, so this
// splits on block index and is bit-identical to dequant_ref. Measured on M1:
// dequantising the 280 tensors a gemma-4 decode pulls through tensor() cost
// ~11.7 s single-threaded -- more than the 6.2 s disk read it depends on -- which
// made dequantise, not I/O or arithmetic, the largest single cost in a token.
// `threads` <= 1 runs serially. Checks `raw_len` exactly as dequant_ref does.
bool dequant_threaded(DType t, const uint8_t* raw, int64_t raw_len,
                      float* out, int64_t n_values, int threads);

// ------------------------------------------------------------------ fused ops
// y = W @ x, W is (n_rows, n_cols) in `dt`. x has n_cols elements, y n_rows.
// `threads` <= 1 runs serially; the caller picks the count. Returns false if
// the type/dimension combination is unsupported, so the caller can fall back
// instead of computing something wrong.
bool matvec(DType dt, const uint8_t* w, const float* x, float* y,
            int64_t n_rows, int64_t n_cols, int threads);

// Y = X @ W^T, X is (n_tokens, n_cols), W is (n_rows, n_cols), Y is
// (n_tokens, n_rows). This is the FFN/attention-projection shape.
bool matmul(DType dt, const uint8_t* w, const float* X, float* Y,
            int64_t n_tokens, int64_t n_rows, int64_t n_cols, int threads);

// Number of hardware threads worth using for this machine. Cached after the
// first call.
int hardware_threads();

}  // namespace aero
