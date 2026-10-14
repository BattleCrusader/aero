// Transformer core primitives: RoPE and grouped-query attention, plus the fp32
// norm/activation entry points the layer loop builds on.
//
// This is the C++ port of the fp32 primitives in engine/transformer.py.
// Correctness here is defined as NUMERICAL AGREEMENT with that Python module on
// identical inputs, and it is checked two ways:
//   * tests/test_transformer_cpp.cpp   -- C++ against independently written
//                                          scalar references (self-contained)
//   * tests/test_transformer_native.py -- C++ against engine/transformer.py,
//                                          printing the observed max difference
//
// SCOPE. NOT the block loop, NOT model assembly, NOT the KV cache. The cache --
// including its fp32/int8/int4 quantisation and ring buffer -- stays in Python
// and hands these functions plain pointers to its k/v buffers. Nothing here owns
// or persists cache state.
//
// LAYOUT. Row-major and unmangled. A "row" is one (token, head) for the RoPE and
// attention entry points, and one token for the norm/softmax ones.
#pragma once

#include <cstddef>
#include <cstdint>

namespace aero {

// RoPE pair conventions, mirroring transformer.ROPE_MODE. The two are
// rotation-invariant in the sense a property test usually checks (q.k depends
// only on relative position), so a property test cannot tell them apart -- which
// is why the mode is explicit and both are checked against the Python.
enum class RopeMode : int {
    Neox        = 0,   // pairs (x[i], x[i+half])  -- Qwen / HF transformers
    Interleaved = 1,   // pairs (x[2i], x[2i+1])   -- original GPT-NeoX code
};

// ---------------------------------------------------------------------- RoPE
// cos/sin tables for RoPE. Returns n_rot, the number of leading channels that
// rotate (always even); cos_out and sin_out each receive n_pos * (n_rot / 2)
// floats. Returns -1 (writing nothing) when the request is malformed, so a
// caller falls back rather than computing a different rotation.
//
// `n_dims` selects PARTIAL rotary: only the first n_dims channels rotate, the
// rest pass through. n_dims <= 0 means the whole head. This is not optional --
// gemma4's windowed layers rotate 256 of their 512 channels.
//
// `freq_factors` (n_ff entries, optional) is PROPORTIONAL RoPE. ggml computes
// the per-pair angle as theta / ff[p], so a factor of 1 changes nothing and a
// huge factor drives the angle to 0 (cos=1, sin=0), leaving that pair
// UNROTATED. gemma4 encodes "rotate only the first 64 pairs" with ff = 1 for the
// first 64 and 1e30 for the remaining 192. n_ff must cover n_rot/2 pairs.
int64_t rope_tables(int64_t head_dim, const int32_t* positions, int64_t n_pos,
                    float base, int64_t n_dims,
                    const float* freq_factors, int64_t n_ff,
                    float* cos_out, float* sin_out);

// Rotate x IN PLACE. x is (n_rows, head_dim) row-major; only the first n_rot
// channels of each row are touched, so the tail passes through unchanged in both
// modes (partial rotary). cos_tab/sin_tab are FULL tables of shape
// (n_pos_table, n_rot/2); the row selected for each x-row is `positions[row]`
// (or `row` itself when positions is null). See RopeMode for the pair order.
void apply_rope(float* x, int64_t n_rows, int64_t head_dim, int64_t n_rot,
                const float* cos_tab, const float* sin_tab,
                const int32_t* positions, RopeMode mode);

// ------------------------------------------------------- norm / activations
// Per-row over the last axis:
//     out[r] = x[r] / sqrt(mean(x[r]^2) + eps) * weight    (RMSNorm)
// `weight` may be null (parameter-free); `out` may alias `x`. Thin wrappers over
// the single-vector aero::rmsnorm in tensor.cpp -- one implementation, not two.
void rmsnorm_rows(const float* x, const float* weight, float* out,
                  int64_t n_rows, int64_t n_cols, float eps);

// L2 variant: the divisor is sqrt(sum(x^2) + eps), i.e. no 1/n. The two differ
// by a constant sqrt(n) per row, which changes attention temperature without
// changing direction -- so both exist rather than one being assumed.
void l2norm_rows(const float* x, const float* weight, float* out,
                 int64_t n_rows, int64_t n_cols, float eps);

// Per-row softmax over the last axis (the Python `axis=-1` form).
void softmax_rows(float* v, int64_t n_rows, int64_t n_cols);

// ------------------------------------------------------------------ attention
// Grouped-query attention with an ONLINE (streaming) softmax, so the peak score
// buffer is O(n_head * block) rather than O(n_head * n_kv) -- the difference
// between fitting and an OOM at long context. Online softmax is algebraically
// identical to the batch form, so this is exact, not an approximation.
//
//   q   (n_q,   n_head,    head_dim)
//   k   (n_kv,  n_head_kv, head_dim)     -- plain pointer; NOT owned, NOT cached
//   v   (n_kv,  n_head_kv, value_dim)
//   out (n_q,   n_head,    value_dim)
//
// GQA: query head h reads kv head h / (n_head / n_head_kv) -- the standard
// "adjacent queries share a kv head" grouping, verified against the Python
// (`np.repeat(kb, kv_rep, axis=1)` in attention_tiled).
//
// Masking: key j is visible to query t iff kpos[j] <= qpos[t], and when
// `window` > 0 additionally iff qpos[t] - kpos[j] < window (sliding window).
// qpos/kpos may be null, meaning 0..n_q-1 / 0..n_kv-1. `block` is the streaming
// key-block size (>= n_kv runs in a single block). Returns false on a shape it
// cannot honour (e.g. n_head not a multiple of n_head_kv).
bool attention_tiled(const float* q, const float* k, const float* v, float* out,
                     int64_t n_q, int64_t n_head, int64_t n_head_kv,
                     int64_t head_dim, int64_t value_dim, float scale,
                     const int32_t* qpos, const int32_t* kpos, int64_t n_kv,
                     int window, int64_t block);

// Single-token decode: one query at the newest cached position (n_kv - 1)
// against the whole cache. Convenience wrapper over attention_tiled with n_q=1.
bool attention(const float* q, const float* k, const float* v, float* out,
               int64_t n_head, int64_t n_head_kv, int64_t head_dim,
               int64_t value_dim, float scale, int64_t n_kv, int window);

}  // namespace aero
