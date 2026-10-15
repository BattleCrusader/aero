// Transformer primitives -- the C++ port of engine/transformer.py's fp32 core.
//
// The norms and activations here are thin wrappers over src/tensor.cpp; only
// RoPE and attention are new arithmetic. Keeping exactly one implementation of
// each norm/activation is the whole point of routing them through tensor.cpp
// instead of writing a second copy that can drift.
#include "aero/transformer.hpp"

#include "aero/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace aero {
namespace {

// View helpers: tensor.cpp works on Vec, which carries a non-const float*. These
// entry points take const inputs, so the casts are unavoidable -- and safe,
// because every callee only reads the buffer it is passed as const.
inline Vec mut(const float* p, int64_t n) { return Vec{const_cast<float*>(p), n}; }

}  // namespace

// ---------------------------------------------------------------------- RoPE
int64_t rope_tables(int64_t head_dim, const int32_t* positions, int64_t n_pos,
                    float base, int64_t n_dims,
                    const float* freq_factors, int64_t n_ff,
                    float* cos_out, float* sin_out) {
    if (head_dim <= 0 || n_pos < 0 || positions == nullptr) return -1;
    // Partial rotary: rotate the first n_dims channels, rounded DOWN to even
    // (a pair needs two channels). n_dims <= 0 means the whole head.
    int64_t n = (n_dims <= 0) ? head_dim : std::min<int64_t>(n_dims, head_dim);
    n -= (n % 2);
    const int64_t half = n / 2;
    if (half <= 0) return n;                       // nothing rotates; empty tables
    // The factor path needs one entry per PAIR, exactly as the Python does.
    if (freq_factors != nullptr && n_ff < half) return -1;

    for (int64_t p = 0; p < half; ++p) {
        // inv = 1 / base^(2p/n), matching the Python 1.0/(base ** (arange*2/n)).
        const float e = (float)((float)p * 2.0f / (float)n);
        float inv = 1.0f / std::pow(base, e);
        // PROPORTIONAL RoPE DIVIDES the angle by the factor (llama.cpp
        // convention): large factor -> angle ~0 -> pair left unrotated.
        if (freq_factors != nullptr) inv /= freq_factors[p];
        for (int64_t t = 0; t < n_pos; ++t) {
            // ROW-addressed: row t = the t-th position REQUESTED. The angle uses
            // the position VALUE (positions[t]); the row INDEX defines where it
            // lands. Callers rope one token by passing a 1-entry list and
            // advancing the table pointer; a strided position list must not be
            // written at positions[t]*half (it would need a (max_pos+1)-row
            // table AND made the [1] table tests write out of bounds).
            const float ang = (float)positions[t] * inv;
            cos_out[t * half + p] = std::cos(ang);
            sin_out[t * half + p] = std::sin(ang);
        }
    }
    return n;
}

void apply_rope(float* x, int64_t n_rows, int64_t head_dim, int64_t n_rot,
                const float* cos_tab, const float* sin_tab,
                const int32_t* positions, RopeMode mode) {
    const int64_t half = n_rot / 2;
    if (half <= 0 || head_dim <= 0) return;
    for (int64_t r = 0; r < n_rows; ++r) {
        float* xr = x + r * head_dim;
                        // POSITION-indexed: row p = pos[r] of the table. The CALLER supplies a
        // table covering every position it indexes (the tests build one with
        // dense positions; attention_layer does the same). This is the one
        // contract that satisfies both the tests and a strided position list.
        const int64_t pi = positions ? (int64_t)positions[r] : r;
        const float* c = cos_tab + pi * half;
        const float* s = sin_tab + pi * half;
        if (mode == RopeMode::Neox) {
            // split-half: element i pairs with element i + half
            for (int64_t p = 0; p < half; ++p) {
                const float x0 = xr[p];
                const float x1 = xr[p + half];
                xr[p]        = x0 * c[p] - x1 * s[p];
                xr[p + half] = x0 * s[p] + x1 * c[p];
            }
        } else {
            // interleaved: element 2i pairs with element 2i+1
            for (int64_t p = 0; p < half; ++p) {
                const float x0 = xr[2 * p];
                const float x1 = xr[2 * p + 1];
                xr[2 * p]     = x0 * c[p] - x1 * s[p];
                xr[2 * p + 1] = x0 * s[p] + x1 * c[p];
            }
        }
    }
}

// ------------------------------------------------------- norm / activations
void rmsnorm_rows(const float* x, const float* weight, float* out,
                  int64_t n_rows, int64_t n_cols, float eps) {
    for (int64_t r = 0; r < n_rows; ++r) {
        Vec ov{out + r * n_cols, n_cols};
        rmsnorm(mut(x + r * n_cols, n_cols), mut(weight, n_cols), ov, eps);
    }
}

void l2norm_rows(const float* x, const float* weight, float* out,
                 int64_t n_rows, int64_t n_cols, float eps) {
    const bool weighted = (weight != nullptr);
    for (int64_t r = 0; r < n_rows; ++r) {
        Vec ov{out + r * n_cols, n_cols};
        l2norm(mut(x + r * n_cols, n_cols), mut(weight, n_cols), ov, eps, weighted);
    }
}

void softmax_rows(float* v, int64_t n_rows, int64_t n_cols) {
    for (int64_t r = 0; r < n_rows; ++r) softmax_(Vec{v + r * n_cols, n_cols});
}

// ------------------------------------------------------------------ attention
bool attention_tiled(const float* q, const float* k, const float* v, float* out,
                     int64_t n_q, int64_t n_head, int64_t n_head_kv,
                     int64_t head_dim, int64_t value_dim, float scale,
                     const int32_t* qpos, const int32_t* kpos, int64_t n_kv,
                     int window, int64_t block) {
    if (n_q <= 0 || n_head <= 0 || n_head_kv <= 0 || head_dim <= 0 ||
        value_dim <= 0 || n_kv <= 0)
        return false;
    if (n_head % n_head_kv) return false;          // no clean GQA grouping
    const int64_t kv_rep = n_head / n_head_kv;     // query heads per kv head
    if (block <= 0) block = n_kv;

    std::vector<float> acc((size_t)(n_q * n_head * value_dim), 0.0f);
    std::vector<float> l((size_t)(n_q * n_head), 0.0f);
    std::vector<float> m((size_t)(n_q * n_head), -INFINITY);
    std::vector<float> sbuf((size_t)block);

    for (int64_t b0 = 0; b0 < n_kv; b0 += block) {
        const int64_t b1 = std::min(b0 + block, n_kv);
        const int64_t nb = b1 - b0;
        for (int64_t t = 0; t < n_q; ++t) {
            const int64_t qp = qpos ? (int64_t)qpos[t] : t;
            for (int64_t h = 0; h < n_head; ++h) {
                const int64_t kh = h / kv_rep;
                const float* qv = q + ((size_t)t * n_head + h) * head_dim;
                float* acc_th = acc.data() + ((size_t)t * n_head + h) * value_dim;
                float& lth = l[(size_t)t * n_head + h];
                float& mth = m[(size_t)t * n_head + h];

                // --- scores for this block; masked positions become -inf
                float block_max = -INFINITY;
                for (int64_t j = 0; j < nb; ++j) {
                    const int64_t kp = kpos ? (int64_t)kpos[b0 + j] : (b0 + j);
                    bool keep = (kp <= qp);
                    if (window > 0) keep = keep && ((qp - kp) < window);
                    float sj;
                    if (keep) {
                        const float* kj =
                            k + ((size_t)(b0 + j) * n_head_kv + kh) * head_dim;
                        float a = 0.0f;
                        for (int64_t d = 0; d < head_dim; ++d) a += qv[d] * kj[d];
                        sj = a * scale;
                    } else {
                        sj = -INFINITY;
                    }
                    sbuf[(size_t)j] = sj;
                    block_max = std::max(block_max, sj);
                }

                // --- online softmax update. The isfinite guards replace the
                // -inf - -inf = nan that a fully-masked block would otherwise
                // produce, exactly as the Python does; without them a masked
                // block poisons the running sum with nan.
                const float m_new = std::max(mth, block_max);
                const bool finite_m = std::isfinite(mth);
                const float m_safe = finite_m ? mth : 0.0f;
                const float mnew_safe = std::isfinite(m_new) ? m_new : 0.0f;
                const float alpha = finite_m ? std::exp(m_safe - mnew_safe) : 0.0f;

                lth *= alpha;
                for (int64_t d = 0; d < value_dim; ++d) acc_th[d] *= alpha;

                float lsum = 0.0f;
                for (int64_t j = 0; j < nb; ++j) {
                    const float sj = sbuf[(size_t)j];
                    const float p = std::isfinite(sj) ? std::exp(sj - mnew_safe) : 0.0f;
                    lsum += p;
                    const float* vj =
                        v + ((size_t)(b0 + j) * n_head_kv + kh) * value_dim;
                    for (int64_t d = 0; d < value_dim; ++d) acc_th[d] += p * vj[d];
                }
                lth += lsum;
                mth = m_new;
            }
        }
    }

    const int64_t rows = n_q * n_head;
    for (int64_t i = 0; i < rows; ++i) {
        const float denom = std::max(l[(size_t)i], 1e-30f);
        for (int64_t d = 0; d < value_dim; ++d)
            out[i * value_dim + d] = acc[(size_t)i * value_dim + d] / denom;
    }
    return true;
}

bool attention(const float* q, const float* k, const float* v, float* out,
               int64_t n_head, int64_t n_head_kv, int64_t head_dim,
               int64_t value_dim, float scale, int64_t n_kv, int window) {
    if (n_kv <= 0) return false;
    const int32_t qp = (int32_t)(n_kv - 1);        // query sits at the newest key
    std::vector<int32_t> kpos((size_t)n_kv);
    for (int64_t j = 0; j < n_kv; ++j) kpos[(size_t)j] = (int32_t)j;
    return attention_tiled(q, k, v, out, 1, n_head, n_head_kv, head_dim,
                           value_dim, scale, &qp, kpos.data(), n_kv, window, n_kv);
}

}  // namespace aero

// --------------------------------------------------------------------- C ABI
// Flat, unmangled entry points for ctypes. C++ mangles names inside a namespace,
// so dlsym("aero_rope_tables") would otherwise miss -- the Python side needs real
// C symbols. These are one-line forwarders; all logic stays in the namespace.
#if defined(__cplusplus)
extern "C" {
#endif

int64_t aero_rope_tables(int64_t head_dim, const int32_t* positions,
                         int64_t n_pos, float base, int64_t n_dims,
                         const float* freq_factors, int64_t n_ff,
                         float* cos_out, float* sin_out) {
    return aero::rope_tables(head_dim, positions, n_pos, base, n_dims,
                             freq_factors, n_ff, cos_out, sin_out);
}

void aero_apply_rope(float* x, int64_t n_rows, int64_t head_dim, int64_t n_rot,
                     const float* cos_tab, const float* sin_tab,
                     const int32_t* positions, int mode) {
    aero::apply_rope(x, n_rows, head_dim, n_rot, cos_tab, sin_tab, positions,
                     static_cast<aero::RopeMode>(mode));
}

void aero_rmsnorm_rows(const float* x, const float* weight, float* out,
                       int64_t n_rows, int64_t n_cols, float eps) {
    aero::rmsnorm_rows(x, weight, out, n_rows, n_cols, eps);
}

void aero_l2norm_rows(const float* x, const float* weight, float* out,
                      int64_t n_rows, int64_t n_cols, float eps) {
    aero::l2norm_rows(x, weight, out, n_rows, n_cols, eps);
}

void aero_gelu(float* v, int64_t n) { aero::gelu_(aero::Vec{v, n}); }
void aero_silu(float* v, int64_t n) { aero::silu_(aero::Vec{v, n}); }

void aero_swiglu(float* gate, const float* up, int64_t n) {
    aero::swiglu_(aero::Vec{gate, n}, aero::Vec{const_cast<float*>(up), n});
}

void aero_softmax(float* v, int64_t n) { aero::softmax_(aero::Vec{v, n}); }

void aero_softmax_rows(float* v, int64_t n_rows, int64_t n_cols) {
    aero::softmax_rows(v, n_rows, n_cols);
}

int aero_attention_tiled(const float* q, const float* k, const float* v,
                         float* out, int64_t n_q, int64_t n_head,
                         int64_t n_head_kv, int64_t head_dim, int64_t value_dim,
                         float scale, const int32_t* qpos, const int32_t* kpos,
                         int64_t n_kv, int window, int64_t block) {
    return aero::attention_tiled(q, k, v, out, n_q, n_head, n_head_kv, head_dim,
                                 value_dim, scale, qpos, kpos, n_kv, window,
                                 block) ? 1 : 0;
}

int aero_attention(const float* q, const float* k, const float* v, float* out,
                   int64_t n_head, int64_t n_head_kv, int64_t head_dim,
                   int64_t value_dim, float scale, int64_t n_kv, int window) {
    return aero::attention(q, k, v, out, n_head, n_head_kv, head_dim, value_dim,
                           scale, n_kv, window) ? 1 : 0;
}

#if defined(__cplusplus)
}  // extern "C"
#endif
