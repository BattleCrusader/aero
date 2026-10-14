// Gated DeltaNet (SSM) inner loops — the C++ side of engine/ssm.py.
//
// The comment conventions the Python file guards are reproduced 1:1 here; see the
// ASSUMPTIONS block in engine/ssm.py for why each one exists. Everything is fp32
// in the same operation order as the numpy reference, so the two agree to fp32
// rounding (measured in tests/test_ssm_native.py).
#include "aero/ssm.hpp"

#include <cmath>
#include <cstring>
#include <vector>

namespace aero {
namespace {

inline float silu_f(float x) { return x / (1.0f + std::exp(-x)); }

inline float sigmoid_f(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// softplus(x) = log1p(exp(-|x|)) + max(x, 0). Stable for large |x|.
inline float softplus_f(float x) {
    return std::log1pf(std::exp(-std::fabs(x))) + (x > 0.0f ? x : 0.0f);
}

// L2-normalise one row of length n, dividing by max(norm, eps) -- bit-for-bit
// what numpy does in engine/ssm.py l2_normalize().
inline void l2_row(const float* x, int n, float eps, float* out) {
    float ss = 0.0f;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    float nv = std::sqrt(ss);
    if (nv < eps) nv = eps;
    for (int i = 0; i < n; ++i) out[i] = x[i] / nv;
}

// y[r] = dot(w.row(r), x); w is (n_out, n_in), row-major. fp32 accumulation.
void matvec_fp32(const float* w, const float* x, float* y, int n_out, int n_in) {
    for (int r = 0; r < n_out; ++r) {
        const float* wr = w + static_cast<size_t>(r) * n_in;
        float s = 0.0f;
        for (int c = 0; c < n_in; ++c) s += wr[c] * x[c];
        y[r] = s;
    }
}

// The gated delta rule for one token, with q/k given as n_k heads and v as n_v
// heads. v-head h reads k-head h % n_k -- the reference TILES k/q up to the
// v-head count, so the grouping is h % n_k, not h // ratio.
void delta_step_grouped(const float* qrow, const float* krow, const float* vrow,
                        const float* a_row, const float* b_row,
                        const float* a_log, const float* dt_bias,
                        float* state, int n_k, int n_v, int d,
                        float scale, float eps, float* out) {
    std::vector<float> qn(static_cast<size_t>(n_v) * d);
    std::vector<float> kn(static_cast<size_t>(n_v) * d);
    std::vector<float> vpred(d), vnew(d);

    for (int h = 0; h < n_v; ++h) {
        const int kk = h % n_k;
        l2_row(qrow + static_cast<size_t>(kk) * d, d, eps,
               qn.data() + static_cast<size_t>(h) * d);
        l2_row(krow + static_cast<size_t>(kk) * d, d, eps,
               kn.data() + static_cast<size_t>(h) * d);
    }

    for (int h = 0; h < n_v; ++h) {
        // g = a_log * softplus(alpha + dt_bias); a decay cannot exceed 1, so
        // clamp g <= 0 (a positive a_log saturates instead of overflowing exp).
        float g = a_log[h] * softplus_f(a_row[h] + dt_bias[h]);
        if (g > 0.0f) g = 0.0f;
        const float dec = std::exp(g);
        const float b = sigmoid_f(b_row[h]);

        float* S = state + static_cast<size_t>(h) * d * d;
        const float* knh = kn.data() + static_cast<size_t>(h) * d;
        const float* qnh = qn.data() + static_cast<size_t>(h) * d;
        const float* vh = vrow + static_cast<size_t>(h) * d;
        float* oh = out + static_cast<size_t>(h) * d;

        for (size_t i = 0; i < static_cast<size_t>(d) * d; ++i) S[i] *= dec;

        // v_pred[e] = sum_i kn[i] * S[i, e]
        for (int e = 0; e < d; ++e) {
            float s = 0.0f;
            for (int i = 0; i < d; ++i) s += knh[i] * S[static_cast<size_t>(i) * d + e];
            vpred[e] = s;
        }
        for (int e = 0; e < d; ++e) vnew[e] = (vh[e] - vpred[e]) * b;

        // rank-1 write: S[i, e] += kn[i] * v_new[e]
        for (int i = 0; i < d; ++i) {
            const float ki = knh[i];
            float* Srow = S + static_cast<size_t>(i) * d;
            for (int e = 0; e < d; ++e) Srow[e] += ki * vnew[e];
        }

        // read: out[e] = scale * sum_i qn[i] * S[i, e]
        for (int e = 0; e < d; ++e) {
            float s = 0.0f;
            for (int i = 0; i < d; ++i) s += qnh[i] * S[static_cast<size_t>(i) * d + e];
            oh[e] = scale * s;
        }
    }
}

// T-token scan with explicit strides (so the forward pass can point straight at
// the packed q/k/v slices of the qkv buffer without copying).
void scan_strided(const float* q, int q_stride, const float* k, int k_stride,
                  const float* v, int v_stride, const float* alpha, int a_stride,
                  const float* beta, int b_stride, const float* a_log,
                  const float* dt_bias, float* state, int T, int n_k, int n_v,
                  int d, float scale, float eps, float* outs, int o_stride) {
    for (int t = 0; t < T; ++t) {
        delta_step_grouped(q + static_cast<size_t>(t) * q_stride,
                           k + static_cast<size_t>(t) * k_stride,
                           v + static_cast<size_t>(t) * v_stride,
                           alpha + static_cast<size_t>(t) * a_stride,
                           beta + static_cast<size_t>(t) * b_stride,
                           a_log, dt_bias, state, n_k, n_v, d, scale, eps,
                           outs + static_cast<size_t>(t) * o_stride);
    }
}

}  // namespace
}  // namespace aero

// The public C ABI. Declared extern "C" in the header; defined at global scope so
// the symbols are unmangled and ctypes finds them by name.
extern "C" {

float aero_ssm_softplus(float x) { return aero::softplus_f(x); }

void aero_ssm_delta_step(const float* q, const float* k, const float* v,
                         const float* alpha, const float* beta,
                         const float* a_log, const float* dt_bias,
                         float* state, int n_heads, int d, float scale, float eps,
                         float* out) {
    // q/k are already tiled: n_k == n_v == n_heads, so h % n_heads == h.
    aero::delta_step_grouped(q, k, v, alpha, beta, a_log, dt_bias, state, n_heads,
                             n_heads, d, scale, eps, out);
}

void aero_ssm_scan(const float* qh, const float* kh, const float* vh,
                   const float* alpha, const float* beta,
                   const float* a_log, const float* dt_bias,
                   float* state, int T, int n_k, int n_v, int d,
                   float scale, float eps, float* outs) {
    aero::scan_strided(qh, n_k * d, kh, n_k * d, vh, n_v * d, alpha, n_v, beta,
                       n_v, a_log, dt_bias, state, T, n_k, n_v, d, scale, eps,
                       outs, n_v * d);
}

void aero_ssm_conv(const float* x, const float* w, float* tail, int* tail_len,
                   float* out, int T, int channels, int kernel) {
    const int tl = *tail_len;
    const int n = tl + T;
    std::vector<float> buf(static_cast<size_t>(n) * channels);
    if (tl > 0) {
        std::memcpy(buf.data(), tail,
                    sizeof(float) * static_cast<size_t>(tl) * channels);
    }
    std::memcpy(buf.data() + static_cast<size_t>(tl) * channels, x,
                sizeof(float) * static_cast<size_t>(T) * channels);

    for (int t = 0; t < T; ++t) {
        const int end = n - T + t + 1;   // inclusive end of the window
        const int start = end - kernel;
        for (int c = 0; c < channels; ++c) {
            float acc = 0.0f;
            for (int j = 0; j < kernel; ++j) {
                const int idx = start + j;
                if (idx >= 0 && idx < n) {
                    // j = 0 is the OLDEST input in the window -> w[:, 0].
                    acc += buf[static_cast<size_t>(idx) * channels + c] *
                           w[static_cast<size_t>(c) * kernel + j];
                }
            }
            out[static_cast<size_t>(t) * channels + c] = acc;
        }
    }

    int keep = kernel - 1;
    if (keep < 0) keep = 0;
    if (keep > n) keep = n;
    if (keep > 0) {
        std::memcpy(tail, buf.data() + static_cast<size_t>(n - keep) * channels,
                    sizeof(float) * static_cast<size_t>(keep) * channels);
    }
    *tail_len = keep;
}

int aero_ssm_forward(const float* x,
                     const float* w_qkv, const float* w_gate,
                     const float* w_alpha, const float* w_beta,
                     const float* a_log, const float* dt_bias,
                     const float* conv1d, const float* norm, const float* w_out,
                     int T, int n_embd, int inner_size, int state_size,
                     int n_group, int conv_kernel, int time_step_rank,
                     float eps, float norm_eps,
                     float* conv_tail, int* tail_len,
                     float* state, float* out) {
    if (inner_size <= 0 || state_size <= 0 || n_group <= 0 || T < 0) return -1;
    if (inner_size % state_size != 0) return -1;
    const int n_v = inner_size / state_size;
    const int n_k = n_group;
    if (n_v % n_k != 0) return -1;
    // delta_rule_step indexes alpha/beta by v-head, so the rank must match.
    if (time_step_rank != n_v) return -1;

    const int q_width = n_k * state_size;
    const int kv_width = n_k * state_size;
    const int qkv_width = q_width + kv_width + inner_size;
    const int d = state_size;
    const float scale = 1.0f / std::sqrt(static_cast<float>(d));

    // 1. fused projection + causal depthwise conv + silu
    std::vector<float> qkv(static_cast<size_t>(T) * qkv_width);
    for (int t = 0; t < T; ++t) {
        aero::matvec_fp32(w_qkv, x + static_cast<size_t>(t) * n_embd,
                          qkv.data() + static_cast<size_t>(t) * qkv_width,
                          qkv_width, n_embd);
    }
    aero_ssm_conv(qkv.data(), conv1d, conv_tail, tail_len, qkv.data(), T,
                  qkv_width, conv_kernel);
    for (size_t i = 0; i < qkv.size(); ++i) qkv[i] = aero::silu_f(qkv[i]);

    // 2. gates
    std::vector<float> alpha(static_cast<size_t>(T) * n_v);
    std::vector<float> beta(static_cast<size_t>(T) * n_v);
    std::vector<float> gate(static_cast<size_t>(T) * inner_size);
    for (int t = 0; t < T; ++t) {
        aero::matvec_fp32(w_alpha, x + static_cast<size_t>(t) * n_embd,
                          alpha.data() + static_cast<size_t>(t) * n_v, n_v, n_embd);
        aero::matvec_fp32(w_beta, x + static_cast<size_t>(t) * n_embd,
                          beta.data() + static_cast<size_t>(t) * n_v, n_v, n_embd);
        aero::matvec_fp32(w_gate, x + static_cast<size_t>(t) * n_embd,
                          gate.data() + static_cast<size_t>(t) * inner_size,
                          inner_size, n_embd);
    }

    // 3. recurrence scan over the packed q/k/v slices of qkv
    std::vector<float> outs(static_cast<size_t>(T) * n_v * d);
    aero::scan_strided(qkv.data(), qkv_width, qkv.data() + q_width, qkv_width,
                       qkv.data() + q_width + kv_width, qkv_width, alpha.data(),
                       n_v, beta.data(), n_v, a_log, dt_bias, state, T, n_k, n_v,
                       d, scale, eps, outs.data(), n_v * d);

    // 4. gated RMSNorm over the head dim, then silu gate. (No output projection
    //    buffer needed: the projection below reads this directly.)
    std::vector<float> o(static_cast<size_t>(T) * inner_size);
    for (int t = 0; t < T; ++t) {
        const float* orow = outs.data() + static_cast<size_t>(t) * inner_size;
        const float* grow = gate.data() + static_cast<size_t>(t) * inner_size;
        float* orow_out = o.data() + static_cast<size_t>(t) * inner_size;
        float ss = 0.0f;
        for (int c = 0; c < inner_size; ++c) ss += orow[c] * orow[c];
        ss /= static_cast<float>(inner_size);   // RMS (unit_offset=True)
        const float denom = std::sqrt(ss + norm_eps);
        for (int c = 0; c < inner_size; ++c) {
            // tile(w.norm, n_v): column c reads norm[c % state_size]
            const float wnorm = norm[c % state_size];
            orow_out[c] = (orow[c] / denom) * wnorm * aero::silu_f(grow[c]);
        }
    }

    // 5. output projection
    for (int t = 0; t < T; ++t) {
        aero::matvec_fp32(w_out, o.data() + static_cast<size_t>(t) * inner_size,
                          out + static_cast<size_t>(t) * n_embd, n_embd, inner_size);
    }
    return 0;
}

}  // extern "C"
