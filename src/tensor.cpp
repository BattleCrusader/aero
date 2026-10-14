// Tensor primitives.
//
// The matvec is the engine's hot loop and is written to be memory-bound, which is
// what it actually is: for a 9216x2560 matrix the arithmetic is 23.6M multiply-adds
// (~4 ms of FLOPs) against 94 MB of fp32 traffic. So the loop is structured to
// stream W once, in row order, and keep x resident in cache -- x is 10 KB and fits
// L1/L2 across the whole matrix, while W does not.
#include "aero/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

namespace aero {
namespace {

// thread::hardware_concurrency() counts SMT siblings on x86; on Apple Silicon it
// is the real core count. Capped, because past the physical cores this loop only
// adds contention -- measured: 8 threads beat 4 beat 1 on this machine.
int default_threads() {
    static const int n = [] {
        unsigned hc = std::thread::hardware_concurrency();
        if (hc == 0) hc = 1;
        return static_cast<int>(std::min(hc, 8u));
    }();
    return n;
}

template <typename Fn>
void parallel_rows(int64_t rows, int threads, Fn fn) {
    const int t = (threads <= 1) ? 1 : std::min<int>(threads, (int)rows);
    if (t <= 1) {
        for (int64_t r = 0; r < rows; ++r) fn(r);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(t);
    for (int k = 0; k < t; ++k)
        pool.emplace_back([&, k] {
            for (int64_t r = k; r < rows; r += t) fn(r);
        });
    for (auto& th : pool) th.join();
}

inline float sigmoidf_(float z) { return 1.0f / (1.0f + std::exp(-z)); }

// erf, Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7). std::erf exists in C++11 but
// is scalar-call overhead in a vectorised loop; this is branch-free and the error
// is far below fp32 resolution.
inline float erff_(float x) {
    const float s = std::copysign(1.0f, x);
    const float ax = std::fabs(x);
    const float t = 1.0f / (1.0f + 0.3275911f * ax);
    const float p = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t
                      - 0.284496736f) * t + 0.254829592f) * t;
    return s * (1.0f - p * std::exp(-ax * ax));
}

}  // namespace

int tensor_default_threads() { return default_threads(); }

// ---------------------------------------------------------------- matvec
void matvec(const Tensor2D& w, const Vec& x, const Vec& y, int threads) {
    const int64_t C = w.cols;
    parallel_rows(w.rows, threads ? threads : default_threads(), [&](int64_t r) {
        const float* wr = w.row(r);
        float acc = 0.0f;
        // unrolled by 4: the loop is memory-bound, so the win is issuing more
        // loads before the adds block on latency
        int64_t c = 0;
        for (; c + 4 <= C; c += 4) {
            acc += wr[c] * x.data[c] + wr[c + 1] * x.data[c + 1]
                 + wr[c + 2] * x.data[c + 2] + wr[c + 3] * x.data[c + 3];
        }
        for (; c < C; ++c) acc += wr[c] * x.data[c];
        y.data[r] = acc;
    });
}

// ---------------------------------------------------------------- matmul
void matmul(const Tensor2D& w, const Tensor2D& x, Tensor2D& y, int threads) {
    // parallelise over OUTPUT ROWS, not tokens: each row's reduction stays inside
    // one thread, so the summation order never changes with the thread count and
    // results are bit-identical whatever `threads` is.
    if (x.rows == 1) {
        matvec(w, Vec{x.data, x.cols}, Vec{y.data, y.rows}, threads);
        return;
    }
    parallel_rows(w.rows, threads ? threads : default_threads(), [&](int64_t r) {
        const float* wr = w.row(r);
        for (int64_t t = 0; t < x.rows; ++t) {
            const float* xt = x.row(t);
            float acc = 0.0f;
            for (int64_t c = 0; c < w.cols; ++c) acc += wr[c] * xt[c];
            y.data[t * y.cols + r] = acc;
        }
    });
}

// ---------------------------------------------------------------- activations
void silu_(Vec v) {
    for (int64_t i = 0; i < v.n; ++i) v.data[i] *= sigmoidf_(v.data[i]);
}

void gelu_(Vec v) {
    constexpr float kInvSqrt2 = 0.7071067811865476f;
    for (int64_t i = 0; i < v.n; ++i) {
        const float z = v.data[i];
        v.data[i] = 0.5f * z * (1.0f + erff_(z * kInvSqrt2));
    }
}

void mul_(Vec a, const Vec& b) {
    for (int64_t i = 0; i < a.n; ++i) a.data[i] *= b.data[i];
}

void swiglu_(Vec gate, const Vec& up) {
    for (int64_t i = 0; i < gate.n; ++i) {
        const float g = gate.data[i];
        gate.data[i] = (g * sigmoidf_(g)) * up.data[i];
    }
}

// ---------------------------------------------------------------- norm
void rmsnorm(const Vec& x, const Vec& weight, Vec& out, float eps) {
    float ss = 0.0f;
    for (int64_t i = 0; i < x.n; ++i) ss += x.data[i] * x.data[i];
    const float inv = 1.0f / std::sqrt(ss / (float)x.n + eps);
    for (int64_t i = 0; i < x.n; ++i)
        out.data[i] = x.data[i] * inv * (weight.data ? weight.data[i] : 1.0f);
}

void l2norm(const Vec& x, const Vec& weight, Vec& out, float eps, bool weighted) {
    // Denominator is max(||x||, eps) -- eps is a FLOOR ON THE NORM, not an additive
    // term inside the sqrt. The two differ: with ||x||=50, max(50,1e-6)=50 while
    // sqrt(2500+1e-6)=50.00000002, a small but CONSISTENT relative offset that
    // looks like fp32 noise and is not. They coincide only when ||x|| >> eps, which
    // is exactly the regime the SSM q/k vectors live in, so this was silent.
    //
    // Matches engine/ssm.py:l2_normalize, which is the definition the model was
    // written against:
    //     n = np.linalg.norm(x); return x / np.maximum(n, eps)
    float ss = 0.0f;
    for (int64_t i = 0; i < x.n; ++i) ss += x.data[i] * x.data[i];
    const float norm = std::sqrt(ss);
    const float denom = norm > eps ? norm : eps;
    const float inv = 1.0f / denom;
    for (int64_t i = 0; i < x.n; ++i)
        out.data[i] = x.data[i] * inv *
                      (weighted && weight.data ? weight.data[i] : 1.0f);
}

void rmsnorm_rows(Tensor2D& t, float eps) {
    for (int64_t r = 0; r < t.rows; ++r) {
        float* p = t.row(r);
        float ss = 0.0f;
        for (int64_t c = 0; c < t.cols; ++c) ss += p[c] * p[c];
        const float inv = 1.0f / std::sqrt(ss / (float)t.cols + eps);
        for (int64_t c = 0; c < t.cols; ++c) p[c] *= inv;
    }
}

// ---------------------------------------------------------------- softmax
void softmax_(Vec v) {
    if (v.n <= 0) return;
    float m = v.data[0];
    for (int64_t i = 1; i < v.n; ++i) m = std::max(m, v.data[i]);
    float s = 0.0f;
    for (int64_t i = 0; i < v.n; ++i) {
        v.data[i] = std::exp(v.data[i] - m);
        s += v.data[i];
    }
    const float inv = (s > 0.0f) ? 1.0f / s : 0.0f;
    for (int64_t i = 0; i < v.n; ++i) v.data[i] *= inv;
}

int argmax(const Vec& v) {
    int best = 0;
    float bv = v.data[0];
    for (int64_t i = 1; i < v.n; ++i)
        if (v.data[i] > bv) { bv = v.data[i]; best = (int)i; }
    return best;
}

// ============================================================ C ABI
//
// Why this block exists at all: every primitive below was already implemented,
// tested and correct -- and completely unreachable from the engine, because
// tensor.cpp exported nothing and was compiled only into its own test binary. That
// is why the forward pass still ran in numpy while "the tensor primitives are
// done" was true. A primitive nobody can call is not ported; it is a private
// implementation detail. These wrappers make the existing functions callable.
//
// Conventions, applied uniformly so ctypes stays simple:
//   * buffers are flat row-major fp32, lengths passed explicitly;
//   * a length of 0 or less is REFUSED (returns 0) rather than treated as empty,
//     because a wrong length is the bug class that already segfaulted this repo;
//   * void returns are 1 on success, 0 on refusal.

}  // namespace aero

extern "C" {

int aero_tensor_threads(void) { return aero::tensor_default_threads(); }

// y[r] = dot(W.row(r), x); W is (n_rows, n_cols), x has n_cols, y gets n_rows.
int aero_tensor_matvec(const float* w, const float* x, float* y,
                       int64_t n_rows, int64_t n_cols, int threads) {
    if (!w || !x || !y || n_rows <= 0 || n_cols <= 0) return 0;
    aero::Tensor2D vw{const_cast<float*>(w), n_rows, n_cols};
    aero::Vec vx{const_cast<float*>(x), n_cols};
    aero::Vec vy{y, n_rows};
    aero::matvec(vw, vx, vy, threads);
    return 1;
}

// Y[t][r] = dot(W.row(r), X.row(t)); W (n_rows,n_cols), X (n_tokens,n_cols),
// Y (n_tokens,n_rows).
int aero_tensor_matmul(const float* w, const float* X, float* Y,
                       int64_t n_tokens, int64_t n_rows, int64_t n_cols,
                       int threads) {
    if (!w || !X || !Y || n_tokens <= 0 || n_rows <= 0 || n_cols <= 0) return 0;
    aero::Tensor2D vw{const_cast<float*>(w), n_rows, n_cols};
    aero::Tensor2D vX{const_cast<float*>(X), n_tokens, n_cols};
    aero::Tensor2D vY{Y, n_tokens, n_rows};
    aero::matmul(vw, vX, vY, threads);
    return 1;
}

// in-place activations
int aero_tensor_silu(float* v, int64_t n) {
    if (!v || n <= 0) return 0;
    aero::silu_(aero::Vec{v, n});
    return 1;
}

int aero_tensor_gelu(float* v, int64_t n) {
    if (!v || n <= 0) return 0;
    aero::gelu_(aero::Vec{v, n});
    return 1;
}

int aero_tensor_softmax(float* v, int64_t n) {
    if (!v || n <= 0) return 0;
    aero::softmax_(aero::Vec{v, n});
    return 1;
}

// a[i] *= b[i]
int aero_tensor_mul(float* a, const float* b, int64_t n) {
    if (!a || !b || n <= 0) return 0;
    aero::mul_(aero::Vec{a, n}, aero::Vec{const_cast<float*>(b), n});
    return 1;
}

// gate[i] = silu(gate[i]) * up[i], in place on gate
int aero_tensor_swiglu(float* gate, const float* up, int64_t n) {
    if (!gate || !up || n <= 0) return 0;
    aero::swiglu_(aero::Vec{gate, n}, aero::Vec{const_cast<float*>(up), n});
    return 1;
}

// out = x / sqrt(mean(x^2) + eps) * weight
int aero_tensor_rmsnorm(const float* x, const float* weight, float* out,
                        int64_t n, float eps) {
    if (!x || !weight || !out || n <= 0) return 0;
    aero::Vec vx{const_cast<float*>(x), n};
    aero::Vec vw{const_cast<float*>(weight), n};
    aero::Vec vo{out, n};
    aero::rmsnorm(vx, vw, vo, eps);
    return 1;
}

// out = x / sqrt(sum(x^2) + eps) * weight   (weighted = 1), or weightless
// (weighted = 0, weight may be NULL) -- the SSM q/k and gemma4 V normalisations.
int aero_tensor_l2norm(const float* x, const float* weight, float* out,
                       int64_t n, float eps, int weighted) {
    if (!x || !out || n <= 0) return 0;
    if (weighted && !weight) return 0;
    aero::Vec vx{const_cast<float*>(x), n};
    aero::Vec vw{const_cast<float*>(weight), n};
    aero::Vec vo{out, n};
    aero::l2norm(vx, vw, vo, eps, weighted != 0);
    return 1;
}

// Per-row weightless RMS over a 2-D block, in place. rows x cols fp32.
int aero_tensor_rmsnorm_rows(float* t, int64_t rows, int64_t cols, float eps) {
    if (!t || rows <= 0 || cols <= 0) return 0;
    aero::Tensor2D v{t, rows, cols};
    aero::rmsnorm_rows(v, eps);
    return 1;
}

int aero_tensor_argmax(const float* v, int64_t n) {
    if (!v || n <= 0) return -1;
    aero::Vec vv{const_cast<float*>(v), n};
    return aero::argmax(vv);
}

}  // extern "C"
