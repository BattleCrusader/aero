// Tensor: a dense fp32 2-D buffer, and the arithmetic the layer loop needs.
//
// Everything the forward pass computes is a small set of primitives over these:
// a matmul, a matvec, a norm, an activation, an elementwise multiply. Keeping
// them in one place means the layer code reads as the maths it implements, and
// means there is exactly one implementation of each to verify.
//
// LAYOUT: row-major, [n_rows, n_cols]. A weight matrix is stored [out, in], so
// y = W x is a row-dot against x -- which is why matvec() is the hot one.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace aero {

// A non-owning view of fp32 data. The engine holds its buffers in one arena and
// hands out views, so nothing here ever allocates except where stated.
struct Tensor2D {
    float* data = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;

    float* row(int64_t r) { return data + r * cols; }
    const float* row(int64_t r) const { return data + r * cols; }
    int64_t n_elem() const { return rows * cols; }
};

// A vector is a 1-D view; separate type so the compiler catches row/vec mixups.
struct Vec {
    float* data = nullptr;
    int64_t n = 0;
};

// ---------------------------------------------------------------- arithmetic
// y[r] = dot(W.row(r), x). The hot loop of the whole engine.
void matvec(const Tensor2D& w, const Vec& x, const Vec& y, int threads);

// Y[t][r] = dot(W.row(r), X.row(t)). Same work, batched over tokens.
void matmul(const Tensor2D& w, const Tensor2D& x, Tensor2D& y, int threads);

// ---------------------------------------------------------------- activations
void silu_(Vec v);                  // in place: v *= sigmoid(v)
void gelu_(Vec v);                 // in place, exact (erf) form
void mul_(Vec a, const Vec& b);    // a *= b, elementwise
void swiglu_(Vec gate, const Vec& up);   // gate = silu(gate) * up, in place on gate

// ---------------------------------------------------------------- norm
// out = x / sqrt(mean(x^2) + eps) * weight      (RMSNorm)
void rmsnorm(const Vec& x, const Vec& weight, Vec& out, float eps);
// out = x / sqrt(sum(x^2) + eps) * weight       (L2; used by the SSM q/k)
void l2norm(const Vec& x, const Vec& weight, Vec& out, float eps, bool weighted);
// Weightless RMS over a 2-D block, per row -- gemma4's V normalisation.
void rmsnorm_rows(Tensor2D& t, float eps);

// ---------------------------------------------------------------- softmax
void softmax_(Vec v);

// ---------------------------------------------------------------- sampling
// Greedy only for now; the sampler proper lives with the model.
int argmax(const Vec& v);

}  // namespace aero
