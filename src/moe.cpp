// Mixture-of-Experts FFN inner loops — the C++ side of engine/moe.py.
//
// Same operation order and fp32 accumulation as the numpy reference, so the two
// agree to fp32 rounding (measured in tests/test_ssm_native.py).
#include "aero/moe.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace aero {
namespace {

inline float silu_f(float x) { return x / (1.0f + std::exp(-x)); }

inline float sigmoid_f(float x) { return 1.0f / (1.0f + std::exp(-x)); }

}  // namespace
}  // namespace aero

// Public C ABI. Declared extern "C" in the header; defined at global scope so the
// symbols are unmangled and ctypes finds them by name.
extern "C" {

int aero_moe_route(const float* logits, int n_expert, int k, int norm_topk_prob,
                   int64_t* idx_out, float* w_out) {
    if (k <= 0 || k > n_expert) return -1;

    std::vector<int> order(n_expert);
    std::iota(order.begin(), order.end(), 0);
    // Descending logit; ties broken by lowest index (the docstring's determinism
    // rule, which a stable descending sort of -logits also gives).
    const float* lg = logits;
    std::partial_sort(order.begin(), order.begin() + k, order.end(),
                      [lg](int a, int b) {
                          if (lg[a] != lg[b]) return lg[a] > lg[b];
                          return a < b;
                      });

    for (int i = 0; i < k; ++i) idx_out[i] = order[i];

    if (norm_topk_prob) {
        // softmax over the SELECTED experts only.
        float m = logits[order[0]];
        for (int i = 1; i < k; ++i) m = std::max(m, logits[order[i]]);
        float sum = 0.0f;
        for (int i = 0; i < k; ++i) {
            const float e = std::exp(logits[order[i]] - m);
            w_out[i] = e;
            sum += e;
        }
        for (int i = 0; i < k; ++i) w_out[i] /= sum;
    } else {
        // Raw softmax probability: denominator over ALL experts, summed in f64.
        float m = logits[0];
        for (int i = 1; i < n_expert; ++i) m = std::max(m, logits[i]);
        double denom = 0.0;
        for (int i = 0; i < n_expert; ++i)
            denom += static_cast<double>(std::exp(logits[i] - m));
        for (int i = 0; i < k; ++i)
            w_out[i] = static_cast<float>(
                static_cast<double>(std::exp(logits[order[i]] - m)) / denom);
    }
    return 0;
}

void aero_moe_expert_accum(const float* h, int n_embd,
                           const float* gate, const float* up, const float* down,
                           int ffn_len, float wt, float* acc) {
    std::vector<float> act(ffn_len);
    for (int j = 0; j < ffn_len; ++j) {
        const float* gj = gate + static_cast<size_t>(j) * n_embd;
        const float* uj = up + static_cast<size_t>(j) * n_embd;
        float g = 0.0f, u = 0.0f;
        for (int c = 0; c < n_embd; ++c) {
            g += h[c] * gj[c];
            u += h[c] * uj[c];
        }
        act[j] = aero::silu_f(g) * u;
    }
    for (int e = 0; e < n_embd; ++e) {
        const float* de = down + static_cast<size_t>(e) * ffn_len;
        float s = 0.0f;
        for (int j = 0; j < ffn_len; ++j) s += act[j] * de[j];
        acc[e] += wt * s;
    }
}

void aero_moe_shared_accum(const float* h, int T, int n_embd,
                           const float* shexp_gate_inp, const float* shexp_gate,
                           const float* shexp_up, const float* shexp_down,
                           int ffn_len, float* out) {
    std::vector<float> act(ffn_len);
    for (int t = 0; t < T; ++t) {
        const float* ht = h + static_cast<size_t>(t) * n_embd;
        // shexp_gate_inp is (1, n_embd): a single router logit per token.
        float gi = 0.0f;
        for (int c = 0; c < n_embd; ++c) gi += ht[c] * shexp_gate_inp[c];
        const float sgate = aero::sigmoid_f(gi);

        for (int j = 0; j < ffn_len; ++j) {
            const float* gj = shexp_gate + static_cast<size_t>(j) * n_embd;
            const float* uj = shexp_up + static_cast<size_t>(j) * n_embd;
            float g = 0.0f, u = 0.0f;
            for (int c = 0; c < n_embd; ++c) {
                g += ht[c] * gj[c];
                u += ht[c] * uj[c];
            }
            act[j] = aero::silu_f(g) * u;
        }
        float* ot = out + static_cast<size_t>(t) * n_embd;
        for (int e = 0; e < n_embd; ++e) {
            const float* de = shexp_down + static_cast<size_t>(e) * ffn_len;
            float s = 0.0f;
            for (int j = 0; j < ffn_len; ++j) s += act[j] * de[j];
            ot[e] += sgate * s;
        }
    }
}

int aero_moe_ffn_block(const float* h, const float* router_logits, int T,
                       int n_embd, int n_expert, int n_expert_used, int ffn_len,
                       int norm_topk_prob, int has_shared,
                       const float* shexp_gate_inp, const float* shexp_gate,
                       const float* shexp_up, const float* shexp_down,
                       aero_moe_fetch_fn fetch, void* ctx, float* out) {
    if (n_expert_used <= 0 || n_expert_used > n_expert) return -1;

    std::vector<float> gbuf(static_cast<size_t>(ffn_len) * n_embd);
    std::vector<float> ubuf(static_cast<size_t>(ffn_len) * n_embd);
    std::vector<float> dbuf(static_cast<size_t>(n_embd) * ffn_len);
    std::vector<int64_t> idx(n_expert_used);
    std::vector<float> wts(n_expert_used);

    for (int t = 0; t < T; ++t) {
        const float* ht = h + static_cast<size_t>(t) * n_embd;
        const float* logits = router_logits + static_cast<size_t>(t) * n_expert;
        if (aero_moe_route(logits, n_expert, n_expert_used, norm_topk_prob,
                           idx.data(), wts.data()) != 0)
            return -1;

        float* acc = out + static_cast<size_t>(t) * n_embd;
        for (int c = 0; c < n_embd; ++c) acc[c] = 0.0f;

        for (int i = 0; i < n_expert_used; ++i) {
            const int e = static_cast<int>(idx[i]);
            if (fetch(ctx, 0, e, gbuf.data()) != 0) return -1;
            if (fetch(ctx, 1, e, ubuf.data()) != 0) return -1;
            if (fetch(ctx, 2, e, dbuf.data()) != 0) return -1;
            aero_moe_expert_accum(ht, n_embd, gbuf.data(), ubuf.data(),
                                  dbuf.data(), ffn_len, wts[i], acc);
        }
    }

    if (has_shared && shexp_gate_inp) {
        aero_moe_shared_accum(h, T, n_embd, shexp_gate_inp, shexp_gate, shexp_up,
                              shexp_down, ffn_len, out);
    }
    return 0;
}

}  // extern "C"
