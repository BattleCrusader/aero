// SSM (gated delta net) + MoE C++ self-test.
//
// Proves the vectorised steps against a naive triple-loop written from the maths
// (not from the implementation), the conv tap order, causality of the whole
// layer, and top-k routing. The cross-check against the PYTHON reference on the
// same inputs is tests/test_ssm_native.py -- that is the one with teeth, since a
// C++/C++ pair shares its reading of the conventions.
//
// Run: make test-cpp   (or ./build/test_ssm_cpp)
#include "aero/moe.hpp"
#include "aero/ssm.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static int CHECKS = 0, FAILS = 0;

static void check(bool ok, const char* what) {
    ++CHECKS;
    std::printf("  %s %s\n", ok ? "ok  " : "FAIL", what);
    if (!ok) ++FAILS;
}

// xorshift -> float in [-1, 1), deterministic across runs.
static uint64_t rng_state = 88172645463325252ULL;
static float frand() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (static_cast<float>(rng_state >> 11) * (1.0f / 9007199254740992.0f)) * 2.0f - 1.0f;
}
static void fill(std::vector<float>& v, float scale = 1.0f) {
    for (auto& x : v) x = frand() * scale;
}

// Naive triple-loop reference for the gated delta rule. Double accumulation, so
// "the fast one matches" means it matches to fp32 rounding, not to its own bugs.
static void naive_step(const float* q, const float* k, const float* v,
                       const float* alpha, const float* beta, const float* a_log,
                       const float* dt_bias, const float* state_in, int nv, int d,
                       float scale, float* out, float* state_out) {
    auto sp = [](double x) { return std::log1p(std::exp(-std::fabs(x))) + (x > 0 ? x : 0.0); };
    auto l2 = [&](const float* vec, std::vector<double>& o) {
        double n = 0;
        for (int i = 0; i < d; ++i) n += (double)vec[i] * vec[i];
        n = std::sqrt(n);
        if (n < 1e-6) n = 1e-6;
        for (int i = 0; i < d; ++i) o[i] = (double)vec[i] / n;
    };
    std::vector<double> qn(d), kn(d), vpred(d), vnew(d);
    for (int h = 0; h < nv; ++h) {
        l2(q + (size_t)h * d, qn);
        l2(k + (size_t)h * d, kn);
        double g = (double)a_log[h] * sp((double)alpha[h] + (double)dt_bias[h]);
        if (g > 0.0) g = 0.0;
        const double dec = std::exp(g);
        const double b = 1.0 / (1.0 + std::exp(-(double)beta[h]));
        const float* S = state_in + (size_t)h * d * d;
        float* So = state_out + (size_t)h * d * d;
        for (int i = 0; i < d; ++i)
            for (int e = 0; e < d; ++e) So[(size_t)i * d + e] = (float)((double)S[(size_t)i * d + e] * dec);
        for (int e = 0; e < d; ++e) {
            double s = 0;
            for (int i = 0; i < d; ++i) s += kn[i] * (double)So[(size_t)i * d + e];
            vpred[e] = s;
        }
        for (int e = 0; e < d; ++e) vnew[e] = ((double)v[(size_t)h * d + e] - vpred[e]) * b;
        for (int i = 0; i < d; ++i)
            for (int e = 0; e < d; ++e) So[(size_t)i * d + e] += (float)(kn[i] * vnew[e]);
        for (int e = 0; e < d; ++e) {
            double s = 0;
            for (int i = 0; i < d; ++i) s += qn[i] * (double)So[(size_t)i * d + e];
            out[(size_t)h * d + e] = (float)((double)scale * s);
        }
    }
}

static float maxdiff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// --------------------------------------------------------------- MoE fetch stub
struct FetchCtx {
    const std::vector<float>* gate;
    const std::vector<float>* up;
    const std::vector<float>* down;
    int n_expert, ffn_len, n_embd;
};
static int fetch_cb(void* ctx_, int which, int expert, float* out) {
    auto* c = static_cast<FetchCtx*>(ctx_);
    const std::vector<float>* src = which == 0 ? c->gate : which == 1 ? c->up : c->down;
    size_t per = (which == 2) ? (size_t)c->n_embd * c->ffn_len : (size_t)c->ffn_len * c->n_embd;
    const float* p = src->data() + (size_t)expert * per;
    for (size_t i = 0; i < per; ++i) out[i] = p[i];
    return 0;
}

int main() {
    std::printf("=== SSM + MoE native self-test ===\n\n");

    // ------------------------------------------------ 1. delta rule vs naive loop
    std::printf("[1] gated delta rule: vectorised step vs a naive triple-loop\n");
    {
        const int nv = 4, d = 6;
        std::vector<float> q(nv * d), k(nv * d), v(nv * d), alpha(nv), beta(nv),
            a_log(nv), dt(nv), state(nv * d * d);
        fill(q); fill(k); fill(v); fill(alpha); fill(beta); fill(dt);
        for (int i = 0; i < nv; ++i) a_log[i] = -std::fabs(frand()) * 0.5f;  // ssm_a is -exp(A_log)
        fill(state, 0.1f);
        const float scale = 1.0f / std::sqrt((float)d);

        std::vector<float> got_o(nv * d), got_s = state;
        aero_ssm_delta_step(q.data(), k.data(), v.data(), alpha.data(), beta.data(),
                            a_log.data(), dt.data(), got_s.data(), nv, d, scale, 1e-6f,
                            got_o.data());
        std::vector<float> want_o(nv * d), want_s(nv * d * d);
        naive_step(q.data(), k.data(), v.data(), alpha.data(), beta.data(), a_log.data(),
                   dt.data(), state.data(), nv, d, scale, want_o.data(), want_s.data());
        const float do_ = maxdiff(got_o, want_o), ds = maxdiff(got_s, want_s);
        std::printf("       max |out diff| = %.3e   max |state diff| = %.3e\n", do_, ds);
        check(do_ < 1e-5f, "output matches the loop reference");
        check(ds < 1e-5f, "state matches the loop reference");
    }

    // --------------------------------------------------------- 2. conv tap order
    std::printf("\n[2] causal depthwise conv: tap order and tail persistence\n");
    {
        const int C = 1, K = 3, T = 4;
        std::vector<float> x = {1, 2, 3, 4};
        std::vector<float> w = {1, 10, 100};   // w[:,0] pairs the OLDEST sample
        std::vector<float> tail((K - 1) * C, 0.0f), out(T * C);
        int tail_len = 0;
        aero_ssm_conv(x.data(), w.data(), tail.data(), &tail_len, out.data(), T, C, K);
        // t=0: window has only x0 -> out = 100*x0 = 100
        // t=2: window x0,x1,x2 -> 1*x0 + 10*x1 + 100*x2 = 1 + 20 + 300 = 321
        check(std::fabs(out[0] - 100.0f) < 1e-4f, "t=0 sees only the first sample");
        check(std::fabs(out[2] - 321.0f) < 1e-4f, "t=2 pairs w[0] with the oldest sample");
        check(tail_len == 2, "tail keeps (kernel-1) samples");

        // second call continues the filter: out for x5 uses x3,x4,x5
        std::vector<float> x2 = {5};
        std::vector<float> out2(C);
        aero_ssm_conv(x2.data(), w.data(), tail.data(), &tail_len, out2.data(), 1, C, K);
        // tail held x3=4, x4 not seen yet -- window is x4,x5 partially. Actually
        // tail = last 2 of the first buffer = {3, 4}; window for t=0 = {3,4,5}.
        check(std::fabs(out2[0] - (1 * 3 + 10 * 4 + 100 * 5)) < 1e-4f,
              "second call folds the tail: continuity across a chunk boundary");
    }

    // ------------------------------------------------------------- 3. routing
    std::printf("\n[3] top-k routing\n");
    {
        const int E = 6, K = 2;
        std::vector<float> logits = {0.1f, 3.0f, -2.0f, 1.5f, 2.9f, 0.0f};
        int64_t idx[2];
        float w[2];
        int rc = aero_moe_route(logits.data(), E, K, 1, idx, w);
        check(rc == 0, "route returns ok");
        check(idx[0] == 1 && idx[1] == 4, "top-2 are the two largest logits (1, 4)");
        check(std::fabs((w[0] + w[1]) - 1.0f) < 1e-5f, "selected weights sum to 1");
        check(w[0] > w[1], "the larger logit gets the larger weight");
        int64_t bad[2];
        check(aero_moe_route(logits.data(), E, 0, 1, bad, w) == -1, "k=0 is refused");
        check(aero_moe_route(logits.data(), E, E + 1, 1, bad, w) == -1, "k>n_expert is refused");
    }

    // -------------------------------------------------------- 4. MoE ffn block
    std::printf("\n[4] MoE expert accumulation (routed + shared)\n");
    {
        const int T = 2, E = 4, K = 2, Ee = 8, F = 5;
        std::vector<float> h(T * Ee), router_logits(T * E), gate((size_t)E * F * Ee),
            up((size_t)E * F * Ee), down((size_t)E * Ee * F);
        fill(h, 0.5f); fill(router_logits); fill(gate, 0.2f); fill(up, 0.2f); fill(down, 0.2f);
        std::vector<float> out(T * Ee, 0.0f);
        FetchCtx ctx{&gate, &up, &down, E, F, Ee};
        int rc = aero_moe_ffn_block(h.data(), router_logits.data(), T, Ee, E, K, F,
                                    1, 0, nullptr, nullptr, nullptr, nullptr,
                                    fetch_cb, &ctx, out.data());
        check(rc == 0, "ffn block returns ok");
        bool finite = true;
        for (float x : out) finite &= std::isfinite(x);
        check(finite, "output is finite");
        bool nonzero = false;
        for (float x : out) nonzero |= (std::fabs(x) > 1e-6f);
        check(nonzero, "output is non-zero (routing actually fired)");
    }

    // ------------------------------------------------------------- 5. forward
    std::printf("\n[5] full layer: causality and shape\n");
    {
        const int n_embd = 16, inner = 8, d = 4, n_group = 1, K = 3, tsr = 2;
        const int nv = inner / d, qw = n_group * d, kvw = n_group * d, qkvw = qw + kvw + inner;
        std::vector<float> w_qkv((size_t)qkvw * n_embd), w_gate((size_t)inner * n_embd),
            w_alpha((size_t)tsr * n_embd), w_beta((size_t)tsr * n_embd), a_log(tsr),
            dt(tsr), conv1d((size_t)qkvw * K), norm(d, 1.0f), w_out((size_t)n_embd * inner);
        fill(w_qkv, 0.2f); fill(w_gate, 0.2f); fill(w_alpha, 0.2f); fill(w_beta, 0.2f);
        fill(a_log, 0.3f); fill(dt); fill(conv1d, 0.3f); fill(w_out, 0.2f);

        const int T = 6;
        std::vector<float> x((size_t)T * n_embd);
        fill(x);
        std::vector<float> out1((size_t)T * n_embd), out2((size_t)T * n_embd),
            tail1((size_t)(K - 1) * qkvw, 0), tail2((size_t)(K - 1) * qkvw, 0);
        std::vector<float> s1((size_t)nv * d * d, 0), s2((size_t)nv * d * d, 0);
        int tl1 = 0, tl2 = 0;
        aero_ssm_forward(x.data(), w_qkv.data(), w_gate.data(), w_alpha.data(), w_beta.data(),
                         a_log.data(), dt.data(), conv1d.data(), norm.data(), w_out.data(),
                         T, n_embd, inner, d, n_group, K, tsr, 1e-6f, 1e-6f,
                         tail1.data(), &tl1, s1.data(), out1.data());
        std::vector<float> xb = x;
        for (int c = 0; c < n_embd; ++c) xb[(size_t)(T - 1) * n_embd + c] += 3.0f;  // perturb last token
        aero_ssm_forward(xb.data(), w_qkv.data(), w_gate.data(), w_alpha.data(), w_beta.data(),
                         a_log.data(), dt.data(), conv1d.data(), norm.data(), w_out.data(),
                         T, n_embd, inner, d, n_group, K, tsr, 1e-6f, 1e-6f,
                         tail2.data(), &tl2, s2.data(), out2.data());
        float early = 0, last = 0;
        for (int t = 0; t < T - 1; ++t)
            for (int c = 0; c < n_embd; ++c)
                early = std::max(early, std::fabs(out1[(size_t)t * n_embd + c] - out2[(size_t)t * n_embd + c]));
        for (int c = 0; c < n_embd; ++c)
            last = std::max(last, std::fabs(out1[(size_t)(T - 1) * n_embd + c] - out2[(size_t)(T - 1) * n_embd + c]));
        std::printf("       earlier-token delta = %.3e   last-token delta = %.3e\n", early, last);
        check(early == 0.0f, "editing the last token leaves earlier outputs bit-identical");
        check(last > 1e-6f, "the last token's own output does change");
    }

    std::printf("\n=== %d checks, %d failures ===\n", CHECKS, FAILS);
    return FAILS == 0 ? 0 : 1;
}
