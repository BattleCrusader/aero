// MoE inner loops vs independently written scalar references.
//
// WHY THIS SUITE EXISTS. src/moe.cpp shipped with only a Python cross-check
// (tests/test_ssm_native.py covers it there, 2.38e-07 / 3.58e-07), and a C++ module
// with no C++ suite is one refactor away from silent rot: the Python test can be
// satisfied while the C ABI's own contract (arg order, k range, fetch protocol)
// drifts. Every primitive is checked here against maths written out longhand, so a
// shared misreading of moe.py's conventions cannot pass.
//
// The conventions that matter, all pinned below:
//   * top-k is by descending logit, ties to the LOWEST index;
//   * norm_topk_prob normalises over the SELECTED experts only -- the un-normalised
//     branch divides by a denominator over ALL experts, summed in f64;
//   * the expert FFN is silu(gate @ h) * (up @ h), then down @ that, scaled by the
//     routing weight, ACCUMULATED into acc;
//   * the shared expert is sigmoid-gated over a 1-row gate_inp;
//   * a fetch callback that returns non-zero ABORTS the block (it must not silently
//     contribute a zero expert).
#include "aero/moe.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

static int g_checks = 0, g_fail = 0;

static void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) { ++g_fail; std::printf("  FAIL %s\n", what.c_str()); }
    else std::printf("  ok   %s\n", what.c_str());
}
static void banner(const char* s) { std::printf("\n%s\n", s); }

static float silu_ref(float x) { return x / (1.0f + std::exp(-x)); }
static float sigmoid_ref(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// A (rows, cols) row-major matrix times a vector, longhand.
static std::vector<float> mv(const std::vector<float>& W, const std::vector<float>& x,
                             int rows, int cols) {
    std::vector<float> y(rows, 0.0f);
    for (int r = 0; r < rows; ++r) {
        float s = 0.0f;
        for (int c = 0; c < cols; ++c) s += W[(size_t)r * cols + c] * x[c];
        y[r] = s;
    }
    return y;
}

static uint64_t g_seed = 99;
static float rnd() {
    g_seed = g_seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return ((int)((g_seed >> 33) & 0xffff) / 32768.0f - 1.0f) * 0.5f;
}
static std::vector<float> rndv(size_t n) {
    std::vector<float> v(n);
    for (auto& x : v) x = rnd();
    return v;
}

// ------------------------------------------------------------------ fetch table
struct Experts {
    int n_expert = 0, ffn_len = 0, n_embd = 0;
    std::vector<std::vector<float>> gate, up, down;   // per expert
    int calls = 0;
    int fail_on_which = -1;                            // to test the abort path
};

static Experts make_experts(int n_expert, int ffn_len, int n_embd) {
    Experts e;
    e.n_expert = n_expert; e.ffn_len = ffn_len; e.n_embd = n_embd;
    for (int i = 0; i < n_expert; ++i) {
        e.gate.push_back(rndv((size_t)ffn_len * n_embd));
        e.up.push_back(rndv((size_t)ffn_len * n_embd));
        e.down.push_back(rndv((size_t)n_embd * ffn_len));
    }
    return e;
}

static int fetch_cb(void* ctx, int which, int expert, float* out) {
    Experts* e = (Experts*)ctx;
    ++e->calls;
    if (which == e->fail_on_which) return 7;           // deliberate failure
    const std::vector<float>& src = which == 0 ? e->gate[expert]
                                 : which == 1 ? e->up[expert] : e->down[expert];
    std::copy(src.begin(), src.end(), out);
    return 0;
}

int main() {
    const int NE = 24, K = 2;                 // 24 experts, pick 2

    banner("[1] routing: top-k by descending logit, ties to the lowest index");
    {
        std::vector<float> lg(NE);
        for (int i = 0; i < NE; ++i) lg[i] = 0.1f * i;          // expert NE-1 wins
        lg[3] = lg[7] = 5.0f;                                   // a tie at the top
        std::vector<int64_t> idx(K);
        std::vector<float> w(K);
        check(aero_moe_route(lg.data(), NE, K, 0, idx.data(), w.data()) == 0,
              "route returns 0");
        // descending: 5.0 at index 3 (tie -> lowest), then 5.0 at 7; expert NE-1 is
        // 2.3 which loses to 5.0
        check(idx[0] == 3 && idx[1] == 7,
              "top-2 are the tied entries, lowest index first (got " +
              std::to_string(idx[0]) + "," + std::to_string(idx[1]) + ")");
    }

    banner("[2] routing weights: the two normalisation conventions");
    {
        std::vector<float> lg = rndv(NE);
        std::vector<int64_t> idx(K);
        std::vector<float> w(K);
        aero_moe_route(lg.data(), NE, K, 1, idx.data(), w.data());
        check(std::fabs(w[0] + w[1] - 1.0f) < 1e-6f,
              "norm_topk_prob=1: the selected weights sum to 1");
        // reference: softmax over the two selected only
        const float m = std::max(lg[idx[0]], lg[idx[1]]);
        const float e0 = std::exp(lg[idx[0]] - m), e1 = std::exp(lg[idx[1]] - m);
        check(std::fabs(w[0] - e0 / (e0 + e1)) < 1e-6f &&
              std::fabs(w[1] - e1 / (e0 + e1)) < 1e-6f,
              "norm_topk_prob=1: weights are the selected-only softmax");

        aero_moe_route(lg.data(), NE, K, 0, idx.data(), w.data());
        float mx = lg[0];
        for (int i = 1; i < NE; ++i) mx = std::max(mx, lg[i]);
        double den = 0.0;
        for (int i = 0; i < NE; ++i) den += std::exp((double)lg[i] - (double)mx);
        const float want0 = (float)(std::exp((double)lg[idx[0]] - mx) / den);
        check(std::fabs(w[0] - want0) < 1e-7f,
              "norm_topk_prob=0: the weight is a FULL-softmax probability");
        check((w[0] + w[1]) < 1.0f,
              "norm_topk_prob=0: the pair does NOT sum to 1 (24 experts share mass)");
    }

    banner("[3] one expert's weighted accumulation, against a scalar reference");
    {
        const int FF = 6, NE_ = 8;
        std::vector<float> h = rndv(NE_);
        auto e = make_experts(1, FF, NE_);
        std::vector<float> acc(NE_, 0.0f);
        const float wt = 0.37f;
        aero_moe_expert_accum(h.data(), NE_, e.gate[0].data(), e.up[0].data(),
                               e.down[0].data(), FF, wt, acc.data());
        // longhand: act = silu(gate@h) * (up@h); y = down @ act
        std::vector<float> gv = mv(e.gate[0], h, FF, NE_);
        std::vector<float> uv = mv(e.up[0], h, FF, NE_);
        std::vector<float> act(FF);
        for (int i = 0; i < FF; ++i) act[i] = silu_ref(gv[i]) * uv[i];
        std::vector<float> y = mv(e.down[0], act, NE_, FF);
        float maxd = 0.0f;
        for (int i = 0; i < NE_; ++i)
            maxd = std::max(maxd, std::fabs(acc[i] - wt * y[i]));
        check(maxd < 1e-6f, "weighted expert contribution matches (max |d| " +
                            std::to_string(maxd) + ")");

        // accumulation: a second expert must ADD, not overwrite
        std::vector<float> before = acc;
        aero_moe_expert_accum(h.data(), NE_, e.gate[0].data(), e.up[0].data(),
                              e.down[0].data(), FF, 1.0f, acc.data());
        float d2 = 0.0f;
        for (int i = 0; i < NE_; ++i)
            d2 = std::max(d2, std::fabs(acc[i] - (before[i] + y[i])));
        check(d2 < 1e-6f, "a second expert ADDS into acc (does not overwrite)");
    }

    banner("[4] the shared expert: sigmoid-gated, added over the block");
    {
        const int T = 3, FF = 5, NE_ = 7;
        std::vector<float> h = rndv((size_t)T * NE_);
        std::vector<float> gi = rndv(NE_);                 // (1, n_embd)
        std::vector<float> g = rndv((size_t)FF * NE_);
        std::vector<float> u = rndv((size_t)FF * NE_);
        std::vector<float> d = rndv((size_t)NE_ * FF);
        std::vector<float> out((size_t)T * NE_, 0.0f);
        aero_moe_shared_accum(h.data(), T, NE_, gi.data(), g.data(), u.data(),
                              d.data(), FF, out.data());
        float maxd = 0.0f;
        for (int t = 0; t < T; ++t) {
            std::vector<float> ht(h.begin() + (size_t)t * NE_,
                                   h.begin() + (size_t)(t + 1) * NE_);
            std::vector<float> gatev = mv(std::vector<float>(gi.begin(), gi.end()),
                                          ht, 1, NE_);
            const float sc = sigmoid_ref(gatev[0]);
            std::vector<float> gv = mv(g, ht, FF, NE_);
            std::vector<float> uv = mv(u, ht, FF, NE_);
            std::vector<float> act(FF);
            for (int i = 0; i < FF; ++i) act[i] = silu_ref(gv[i]) * uv[i];
            std::vector<float> y = mv(d, act, NE_, FF);
            for (int i = 0; i < NE_; ++i)
                maxd = std::max(maxd, std::fabs(out[(size_t)t * NE_ + i] - sc * y[i]));
        }
        check(maxd < 1e-6f, "shared expert matches sigmoid(gate_inp@h) * ffn(h) "
                            "(max |d| " + std::to_string(maxd) + ")");
    }

    banner("[5] the full block: routing + fetch + accumulation");
    {
        const int T = 2, FF = 4, NE_ = 10, K2 = 3;
        std::vector<float> h = rndv((size_t)T * NE_);
        std::vector<float> rl = rndv((size_t)T * NE_);          // (T, n_expert)
        std::vector<float> gi = rndv(NE_);
        std::vector<float> g = rndv((size_t)FF * NE_);
        std::vector<float> u = rndv((size_t)FF * NE_);
        std::vector<float> d = rndv((size_t)NE_ * FF);
        std::vector<float> out((size_t)T * NE_, 0.0f);

        // NOTE the convention: n_expert is the FILE's expert count and the router
        // logits are (T, n_expert) -- here 24 experts with 10 in the fetch table would
        // be a mismatch, so this block uses n_expert = the table's size.
        Experts e = make_experts(NE_, FF, NE_);
        e.calls = 0;
        int rc = aero_moe_ffn_block(h.data(), rl.data(), T, NE_, NE_, K2, FF, 1, 1,
                                   gi.data(), g.data(), u.data(), d.data(),
                                   fetch_cb, &e, out.data());
        check(rc == 0, "ffn_block returns 0 with a working fetch");
        check(e.calls == T * K2 * 3,
              "the fetch was called T*k*3 times (gate, up, down per routed expert) "
              "-> " + std::to_string(e.calls));

        // reference: route each row, accumulate each expert, add the shared expert
        std::vector<float> want((size_t)T * NE_, 0.0f);
        for (int t = 0; t < T; ++t) {
            std::vector<float> row(rl.begin() + (size_t)t * NE_,
                                   rl.begin() + (size_t)(t + 1) * NE_);
            std::vector<int64_t> idx(K2);
            std::vector<float> w(K2);
            aero_moe_route(row.data(), NE_, K2, 1, idx.data(), w.data());
            std::vector<float> ht(h.begin() + (size_t)t * NE_,
                                  h.begin() + (size_t)(t + 1) * NE_);
            for (int kk = 0; kk < K2; ++kk) {
                const int ex = (int)idx[kk];
                std::vector<float> acc(NE_, 0.0f);
                aero_moe_expert_accum(ht.data(), NE_, e.gate[ex].data(),
                                      e.up[ex].data(), e.down[ex].data(), FF,
                                      w[kk], acc.data());
                for (int i = 0; i < NE_; ++i) want[(size_t)t * NE_ + i] += acc[i];
            }
        }
        std::vector<float> shared((size_t)T * NE_, 0.0f);
        aero_moe_shared_accum(h.data(), T, NE_, gi.data(), g.data(), u.data(),
                              d.data(), FF, shared.data());
        float maxd = 0.0f;
        for (size_t i = 0; i < want.size(); ++i)
            maxd = std::max(maxd, std::fabs(out[i] - (want[i] + shared[i])));
        check(maxd < 1e-5f, "block = routed + shared, against the longhand reference "
                            "(max |d| " + std::to_string(maxd) + ")");
    }

    banner("[6] refusals: a failing fetch must ABORT, not contribute a zero expert");
    {
        const int T = 1, FF = 4, NE_ = 6, K2 = 2;
        std::vector<float> h = rndv((size_t)T * NE_);
        std::vector<float> rl = rndv((size_t)T * NE_);
        Experts e = make_experts(NE_, FF, NE_);
        e.fail_on_which = 1;                                 // "up" fails
        std::vector<float> out((size_t)T * NE_, 0.0f);
        const int rc = aero_moe_ffn_block(h.data(), rl.data(), T, NE_, NE_, K2, FF,
                                         0, 0, nullptr, nullptr, nullptr, nullptr,
                                         fetch_cb, &e, out.data());
        check(rc != 0, "a failing fetch makes the block return non-zero (rc=" +
                       std::to_string(rc) + ")");
        bool all_zero = true;
        for (float v : out) if (v != 0.0f) all_zero = false;
        check(all_zero, "and the output was not partially written (no silent zeros)");
    }

    banner("[7] refusals: k outside [1, n_expert]");
    {
        std::vector<float> lg = rndv(4);
        int64_t idx[8];
        float w[8];
        check(aero_moe_route(lg.data(), 4, 0, 0, idx, w) != 0, "k=0 refused");
        check(aero_moe_route(lg.data(), 4, 5, 0, idx, w) != 0, "k>n_expert refused");
        check(aero_moe_route(lg.data(), 4, 4, 0, idx, w) == 0, "k==n_expert accepted");
    }

    std::printf("\n=== %d checks, %d failures ===\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
