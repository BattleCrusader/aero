// Transformer primitives self-test (RoPE, norms, attention).
//
// House rule: every primitive is checked against a reference derived
// INDEPENDENTLY in this file -- a scalar loop written straight from the
// definition, sharing no code with the implementation. Checking a vectorised
// routine against another vectorised routine proves nothing. The cross-language
// check against engine/transformer.py lives in tests/test_transformer_native.py.
//
// Run: make test-cpp
#include "aero/transformer.hpp"
#include "aero/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace aero;

static int g_checks = 0, g_fails = 0;
static void check(bool ok, const char* what) {
    ++g_checks;
    std::printf(ok ? "  ok   %s\n" : "  FAIL %s\n", what);
    if (!ok) ++g_fails;
}
static void banner(const char* s) { std::printf("\n%s\n", s); }

// relative closeness, matching the tolerance philosophy of the other suites:
// differences are fp32 accumulation order, not algorithmic error.
static double maxabs(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        d = std::max(d, (double)std::fabs(a[i] - b[i]));
    return d;
}

// ---------------------------------------------------------------- references
// Scalar RoPE tables: inv[p] = 1/base^(2p/n), divided by ff[p] when present.
static void ref_rope_tables(int64_t head_dim, const int32_t* pos, int64_t n_pos,
                            double base, int64_t n_dims, const float* ff,
                            int64_t half, std::vector<float>& cos,
                            std::vector<float>& sin) {
    int64_t n = (n_dims <= 0) ? head_dim : std::min<int64_t>(n_dims, head_dim);
    n -= n % 2;
    for (int64_t p = 0; p < half; ++p) {
        double inv = 1.0 / std::pow(base, 2.0 * (double)p / (double)n);
        if (ff) inv /= (double)ff[p];
        for (int64_t t = 0; t < n_pos; ++t) {
            const double ang = (double)pos[t] * inv;
            cos[(size_t)(t * half + p)] = (float)std::cos(ang);
            sin[(size_t)(t * half + p)] = (float)std::sin(ang);
        }
    }
}

// Scalar attention by the batch (materialise-all-scores) definition. Uses a
// different algorithm from the online-softmax implementation precisely so it is
// an independent check, not a paraphrase.
static void ref_attention(const std::vector<float>& q, const std::vector<float>& k,
                          const std::vector<float>& v, int64_t n_q,
                          int64_t H, int64_t HK, int64_t D, int64_t DV,
                          float scale, const int32_t* qpos, const int32_t* kpos,
                          int64_t S, int window, std::vector<float>& out) {
    const int64_t rep = H / HK;
    out.assign((size_t)(n_q * H * DV), 0.0f);
    for (int64_t t = 0; t < n_q; ++t) {
        for (int64_t h = 0; h < H; ++h) {
            const int64_t kh = h / rep;
            std::vector<double> sc((size_t)S);
            double m = -1e300;
            for (int64_t j = 0; j < S; ++j) {
                const int64_t kp = kpos ? kpos[j] : j;
                const int64_t qp = qpos ? qpos[t] : t;
                bool keep = (kp <= qp);
                if (window > 0) keep = keep && ((qp - kp) < window);
                double a = 0.0;
                if (keep) {
                    for (int64_t d = 0; d < D; ++d)
                        a += (double)q[(size_t)((t * H + h) * D + d)] *
                             (double)k[(size_t)((j * HK + kh) * D + d)];
                    a *= (double)scale;
                } else {
                    a = -1e300;
                }
                sc[(size_t)j] = a;
                m = std::max(m, a);
            }
            double s = 0.0;
            for (int64_t j = 0; j < S; ++j)
                s += (sc[(size_t)j] > -1e299) ? std::exp(sc[(size_t)j] - m) : 0.0;
            for (int64_t d = 0; d < DV; ++d) {
                double acc = 0.0;
                for (int64_t j = 0; j < S; ++j) {
                    const double p = (sc[(size_t)j] > -1e299)
                                         ? std::exp(sc[(size_t)j] - m) : 0.0;
                    acc += p * (double)v[(size_t)((j * HK + kh) * DV + d)];
                }
                out[(size_t)((t * H + h) * DV + d)] = (float)(acc / s);
            }
        }
    }
}

int main() {
    std::printf("=== aero transformer primitives ===\n");
    std::mt19937 rng(20261015);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);

    // ------------------------------------------------------------ rope_tables
    banner("[1] rope_tables vs a scalar reference (full, partial, proportional)");
    {
        const int64_t H = 64;
        std::vector<int32_t> pos(5);
        for (int i = 0; i < 5; ++i) pos[i] = i * 3;

        struct Case { int64_t n_dims; const char* name; };
        for (const Case c : {Case{0, "full rotary"}, Case{16, "partial (16 of 64)"}}) {
            int64_t n = (c.n_dims <= 0) ? H : c.n_dims;
            n -= n % 2;
            const int64_t half = n / 2;
            std::vector<float> cos((size_t)(5 * half)), sin((size_t)(5 * half));
            const int64_t got = rope_tables(H, pos.data(), 5, 10000.0f, c.n_dims,
                                            nullptr, 0, cos.data(), sin.data());
            std::vector<float> rcos((size_t)(5 * half)), rsin((size_t)(5 * half));
            ref_rope_tables(H, pos.data(), 5, 10000.0, c.n_dims, nullptr, half,
                            rcos, rsin);
            const double dc = maxabs(cos, rcos), ds = maxabs(sin, rsin);
            char msg[128];
            std::snprintf(msg, sizeof(msg),
                          "%s: n_rot=%lld, cos/sin max diff %.2e / %.2e",
                          c.name, (long long)got, dc, ds);
            check(got == n && dc < 1e-6 && ds < 1e-6, msg);
        }

        // proportional RoPE: a huge factor must leave its pairs UNROTATED
        const int64_t half = H / 2;
        std::vector<float> ff((size_t)half, 1.0f);
        for (int64_t p = 8; p < half; ++p) ff[(size_t)p] = 1e30f;
        std::vector<float> cos((size_t)(5 * half)), sin((size_t)(5 * half));
        rope_tables(H, pos.data(), 5, 10000.0f, 0, ff.data(), half, cos.data(),
                    sin.data());
        bool unrot = true;
        for (int64_t t = 0; t < 5; ++t)
            for (int64_t p = 8; p < half; ++p) {
                if (std::fabs(cos[(size_t)(t * half + p)] - 1.0f) > 1e-5f) unrot = false;
                if (std::fabs(sin[(size_t)(t * half + p)]) > 1e-5f) unrot = false;
            }
        check(unrot, "freq_factor 1e30 drives cos->1, sin->0 (pair left unrotated)");
        bool firstrot = false;
        for (int64_t t = 1; t < 5; ++t)
            if (std::fabs(cos[(size_t)(t * half + 0)] - 1.0f) > 1e-6f) firstrot = true;
        check(firstrot, "...while factor-1 pairs still rotate (the table is not a no-op)");

        check(rope_tables(H, pos.data(), 5, 10000.0f, 0, ff.data(), half - 1,
                          cos.data(), sin.data()) == -1,
              "too-short freq_factors is refused (-1), not silently truncated");
    }

    // ------------------------------------------------------------ apply_rope
    banner("[2] apply_rope: neox and interleaved, and the tail passes through");
    {
        const int64_t rows = 6, Hd = 32, n = 32, half = n / 2;
        std::vector<int32_t> pos((size_t)rows);
        for (int64_t r = 0; r < rows; ++r) pos[(size_t)r] = (int32_t)(r * 2);
        const int64_t maxpos = pos[rows - 1];
        std::vector<float> cos((size_t)((maxpos + 1) * half)),
            sin((size_t)((maxpos + 1) * half));
        rope_tables(Hd, pos.data(), rows, 10000.0f, 0, nullptr, 0,
                    cos.data(), sin.data());  // note: table only for len(rows)
        // rebuild a full table for all positions we index
        std::vector<int32_t> allpos((size_t)(maxpos + 1));
        for (int64_t i = 0; i <= maxpos; ++i) allpos[(size_t)i] = (int32_t)i;
        rope_tables(Hd, allpos.data(), maxpos + 1, 10000.0f, 0, nullptr, 0,
                    cos.data(), sin.data());

        std::vector<float> x((size_t)(rows * Hd));
        for (auto& v : x) v = d(rng);

        for (int mode = 0; mode < 2; ++mode) {
            std::vector<float> got = x;
            apply_rope(got.data(), rows, Hd, n, cos.data(), sin.data(), pos.data(),
                       mode == 0 ? RopeMode::Neox : RopeMode::Interleaved);
            // scalar reference
            std::vector<float> ref = x;
            for (int64_t r = 0; r < rows; ++r) {
                const int32_t p = pos[(size_t)r];
                for (int64_t k2 = 0; k2 < half; ++k2) {
                    const float c = cos[(size_t)(p * half + k2)];
                    const float s = sin[(size_t)(p * half + k2)];
                    float x0, x1;
                    int64_t i0, i1;
                    if (mode == 0) { i0 = k2; i1 = k2 + half; }
                    else           { i0 = 2 * k2; i1 = 2 * k2 + 1; }
                    x0 = x[(size_t)(r * Hd + i0)];
                    x1 = x[(size_t)(r * Hd + i1)];
                    ref[(size_t)(r * Hd + i0)] = x0 * c - x1 * s;
                    ref[(size_t)(r * Hd + i1)] = x0 * s + x1 * c;
                }
            }
            const double diff = maxabs(got, ref);
            char msg[128];
            std::snprintf(msg, sizeof(msg),
                          "%s matches the scalar reference (max diff %.2e)",
                          mode == 0 ? "neox" : "interleaved", diff);
            check(diff < 1e-5, msg);
        }

        // partial rotary: rotate only 8 of 32 channels, tail untouched
        std::vector<float> got = x;
        apply_rope(got.data(), rows, Hd, 8, cos.data(), sin.data(), pos.data(),
                   RopeMode::Neox);
        bool tail_same = true;
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t i = 8; i < Hd; ++i)
                if (got[(size_t)(r * Hd + i)] != x[(size_t)(r * Hd + i)]) tail_same = false;
        bool head_changed = false;
        for (int64_t r = 0; r < rows; ++r)
            for (int64_t i = 0; i < 8; ++i)
                if (got[(size_t)(r * Hd + i)] != x[(size_t)(r * Hd + i)]) head_changed = true;
        check(tail_same && head_changed,
              "n_rot=8 rotates the first 8 channels and leaves channels 8..31 alone");
    }

    // ----------------------------------------------- relative-position property
    banner("[3] RoPE: q.k depends only on the RELATIVE position (the defining property)");
    {
        const int64_t Hd = 16, half = Hd / 2;
        std::vector<int32_t> allpos(32);
        for (int i = 0; i < 32; ++i) allpos[(size_t)i] = i;
        std::vector<float> cos((size_t)(32 * half)), sin((size_t)(32 * half));
        rope_tables(Hd, allpos.data(), 32, 10000.0f, 0, nullptr, 0,
                    cos.data(), sin.data());

        std::vector<float> q0((size_t)Hd), k0((size_t)Hd);
        for (auto& z : q0) z = d(rng);
        for (auto& z : k0) z = d(rng);

        auto dot_at = [&](int64_t qp, int64_t kp) {
            std::vector<float> q = q0, k = k0;
            int32_t p1 = (int32_t)qp, p2 = (int32_t)kp;
            apply_rope(q.data(), 1, Hd, Hd, cos.data(), sin.data(), &p1, RopeMode::Neox);
            apply_rope(k.data(), 1, Hd, Hd, cos.data(), sin.data(), &p2, RopeMode::Neox);
            double a = 0.0;
            for (int64_t i = 0; i < Hd; ++i) a += (double)q[(size_t)i] * k[(size_t)i];
            return a;
        };
        const double base = dot_at(0, 5);
        double worst = 0.0;
        for (int s = 1; s <= 20; ++s) worst = std::max(worst, std::fabs(dot_at(s, s + 5) - base));
        char msg[128];
        std::snprintf(msg, sizeof(msg),
                      "score at offset +5 is constant across 20 absolute shifts (max dev %.2e)",
                      worst);
        check(worst < 1e-4, msg);
    }

    // ------------------------------------------------------------ norms
    banner("[4] rmsnorm_rows / l2norm_rows vs the definition (and they differ)");
    {
        const int64_t R = 4, C = 48;
        std::vector<float> x((size_t)(R * C)), w((size_t)C);
        for (auto& v : x) v = d(rng) * 2.0f;
        for (auto& v : w) v = d(rng) + 2.0f;
        std::vector<float> rms((size_t)(R * C)), l2((size_t)(R * C));
        rmsnorm_rows(x.data(), w.data(), rms.data(), R, C, 1e-6f);
        l2norm_rows(x.data(), w.data(), l2.data(), R, C, 1e-6f);

        std::vector<float> rref((size_t)(R * C)), lref((size_t)(R * C));
        for (int64_t r = 0; r < R; ++r) {
            double ss = 0.0;
            for (int64_t c = 0; c < C; ++c) ss += (double)x[(size_t)(r * C + c)] * x[(size_t)(r * C + c)];
            for (int64_t c = 0; c < C; ++c) {
                const double xv = x[(size_t)(r * C + c)];
                rref[(size_t)(r * C + c)] = (float)(xv / std::sqrt(ss / C + 1e-6) * w[(size_t)c]);
                lref[(size_t)(r * C + c)] = (float)(xv / std::sqrt(ss + 1e-6) * w[(size_t)c]);
            }
        }
        char msg[128];
        std::snprintf(msg, sizeof(msg), "rmsnorm matches mean(x^2) definition (max diff %.2e)",
                      maxabs(rms, rref));
        check(maxabs(rms, rref) < 1e-4, msg);
        std::snprintf(msg, sizeof(msg), "l2norm matches sum(x^2) definition (max diff %.2e)",
                      maxabs(l2, lref));
        check(maxabs(l2, lref) < 1e-4, msg);
        check(maxabs(rms, l2) > 1e-2, "...and the two genuinely differ (sqrt(C) apart)");

        // weight=NULL must behave as a unit weight, not read through null
        std::vector<float> now((size_t)(R * C));
        rmsnorm_rows(x.data(), nullptr, now.data(), R, C, 1e-6f);
        std::vector<float> unitw((size_t)C, 1.0f), ref1((size_t)(R * C));
        rmsnorm_rows(x.data(), unitw.data(), ref1.data(), R, C, 1e-6f);
        check(maxabs(now, ref1) == 0.0, "weight=NULL is bit-identical to an all-ones weight");
    }

    // ------------------------------------------------------------ activations
    banner("[5] gelu / silu / swiglu / softmax over flat buffers");
    {
        const int64_t N = 257;
        std::vector<float> g((size_t)N), s((size_t)N);
        for (int64_t i = 0; i < N; ++i) g[(size_t)i] = -6.0f + 12.0f * (float)i / (float)(N - 1);
        s = g;
        gelu_(Vec{g.data(), N});
        silu_(Vec{s.data(), N});
        bool ok = true;
        for (int64_t i = 0; i < N; ++i) {
            const double z = -6.0 + 12.0 * (double)i / (double)(N - 1);
            const float ref = (float)(0.5 * z * (1.0 + std::erf(z / std::sqrt(2.0))));
            if (std::fabs(g[(size_t)i] - ref) > 2e-4f) ok = false;
        }
        check(ok, "gelu matches 0.5*z*(1+erf(z/sqrt2)) pointwise");

        std::vector<float> gate((size_t)N), up((size_t)N);
        for (auto& v : gate) v = d(rng) * 3.0f;
        for (auto& v : up) v = d(rng) * 3.0f;
        std::vector<float> ref((size_t)N);
        for (int64_t i = 0; i < N; ++i) {
            const double z = gate[(size_t)i];
            ref[(size_t)i] = (float)((z / (1.0 + std::exp(-z))) * (double)up[(size_t)i]);
        }
        swiglu_(Vec{gate.data(), N}, Vec{up.data(), N});
        char msg[128];
        std::snprintf(msg, sizeof(msg), "swiglu == silu(gate)*up (max diff %.2e)",
                      maxabs(gate, ref));
        check(maxabs(gate, ref) < 1e-5, msg);

        // softmax_rows: each row sums to 1, and huge logits do not overflow
        const int64_t R = 3, C = 8;
        std::vector<float> m((size_t)(R * C));
        for (auto& v : m) v = d(rng) * 5.0f;
        std::vector<float> mref((size_t)(R * C)), mcopy = m;
        for (int64_t r = 0; r < R; ++r) {
            float mx = mcopy[(size_t)(r * C)];
            for (int64_t c = 1; c < C; ++c) mx = std::max(mx, mcopy[(size_t)(r * C + c)]);
            double sum = 0.0;
            for (int64_t c = 0; c < C; ++c) sum += std::exp(mcopy[(size_t)(r * C + c)] - mx);
            for (int64_t c = 0; c < C; ++c)
                mref[(size_t)(r * C + c)] = (float)(std::exp(mcopy[(size_t)(r * C + c)] - mx) / sum);
        }
        softmax_rows(m.data(), R, C);
        std::snprintf(msg, sizeof(msg), "softmax_rows matches, per row (max diff %.2e)",
                      maxabs(m, mref));
        check(maxabs(m, mref) < 1e-5, msg);
        bool sums = true;
        for (int64_t r = 0; r < R; ++r) {
            float t = 0.0f;
            for (int64_t c = 0; c < C; ++c) t += m[(size_t)(r * C + c)];
            if (std::fabs(t - 1.0f) > 1e-5f) sums = false;
        }
        check(sums, "each row sums to 1");
    }

    // ------------------------------------------------------------ attention
    banner("[6] attention_tiled vs a batch (materialise-all-scores) reference");
    {
        const int64_t n_q = 5, H = 4, HK = 2, D = 8, DV = 8, S = 13;
        std::vector<float> q((size_t)(n_q * H * D)), k((size_t)(S * HK * D)),
            v((size_t)(S * HK * DV));
        for (auto& z : q) z = d(rng);
        for (auto& z : k) z = d(rng);
        for (auto& z : v) z = d(rng);
        const float scale = 1.0f / std::sqrt((float)D);
        std::vector<int32_t> qpos((size_t)n_q), kpos((size_t)S);
        for (int64_t t = 0; t < n_q; ++t) qpos[(size_t)t] = (int32_t)t;
        for (int64_t j = 0; j < S; ++j) kpos[(size_t)j] = (int32_t)j;

        std::vector<float> ref;
        ref_attention(q, k, v, n_q, H, HK, D, DV, scale, qpos.data(), kpos.data(), S,
                      0, ref);

        for (int64_t block : {(int64_t)S, (int64_t)1, (int64_t)4}) {   // one block, and streamed
            std::vector<float> got((size_t)(n_q * H * DV));
            const bool ok = attention_tiled(q.data(), k.data(), v.data(), got.data(),
                                            n_q, H, HK, D, DV, scale, qpos.data(),
                                            kpos.data(), S, 0, block);
            char msg[128];
            std::snprintf(msg, sizeof(msg),
                          "block=%lld matches the batch softmax (max diff %.2e)",
                          (long long)block, maxabs(got, ref));
            check(ok && maxabs(got, ref) < 1e-4, msg);
        }
    }

    banner("[7] causality, sliding window, and GQA grouping");
    {
        const int64_t n_q = 4, H = 2, HK = 2, D = 8, DV = 8, S = 9;
        std::vector<float> q((size_t)(n_q * H * D)), k((size_t)(S * HK * D)),
            v((size_t)(S * HK * DV));
        for (auto& z : q) z = d(rng);
        for (auto& z : k) z = d(rng);
        for (auto& z : v) z = d(rng);
        const float scale = 1.0f / std::sqrt((float)D);
        std::vector<int32_t> qpos((size_t)n_q), kpos((size_t)S);
        for (int64_t t = 0; t < n_q; ++t) qpos[(size_t)t] = (int32_t)t;
        for (int64_t j = 0; j < S; ++j) kpos[(size_t)j] = (int32_t)j;

        std::vector<float> a((size_t)(n_q * H * DV)), b((size_t)(n_q * H * DV));
        attention_tiled(q.data(), k.data(), v.data(), a.data(), n_q, H, HK, D, DV,
                        scale, qpos.data(), kpos.data(), S, 0, S);
        // Perturb key at position 3 (both its kv heads). Query t=3 sees it and
        // MUST move; queries t<3 must not, and must be BIT-identical -- a causal
        // mask that merely shrinks the change is a bug, not a mask.
        std::vector<float> k2 = k;
        for (int64_t j = 0; j < HK; ++j)
            for (int64_t d = 0; d < D; ++d) k2[(size_t)((3 * HK + j) * D + d)] += 7.0f;
        attention_tiled(q.data(), k2.data(), v.data(), b.data(), n_q, H, HK, D, DV,
                        scale, qpos.data(), kpos.data(), S, 0, S);
        bool earlier_same = true, later_moved = false;
        for (int64_t t = 0; t < n_q; ++t)
            for (int64_t h = 0; h < H; ++h)
                for (int64_t dd = 0; dd < DV; ++dd) {
                    const float av = a[(size_t)((t * H + h) * DV + dd)];
                    const float bv = b[(size_t)((t * H + h) * DV + dd)];
                    if (t < 3 && av != bv) earlier_same = false;
                    if (t >= 3 && std::fabs(av - bv) > 1e-4f) later_moved = true;
                }
        check(earlier_same, "changing key 3 leaves queries 0..2 BIT-identical (causal)");
        check(later_moved, "...while query 3 does see it (the mask is causal, not blind)");

        // sliding window of 3: query t sees only keys t-2..t. Compare t=3 against a
        // hand-built reference that keeps just those keys.
        std::vector<float> wref;
        ref_attention(q, k, v, n_q, H, HK, D, DV, scale, qpos.data(), kpos.data(), S,
                      3, wref);
        std::vector<float> wgot((size_t)(n_q * H * DV));
        attention_tiled(q.data(), k.data(), v.data(), wgot.data(), n_q, H, HK, D, DV,
                        scale, qpos.data(), kpos.data(), S, 3, S);
        char msg[128];
        std::snprintf(msg, sizeof(msg), "window=3 matches the masked reference (max diff %.2e)",
                      maxabs(wgot, wref));
        check(maxabs(wgot, wref) < 1e-4, msg);
        // and the window actually bites: it must differ from the unwindowed run
        check(maxabs(wgot, a) > 1e-3, "...and differs from no-window (the mask is live)");

        // GQA: with H=4, HK=2, heads 0/1 must use kv head 0, heads 2/3 kv head 1.
        const int64_t H2 = 4, HK2 = 2;
        std::vector<float> q2((size_t)(1 * H2 * D)), k3((size_t)(S * HK2 * D)),
            v3((size_t)(S * HK2 * DV));
        for (auto& z : q2) z = d(rng);
        for (auto& z : k3) z = d(rng);
        for (auto& z : v3) z = d(rng);
        std::vector<float> o2((size_t)(1 * H2 * DV));
        attention_tiled(q2.data(), k3.data(), v3.data(), o2.data(), 1, H2, HK2, D, DV,
                        scale, nullptr, nullptr, S, 0, S);
        // heads 0 and 1 share kv head 0, so their outputs must be equal; 0 and 2 must not
        auto col = [&](int64_t h) {
            std::vector<float> c((size_t)DV);
            for (int64_t dd = 0; dd < DV; ++dd) c[(size_t)dd] = o2[(size_t)(h * DV + dd)];
            return c;
        };
        check(maxabs(col(0), col(1)) == 0.0, "query heads 0,1 share kv head 0 (identical output)");
        check(maxabs(col(2), col(3)) == 0.0, "query heads 2,3 share kv head 1");
        check(maxabs(col(0), col(2)) > 1e-3, "...and the two groups differ (grouping is real)");
    }

    banner("[8] single-token decode path");
    {
        const int64_t H = 4, HK = 2, D = 8, DV = 8, S = 7;
        std::vector<float> q((size_t)(H * D)), k((size_t)(S * HK * D)),
            v((size_t)(S * HK * DV));
        for (auto& z : q) z = d(rng);
        for (auto& z : k) z = d(rng);
        for (auto& z : v) z = d(rng);
        const float scale = 1.0f / std::sqrt((float)D);

        std::vector<float> got((size_t)(H * DV));
        const bool ok = attention(q.data(), k.data(), v.data(), got.data(), H, HK, D,
                                  DV, scale, S, 0);
        // reference: the full attention_tiled with one query at position S-1
        std::vector<int32_t> kpos((size_t)S);
        for (int64_t j = 0; j < S; ++j) kpos[(size_t)j] = (int32_t)j;
        std::vector<int32_t> qp{(int32_t)(S - 1)};
        std::vector<float> ref((size_t)(H * DV));
        attention_tiled(q.data(), k.data(), v.data(), ref.data(), 1, H, HK, D, DV,
                        scale, qp.data(), kpos.data(), S, 0, S);
        char msg[128];
        std::snprintf(msg, sizeof(msg), "decode == tiled(1 query at S-1) (max diff %.2e)",
                      maxabs(got, ref));
        check(ok && maxabs(got, ref) == 0.0, msg);
    }

    banner("[9] it refuses what it cannot do");
    {
        std::vector<float> q(16, 0.0f), k(16, 0.0f), v(16, 0.0f), o(16, 0.0f);
        check(!attention_tiled(q.data(), k.data(), v.data(), o.data(), 1, 3, 2, 4, 4,
                               1.0f, nullptr, nullptr, 4, 0, 4),
              "n_head not a multiple of n_head_kv returns false");
        check(attention_tiled(q.data(), k.data(), v.data(), o.data(), 1, 2, 2, 4, 4,
                              1.0f, nullptr, nullptr, 4, 0, 4),
              "a well-formed shape is accepted");
        check(!attention_tiled(q.data(), k.data(), v.data(), o.data(), 1, 2, 2, 4, 4,
                               1.0f, nullptr, nullptr, 0, 0, 0),
              "n_kv=0 returns false rather than dividing by an empty cache");
    }

    std::printf("\n=== %d checks, %d failures ===\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
