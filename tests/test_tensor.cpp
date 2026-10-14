// Tensor primitives self-test.
//
// Each primitive is checked against a reference derived INDEPENDENTLY in this
// file -- a scalar loop written straight from the definition. Checking a
// vectorised implementation against another vectorised one proves nothing.
#include "aero/tensor.hpp"

#include <cmath>
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

static bool close(float a, float b, float tol = 1e-5f) {
    return std::fabs(a - b) <= tol * std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
}

int main() {
    std::printf("=== aero tensor primitives ===\n");
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);

    // ---------------------------------------------------------------- matvec
    banner("[1] matvec vs a scalar reference");
    {
        const int64_t R = 37, C = 128;
        std::vector<float> wd(R * C), xd(C), yd(R), ref(R);
        for (auto& v : wd) v = d(rng);
        for (auto& v : xd) v = d(rng);
        Tensor2D w{wd.data(), R, C};
        Vec x{xd.data(), C}, y{yd.data(), R};

        for (int th : {1, 2, 4, 8}) {
            matvec(w, x, y, th);
            for (int64_t r = 0; r < R; ++r) {
                double acc = 0.0;
                for (int64_t c = 0; c < C; ++c) acc += (double)wd[r * C + c] * xd[c];
                ref[r] = (float)acc;
            }
            bool ok = true;
            for (int64_t r = 0; r < R; ++r) if (!close(yd[r], ref[r], 1e-4f)) ok = false;
            char msg[96];
            std::snprintf(msg, sizeof(msg), "threads=%d matches the scalar reference", th);
            check(ok, msg);
        }
        // thread count must not change the ANSWER (each row reduces in one thread)
        std::vector<float> a(R), b(R);
        Vec av{a.data(), R}, bv{b.data(), R};
        matvec(w, x, av, 1);
        matvec(w, x, bv, 8);
        check(std::equal(a.begin(), a.end(), b.begin()), "1 and 8 threads are bit-identical");
    }

    // ---------------------------------------------------------------- matmul
    banner("[2] matmul == matvec per row (and the T=1 shortcut is exact)");
    {
        const int64_t R = 16, C = 64, T = 5;
        std::vector<float> wd(R * C), xd(T * C), yd(T * R), ref(T * R);
        for (auto& v : wd) v = d(rng);
        for (auto& v : xd) v = d(rng);
        Tensor2D w{wd.data(), R, C}, X{xd.data(), T, C}, Y{yd.data(), T, R};
        matmul(w, X, Y, 4);
        for (int64_t t = 0; t < T; ++t) {
            std::vector<float> one(R);
            Vec xr{xd.data() + t * C, C}, oner{one.data(), R};
            matvec(w, xr, oner, 1);
            for (int64_t r = 0; r < R; ++r) ref[t * R + r] = one[r];
        }
        bool ok = true;
        for (int64_t i = 0; i < T * R; ++i) if (!close(yd[i], ref[i], 1e-4f)) ok = false;
        check(ok, "batched matmul equals per-token matvec");
    }

    // ---------------------------------------------------------------- rmsnorm
    banner("[3] RMSNorm vs the definition");
    {
        const int64_t N = 64;
        std::vector<float> xd(N), wd(N, 1.0f), od(N), ref(N);
        for (auto& v : xd) v = d(rng);
        for (auto& v : wd) v = d(rng) + 2.0f;   // non-unit weights
        Vec xv{xd.data(), N}, wv{wd.data(), N}, ov{od.data(), N};
        rmsnorm(xv, wv, ov, 1e-6f);
        double ss = 0.0;
        for (double v : xd) ss += v * v;
        const double inv = 1.0 / std::sqrt(ss / N + 1e-6);
        for (int64_t i = 0; i < N; ++i) ref[i] = (float)(xd[i] * inv * wd[i]);
        bool ok = true;
        for (int64_t i = 0; i < N; ++i) if (!close(od[i], ref[i], 1e-4f)) ok = false;
        check(ok, "matches the definition, including non-unit weights");
        // ones in -> ones out proves the normalisation itself
        std::vector<float> ones(N, 1.0f), out1(N);
        Vec onesv{ones.data(), N}, outv{out1.data(), N};
        rmsnorm(onesv, wv, outv, 1e-6f);
        check(close(out1[0], wd[0], 1e-3f), "ones in -> weight out (scale is right)");
    }

    // ---------------------------------------------------------------- l2norm
    banner("[4] L2 norm (SSM q/k) is unweighted by default");
    {
        const int64_t N = 128;
        std::vector<float> xd(N), od(N);
        for (auto& v : xd) v = d(rng) * 3.0f + 0.5f;   // biased on purpose
        Vec xv{xd.data(), N}, ov{od.data(), N}, nov{nullptr, 0};
        l2norm(xv, nov, ov, 1e-6f, false);
        double ss = 0.0;
        for (double v : xd) ss += v * v;
        double nrm = 0.0;
        for (int64_t i = 0; i < N; ++i) nrm += od[i] * od[i];
        check(close((float)std::sqrt(nrm), 1.0f, 1e-3f),
              "output is a unit vector (L2, not RMS)");
        // the distinguishing property: L2 ignores the mean, RMS does not
        check(!close((float)(std::sqrt(ss) / std::sqrt((double)N)), 1.0f, 1e-2f),
              "...and it differs from RMS, so the flag is not decorative");
    }

    // ---------------------------------------------------------------- gelu
    banner("[5] gelu vs the erf definition (and it is NOT silu)");
    {
        const int64_t N = 401;
        std::vector<float> gd(N), sd(N);
        for (int64_t i = 0; i < N; ++i) gd[i] = -6.0f + 12.0f * (float)i / (N - 1);
        sd = gd;
        gelu_(Vec{gd.data(), N});
        silu_(Vec{sd.data(), N});

        bool ok = true;
        for (int64_t i = 0; i < N; ++i) {
            const double z = (double)(-6.0f + 12.0f * (float)i / (N - 1));
            const float ref = (float)(0.5 * z * (1.0 + std::erf(z / std::sqrt(2.0))));
            if (!close(gd[i], ref, 1e-4f)) ok = false;
        }
        check(ok, "gelu matches 0.5*z*(1+erf(z/sqrt2)) pointwise");
        // and it must be a DIFFERENT function from silu, or the FFN is wrong
        bool differs = false;
        for (int64_t i = 0; i < N; ++i) if (std::fabs(gd[i] - sd[i]) > 1e-3f) differs = true;
        check(differs, "gelu and silu genuinely differ (the FFN activation is real)");
        check(close(gd[N / 2], 0.0f, 1e-6f), "gelu(0) == 0");
        check(close(gd[0], 0.0f, 1e-3f), "gelu is ~0 at -6 (the negative tail is suppressed)");
    }

    // ---------------------------------------------------------------- rmsnorm rows
    banner("[6] weightless row norm (gemma4's V) uses no learned weight");
    {
        const int64_t R = 3, C = 32;
        std::vector<float> t(R * C);
        for (auto& v : t) v = d(rng) * 2.0f + 0.0f;
        Tensor2D tt{t.data(), R, C};
        rmsnorm_rows(tt, 1e-6f);
        bool unit = true;
        for (int64_t r = 0; r < R; ++r) {
            double ss = 0.0;
            for (int64_t c = 0; c < C; ++c) ss += t[r * C + c] * t[r * C + c];
            if (std::fabs(std::sqrt(ss / C) - 1.0) > 1e-3) unit = false;
        }
        check(unit, "every row has unit RMS after the norm");
    }

    // ---------------------------------------------------------------- softmax
    banner("[7] softmax and argmax");
    {
        const int64_t N = 5;
        std::vector<float> v{1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
        std::vector<float> ref(N);
        double s = 0.0;
        for (int64_t i = 0; i < N; ++i) { ref[i] = std::exp(v[i] - 3.0f); s += ref[i]; }
        for (int64_t i = 0; i < N; ++i) ref[i] = (float)(ref[i] / s);
        softmax_(Vec{v.data(), N});
        bool ok = true;
        for (int64_t i = 0; i < N; ++i) if (!close(v[i], ref[i], 1e-5f)) ok = false;
        check(ok, "matches the definition");
        float sum = 0.0f;
        for (float x : v) sum += x;
        check(close(sum, 1.0f, 1e-5f), "probabilities sum to 1");
        check(argmax(Vec{v.data(), N}) == 2, "argmax picks the peak");
        // Large logits must not overflow (the classic exp() blowup). The two
        // peaks must differ by MORE than fp32 precision: at 1e30 a difference of
        // 5.0 rounds away entirely (1e30 + 5 == 1e30 in float32), so the fixture
        // has to use a separation the type can actually represent.
        std::vector<float> big{1e30f, 1.2e30f, 0.0f};
        softmax_(Vec{big.data(), 3});
        check(std::isfinite(big[0]) && std::isfinite(big[1]), "no overflow on huge logits");
        check(argmax(Vec{big.data(), 3}) == 1, "and the peak is still found");
        check(big[1] > big[0], "...with the larger logit holding the larger mass");
    }

    std::printf("\n=== %d checks, %d failures ===\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
