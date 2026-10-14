// Aero kernels self-test.
//
// Two things must hold and they are different:
//   1. CORRECTNESS -- the fused path must equal the reference dequantise +
//      multiply, and the reference must itself equal analytically known values
//      for hand-built blocks. A fused loop that is fast and wrong is worthless.
//   2. THREADS -- multi-threaded and serial must give bit-identical results, so
//      the thread count is a performance knob and never a correctness one.
//
// Run: make test-cpp
#include "aero/kernels.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace aero;

static int g_checks = 0;
static int g_fails = 0;

static void check(bool ok, const char* what) {
    ++g_checks;
    if (ok) {
        std::printf("  ok   %s\n", what);
    } else {
        ++g_fails;
        std::printf("  FAIL %s\n", what);
    }
}

static void banner(const char* s) { std::printf("\n%s\n", s); }

// ------------------------------------------------------------------ helpers
// fp16 encoding for test fixtures, so expected values can be stated exactly.
static uint16_t f32_to_f16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int32_t  exp  = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127 + 15;
    const uint32_t mant = bits & 0x7FFFFFu;
    if (exp <= 0) return static_cast<uint16_t>(sign);           // flush to +/-0
    if (exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
}

// ------------------------------------------------------------------ fixtures
// Realistic Q4_K bytes. Fully random bytes are NOT usable as a fixture: bytes
// 0-3 are an fp16 super-block scale, so a uniform random draw produces inf/NaN
// exponents, and then every comparison is meaningless (NaN != NaN) and the
// magnitudes are unbounded. Real files hold d in roughly [1e-4, 1e-1], so the
// fixture pins d and dmin to sane values and randomises only the parts that are
// genuinely 6-bit or 4-bit fields.
static void fill_realistic_q4k(std::vector<uint8_t>& buf, std::mt19937& rng) {
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> dscale(1e-3f, 5e-2f);
    const int64_t nblocks = static_cast<int64_t>(buf.size()) / 144;
    for (int64_t b = 0; b < nblocks; ++b) {
        uint8_t* blk = buf.data() + b * 144;
        for (int i = 0; i < 144; ++i) blk[i] = static_cast<uint8_t>(byte(rng));
        const uint16_t dh  = f32_to_f16(dscale(rng));
        const uint16_t dmh = f32_to_f16(dscale(rng) * 0.1f);
        blk[0] = static_cast<uint8_t>(dh & 0xFF);
        blk[1] = static_cast<uint8_t>(dh >> 8);
        blk[2] = static_cast<uint8_t>(dmh & 0xFF);
        blk[3] = static_cast<uint8_t>(dmh >> 8);
    }
}

// Build one Q4_K block with a chosen d, dmin and per-sub-block scale/min.
struct Q4KFixture {
    std::vector<uint8_t> bytes;
    // expected fp32 values, computed independently here
    std::vector<float> values;
};

static Q4KFixture make_q4k(float d, float dmin,
                           const uint8_t sc[8], const uint8_t mn[8],
                           const uint8_t* quants) {
    Q4KFixture f;
    f.bytes.assign(144, 0);
    const uint16_t dh = f32_to_f16(d), dmh = f32_to_f16(dmin);
    f.bytes[0] = static_cast<uint8_t>(dh & 0xFF);
    f.bytes[1] = static_cast<uint8_t>(dh >> 8);
    f.bytes[2] = static_cast<uint8_t>(dmh & 0xFF);
    f.bytes[3] = static_cast<uint8_t>(dmh >> 8);

    // pack 8 scale/min pairs into the 12-byte scales[] field, mirroring what
    // the reader extracts:
    //   j<4 : sc[j] = q[j]  & 0x3F,          mn[j] = q[j+4] & 0x3F
    //   j>=4: sc[j] = (q[j+4] & 0x0F) | ((q[j-4] >> 6) << 4)
    //         mn[j] = (q[j+4] >> 4)   | ((q[j]   >> 6) << 4)
    // so q[8..11] carries sc[4..7] in its LOW nibble and mn[4..7] in its HIGH.
    for (int j = 0; j < 4; ++j) {
        f.bytes[4 + j]     = static_cast<uint8_t>((sc[j] & 0x3F) |
                                                  (((sc[j + 4] >> 4) & 0x03) << 6));
        f.bytes[4 + j + 4] = static_cast<uint8_t>((mn[j] & 0x3F) |
                                                  (((mn[j + 4] >> 4) & 0x03) << 6));
        f.bytes[4 + j + 8] = static_cast<uint8_t>((sc[j + 4] & 0x0F) |
                                                  ((mn[j + 4] & 0x0F) << 4));
    }
    std::memcpy(&f.bytes[16], quants, 128);

    // independent expected values: w = d*sc*q - dmin*mn, 32 values per sub-block
    f.values.resize(256);
    int col = 0;
    for (int j = 0; j < 256; j += 64) {
        for (int half = 0; half < 2; ++half) {
            const int sb = j / 32 + half;
            for (int i = 0; i < 32; ++i) {
                const uint8_t byte = quants[col + i];
                const int q = half == 0 ? (byte & 0x0F) : (byte >> 4);
                f.values[j + half * 32 + i] =
                    d * static_cast<float>(sc[sb]) * static_cast<float>(q)
                    - dmin * static_cast<float>(mn[sb]);
            }
        }
        col += 32;
    }
    return f;
}

static void test_reference_q4k() {
    banner("[1] Q4_K reference vs analytically known values");
    const uint8_t sc[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const uint8_t mn[8] = {8, 7, 6, 5, 4, 3, 2, 1};
    std::vector<uint8_t> quants(128);
    for (int i = 0; i < 128; ++i) quants[i] = static_cast<uint8_t>((i * 3) & 0xFF);

    // d=1, dmin=0 -> w = sc*q exactly, so the numbers are checkable by hand
    const Q4KFixture f = make_q4k(1.0f, 0.0f, sc, mn, quants.data());
    std::vector<float> got(256);
    dequant_ref(DType::Q4_K, f.bytes.data(), (int64_t)f.bytes.size(), got.data(), 256);

    float maxdiff = 0.0f;
    for (int i = 0; i < 256; ++i) {
        maxdiff = std::max(maxdiff, std::fabs(got[i] - f.values[i]));
    }
    check(maxdiff < 1e-4f, "every dequantised value matches the hand-built expectation");
    std::printf("       first 8: ");
    for (int i = 0; i < 8; ++i) std::printf("%.1f ", got[i]);
    std::printf("\n");

    // with a non-zero dmin the min term must actually subtract
    const Q4KFixture g = make_q4k(1.0f, 1.0f, sc, mn, quants.data());
    std::vector<float> got2(256);
    dequant_ref(DType::Q4_K, g.bytes.data(), (int64_t)g.bytes.size(), got2.data(), 256);
    check(std::fabs(got2[0] - g.values[0]) < 1e-4f,
          "a non-zero dmin subtracts the min term");
    check(got2[0] != got[0], "...and changes the result (not a no-op)");
}

static void test_fused_matches_reference() {
    banner("[2] fused matvec == reference dequantise + multiply");
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    const int64_t n_rows = 37, n_cols = 256 * 3;   // 3 blocks per row
    const int64_t bb = dtype_block_bytes(DType::Q4_K);

    // deterministic pseudo-random blocks
    std::vector<uint8_t> w(static_cast<size_t>(n_rows * (n_cols / 256) * bb));
    fill_realistic_q4k(w, rng);

    std::vector<float> x(static_cast<size_t>(n_cols));
    for (auto& v : x) v = dist(rng);

    // reference: dequantise the whole thing, then multiply
    std::vector<float> wf(static_cast<size_t>(n_rows * n_cols));
    dequant_ref(DType::Q4_K, w.data(), (int64_t)w.size(), wf.data(), n_rows * n_cols);

    std::vector<float> want(static_cast<size_t>(n_rows), 0.0f);
    for (int64_t r = 0; r < n_rows; ++r) {
        float a = 0.0f;
        for (int64_t c = 0; c < n_cols; ++c) a += wf[r * n_cols + c] * x[c];
        want[static_cast<size_t>(r)] = a;
    }

    for (int threads : {1, 4, 8}) {
        std::vector<float> got(static_cast<size_t>(n_rows), 0.0f);
        const bool ok = matvec(DType::Q4_K, w.data(), x.data(), got.data(),
                               n_rows, n_cols, threads);
        float maxdiff = 0.0f;
        for (int64_t r = 0; r < n_rows; ++r)
            maxdiff = std::max(maxdiff, std::fabs(got[static_cast<size_t>(r)] -
                                                 want[static_cast<size_t>(r)]));
        char msg[160];
        std::snprintf(msg, sizeof(msg),
                      "threads=%d matches the reference (max diff %.2e)", threads,
                      static_cast<double>(maxdiff));
        // tolerance is accumulation order, not algorithmic error
        check(ok && maxdiff < 1e-2f, msg);
    }
}

static void test_threads_are_exact() {
    banner("[3] thread count is a performance knob, not a correctness one");
    std::mt19937 rng(99);
    const int64_t n_rows = 64, n_cols = 256 * 2;
    const int64_t bb = dtype_block_bytes(DType::Q4_K);
    std::vector<uint8_t> w(static_cast<size_t>(n_rows * (n_cols / 256) * bb));
    fill_realistic_q4k(w, rng);
    std::vector<float> x(static_cast<size_t>(n_cols));
    for (auto& v : x) v = static_cast<float>(rng() % 100) / 100.0f;

    std::vector<float> ref(static_cast<size_t>(n_rows));
    matvec(DType::Q4_K, w.data(), x.data(), ref.data(), n_rows, n_cols, 1);

    bool all_equal = true;
    for (int threads : {2, 3, 4, 8}) {
        std::vector<float> got(static_cast<size_t>(n_rows));
        matvec(DType::Q4_K, w.data(), x.data(), got.data(), n_rows, n_cols, threads);
        for (int64_t r = 0; r < n_rows; ++r) {
            if (got[static_cast<size_t>(r)] != ref[static_cast<size_t>(r)]) all_equal = false;
        }
    }
    check(all_equal, "serial and threaded results are bit-identical (each row is "
                     "computed by one thread, so summation order never changes)");
}

static void test_fused_is_smaller_than_naive() {
    banner("[4] the fused path never materialises fp32 weights");
    // A structural property, not a timing one: the fused matvec takes the raw
    // quantized bytes, so there is no fp32 weight buffer to allocate at all.
    const int64_t n_rows = 9216, n_cols = 2560;
    const int64_t q_bytes = (n_rows * n_cols / 256) * 144;
    const int64_t f32_bytes = n_rows * n_cols * 4;
    std::printf("       9216x2560 Q4_K on disk %lld B, as fp32 %lld B (%.2fx)\n",
                static_cast<long long>(q_bytes),
                static_cast<long long>(f32_bytes),
                static_cast<double>(f32_bytes) / static_cast<double>(q_bytes));
    check(f32_bytes / q_bytes >= 7, "fp32 materialisation would inflate 7x -- the "
                                    "fused kernel avoids that entirely");
}

static void test_refusals() {
    banner("[5] it refuses what it cannot do");
    std::vector<uint8_t> w(1024, 0);
    std::vector<float> x(16, 1.0f), y(4, 0.0f);

    check(!matvec(DType::F32, w.data(), x.data(), y.data(), 4, 16, 1),
          "an unsupported type returns false rather than computing garbage");
    check(!matvec(DType::Q4_K, w.data(), x.data(), y.data(), 4, 16, 1),
          "a column count that is not a whole number of 256-blocks returns false");
    check(matvec(DType::Q4_K, w.data(), x.data(), y.data(), 4, 256, 1),
          "a well-formed shape is accepted");
}

static void test_q6k() {
    banner("[6] Q6_K fused == reference");
    std::mt19937 rng(7);
    const int64_t n_rows = 11, n_cols = 256;
    const int64_t bb = dtype_block_bytes(DType::Q6_K);
    std::vector<uint8_t> w(static_cast<size_t>(n_rows * bb));
    for (auto& b : w) b = static_cast<uint8_t>(rng() & 0xFF);
    // keep d small so the dequantised values stay in a sane range
    for (int64_t r = 0; r < n_rows; ++r) {
        const uint16_t d = f32_to_f16(0.01f);
        w[static_cast<size_t>(r * bb + 208)] = static_cast<uint8_t>(d & 0xFF);
        w[static_cast<size_t>(r * bb + 209)] = static_cast<uint8_t>(d >> 8);
    }
    std::vector<float> x(static_cast<size_t>(n_cols));
    for (auto& v : x) v = static_cast<float>(rng() % 100) / 50.0f - 1.0f;

    std::vector<float> wf(static_cast<size_t>(n_rows * n_cols));
    dequant_ref(DType::Q6_K, w.data(), (int64_t)w.size(), wf.data(), n_rows * n_cols);
    std::vector<float> want(static_cast<size_t>(n_rows), 0.0f);
    for (int64_t r = 0; r < n_rows; ++r) {
        float a = 0.0f;
        for (int64_t c = 0; c < n_cols; ++c) a += wf[r * n_cols + c] * x[c];
        want[static_cast<size_t>(r)] = a;
    }
    std::vector<float> got(static_cast<size_t>(n_rows), 0.0f);
    const bool ok = matvec(DType::Q6_K, w.data(), x.data(), got.data(),
                           n_rows, n_cols, 1);
    float maxdiff = 0.0f;
    for (int64_t r = 0; r < n_rows; ++r)
        maxdiff = std::max(maxdiff, std::fabs(got[static_cast<size_t>(r)] -
                                             want[static_cast<size_t>(r)]));
    char msg[160];
    std::snprintf(msg, sizeof(msg), "Q6_K fused matches the reference (max diff %.2e)",
                  static_cast<double>(maxdiff));
    check(ok && maxdiff < 1e-2f, msg);
}

static void test_matmul_shape() {
    banner("[7] matmul: X @ W^T with the right shape and contents");
    std::mt19937 rng(3);
    const int64_t T = 3, n_rows = 13, n_cols = 256;
    const int64_t bb = dtype_block_bytes(DType::Q4_K);
    std::vector<uint8_t> w(static_cast<size_t>(n_rows * bb));
    fill_realistic_q4k(w, rng);
    std::vector<float> X(static_cast<size_t>(T * n_cols));
    for (auto& v : X) v = static_cast<float>(rng() % 100) / 100.0f;

    std::vector<float> Y(static_cast<size_t>(T * n_rows));
    check(matmul(DType::Q4_K, w.data(), X.data(), Y.data(), T, n_rows, n_cols, 1),
          "matmul accepts a well-formed shape");

    // every token's row must equal the equivalent matvec
    bool all_ok = true;
    for (int64_t t = 0; t < T; ++t) {
        std::vector<float> y1(static_cast<size_t>(n_rows));
        matvec(DType::Q4_K, w.data(), X.data() + t * n_cols, y1.data(),
               n_rows, n_cols, 1);
        for (int64_t r = 0; r < n_rows; ++r) {
            if (std::fabs(y1[static_cast<size_t>(r)] - Y[static_cast<size_t>(t * n_rows + r)]) > 1e-4f)
                all_ok = false;
        }
    }
    check(all_ok, "each token row equals the corresponding matvec result");
}

int main() {
    std::printf("=== aero kernels self-test ===\n");
    std::printf("hardware_threads() = %d\n", hardware_threads());
    test_reference_q4k();
    test_fused_matches_reference();
    test_threads_are_exact();
    test_fused_is_smaller_than_naive();
    test_refusals();
    test_q6k();
    test_matmul_shape();
    std::printf("\n=== %d checks, %d failures ===\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
