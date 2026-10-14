// Aero compute kernels — implementation.
//
// The two things that make this fast, and both are deliberate:
//
//   1. FUSED DEQUANTISE. The fp32 weight never exists. Q4_K gives
//      w[i] = d*sc*q[i] - dmin*m, so
//          sum_i w[i]*x[i] = d*sc * sum_i q[i]*x[i] - dmin*m * sum_i x[i]
//      and the inner loop only ever touches the 4-bit q values plus one x
//      partial per sub-block. That removes a 7.1x memory expansion per block.
//
//   2. SCALAR-FIRST, VECTOR-READY. The loops are written so a NEON path can
//      drop in, but correctness is established against `dequant_ref` first --
//      a vectorised inner loop that is wrong is worse than a slow one that is
//      right, and this project has already been bitten by exactly that.
#include "aero/kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <thread>
#include <vector>

namespace aero {

namespace {

// ------------------------------------------------------------------ fp16
// Scalar and exact. Only ever applied to per-block scales, not bulk data.
inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;                       // +/-0
        } else {
            // subnormal: normalise
            uint32_t e = 127 - 15 + 1;
            uint32_t m = mant;
            while ((m & 0x400u) == 0) { m <<= 1; --e; }
            bits = sign | (e << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13);   // inf / nan
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline float bf16_to_f32(uint16_t h) {
    const uint32_t bits = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// ------------------------------------------------------------------ Q4_K
// 256 values in 144 bytes: d(2) dmin(2) scales(12) qs(128).
// 8 sub-blocks of 32; scales[] packs 6-bit scale in one byte and 6-bit min in
// another, for sub-blocks 0..3 and 4..7 respectively.
inline void q4k_scales(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63u;
        *m = q[j + 4] & 63u;
    } else {
        *d = static_cast<uint8_t>((q[j + 4] & 0x0Fu) | ((q[j - 4] >> 6) << 4));
        *m = static_cast<uint8_t>((q[j + 4] >> 4)    | ((q[j]     >> 6) << 4));
    }
}

// One Q4_K super-block dotted against 256 contiguous x values. Never expands w.
inline float q4k_block_dot(const uint8_t* blk, const float* x) {
    const float d    = f16_to_f32(static_cast<uint16_t>(blk[0] | (blk[1] << 8)));
    const float dmin = f16_to_f32(static_cast<uint16_t>(blk[2] | (blk[3] << 8)));
    const uint8_t* scales = blk + 4;
    const uint8_t* qs     = blk + 16;

    float acc = 0.0f;
    int col = 0;
    int is  = 0;
    for (int j = 0; j < 256; j += 64) {
        uint8_t s1, m1, s2, m2;
        q4k_scales(is + 0, scales, &s1, &m1);
        q4k_scales(is + 1, scales, &s2, &m2);

        // low nibbles -> x[j .. j+32), high nibbles -> x[j+32 .. j+64)
        float qx1 = 0.0f, sx1 = 0.0f;
        float qx2 = 0.0f, sx2 = 0.0f;
        for (int i = 0; i < 32; ++i) {
            const uint8_t byte = qs[col + i];
            const float xlo = x[j + i];
            const float xhi = x[j + 32 + i];
            qx1 += static_cast<float>(byte & 0x0F) * xlo;
            sx1 += xlo;
            qx2 += static_cast<float>(byte >> 4)   * xhi;
            sx2 += xhi;
        }
        acc += (d * static_cast<float>(s1)) * qx1 - (dmin * static_cast<float>(m1)) * sx1;
        acc += (d * static_cast<float>(s2)) * qx2 - (dmin * static_cast<float>(m2)) * sx2;

        col += 32;
        is  += 2;
    }
    return acc;
}

// ------------------------------------------------------------------ Q6_K
// 256 values in 210 bytes: ql(128) qh(64) scales(16, int8) d(2 fp16).
//
// LAYOUT (verified against the Python reader, which is itself checked against
// real files -- getting this from the shape alone is how it was wrong first
// time): the 256 values are 2 halves of 128, and within a half 4 groups of 32.
// For half h and l in [0,32):
//     index h*128 + l      : ql[h*64 + l     ] low nibble,  qh[h*32+l] >> 0, scale h*8 + (l>>4) + 0
//     index h*128 + l + 32 : ql[h*64 + l + 32] low nibble,  qh[h*32+l] >> 2, scale h*8 + (l>>4) + 2
//     index h*128 + l + 64 : ql[h*64 + l     ] high nibble, qh[h*32+l] >> 4, scale h*8 + (l>>4) + 4
//     index h*128 + l + 96 : ql[h*64 + l + 32] high nibble, qh[h*32+l] >> 6, scale h*8 + (l>>4) + 6
// Each quant is 6 bits, taken as (low 4) | (high 2 << 4), then offset by -32.
inline float q6k_block_dot(const uint8_t* blk, const float* x) {
    const uint8_t* ql = blk;                                          // 128
    const uint8_t* qh = blk + 128;                                    // 64
    const int8_t*  sc = reinterpret_cast<const int8_t*>(blk + 192);   // 16
    const float d = f16_to_f32(static_cast<uint16_t>(blk[208] | (blk[209] << 8)));

    float acc = 0.0f;
    for (int half = 0; half < 2; ++half) {
        const uint8_t* qlh = ql + half * 64;
        const uint8_t* qhh = qh + half * 32;
        const int8_t*  sch = sc + half * 8;
        const int base = half * 128;
        for (int l = 0; l < 32; ++l) {
            const int is_ = l >> 4;
            const uint8_t lo = qlh[l];
            const uint8_t hi = qlh[l + 32];
            const uint8_t hb = qhh[l];
            const float v0 = static_cast<float>(((lo & 0x0F) | (((hb >> 0) & 3) << 4)) - 32);
            const float v1 = static_cast<float>(((hi & 0x0F) | (((hb >> 2) & 3) << 4)) - 32);
            const float v2 = static_cast<float>(((lo >> 4)   | (((hb >> 4) & 3) << 4)) - 32);
            const float v3 = static_cast<float>(((hi >> 4)   | (((hb >> 6) & 3) << 4)) - 32);
            acc += d * static_cast<float>(sch[is_ + 0]) * v0 * x[base + l];
            acc += d * static_cast<float>(sch[is_ + 2]) * v1 * x[base + l + 32];
            acc += d * static_cast<float>(sch[is_ + 4]) * v2 * x[base + l + 64];
            acc += d * static_cast<float>(sch[is_ + 6]) * v3 * x[base + l + 96];
        }
    }
    return acc;
}

// ------------------------------------------------------------------ dispatch
inline float block_dot(DType t, const uint8_t* blk, const float* x) {
    switch (t) {
        case DType::Q4_K: return q4k_block_dot(blk, x);
        case DType::Q6_K: return q6k_block_dot(blk, x);
        default:          return 0.0f;
    }
}

// Run fn(i) for i in [0,n) across `threads` threads.
template <typename Fn>
void parallel_for(int64_t n, int threads, Fn fn) {
    if (threads <= 1 || n <= 1) {
        for (int64_t i = 0; i < n; ++i) fn(i);
        return;
    }
    const int t = static_cast<int>(std::min<int64_t>(threads, n));
    std::vector<std::thread> pool;
    pool.reserve(static_cast<size_t>(t));
    for (int k = 0; k < t; ++k) {
        pool.emplace_back([&, k] {
            for (int64_t i = k; i < n; i += t) fn(i);
        });
    }
    for (auto& th : pool) th.join();
}

}  // namespace

const char* dtype_name(DType t) {
    switch (t) {
        case DType::F32:  return "F32";
        case DType::F16:  return "F16";
        case DType::Q4_K: return "Q4_K";
        case DType::Q6_K: return "Q6_K";
        case DType::BF16: return "BF16";
    }
    return "unknown";
}

int64_t dtype_block_bytes(DType t) {
    switch (t) {
        case DType::F32:  return 4;
        case DType::F16:  return 2;
        case DType::BF16: return 2;
        case DType::Q4_K: return 144;
        case DType::Q6_K: return 210;
    }
    return 0;
}

int64_t dtype_block_values(DType t) {
    switch (t) {
        case DType::F32:  return 1;
        case DType::F16:  return 1;
        case DType::BF16: return 1;
        case DType::Q4_K: return 256;
        case DType::Q6_K: return 256;
    }
    return 0;
}

// ------------------------------------------------------------------ reference
void dequant_ref(DType t, const uint8_t* raw, float* out, int64_t n_values) {
    const int64_t bv = dtype_block_values(t);
    const int64_t bb = dtype_block_bytes(t);
    const int64_t nblocks = n_values / bv;

    if (t == DType::F32) {
        std::memcpy(out, raw, static_cast<size_t>(n_values) * sizeof(float));
        return;
    }
    if (t == DType::F16 || t == DType::BF16) {
        const uint16_t* h = reinterpret_cast<const uint16_t*>(raw);
        for (int64_t i = 0; i < n_values; ++i) {
            out[i] = (t == DType::F16) ? f16_to_f32(h[i]) : bf16_to_f32(h[i]);
        }
        return;
    }
    if (t == DType::Q4_K) {
        for (int64_t b = 0; b < nblocks; ++b) {
            const uint8_t* blk = raw + b * bb;
            const float d    = f16_to_f32(static_cast<uint16_t>(blk[0] | (blk[1] << 8)));
            const float dmin = f16_to_f32(static_cast<uint16_t>(blk[2] | (blk[3] << 8)));
            const uint8_t* scales = blk + 4;
            const uint8_t* qs     = blk + 16;
            float* o = out + b * 256;
            int col = 0, is = 0;
            for (int j = 0; j < 256; j += 64) {
                uint8_t s1, m1, s2, m2;
                q4k_scales(is + 0, scales, &s1, &m1);
                q4k_scales(is + 1, scales, &s2, &m2);
                const float f1 = d * static_cast<float>(s1);
                const float g1 = dmin * static_cast<float>(m1);
                const float f2 = d * static_cast<float>(s2);
                const float g2 = dmin * static_cast<float>(m2);
                for (int i = 0; i < 32; ++i) {
                    const uint8_t byte = qs[col + i];
                    o[j + i]        = f1 * static_cast<float>(byte & 0x0F) - g1;
                    o[j + 32 + i]   = f2 * static_cast<float>(byte >> 4)   - g2;
                }
                col += 32; is += 2;
            }
        }
        return;
    }
    if (t == DType::Q6_K) {
        for (int64_t b = 0; b < nblocks; ++b) {
            const uint8_t* blk = raw + b * bb;
            const uint8_t* ql = blk;
            const uint8_t* qh = blk + 128;
            const int8_t*  sc = reinterpret_cast<const int8_t*>(blk + 192);
            const float d = f16_to_f32(static_cast<uint16_t>(blk[208] | (blk[209] << 8)));
            float* o = out + b * 256;
            // same 2-halves x 4-groups layout as q6k__dot
            for (int half = 0; half < 2; ++half) {
                const uint8_t* qlh = ql + half * 64;
                const uint8_t* qhh = qh + half * 32;
                const int8_t*  sch = sc + half * 8;
                const int base = half * 128;
                for (int l = 0; l < 32; ++l) {
                    const int is_ = l >> 4;
                    const uint8_t lo = qlh[l];
                    const uint8_t hi = qlh[l + 32];
                    const uint8_t hb = qhh[l];
                    o[base + l]      = d * static_cast<float>(sch[is_ + 0]) *
                                       static_cast<float>(((lo & 0x0F) | (((hb >> 0) & 3) << 4)) - 32);
                    o[base + l + 32] = d * static_cast<float>(sch[is_ + 2]) *
                                       static_cast<float>(((hi & 0x0F) | (((hb >> 2) & 3) << 4)) - 32);
                    o[base + l + 64] = d * static_cast<float>(sch[is_ + 4]) *
                                       static_cast<float>(((lo >> 4)   | (((hb >> 4) & 3) << 4)) - 32);
                    o[base + l + 96] = d * static_cast<float>(sch[is_ + 6]) *
                                       static_cast<float>(((hi >> 4)   | (((hb >> 6) & 3) << 4)) - 32);
                }
            }
        }
        return;
    }
}

// ------------------------------------------------------------------ fusion
bool matvec(DType dt, const uint8_t* w, const float* x, float* y,
            int64_t n_rows, int64_t n_cols, int threads) {
    if (dt != DType::Q4_K && dt != DType::Q6_K) return false;
    const int64_t bv = dtype_block_values(dt);
    const int64_t bb = dtype_block_bytes(dt);
    if (n_cols % bv != 0) return false;
    const int64_t row_bytes = (n_cols / bv) * bb;

    parallel_for(n_rows, threads, [&](int64_t r) {
        const uint8_t* row = w + r * row_bytes;
        float acc = 0.0f;
        for (int64_t c = 0; c < n_cols; c += bv) {
            acc += block_dot(dt, row + (c / bv) * bb, x + c);
        }
        y[r] = acc;
    });
    return true;
}

bool matmul(DType dt, const uint8_t* w, const float* X, float* Y,
            int64_t n_tokens, int64_t n_rows, int64_t n_cols, int threads) {
    if (dt != DType::Q4_K && dt != DType::Q6_K) return false;
    const int64_t bv = dtype_block_values(dt);
    const int64_t bb = dtype_block_bytes(dt);
    if (n_cols % bv != 0) return false;
    const int64_t row_bytes = (n_cols / bv) * bb;

    // parallelise over rows; each row is dotted against every token
    parallel_for(n_rows, threads, [&](int64_t r) {
        const uint8_t* row = w + r * row_bytes;
        for (int64_t t = 0; t < n_tokens; ++t) {
            const float* x = X + t * n_cols;
            float acc = 0.0f;
            for (int64_t c = 0; c < n_cols; c += bv) {
                acc += block_dot(dt, row + (c / bv) * bb, x + c);
            }
            Y[t * n_rows + r] = acc;
        }
    });
    return true;
}

int hardware_threads() {
    static const int n = [] {
        unsigned hc = std::thread::hardware_concurrency();
        if (hc == 0) hc = 1;
        return static_cast<int>(std::min(hc, 8u));
    }();
    return n;
}

}  // namespace aero

// --------------------------------------------------------------------- C ABI
// Flat, unmangled entry points for ctypes. C++ mangles names inside a namespace,
// so dlsym("matvec") would otherwise fail -- the Python side needs real C
// symbols. These are one-line forwarders; all logic stays in the namespace.
#if defined(__cplusplus)
extern "C" {
#endif

int64_t aero_dtype_block_bytes(int t) {
    return aero::dtype_block_bytes(static_cast<aero::DType>(t));
}
int64_t aero_dtype_block_values(int t) {
    return aero::dtype_block_values(static_cast<aero::DType>(t));
}
int aero_hardware_threads(void) { return aero::hardware_threads(); }

int aero_dequant_ref(int t, const uint8_t* raw, float* out, int64_t n_values) {
    switch (static_cast<aero::DType>(t)) {
        case aero::DType::F32:
        case aero::DType::F16:
        case aero::DType::Q4_K:
        case aero::DType::Q6_K:
        case aero::DType::BF16:
            aero::dequant_ref(static_cast<aero::DType>(t), raw, out, n_values);
            return 1;
        default:
            return 0;
    }
}

int aero_matvec(int t, const uint8_t* w, const float* x, float* y,
                int64_t n_rows, int64_t n_cols, int threads) {
    return aero::matvec(static_cast<aero::DType>(t), w, x, y,
                        n_rows, n_cols, threads) ? 1 : 0;
}

int aero_matmul(int t, const uint8_t* w, const float* X, float* Y,
                int64_t n_tokens, int64_t n_rows, int64_t n_cols, int threads) {
    return aero::matmul(static_cast<aero::DType>(t), w, X, Y,
                        n_tokens, n_rows, n_cols, threads) ? 1 : 0;
}

#if defined(__cplusplus)
}  // extern "C"
#endif

