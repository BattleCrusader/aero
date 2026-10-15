// The KV cache, native. Port of engine/transformer.py's KVCache.
//
// WHY THIS IS WORTH PORTING: the cache is touched per token per layer, and its
// inner loops (quantise, pack, unpack, ring rotation) are pure elementwise work --
// the exact shape numpy wastes time dispatching on. The Python class remains the
// reference; tests/test_kv_native.py cross-checks these functions against it on
// identical inputs and prints the observed difference.
//
// SEMANTIC CONTRACT (from the Python, deviations = bugs):
//   * int8 : symmetric per-(position,head) scale, q = round(x/s) clipped to
//            [-127,127]; scale = amax/127, or 1.0 where amax==0.
//   * int4 : same, but range [-7,7], scale amax/7, two 4-bit values per byte,
//            LOW NIBBLE FIRST, stored as (v & 0x0F) (two's complement on unpack).
//   * ring : a windowed layer's buffer holds the most recent (window+slack)
//            positions and WRAPS; reads must rotate by (end % cap) so index 0 is
//            the OLDEST retained position.
//   * append does NOT advance len -- commit() does -- so a whole step (every
//     layer) lands at the same offset.
//   * the quantisation error (rel, vs the span's input max) is accumulated and
//     reported, never assumed zero.
//
// Layout: one flat buffer per layer per side, (cap, heads, dim) row-major, so the
// C ABI stays pointer+lengths and ctypes never sees a C++ type.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace aero {

enum class KvDType { FP32 = 0, INT8 = 1, INT4 = 2 };

// One layer's slot for one side (K or V).
struct KvSlot {
    int64_t cap = 0;          // positions this slot holds
    int64_t heads = 0;        // per-layer kv head count (may differ per layer)
    int64_t dim = 0;          // per-layer vector width (kdim/vdim)
    KvDType dt = KvDType::FP32;
    std::vector<float> f32;   // FP32
    std::vector<int8_t> i8;   // INT8 values
    std::vector<uint8_t> i4;  // INT4 packed values (dim/2 per vector)
    std::vector<float> scale; // INT8/INT4: (cap, heads)
    bool ring = false;

    int64_t vec_bytes() const {
        if (dt == KvDType::INT4) return dim / 2;
        if (dt == KvDType::INT8) return dim;
        return dim * 4;
    }
    int64_t bytes() const {
        const int64_t vecs = cap * heads;
        int64_t n = vecs * vec_bytes();
        if (dt != KvDType::FP32) n += cap * heads * int64_t(sizeof(float));
        return n;
    }
};

// Quantise helpers, matching the Python exactly.
inline void kv_quant_i8(const float* x, int64_t dim, int8_t* q, float* s) {
    float amax = 0.0f;
    for (int64_t i = 0; i < dim; ++i) amax = std::max(amax, std::fabs(x[i]));
    const float sc = amax > 0.0f ? amax / 127.0f : 1.0f;
    *s = sc;
    for (int64_t i = 0; i < dim; ++i) {
        float v = std::round(x[i] / sc);
        if (v > 127.0f) v = 127.0f;
        if (v < -127.0f) v = -127.0f;
        q[i] = static_cast<int8_t>(v);
    }
}

inline void kv_quant_i4(const float* x, int64_t dim, uint8_t* q, float* s) {
    float amax = 0.0f;
    for (int64_t i = 0; i < dim; ++i) amax = std::max(amax, std::fabs(x[i]));
    const float sc = amax > 0.0f ? amax / 7.0f : 1.0f;
    *s = sc;
    // pack two per byte, LOW NIBBLE FIRST, clipped to [-7,7]
    for (int64_t i = 0; i < dim; i += 2) {
        float a = std::round(x[i] / sc);
        float b = std::round(x[i + 1] / sc);
        if (a > 7.0f) a = 7.0f;
        if (a < -7.0f) a = -7.0f;
        if (b > 7.0f) b = 7.0f;
        if (b < -7.0f) b = -7.0f;
        const uint8_t lo = static_cast<uint8_t>(static_cast<int>(a) & 0x0F);
        const uint8_t hi = static_cast<uint8_t>(static_cast<int>(b) & 0x0F);
        q[i / 2] = static_cast<uint8_t>(lo | (hi << 4));
    }
}

inline void kv_unpack_i4(const uint8_t* q, int64_t dim, float* out) {
    for (int64_t i = 0; i < dim; i += 2) {
        const uint8_t byte = q[i / 2];
        const int lo = static_cast<int>(byte & 0x0F);
        const int hi = static_cast<int>((byte >> 4) & 0x0F);
        // sign-extend from 4 bits: values are in [-8, 7]
        out[i] = static_cast<float>(lo >= 8 ? lo - 16 : lo);
        out[i + 1] = static_cast<float>(hi >= 8 ? hi - 16 : hi);
    }
}

// Write one (position, head) vector of one side. `pos` is the slot-relative
// physical index (ring wrap handled by the caller, which owns the spans).
inline void kv_slot_write(KvSlot& sl, int64_t phys, const float* x) {
    const int64_t off = phys * sl.heads * sl.vec_bytes();
    if (sl.dt == KvDType::FP32) {
        std::memcpy(sl.f32.data() + off, x, sl.dim * sizeof(float));
        return;
    }
    const int64_t so = phys * sl.heads;
    if (sl.dt == KvDType::INT8) {
        kv_quant_i8(x, sl.dim, sl.i8.data() + off, sl.scale.data() + so);
        return;
    }
    kv_quant_i4(x, sl.dim, sl.i4.data() + off, sl.scale.data() + so);
}

// Materialise one (position, head) vector as fp32.
inline void kv_slot_read(const KvSlot& sl, int64_t phys, float* out) {
    const int64_t off = phys * sl.heads * sl.vec_bytes();
    const int64_t so = phys * sl.heads;
    if (sl.dt == KvDType::FP32) {
        std::memcpy(out, sl.f32.data() + off, sl.dim * sizeof(float));
        return;
    }
    if (sl.dt == KvDType::INT8) {
        for (int64_t i = 0; i < sl.dim; ++i) out[i] = float(sl.i8[off + i]) * sl.scale[so];
        return;
    }
    for (int64_t i = 0; i < sl.dim; i += 2) {
        const uint8_t byte = sl.i4[off + i / 2];
        const int lo = static_cast<int>(byte & 0x0F);
        const int hi = static_cast<int>((byte >> 4) & 0x0F);
        out[i] = float(lo >= 8 ? lo - 16 : lo) * sl.scale[so];
        out[i + 1] = float(hi >= 8 ? hi - 16 : hi) * sl.scale[so];
    }
}

}  // namespace aero