// KV cache primitives, C ABI. See include/aero/kv.hpp for the contract, which
// mirrors engine/transformer.py's KVCache exactly (quantisation formulas, nibble
// order, ring rotation, the append/commit split).
#include "aero/kv.hpp"

#include <new>

extern "C" {

// ---- quantise one vector -------------------------------------------------
// x (dim,) -> q packed according to dt; writes ONE scale (fp32).
// dt: 0 = fp32 (memcpy), 1 = int8, 2 = int4-packed.
int aero_kv_write_vec(int dt, const float* x, int64_t dim, void* q, float* scale) {
    if (!x || !q || !scale || dim <= 0) return 0;
    if (dt == 0) {
        std::memcpy(q, x, dim * sizeof(float));
    } else if (dt == 1) {
        aero::kv_quant_i8(x, dim, static_cast<int8_t*>(q), scale);
    } else if (dt == 2) {
        if (dim % 2) return 0;
        aero::kv_quant_i4(x, dim, static_cast<uint8_t*>(q), scale);
    } else {
        return 0;
    }
    return 1;
}

// ---- materialise one vector ----------------------------------------------
int aero_kv_read_vec(int dt, const void* q, const float* scale, int64_t dim,
                     float* out) {
    if (!q || !scale || !out || dim <= 0) return 0;
    if (dt == 0) {
        std::memcpy(out, q, dim * sizeof(float));
    } else if (dt == 1) {
        const int8_t* v = static_cast<const int8_t*>(q);
        for (int64_t i = 0; i < dim; ++i) out[i] = float(v[i]) * (*scale);
    } else if (dt == 2) {
        if (dim % 2) return 0;
        aero::kv_unpack_i4(static_cast<const uint8_t*>(q), dim, out);
        for (int64_t i = 0; i < dim; ++i) out[i] *= (*scale);
    } else {
        return 0;
    }
    return 1;
}

// ---- ring rotation --------------------------------------------------------
// A ring's physical layout wraps; the chronological order needs index rotation:
// out[0] = physical (start), out[1] = start+1 ... wrapping. Emits an index array
// of `cap` physical indices in chronological order for the FIRST `n` positions.
// `start = end % cap` when wrapped, 0 otherwise.
int aero_kv_ring_indices(int64_t cap, int64_t n, int64_t* out) {
    if (cap <= 0 || n < 0 || !out) return 0;
    if (n > cap) return 0;
    for (int64_t i = 0; i < n; ++i) out[i] = i;   // caller supplies start==0 case
    return 1;
}

// The wrapped rotation, given the ring's start offset.
int aero_kv_ring_rotate(int64_t cap, int64_t start, int64_t n, int64_t* out) {
    if (cap <= 0 || n < 0 || !out || start < 0 || start >= cap) return 0;
    if (n > cap) return 0;
    for (int64_t i = 0; i < n; ++i) out[i] = (start + i) % cap;
    return 1;
}

// ---- error accounting -----------------------------------------------------
// Relative quantisation error of one written span: max|x - dequant(x)| / max|x|.
int aero_kv_span_error(int dt, const void* q, const float* scale, int64_t dim,
                       const float* x, float* out) {
    if (!q || !scale || !x || !out || dim <= 0) return 0;
    float maxerr = 0.0f, xmax = 0.0f;
    float* back = new float[dim];
    if (!aero_kv_read_vec(dt, q, scale, dim, back)) { delete[] back; return 0; }
    for (int64_t i = 0; i < dim; ++i) {
        const float e = std::fabs(back[i] - x[i]);
        if (e > maxerr) maxerr = e;
        const float a = std::fabs(x[i]);
        if (a > xmax) xmax = a;
    }
    delete[] back;
    *out = maxerr / (xmax > 0.0f ? xmax : 1e-9f);
    return 1;
}

// ---- slot lifetime + access ----------------------------------------------
// The attention layer (src/attention_layer.cpp) takes KvSlot* and writes/reads
// through them, so the runtime needs a way to own slots without seeing a C++ type.
// These are the allocator and the accessors; the struct stays opaque (void*).
//
// dt: 0 = fp32, 1 = int8, 2 = int4-packed. `ring` marks a sliding-window slot, whose
// physical layout wraps and whose reads must be rotated by (end % cap).
void* aero_kv_slot_new(int64_t cap, int64_t heads, int64_t dim, int dt, int ring) {
    if (cap <= 0 || heads <= 0 || dim <= 0 || dt < 0 || dt > 2) return nullptr;
    if (dt == 2 && (dim % 2)) return nullptr;      // int4 packs pairs
    auto* sl = new (std::nothrow) aero::KvSlot();
    if (!sl) return nullptr;
    sl->cap = cap;
    sl->heads = heads;
    sl->dim = dim;
    sl->dt = static_cast<aero::KvDType>(dt);
    sl->ring = ring != 0;
    const int64_t vecs = cap * heads;
    if (dt == 0) sl->f32.assign(static_cast<size_t>(vecs * dim), 0.0f);
    else if (dt == 1) sl->i8.assign(static_cast<size_t>(vecs * dim), 0);
    else sl->i4.assign(static_cast<size_t>(vecs * (dim / 2)), 0);
    if (dt != 0) sl->scale.assign(static_cast<size_t>(vecs), 0.0f);
    return sl;
}

void aero_kv_slot_free(void* s) { delete static_cast<aero::KvSlot*>(s); }

int64_t aero_kv_slot_bytes(void* s) {
    auto* sl = static_cast<aero::KvSlot*>(s);
    return sl ? sl->bytes() : -1;
}

// Write one (position, head) vector at a PHYSICAL index (the ring wrap is the
// caller's business, exactly as in the Python _write_side spans).
int aero_kv_slot_write(void* s, int64_t phys, const float* x) {
    auto* sl = static_cast<aero::KvSlot*>(s);
    if (!sl || !x || phys < 0 || phys >= sl->cap) return 0;
    aero::kv_slot_write(*sl, phys, x);
    return 1;
}

int aero_kv_slot_read(void* s, int64_t phys, float* out) {
    auto* sl = static_cast<aero::KvSlot*>(s);
    if (!sl || !out || phys < 0 || phys >= sl->cap) return 0;
    aero::kv_slot_read(*sl, phys, out);
    return 1;
}

}  // extern "C"