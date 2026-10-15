// The native attention layer's implementation. Contract: include/aero/attention_layer.hpp
// (mirror of engine/transformer.py's attention(); a deviation is a bug).
#include "aero/attention_layer.hpp"
#include "aero/tensor.hpp"
#include "aero/transformer.hpp"

#include <algorithm>
#include <numeric>

namespace aero {
bool attention_layer(const AttentionLayerArgs& a, float* out) {
    if (!a.x || !a.qpos || !a.q_w || !a.k_w || !a.o_w || !out) return false;
    if (a.T <= 0 || a.n_embd <= 0 || a.n_head <= 0 || a.n_head_kv <= 0) return false;
    if (a.head_dim <= 0 || a.value_dim <= 0) return false;
    if (a.n_head % a.n_head_kv) return false;
            const int64_t H = a.n_head, HK = a.n_head_kv;
    const int64_t D = a.head_dim, DV = a.value_dim;
    const int64_t HD = H * D;
    const float scale = a.scale > 0.0f ? a.scale : 1.0f / std::sqrt(float(D));

    // ---- 1. projections: blocked quantised matmul, bounded scratch ----------
    // `src` is the INPUT to project (a.x for the q/k/v projections, the attention
    // output for the o-projection). The first version captured a.x by reference
    // and projected THE LAYER INPUT for the o-projection too -- the attention
    // output was silently thrown away. Every stage up to the gate matched the
    // python exactly while the final result was nonsense, which is what made this
    // take so long to find: the pipeline was right, the LAST projection read the
    // wrong buffer.
    // `cols` is the projection's INPUT width: n_embd for q/k/v, but the O
    // projection maps HD -> n_embd, so its cols = HD. The first version hardcoded
    // cols = a.n_embd for every projection, which silently truncated the o-proj's
    // input to 2560 of its 4096 elements -- right shapes, wrong maths, exactly
    // the failure class this module exists to close.
    auto proj = [&](DType dt, const void* w, int64_t rows, int64_t cols,
                    const float* src_in, float* y) -> bool {
        const int64_t bv = dtype_block_values(dt);
        const int64_t bb = dtype_block_bytes(dt);
        if (bv <= 0 || bb <= 0 || (cols % bv)) return false;
        const int64_t row_bytes = (cols / bv) * bb;
        const int64_t br = 64;
        std::vector<float> wblk, yblk;
        for (int64_t r0 = 0; r0 < rows; r0 += br) {
            const int64_t nr = std::min(br, rows - r0);
            wblk.resize((size_t)(nr * cols));
            if (!dequant_ref(dt, static_cast<const uint8_t*>(w) + r0 * row_bytes,
                             nr * row_bytes, wblk.data(), nr * cols))
                return false;
            yblk.assign((size_t)(a.T * nr), 0.0f);
            Tensor2D W{wblk.data(), nr, cols};
            Tensor2D X{const_cast<float*>(src_in), a.T, cols};
            Tensor2D Y{yblk.data(), a.T, nr};
            matmul(W, X, Y, a.threads);
            for (int64_t t = 0; t < a.T; ++t)
                std::memcpy(y + t * rows + r0, yblk.data() + t * nr,
                            (size_t)(nr * sizeof(float)));
        }
        return true;
    };

    std::vector<float> q_raw((size_t)(a.T * a.q_rows));
    if (!proj(a.q_dt, a.q_w, a.q_rows, a.n_embd, a.x, q_raw.data())) return false;
    std::vector<float> k_raw((size_t)(a.T * a.k_rows));
    if (!proj(a.k_dt, a.k_w, a.k_rows, a.n_embd, a.x, k_raw.data())) return false;
    std::vector<float> v_raw;
    if (a.v_rows > 0) {
        v_raw.resize((size_t)(a.T * a.v_rows));
        if (!proj(a.v_dt, a.v_w, a.v_rows, a.n_embd, a.x, v_raw.data())) return false;
    } else {
        v_raw = k_raw;                    // V := K (gemma4's V-less global layers)
    }

    // ---- 2. gate split: PER-HEAD INTERLEAVED [q_h | gate_h] -----------------
    std::vector<float> gate_in;
    if (a.has_gate) {
        gate_in.resize((size_t)(a.T * HD));
        std::vector<float> q2((size_t)(a.T * HD));
        for (int64_t t = 0; t < a.T; ++t)
            for (int64_t h = 0; h < H; ++h) {
                const float* src = q_raw.data() + t * 2 * HD + h * 2 * D;
                std::memcpy(q2.data() + t * HD + h * D, src,
                            (size_t)(D * sizeof(float)));
                std::memcpy(gate_in.data() + t * HD + h * D, src + D,
                            (size_t)(D * sizeof(float)));
            }
        q_raw.swap(q2);
    }


    // ---- 3. per-head q/k norms (BEFORE rope) --------------------------------
    if (a.q_norm)
        for (int64_t t = 0; t < a.T; ++t)
            for (int64_t h = 0; h < H; ++h) {
                // rmsnorm takes Vec& (it writes through), so the output needs an
                // lvalue, not a temporary -- the same constraint that shaped the
                // C ABI in src/tensor.cpp.
                Vec xv{q_raw.data() + t * HD + h * D, D};
                Vec wv{const_cast<float*>(a.q_norm), D};
                Vec ov{q_raw.data() + t * HD + h * D, D};
                rmsnorm(xv, wv, ov, a.eps);
            }
    if (a.k_norm)
        for (int64_t t = 0; t < a.T; ++t)
            for (int64_t h = 0; h < HK; ++h) {
                Vec xv{k_raw.data() + t * a.k_rows + h * D, D};
                Vec wv{const_cast<float*>(a.k_norm), D};
                Vec ov{k_raw.data() + t * a.k_rows + h * D, D};
                rmsnorm(xv, wv, ov, a.eps);
            }
    if (a.v_weightless_norm && a.v_rows > 0)
        for (int64_t t = 0; t < a.T; ++t)
            for (int64_t h = 0; h < HK; ++h) {
                Tensor2D t_v{v_raw.data() + t * a.v_rows + h * DV, 1, DV};
                rmsnorm_rows(t_v, a.eps);
            }

    // ---- 4. rope (partial) on q and k --------------------------------------
    //
    // CONTRACT: apply_rope reads the table by POSITION (cos[pos[r]*half]). So the
    // table must cover max_pos+1 rows; build it ONCE with a dense position list
    // 0..max_pos per call (the tests use exactly this pattern). The first version
    // passed a (T)-row table, so a single token at position 7 read row 7 of a
    // 1-row table: SIGBUS. The second read by ROW instead, which broke the
    // transformer tests' contract. Dense-table construction satisfies both.
    {
        const int64_t n_rot = a.rope_dims > 0 ? a.rope_dims : D;
        const int64_t half = n_rot / 2;
        int64_t max_pos = a.qpos[0];
        for (int64_t t = 1; t < a.T; ++t)
            max_pos = std::max(max_pos, (int64_t)a.qpos[t]);
        std::vector<float> cos_tab((size_t)((max_pos + 1) * half)), sin_tab(
            (size_t)((max_pos + 1) * half));
        std::vector<int32_t> allpos((size_t)(max_pos + 1));
        for (int64_t i = 0; i <= max_pos; ++i) allpos[(size_t)i] = (int32_t)i;
        rope_tables(n_rot, allpos.data(), max_pos + 1, a.rope_base, n_rot,
                    a.freq_factors, a.n_ff, cos_tab.data(), sin_tab.data());
        apply_rope(q_raw.data(), a.T * H, D, n_rot, cos_tab.data(), sin_tab.data(),
                   a.qpos, RopeMode::Neox);
        apply_rope(k_raw.data(), a.T * HK, D, n_rot, cos_tab.data(), sin_tab.data(),
                   a.qpos, RopeMode::Neox);
    }

    // ---- 5. KV append (native quantisers) + read-back of the span ----------
    std::vector<float> k_all, v_all;
    int64_t n_kv;
    if (a.k_slot && a.v_slot) {
        for (int64_t t = 0; t < a.T; ++t) {
            const int64_t phys = a.kv_phys + t;
            for (int64_t h = 0; h < HK; ++h) {
                KvSlot& ks = *a.k_slot;
                const int64_t off = phys * ks.heads * ks.vec_bytes() + h * ks.vec_bytes();
                const int64_t so = phys * ks.heads + h;
                const float* krow = k_raw.data() + t * a.k_rows + h * D;
                if (ks.dt == KvDType::FP32)
                    std::memcpy(ks.f32.data() + off, krow, (size_t)(D * sizeof(float)));
                else if (ks.dt == KvDType::INT8)
                    kv_quant_i8(krow, D, ks.i8.data() + off, ks.scale.data() + so);
                else
                    kv_quant_i4(krow, D, ks.i4.data() + off, ks.scale.data() + so);
            }
            for (int64_t h = 0; h < HK; ++h) {
                KvSlot& vs = *a.v_slot;
                const int64_t off = phys * vs.heads * vs.vec_bytes() + h * vs.vec_bytes();
                const int64_t so = phys * vs.heads + h;
                const float* vrow = v_raw.data() + t * a.v_rows + h * DV;
                if (vs.dt == KvDType::FP32)
                    std::memcpy(vs.f32.data() + off, vrow, (size_t)(DV * sizeof(float)));
                else if (vs.dt == KvDType::INT8)
                    kv_quant_i8(vrow, DV, vs.i8.data() + off, vs.scale.data() + so);
                else
                    kv_quant_i4(vrow, DV, vs.i4.data() + off, vs.scale.data() + so);
            }
        }
        n_kv = a.kv_phys + a.T;
        k_all.resize((size_t)(n_kv * HK * D));
        v_all.resize((size_t)(n_kv * HK * DV));
        for (int64_t p = 0; p < n_kv; ++p)
            for (int64_t h = 0; h < HK; ++h) {
                kv_slot_read(*a.k_slot, p, k_all.data() + (p * HK + h) * D);
                kv_slot_read(*a.v_slot, p, v_all.data() + (p * HK + h) * DV);
            }
    } else {
        k_all = k_raw;
        v_all = v_raw;
        n_kv = a.T;
    }

    // ---- 6. GQA/causal/windowed attention ----------------------------------
    std::vector<float> att((size_t)(a.T * HD), 0.0f);
    if (!attention_tiled(q_raw.data(), k_all.data(), v_all.data(), att.data(),
                         a.T, H, HK, D, DV, scale, a.qpos, nullptr, n_kv, a.window,
                         1024))
        return false;

    // ---- 7. gate (python's default branch: SIGMOID) ------------------------
    if (a.has_gate)
        for (size_t i = 0; i < gate_in.size(); ++i)
            att[i] *= 1.0f / (1.0f + std::exp(-gate_in[i]));

    // ---- 8. output projection ----------------------------------------------
    // cols = HD: the o projection maps the attention width, not n_embd.
    if (!proj(a.o_dt, a.o_w, a.o_rows, a.T * HD, att.data(), out))
        return false;
    return true;
}

}  // namespace aero

extern "C" {

int aero_attention_qwen(const float* x, int64_t T, int64_t n_embd,
                        const int32_t* qpos,
                        int q_dt, const void* q_w, int64_t q_rows, int has_gate,
                        int k_dt, const void* k_w, int64_t k_rows,
                        int v_dt, const void* v_w, int64_t v_rows,
                        int o_dt, const void* o_w, int64_t o_rows,
                        const float* q_norm, const float* k_norm,
                        int v_weightless_norm, float eps,
                        float rope_base, int64_t rope_dims,
                        const float* freq_factors, int64_t n_ff,
                        int64_t n_head, int64_t n_head_kv, int64_t head_dim,
                        int64_t value_dim, float scale, int window,
                        int gate, int threads,
                        void* k_slot, void* v_slot, int64_t kv_phys,
                        float* out) {
    aero::AttentionLayerArgs a{};
    a.x = x; a.T = T; a.n_embd = n_embd; a.qpos = qpos;
    a.q_dt = (aero::DType)q_dt; a.q_w = q_w; a.q_rows = q_rows;
    a.has_gate = has_gate != 0;
    a.k_dt = (aero::DType)k_dt; a.k_w = k_w; a.k_rows = k_rows;
    a.v_dt = (aero::DType)v_dt; a.v_w = v_w; a.v_rows = v_rows;
    a.o_dt = (aero::DType)o_dt; a.o_w = o_w; a.o_rows = o_rows;
    a.q_norm = q_norm; a.k_norm = k_norm;
    a.v_weightless_norm = v_weightless_norm != 0;
    a.eps = eps;
    a.rope_base = rope_base; a.rope_dims = rope_dims;
    a.freq_factors = freq_factors; a.n_ff = n_ff;
    a.n_head = n_head; a.n_head_kv = n_head_kv;
    a.head_dim = head_dim; a.value_dim = value_dim;
    a.scale = scale; a.window = window;
    a.gate = (aero::GateMode)gate; a.threads = threads;
    a.k_slot = (aero::KvSlot*)k_slot; a.v_slot = (aero::KvSlot*)v_slot;
    a.kv_phys = kv_phys;
    return aero::attention_layer(a, out) ? 1 : 0;
}

}  // extern "C"
