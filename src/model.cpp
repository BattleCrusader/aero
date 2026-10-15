// The native block loop. See include/aero/model.hpp for why this exists and why
// the per-architecture primitives are pluggable.
#include "aero/model.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include "aero/kernels.hpp"
#include "aero/tensor.hpp"

namespace aero {
namespace {

inline std::string blk(int layer) {
    return "blk." + std::to_string(layer) + ".";
}

inline bool is_fused(GGMLType t) {
    return t == GGMLType::Q4_K || t == GGMLType::Q6_K || t == GGMLType::IQ4_XS;
}

// rmsnorm over ONE row of a (T x n) activation matrix, into `h`.
void norm_row(const float* row, const float* w, float* out, int64_t n, float eps) {
    Vec xv{const_cast<float*>(row), n}, wv{const_cast<float*>(w), n}, ov{out, n};
    rmsnorm(xv, wv, ov, eps);
}

}  // namespace

const char* model_arch_name(int arch) {
    return arch == 0 ? "gemma4" : "qwen35";
}

ModelNative::ModelNative(const std::string& path, const Geom& g,
                         const std::vector<LayerGeom>& layers, int threads)
    : r_(new GgufReader(path)), g_(g), layers_(layers) {
    g_.threads = threads > 0 ? threads : 1;
    if ((int)layers_.size() != g_.n_layer)
        throw std::runtime_error("layer geom count " + std::to_string(layers_.size()) +
                                 " != n_layer " + std::to_string(g_.n_layer));
    // The dense-FFN width comes from the TENSOR, not from feed_forward_length:
    // this file class has no feed_forward_length at all in some revisions, and a
    // routed expert is narrower than the dense FFN. Only needed for dense layers.
    const TensorInfo* fg = r_->find(blk(0) + "ffn_gate.weight");
    ffn_len_ = fg ? fg->rows() : 0;
}

ModelNative::~ModelNative() = default;

void ModelNative::ensure(int64_t T) {
    if (cap_T_ >= T && !x_.empty()) return;
    cap_T_ = T;
    x_.assign((size_t)(T * g_.n_embd), 0.0f);
    h_.assign((size_t)(T * g_.n_embd), 0.0f);
    o_.assign((size_t)(T * g_.n_embd), 0.0f);
    d_.assign((size_t)(T * g_.n_embd), 0.0f);
    int64_t fl = ffn_len_ > 0 ? ffn_len_ : 1;
    gate_.assign((size_t)(T * fl), 0.0f);
    up_.assign((size_t)(T * fl), 0.0f);
}

// ------------------------------------------------------------------ matmul
// The projection primitive. Fused dequant+matmul straight off the mapping when
// the type has a kernel; otherwise dequantise in row blocks and use the fp32
// tensor matmul, so an unquantised weight still computes (bounded scratch).
bool ModelNative::mm(const std::string& name, const float* X, float* Y, int64_t T) {
    const TensorInfo* ti = r_->find(name);
    if (!ti || !ti->is_2d()) return false;
    const int64_t R = ti->rows(), C = ti->cols();
    if (ti->n_elements != R * C) return false;

    if (is_fused(ti->type)) {
        return aero::matmul((DType)(int)ti->type, r_->tensor_data(*ti), X, Y,
                            T, R, C, g_.threads);
    }

    // fp32 / f16 / bf16 fallback: block over the output rows.
    const DType dt = (DType)(int)ti->type;
    const int64_t bv = dtype_block_values(dt), bb = dtype_block_bytes(dt);
    if (bv <= 0 || bb <= 0 || (C % bv)) return false;
    const int64_t row_bytes = (C / bv) * bb;
    const uint8_t* base = r_->tensor_data(*ti);
    const int64_t br = g_.block_rows > 0 ? g_.block_rows : 64;
    std::vector<float> wblk, yblk;
    for (int64_t r0 = 0; r0 < R; r0 += br) {
        const int64_t nr = std::min<int64_t>(br, R - r0);
        wblk.resize((size_t)(nr * C));
        if (!dequant_ref(dt, base + r0 * row_bytes, nr * row_bytes, wblk.data(),
                         nr * C))
            return false;
        yblk.resize((size_t)(T * nr));
        Tensor2D W{wblk.data(), nr, C};
        Tensor2D Xm{const_cast<float*>(X), T, C};
        Tensor2D Ym{yblk.data(), T, nr};
        aero::matmul(W, Xm, Ym, g_.threads);
        for (int64_t t = 0; t < T; ++t)
            std::memcpy(Y + t * R + r0, yblk.data() + t * nr,
                        (size_t)(nr * sizeof(float)));
    }
    return true;
}

// ------------------------------------------------------------------ FFN
bool ModelNative::ffn(int layer, int T, const float* h2, float* out) {
    if (layers_[layer].is_moe) {
        if (!moe_) return false;
        return moe_(moe_ctx_, layer, h2, (int)T, g_.n_embd, out) != 0;
    }
    if (ffn_len_ <= 0) return false;
    const std::string p = blk(layer);
    if (!mm(p + "ffn_gate.weight", h2, gate_.data(), T)) return false;
    if (!mm(p + "ffn_up.weight", h2, up_.data(), T)) return false;
    swiglu_(Vec{gate_.data(), T * ffn_len_}, Vec{up_.data(), T * ffn_len_});
    if (!mm(p + "ffn_down.weight", gate_.data(), out, T)) return false;
    return true;
}

// ------------------------------------------------------------------ blocks
bool ModelNative::gemma_block(int layer, int T) {
    const std::string p = blk(layer);
    const int64_t n = g_.n_embd;
    const float scale = layers_[layer].out_scale;
    // layer_output_scale placement. "sum" (the reference) scales the WHOLE layer
    // output including the residual; scaling only the deltas damps the residual
    // stream by the per-layer factor 48 times over, which is a different function.
    float ds = 1.0f, ss = 1.0f;
    if (g_.gemma_out_scale == 1) ds = scale;
    else if (g_.gemma_out_scale == 2) ss = scale;

    const TensorInfo* an = r_->find(p + "attn_norm.weight");
    const TensorInfo* pan = r_->find(p + "post_attention_norm.weight");
    const TensorInfo* fn = r_->find(p + "ffn_norm.weight");
    const TensorInfo* pfn = r_->find(p + "post_ffw_norm.weight");
    if (!an || !pan || !fn || !pfn) return false;
    if (an->type != GGMLType::F32 || pan->type != GGMLType::F32 ||
        fn->type != GGMLType::F32 || pfn->type != GGMLType::F32)
        return false;   // every file this engine handles stores norms fp32
    const float* an_w = reinterpret_cast<const float*>(r_->tensor_data(*an));
    const float* pan_w = reinterpret_cast<const float*>(r_->tensor_data(*pan));
    const float* fn_w = reinterpret_cast<const float*>(r_->tensor_data(*fn));
    const float* pfn_w = reinterpret_cast<const float*>(r_->tensor_data(*pfn));

    if (!attn_) return false;
    for (int64_t t = 0; t < T; ++t)
        norm_row(x_.data() + t * n, an_w, h_.data() + t * n, n, g_.eps);
    if (!attn_(attn_ctx_, layer, h_.data(), positions_, (int)T, g_.n_embd,
               o_.data()))
        return false;

    // a = post_attention_norm(attn_out); attn_out = x + ds*a
    for (int64_t t = 0; t < T; ++t)
        norm_row(o_.data() + t * n, pan_w, h_.data() + t * n, n, g_.eps);
    for (int64_t i = 0; i < T * n; ++i) x_[i] += ds * h_[i];

    // h2 = ffn_norm(attn_out); d = post_ffw_norm(ffn(h2))
    for (int64_t t = 0; t < T; ++t)
        norm_row(x_.data() + t * n, fn_w, h_.data() + t * n, n, g_.eps);
    if (ffn_len_ <= 0) return false;
    if (!mm(p + "ffn_gate.weight", h_.data(), gate_.data(), T)) return false;
    if (!mm(p + "ffn_up.weight", h_.data(), up_.data(), T)) return false;
    // GELU, not SiLU: gemma4's FFN is gelu(gate) * up.
    gelu_(Vec{gate_.data(), T * ffn_len_});
    mul_(Vec{gate_.data(), T * ffn_len_}, Vec{up_.data(), T * ffn_len_});
    if (!mm(p + "ffn_down.weight", gate_.data(), d_.data(), T)) return false;
    for (int64_t t = 0; t < T; ++t)
        norm_row(d_.data() + t * n, pfn_w, d_.data() + t * n, n, g_.eps);

    for (int64_t i = 0; i < T * n; ++i) x_[i] = (x_[i] + ds * d_[i]) * ss;
    return true;
}

bool ModelNative::qwen_block(int layer, int T) {
    const std::string p = blk(layer);
    const int64_t n = g_.n_embd;
    const TensorInfo* an = r_->find(p + "attn_norm.weight");
    const TensorInfo* pan = r_->find(p + "post_attention_norm.weight");
    if (!an || !pan) return false;
    if (an->type != GGMLType::F32 || pan->type != GGMLType::F32) return false;
    const float* an_w = reinterpret_cast<const float*>(r_->tensor_data(*an));
    const float* pan_w = reinterpret_cast<const float*>(r_->tensor_data(*pan));

    for (int64_t t = 0; t < T; ++t)
        norm_row(x_.data() + t * n, an_w, h_.data() + t * n, n, g_.eps);

    if (layers_[layer].kind == 1) {
        if (!ssm_) return false;
        if (!ssm_(ssm_ctx_, layer, h_.data(), (int)T, g_.n_embd, o_.data()))
            return false;
    } else {
        if (!attn_) return false;
        if (!attn_(attn_ctx_, layer, h_.data(), positions_, (int)T, g_.n_embd,
                   o_.data()))
            return false;
    }

    if (g_.block_on_output) {
        // post_attention_norm normalises the ATTENTION OUTPUT; the MLP runs on
        // that; the single residual add comes after. (The other reading keeps
        // shapes valid and values finite while destroying the learned function.)
        for (int64_t t = 0; t < T; ++t)
            norm_row(o_.data() + t * n, pan_w, h_.data() + t * n, n, g_.eps);
        if (!ffn(layer, T, h_.data(), d_.data())) return false;
        for (int64_t i = 0; i < T * n; ++i) x_[i] += d_[i];
    } else {
        for (int64_t i = 0; i < T * n; ++i) x_[i] += o_[i];
        for (int64_t t = 0; t < T; ++t)
            norm_row(x_.data() + t * n, pan_w, h_.data() + t * n, n, g_.eps);
        if (!ffn(layer, T, h_.data(), d_.data())) return false;
        for (int64_t i = 0; i < T * n; ++i) x_[i] += d_[i];
    }
    return true;
}

// ------------------------------------------------------------------ forward
bool ModelNative::forward(const int32_t* tokens, int n_tokens,
                          const int64_t* positions, float* logits_out) {
    if (!tokens || !positions || !logits_out || n_tokens <= 0) return false;
    ensure(n_tokens);
    positions_ = positions;
    const int64_t n = g_.n_embd;

    // ---- embedding (one row dequantised per token, straight off the mapping)
    const TensorInfo* emb = r_->find("token_embd.weight");
    if (!emb) return false;
    if (emb->cols() != n) return false;
    {
        const DType dt = (DType)(int)emb->type;
        const int64_t bv = dtype_block_values(dt), bb = dtype_block_bytes(dt);
        if (bv <= 0 || bb <= 0 || (n % bv)) return false;
        const int64_t rb = (n / bv) * bb;
        const uint8_t* base = r_->tensor_data(*emb);
        const int64_t nv = emb->rows();
        for (int i = 0; i < n_tokens; ++i) {
            const int64_t tok = tokens[i];
            if (tok < 0 || tok >= nv) return false;
            if (!dequant_ref(dt, base + tok * rb, rb, x_.data() + (int64_t)i * n, n))
                return false;
        }
    }
    if (g_.embed_scale > 0.0f)
        for (int64_t i = 0; i < n_tokens * n; ++i) x_[i] *= g_.embed_scale;

    // ---- the blocks
    for (int layer = 0; layer < g_.n_layer; ++layer) {
        const bool ok = (g_.arch == 0) ? gemma_block(layer, n_tokens)
                                       : qwen_block(layer, n_tokens);
        if (!ok) return false;
    }

    // ---- final norm + head, last position only
    const TensorInfo* on = r_->find("output_norm.weight");
    if (!on || on->type != GGMLType::F32) return false;
    const float* on_w = reinterpret_cast<const float*>(r_->tensor_data(*on));
    norm_row(x_.data() + (int64_t)(n_tokens - 1) * n, on_w, h_.data(), n, g_.eps);

    const std::string head = g_.tied_output ? "token_embd.weight" : "output.weight";
    if (!mm(head, h_.data(), logits_out, 1)) return false;

    // gemma4 clamps the logit range; qwen does not. Applying it when the file
    // does not ask for it would silently flatten the distribution.
    if (g_.softcap > 0.0f) {
        const float cap = g_.softcap;
        for (int64_t i = 0; i < g_.vocab; ++i)
            logits_out[i] = cap * std::tanh(logits_out[i] / cap);
    }
    return true;
}

}  // namespace aero

// ============================================================ C ABI
extern "C" {

aero_model_t aero_model_open(const char* path, const aero_geom* g,
                             const aero_layer_geom* layers, char* err, int errlen) {
    if (!path || !g || !layers) {
        if (err) std::snprintf(err, (size_t)errlen, "null argument");
        return nullptr;
    }
    try {
        aero::Geom gg;
        gg.arch = g->arch; gg.n_layer = g->n_layer; gg.n_embd = g->n_embd;
        gg.vocab = g->vocab; gg.eps = g->eps; gg.tied_output = g->tied_output;
        gg.softcap = g->softcap; gg.embed_scale = g->embed_scale;
        gg.block_on_output = g->block_on_output;
        gg.gemma_out_scale = g->gemma_out_scale;
        gg.block_rows = g->block_rows;
        std::vector<aero::LayerGeom> lg((size_t)g->n_layer);
        for (int i = 0; i < g->n_layer; ++i) {
            lg[i].kind = layers[i].kind;
            lg[i].is_moe = layers[i].is_moe;
            lg[i].out_scale = layers[i].out_scale;
        }
        auto* m = new aero::ModelNative(path, gg, lg, g->threads);
        return (aero_model_t)m;
    } catch (const std::exception& e) {
        if (err) std::snprintf(err, (size_t)errlen, "%s", e.what());
        return nullptr;
    }
}

void aero_model_free(aero_model_t m) { delete (aero::ModelNative*)m; }

void aero_model_set_attn(aero_model_t m, aero_attn_fn fn, void* ctx) {
    if (m) ((aero::ModelNative*)m)->set_attn((aero::AttnFn)fn, ctx);
}
void aero_model_set_ssm(aero_model_t m, aero_ssm_fn fn, void* ctx) {
    if (m) ((aero::ModelNative*)m)->set_ssm((aero::SsmFn)fn, ctx);
}
void aero_model_set_moe(aero_model_t m, aero_moe_fn fn, void* ctx) {
    if (m) ((aero::ModelNative*)m)->set_moe((aero::MoeFn)fn, ctx);
}

int aero_model_forward(aero_model_t m, const int32_t* tokens, int n_tokens,
                       const int64_t* positions, float* logits) {
    if (!m) return 0;
    return ((aero::ModelNative*)m)->forward(tokens, n_tokens, positions, logits) ? 1 : 0;
}

int aero_model_n_layer(aero_model_t m) {
    return m ? ((aero::ModelNative*)m)->geom().n_layer : 0;
}
int aero_model_n_embd(aero_model_t m) {
    return m ? ((aero::ModelNative*)m)->geom().n_embd : 0;
}
int aero_model_vocab(aero_model_t m) {
    return m ? ((aero::ModelNative*)m)->geom().vocab : 0;
}
const char* aero_model_arch(aero_model_t m) {
    return m ? aero::model_arch_name(((aero::ModelNative*)m)->geom().arch) : "";
}

}  // extern "C"
