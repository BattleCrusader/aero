// gemma4 architecture descriptor, native. See include/aero/arch.hpp for why the
// per-layer variation matters and for the file facts these values came from.

#include "aero/arch.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace aero {

namespace {

const char* const kBlockTensors[] = {
    "attn_norm.weight",
    "ffn_norm.weight",
    "post_attention_norm.weight",
    "post_ffw_norm.weight",
    "layer_output_scale.weight",
};

const int kBlockTensorCount = (int)(sizeof(kBlockTensors) / sizeof(kBlockTensors[0]));

}  // namespace

std::vector<int> Gemma4Arch::kv_heads_for(const std::vector<int>& raw, int n_layer) {
    // A scalar widens to every layer; a list must match the layer count. gemma4
    // REQUIRES the key: defaulting it would give every layer some other layer's
    // head count.
    if (raw.empty())
        throw std::invalid_argument("gemma4 requires attention.head_count_kv");
    if ((int)raw.size() != 1 && (int)raw.size() != n_layer)
        throw std::invalid_argument("head_count_kv has " + std::to_string(raw.size()) +
                                    " entries for " + std::to_string(n_layer) + " layers");
    if ((int)raw.size() == 1) return std::vector<int>((size_t)n_layer, raw[0]);
    return raw;
}

std::vector<int> Gemma4Arch::pattern_for(const std::vector<int>& raw, int n_layer) {
    // Empty raw == the metadata key is absent == no pattern (all layers global).
    if (raw.empty()) return {};
    if ((int)raw.size() != 1 && (int)raw.size() != n_layer)
        throw std::invalid_argument("sliding_window_pattern has " +
                                    std::to_string(raw.size()) + " entries for " +
                                    std::to_string(n_layer) + " layers");
    if ((int)raw.size() == 1) return std::vector<int>((size_t)n_layer, raw[0] != 0 ? 1 : 0);
    std::vector<int> out((size_t)n_layer);
    for (int i = 0; i < n_layer; ++i) out[(size_t)i] = raw[(size_t)i] != 0 ? 1 : 0;
    return out;
}

Gemma4Arch::Gemma4Arch(const Gemma4Config& cfg, const std::vector<int>& kv_raw,
                       const std::vector<int>& pattern_raw)
    : cfg_(cfg) {
    if (cfg_.n_layer <= 0)
        throw std::invalid_argument("n_layer must be positive");
    // A zero cap means "no softcap" in this ABI; a NEGATIVE one is a real error
    // (the Python raises for cap <= 0; 0 as a sentinel would otherwise be valid).
    if (cfg_.softcap < 0.0f)
        throw std::invalid_argument("softcap must not be negative");
    kv_ = kv_heads_for(kv_raw, cfg_.n_layer);
    pattern_ = pattern_for(pattern_raw, cfg_.n_layer);

    for (int i = 0; i < cfg_.n_layer; ++i) {
        if (!pattern_.empty() && pattern_[(size_t)i])
            window_layers_.push_back(i);
        else
            global_layers_.push_back(i);
    }
}

bool Gemma4Arch::is_swa(int layer) const {
    if (pattern_.empty()) return false;
    if (layer < 0 || layer >= cfg_.n_layer) return false;
    return pattern_[(size_t)layer] != 0;
}

int Gemma4Arch::head_dim_at(int layer) const {
    // gemma4 rotates a different width on each layer kind: 256 windowed, 512 full.
    if (cfg_.head_dim_swa > 0 && is_swa(layer)) return cfg_.head_dim_swa;
    return cfg_.head_dim;
}

float Gemma4Arch::rope_base_at(int layer) const {
    if (cfg_.rope_base_swa > 0.0f && is_swa(layer)) return cfg_.rope_base_swa;
    return cfg_.rope_base;
}

int Gemma4Arch::rope_dims_at(int layer) const {
    if (cfg_.rope_dims_swa > 0 && is_swa(layer)) return cfg_.rope_dims_swa;
    return cfg_.rope_dims;
}

int Gemma4Arch::kv_heads_at(int layer) const {
    if (layer < 0 || layer >= cfg_.n_layer) return -1;
    return kv_[(size_t)layer];
}

int Gemma4Arch::window_at(int layer) const {
    // A global layer MUST get 0, not the window value: passing the window to a
    // global layer silently truncates its context to 1024.
    if (cfg_.sliding_window <= 0 || pattern_.empty()) return 0;
    if (layer < 0 || layer >= cfg_.n_layer) return 0;
    return pattern_[(size_t)layer] ? cfg_.sliding_window : 0;
}

float Gemma4Arch::embed_scale() const {
    // The token embedding is scaled by sqrt(n_embd) before the blocks.
    return std::sqrt((float)cfg_.n_embd);
}

float Gemma4Arch::softcap() const {
    return cfg_.softcap > 0.0f ? cfg_.softcap : 0.0f;
}

const char* Gemma4Arch::activation() const { return "gelu"; }

const char* Gemma4Arch::block_mode() const { return "after_add"; }

int Gemma4Arch::out_scale_mode() const { return 2; }  // whole layer output

const char* const* Gemma4Arch::block_tensors(int* count) {
    if (count != nullptr) *count = kBlockTensorCount;
    return kBlockTensors;
}

void softcap_logits(float* logits, int n, float cap) {
    // cap * tanh(logits / cap). Bounds the logit range, which bounds the
    // softmax's sensitivity; without it the tail of a large vocab dominates.
    if (logits == nullptr || n < 0)
        throw std::invalid_argument("softcap_logits: bad arguments");
    if (!(cap > 0.0f))
        throw std::invalid_argument("softcap must be positive");
    for (int i = 0; i < n; ++i)
        logits[i] = cap * std::tanh(logits[i] / cap);
}

float scale_element(const float* vec, int n) {
    if (vec == nullptr || n != 1)
        throw std::invalid_argument("layer_output_scale must have exactly 1 element");
    return vec[0];
}

}  // namespace aero

// ---------------------------------------------------------------------- C ABI
// Helpers at file scope, NOT inside extern "C" (a namespace declared inside a
// linkage block gives its members C linkage and exports them as global symbols).
namespace {

const aero::Gemma4Arch* as_arch(aero_arch_t a) {
    return reinterpret_cast<const aero::Gemma4Arch*>(a);
}

void set_err(char* err, int errlen, const char* msg) {
    if (err == nullptr || errlen <= 0) return;
    std::snprintf(err, (size_t)errlen, "%s", msg);
}

std::vector<int> gather_i32(const int32_t* raw, int n) {
    std::vector<int> v;
    if (raw == nullptr || n <= 0) return v;
    v.reserve((size_t)n);
    for (int i = 0; i < n; ++i) v.push_back((int)raw[i]);
    return v;
}

}  // namespace

extern "C" {

int aero_gemma4_kv_heads_for(const int32_t* raw, int n_raw, int n_layer, int32_t* out) {
    if (out == nullptr || n_layer <= 0) return AERO_ARCH_ERR;
    try {
        std::vector<int> v = aero::Gemma4Arch::kv_heads_for(gather_i32(raw, n_raw), n_layer);
        for (size_t i = 0; i < v.size(); ++i) out[i] = (int32_t)v[i];
        return (int)v.size();
    } catch (const std::exception&) {
        return AERO_ARCH_ERR;
    }
}

int aero_gemma4_pattern_for(const int32_t* raw, int n_raw, int n_layer, int32_t* out) {
    if (out == nullptr || n_layer <= 0) return AERO_ARCH_ERR;
    try {
        std::vector<int> v = aero::Gemma4Arch::pattern_for(gather_i32(raw, n_raw), n_layer);
        // An ABSENT pattern widens to NO entries, not n_layer zeros: that is the
        // only way "the metadata key is missing" stays distinguishable from "the
        // key says every layer is global". Returning n_layer here, and indexing
        // v[i] for i < n_layer, is an out-of-bounds read (it segfaulted once).
        for (size_t i = 0; i < v.size(); ++i) out[i] = (int32_t)v[i];
        return (int)v.size();
    } catch (const std::exception&) {
        return AERO_ARCH_ERR;
    }
}

void aero_gemma4_defaults(aero_gemma4_config* c) {
    if (c == nullptr) return;
    std::memset(c, 0, sizeof(*c));
    c->n_layer = 48;
    c->n_embd = 3840;
    c->n_head = 16;
    c->head_dim = 512;        // gemma4.attention.key_length
    c->value_dim = 512;       // gemma4.attention.value_length
    c->ffn_len = 15360;       // gemma4.feed_forward_length
    c->eps = 1e-6f;           // gemma4.attention.layer_norm_rms_epsilon
    c->sliding_window = 1024; // gemma4.attention.sliding_window
    c->softcap = 30.0f;       // gemma4.final_logit_softcapping
    c->rope_base = 1000000.0f;
    c->rope_dims = 512;
    c->head_dim_swa = 256;    // gemma4.attention.key_length_swa
    c->rope_dims_swa = 256;   // gemma4.rope.dimension_count_swa
    c->rope_base_swa = 10000.0f;
    c->attn_scale = 1.0f;     // no 1/sqrt(head_dim) pre-scaling
    c->n_kv_shared = 0;
    c->has_per_layer_embd = 0;
    c->n_layer_embd = 0;
}

int aero_gemma4_default_kv(int32_t* out, int cap) {
    // 8 KV heads everywhere except the 8 global layers (5,11,..,47), which have 1.
    if (out == nullptr || cap < 48) return AERO_ARCH_ERR;
    for (int i = 0; i < 48; ++i) out[i] = ((i % 6) == 5) ? 1 : 8;
    return 48;
}

int aero_gemma4_default_pattern(int32_t* out, int cap) {
    // 5 windowed then 1 global, repeating: [T,T,T,T,T,F, ...].
    if (out == nullptr || cap < 48) return AERO_ARCH_ERR;
    for (int i = 0; i < 48; ++i) out[i] = ((i % 6) == 5) ? 0 : 1;
    return 48;
}

aero_arch_t aero_gemma4_new(const aero_gemma4_config* c, const int32_t* kv_raw,
                            int n_kv_raw, const int32_t* pattern_raw,
                            int n_pattern_raw, char* err, int errlen) {
    if (c == nullptr) {
        set_err(err, errlen, "config is null");
        return nullptr;
    }
    try {
        aero::Gemma4Config cfg;
        cfg.n_layer = c->n_layer;
        cfg.n_embd = c->n_embd;
        cfg.n_head = c->n_head;
        cfg.head_dim = c->head_dim;
        cfg.value_dim = c->value_dim;
        cfg.ffn_len = c->ffn_len;
        cfg.eps = c->eps;
        cfg.sliding_window = c->sliding_window;
        cfg.softcap = c->softcap;
        cfg.rope_base = c->rope_base;
        cfg.rope_dims = c->rope_dims;
        cfg.head_dim_swa = c->head_dim_swa;
        cfg.rope_dims_swa = c->rope_dims_swa;
        cfg.rope_base_swa = c->rope_base_swa;
        cfg.attn_scale = c->attn_scale;
        cfg.n_kv_shared = c->n_kv_shared;
        cfg.has_per_layer_embd = c->has_per_layer_embd;
        cfg.n_layer_embd = c->n_layer_embd;
        auto* a = new aero::Gemma4Arch(cfg, gather_i32(kv_raw, n_kv_raw),
                                       gather_i32(pattern_raw, n_pattern_raw));
        return reinterpret_cast<aero_arch_t>(a);
    } catch (const std::exception& e) {
        set_err(err, errlen, e.what());
        return nullptr;
    }
}

void aero_arch_free(aero_arch_t a) { delete reinterpret_cast<aero::Gemma4Arch*>(a); }

int aero_arch_n_layer(aero_arch_t a) {
    return a ? as_arch(a)->n_layer() : AERO_ARCH_ERR;
}
int aero_arch_n_embd(aero_arch_t a) {
    return a ? as_arch(a)->n_embd() : AERO_ARCH_ERR;
}
int aero_arch_n_head(aero_arch_t a) { return a ? as_arch(a)->n_head() : AERO_ARCH_ERR; }
int aero_arch_head_dim(aero_arch_t a) {
    return a ? as_arch(a)->head_dim() : AERO_ARCH_ERR;
}
int aero_arch_ffn_len(aero_arch_t a) { return a ? as_arch(a)->ffn_len() : AERO_ARCH_ERR; }
float aero_arch_eps(aero_arch_t a) { return a ? as_arch(a)->eps() : 0.0f; }

int aero_arch_is_swa(aero_arch_t a, int layer) {
    if (a == nullptr || layer < 0 || layer >= as_arch(a)->n_layer()) return AERO_ARCH_ERR;
    return as_arch(a)->is_swa(layer) ? 1 : 0;
}

int aero_arch_head_dim_at(aero_arch_t a, int layer) {
    if (a == nullptr || layer < 0 || layer >= as_arch(a)->n_layer()) return AERO_ARCH_ERR;
    return as_arch(a)->head_dim_at(layer);
}

float aero_arch_rope_base_at(aero_arch_t a, int layer) {
    if (a == nullptr || layer < 0 || layer >= as_arch(a)->n_layer()) return 0.0f;
    return as_arch(a)->rope_base_at(layer);
}

int aero_arch_rope_dims_at(aero_arch_t a, int layer) {
    if (a == nullptr || layer < 0 || layer >= as_arch(a)->n_layer()) return AERO_ARCH_ERR;
    return as_arch(a)->rope_dims_at(layer);
}

int aero_arch_kv_heads_at(aero_arch_t a, int layer) {
    if (a == nullptr || layer < 0 || layer >= as_arch(a)->n_layer()) return AERO_ARCH_ERR;
    return as_arch(a)->kv_heads_at(layer);
}

int aero_arch_window_at(aero_arch_t a, int layer) {
    if (a == nullptr || layer < 0 || layer >= as_arch(a)->n_layer()) return AERO_ARCH_ERR;
    return as_arch(a)->window_at(layer);
}

int aero_arch_window_layers(aero_arch_t a, int32_t* out, int cap) {
    if (a == nullptr || out == nullptr) return AERO_ARCH_ERR;
    const std::vector<int>& v = as_arch(a)->window_layers();
    if (cap < (int)v.size()) return AERO_ARCH_ERR;
    for (size_t i = 0; i < v.size(); ++i) out[i] = (int32_t)v[i];
    return (int)v.size();
}

int aero_arch_global_layers(aero_arch_t a, int32_t* out, int cap) {
    if (a == nullptr || out == nullptr) return AERO_ARCH_ERR;
    const std::vector<int>& v = as_arch(a)->global_layers();
    if (cap < (int)v.size()) return AERO_ARCH_ERR;
    for (size_t i = 0; i < v.size(); ++i) out[i] = (int32_t)v[i];
    return (int)v.size();
}

float aero_arch_attn_scale(aero_arch_t a) { return a ? as_arch(a)->attn_scale() : 0.0f; }
float aero_arch_embed_scale(aero_arch_t a) { return a ? as_arch(a)->embed_scale() : 0.0f; }
float aero_arch_softcap(aero_arch_t a) { return a ? as_arch(a)->softcap() : 0.0f; }
const char* aero_arch_activation(aero_arch_t a) {
    return a ? as_arch(a)->activation() : "";
}
const char* aero_arch_block_mode(aero_arch_t a) { return a ? as_arch(a)->block_mode() : ""; }
int aero_arch_out_scale_mode(aero_arch_t a) {
    return a ? as_arch(a)->out_scale_mode() : AERO_ARCH_ERR;
}

int aero_arch_softcap_logits(float* logits, int n, float cap) {
    if (logits == nullptr || n < 0 || !(cap > 0.0f)) return AERO_ARCH_ERR;
    try {
        aero::softcap_logits(logits, n, cap);
    } catch (const std::exception&) {
        return AERO_ARCH_ERR;
    }
    return 0;
}

int aero_arch_scale_of(const float* vec, int n, float* out) {
    if (vec == nullptr || out == nullptr || n != 1) return AERO_ARCH_ERR;
    try {
        *out = aero::scale_element(vec, n);
    } catch (const std::exception&) {
        return AERO_ARCH_ERR;
    }
    return 0;
}

int aero_arch_block_tensors(const char** out, int cap) {
    int count = 0;
    const char* const* names = aero::Gemma4Arch::block_tensors(&count);
    if (out == nullptr || cap < count) return AERO_ARCH_ERR;
    for (int i = 0; i < count; ++i) out[i] = names[i];
    return count;
}

}  // extern "C"
