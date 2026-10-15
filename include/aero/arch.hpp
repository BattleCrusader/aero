// gemma4 architecture descriptor, native. A port of engine/gemma.py.
//
// WHY A DESCRIPTOR AND NOT JUST CONSTANTS IN THE BLOCK LOOP. gemma4's geometry
// VARIES PER LAYER, and every one of these differences is a silent-wrongness bug
// if it is collapsed to a scalar:
//
//   * head_dim     : 256 on sliding-window layers, 512 on the full-attention ones
//                    (gemma4.attention.key_length = 512, ..._swa = 256 in the file);
//   * rope base    : 10000 on SWA layers, 1000000 on the full ones
//                    (rope.freq_base = 1000000, rope.freq_base_swa = 10000);
//   * rope dims    : 256 SWA vs 512 full (rope.dimension_count / ..._swa);
//   * kv heads     : 8 SWA vs 1 full -- and in the FILE this is a 48-element
//                    LIST, not a scalar. A scalar read applies layer 0's value to
//                    the whole model and produces fluent nonsense;
//   * window       : 1024 on SWA layers and NO window (0) on the global ones.
//                    Handing a global layer the window silently truncates its
//                    context to 1024.
//
// Verified against the real file (`gemma-4-12b-it-IQ4_XS`, 48 layers):
//   block_count 48, embedding_length 3840, head_count 16,
//   key_length 512 / key_length_swa 256, value_length 512 / 256,
//   feed_forward_length 15360, sliding_window 1024,
//   sliding_window_pattern = [T,T,T,T,T,F, ...] (40 windowed, 8 global: 5,11,..,47),
//   head_count_kv = [8,8,8,8,8,1, ...] (1 exactly on the 8 global layers),
//   final_logit_softcapping 30.0, layer_norm_rms_epsilon ~1e-6,
//   attention.shared_kv_layers 0, embedding_length_per_layer_input 0.
//
// FACTS THAT ARE NOT IN THE FILE BUT ARE PART OF THE ARCHITECTURE, and which the
// block loop must honour (each one is a different function if got wrong):
//   * activation is GELU, not SiLU;
//   * the token embedding is scaled by sqrt(n_embd) before the blocks;
//   * `layer_output_scale` multiplies the WHOLE layer output (both the attention
//     and the FFN contributions), not just one of the two deltas;
//   * the ordering is 'after_add': the attention output is normalised and then
//     added -- post_attention_norm / post_ffw_norm sit AFTER the add, unlike
//     qwen's 'on_output' placement;
//   * the attention scale is 1.0 -- there is NO 1/sqrt(head_dim) pre-scaling;
//   * logits are soft-capped: logits = cap * tanh(logits / cap).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace aero {

// What the block loop needs per layer; the two arrays are one entry per layer.
struct Gemma4Config {
    int n_layer = 0;
    int n_embd = 0;
    int n_head = 0;
    int head_dim = 0;              // full-attention head dim (512 in the file)
    int value_dim = 0;
    int ffn_len = 0;
    float eps = 1e-6f;
    int sliding_window = 0;        // 0 = no sliding window
    float softcap = 0.0f;          // 0 = none; < 0 is refused (mirrors 0.0 raising)
    float rope_base = 1000000.0f;  // full-attention rotary base
    int rope_dims = 512;
    // SWA variants. <= 0 means "the file did not give a separate SWA value", in
    // which case the full-attention value is used for every layer.
    int head_dim_swa = 256;
    int rope_dims_swa = 256;
    float rope_base_swa = 10000.0f;
    float attn_scale = 1.0f;       // 1.0 for gemma4: no 1/sqrt(D)
    int n_kv_shared = 0;
    int has_per_layer_embd = 0;
    int n_layer_embd = 0;
};

// Per-layer geometry answers. Constructed once from the file's metadata (the
// caller already parsed and VERIFIED it -- this module deliberately does not
// re-derive architecture facts from a path; one parser, one set of rules, the
// same stance include/aero/model.hpp takes).
class Gemma4Arch {
public:
    // `kv_raw` / `pattern_raw` are the RAW metadata values: a 1-element vector is
    // the scalar spelling and widens to every layer; an n_layer-element vector is
    // the per-layer list. An empty `pattern_raw` means "no pattern" (every layer
    // is global). Throws std::invalid_argument on a length mismatch.
    Gemma4Arch(const Gemma4Config& cfg, const std::vector<int>& kv_raw,
               const std::vector<int>& pattern_raw);

    static std::vector<int> kv_heads_for(const std::vector<int>& raw, int n_layer);
    static std::vector<int> pattern_for(const std::vector<int>& raw, int n_layer);

    const Gemma4Config& config() const { return cfg_; }
    int n_layer() const { return cfg_.n_layer; }
    int n_embd() const { return cfg_.n_embd; }
    int n_head() const { return cfg_.n_head; }
    int head_dim() const { return cfg_.head_dim; }
    int ffn_len() const { return cfg_.ffn_len; }
    float eps() const { return cfg_.eps; }

    bool is_swa(int layer) const;
    int head_dim_at(int layer) const;
    float rope_base_at(int layer) const;
    int rope_dims_at(int layer) const;
    int kv_heads_at(int layer) const;
    // Window for this layer, or 0 for a global layer. A global layer MUST get 0,
    // never the window value.
    int window_at(int layer) const;

    const std::vector<int>& kv_heads() const { return kv_; }
    const std::vector<int>& sliding_pattern() const { return pattern_; }
    const std::vector<int>& window_layers() const { return window_layers_; }
    const std::vector<int>& global_layers() const { return global_layers_; }

    float attn_scale() const { return cfg_.attn_scale; }
    float embed_scale() const;      // sqrt(n_embd)
    float softcap() const;          // 0 when there is none
    const char* activation() const;    // "gelu" -- NOT silu
    const char* block_mode() const;    // "after_add"
    int out_scale_mode() const;        // 2 = whole layer output

    // gemma4's block tensors, in the order the block consumes them, WITHOUT the
    // "blk.<layer>." prefix. `count` is the array length.
    static const char* const* block_tensors(int* count);

private:
    Gemma4Config cfg_;
    std::vector<int> kv_;
    std::vector<int> pattern_;
    std::vector<int> window_layers_;
    std::vector<int> global_layers_;
};

// cap * tanh(logits / cap). Throws std::invalid_argument for cap <= 0, matching
// engine.gemma.softcap_logits. `cap` is the raw value; pass the config's
// softcap() which is 0 for "none" and handled by the caller as a no-op.
void softcap_logits(float* logits, int n, float cap);

// layer_output_scale is stored as a 1-element tensor; broadcasting a vector of
// the wrong length would silently change the block's output scale. Throws
// std::invalid_argument unless n == 1.
float scale_element(const float* vec, int n);

}  // namespace aero

// ---------------------------------------------------------------------- C ABI
extern "C" {

// Out-of-range layer / unsupported query. Distinct from every legitimate answer
// (head_dim and rope dims are positive, kv heads are >= 1).
#define AERO_ARCH_ERR (-1)

// Mirrored field-for-field by ctypes in the test. 0 means "the file did not say".
typedef struct {
    int32_t n_layer, n_embd, n_head, head_dim, value_dim, ffn_len;
    float eps;
    int32_t sliding_window;
    float softcap;
    float rope_base;
    int32_t rope_dims;
    int32_t head_dim_swa, rope_dims_swa;
    float rope_base_swa;
    float attn_scale;
    int32_t n_kv_shared, has_per_layer_embd, n_layer_embd;
} aero_gemma4_config;

typedef void* aero_arch_t;

// Wideners. A 1-element `raw` is the scalar spelling. Both return the number of
// entries written, or a negative AERO_ARCH_ERR on a bad length / bad args; `out`
// must hold n_layer entries.
//   * kv heads are REQUIRED: a scalar widens to n_layer, a list must already be
//     n_layer long, and an absent value is refused -- so the return is n_layer.
//   * the sliding pattern may be ABSENT (empty `raw`), and then the answer is 0
//     entries: "the key is missing" must stay distinguishable from "the key says
//     every layer is global". Do not read n_layer entries off a 0 return.
int aero_gemma4_kv_heads_for(const int32_t* raw, int n_raw, int n_layer, int32_t* out);
int aero_gemma4_pattern_for(const int32_t* raw, int n_raw, int n_layer, int32_t* out);

// gemma-4-12b's geometry, read off the real file (see the header comment).
// kv/pattern are NOT filled here -- they are per-layer lists, pass them to
// aero_gemma4_new. Use aero_gemma4_default_kv/pattern for the 12b values.
void aero_gemma4_defaults(aero_gemma4_config* c);
int aero_gemma4_default_kv(int32_t* out, int cap);       // 48 entries, 8/1 split
int aero_gemma4_default_pattern(int32_t* out, int cap);  // 48 entries, 5-on-1-off

// Build a descriptor. Returns NULL and writes a reason into `err` on bad
// geometry (a kv/PATTERN length that is neither 1 nor n_layer, a missing
// head_count_kv, a negative softcap). The caller keeps ownership of the arrays.
aero_arch_t aero_gemma4_new(const aero_gemma4_config* c, const int32_t* kv_raw,
                            int n_kv_raw, const int32_t* pattern_raw,
                            int n_pattern_raw, char* err, int errlen);
void aero_arch_free(aero_arch_t a);

int aero_arch_n_layer(aero_arch_t a);
int aero_arch_n_embd(aero_arch_t a);
int aero_arch_n_head(aero_arch_t a);
int aero_arch_head_dim(aero_arch_t a);
int aero_arch_ffn_len(aero_arch_t a);
float aero_arch_eps(aero_arch_t a);

int aero_arch_is_swa(aero_arch_t a, int layer);
int aero_arch_head_dim_at(aero_arch_t a, int layer);
float aero_arch_rope_base_at(aero_arch_t a, int layer);
int aero_arch_rope_dims_at(aero_arch_t a, int layer);
int aero_arch_kv_heads_at(aero_arch_t a, int layer);
int aero_arch_window_at(aero_arch_t a, int layer);   // 0 = global layer

// Layer lists. Return the count written, or AERO_ARCH_ERR when `cap` is too
// small (nothing is written in that case).
int aero_arch_window_layers(aero_arch_t a, int32_t* out, int cap);
int aero_arch_global_layers(aero_arch_t a, int32_t* out, int cap);

float aero_arch_attn_scale(aero_arch_t a);
float aero_arch_embed_scale(aero_arch_t a);
float aero_arch_softcap(aero_arch_t a);
const char* aero_arch_activation(aero_arch_t a);
const char* aero_arch_block_mode(aero_arch_t a);
int aero_arch_out_scale_mode(aero_arch_t a);

// In-place softcap. Returns 0 on success, AERO_ARCH_ERR on a non-positive cap
// or bad arguments (n < 0, NULL). A cap of 0 is "no cap" only through
// aero_arch_softcap()/the caller; this function refuses it, as the Python does.
int aero_arch_softcap_logits(float* logits, int n, float cap);

// One-element accessor for layer_output_scale. Returns 0 on success.
int aero_arch_scale_of(const float* vec, int n, float* out);

// The block tensor suffixes, in consumption order (5 of them). Returns the count.
int aero_arch_block_tensors(const char** out, int cap);

}  // extern "C"
