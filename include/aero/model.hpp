// Model: the per-layer block loop, native.
//
// WHAT THIS OWNS. Everything between the token embedding and the final logits
// that is not itself a per-architecture primitive:
//
//     embed -> N x [ norm -> (attn | ssm) -> residual/norm -> FFN -> residual ]
//           -> final norm -> tied head -> logit soft-cap
//
// and, crucially, ALL the projections (q/k/v/o, gate/up/down, the head) go
// through the FUSED dequant+matmul in kernels.hpp, reading the weight bytes
// straight out of the mmap. The Python block loop paid, per matmul,
//   * a pread() of the whole tensor through a Python file handle, and
//   * a ctypes from_buffer_copy of the same bytes,
// on every one of ~145 matmuls a gemma-4-12b decode token issues. Neither copy
// exists here: GgufReader::tensor_data() is a pointer into the mapping and the
// kernel streams it in place. That is the whole point of "native end-to-end" --
// not faster arithmetic (the fused kernel was already native), but no bytes
// crossing the language boundary.
//
// WHY THE PRIMITIVES ARE PLUGGABLE. RoPE+attention and the SSM/MoE layers are
// separate modules under concurrent development. Rather than reimplement them
// (a second implementation to keep correct is worse than a seam), this module
// declares their entry points as function pointers and calls whatever is
// registered. A caller that has no native implementation registers the verified
// Python one through a ctypes callback; the block loop then runs natively with
// that one primitive delegated. When the native version lands, it registers
// instead and nothing else changes.
//
// HONESTY: if a block needs a primitive that is not registered, forward()
// REFUSES (returns 0) rather than computing something wrong. The caller falls
// back. A silent zero-filled attention output would produce finite, fluent
// nonsense -- the failure mode this project keeps paying for.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "aero/gguf.hpp"

namespace aero {

// ------------------------------------------------------------------ geometry
// Filled by the caller (which already parsed and VERIFIED the config in
// engine/model.py). This module deliberately does not re-derive
// architecture facts from the file: one parser, one set of verified rules.
struct Geom {
    // 0 = gemma4, 1 = qwen35/qwen35moe
    int arch = 0;
    int n_layer = 0;
    int n_embd = 0;
    int vocab = 0;
    float eps = 1e-6f;
    int tied_output = 1;          // head uses token_embd.weight
    float softcap = 0.0f;         // <= 0 -> no soft-cap (qwen must NOT get one)
    float embed_scale = 0.0f;     // <= 0 -> no embedding scale
    int block_on_output = 0;      // qwen: BLOCK_MODE == "on_output"
    // gemma layer_output_scale placement: 0 none, 1 deltas, 2 sum (whole output)
    int gemma_out_scale = 2;
    int threads = 1;
    int block_rows = 64;          // row block for the fp32 fallback only
};

// 0 = attention, 1 = ssm
struct LayerGeom {
    int kind = 0;
    int is_moe = 0;               // route this layer's FFN through the moe hook
    float out_scale = 1.0f;       // gemma layer_output_scale (1.0 otherwise)
};

// ------------------------------------------------------------------ primitives
// Each hook is handed the ALREADY-NORMED sublayer input (T x n_embd) and must
// write the sublayer OUTPUT (T x n_embd) -- attention output before the
// residual, SSM output before the residual. `ctx` is opaque to this module.
//
// The hook owns its own state (KV cache, recurrent state), keyed by
// (ctx, layer). This module never touches it.
using AttnFn = int (*)(void* ctx, int layer, const float* h, const int64_t* pos,
                       int T, int n_embd, float* out);
using SsmFn = int (*)(void* ctx, int layer, const float* h, int T, int n_embd,
                      float* out);
using MoeFn = int (*)(void* ctx, int layer, const float* h, int T, int n_embd,
                      float* out);

// ------------------------------------------------------------------ the model
class ModelNative {
public:
    ModelNative(const std::string& path, const Geom& g,
                const std::vector<LayerGeom>& layers, int threads);
    ~ModelNative();

    // Registered primitive hooks. Any of these may be left null; forward()
    // then refuses the layers that need them.
    void set_attn(AttnFn fn, void* ctx) { attn_ = fn; attn_ctx_ = ctx; }
    void set_ssm(SsmFn fn, void* ctx) { ssm_ = fn; ssm_ctx_ = ctx; }
    void set_moe(MoeFn fn, void* ctx) { moe_ = fn; moe_ctx_ = ctx; }

    // tokens -> logits for the LAST position (vocab floats). Returns false when
    // a needed primitive is missing or the file lacks a tensor this layout
    // requires, so the caller can fall back to another path.
    bool forward(const int32_t* tokens, int n_tokens, const int64_t* positions,
                 float* logits_out);

    const Geom& geom() const { return g_; }
    const GgufReader& reader() const { return *r_; }

private:
    bool gemma_block(int layer, int T);
    bool qwen_block(int layer, int T);
    bool ffn(int layer, int T, const float* h2, float* out);
    bool mm(const std::string& name, const float* X, float* Y, int64_t T);

    std::unique_ptr<GgufReader> r_;
    Geom g_;
    std::vector<LayerGeom> layers_;
    AttnFn attn_ = nullptr; void* attn_ctx_ = nullptr;
    SsmFn ssm_ = nullptr;   void* ssm_ctx_ = nullptr;
    MoeFn moe_ = nullptr;   void* moe_ctx_ = nullptr;

    // scratch, sized for the largest activation
    std::vector<float> x_, h_, o_, d_;      // n_embd wide
    std::vector<float> gate_, up_;          // ffn_len wide
    int64_t ffn_len_ = 0;
    const int64_t* positions_ = nullptr;
    int64_t cap_T_ = 0;
    void ensure(int64_t T);
};

const char* model_arch_name(int arch);

}  // namespace aero

// ---------------------------------------------------------------------- C ABI
extern "C" {

// Opaque handle so ctypes never sees a C++ type.
typedef void* aero_model_t;

typedef int (*aero_attn_fn)(void* ctx, int layer, const float* h,
                            const int64_t* pos, int T, int n_embd, float* out);
typedef int (*aero_ssm_fn)(void* ctx, int layer, const float* h, int T,
                           int n_embd, float* out);
typedef int (*aero_moe_fn)(void* ctx, int layer, const float* h, int T,
                           int n_embd, float* out);

// Geometry structs, mirrored field-for-field by ctypes in engine/native.py.
typedef struct {
    int32_t arch, n_layer, n_embd, vocab;
    float eps;
    int32_t tied_output;
    float softcap, embed_scale;
    int32_t block_on_output, gemma_out_scale, threads, block_rows;
} aero_geom;

typedef struct {
    int32_t kind, is_moe;
    float out_scale;
} aero_layer_geom;

// Open a model. Returns NULL on failure and writes the reason into `err`.
aero_model_t aero_model_open(const char* path, const aero_geom* g,
                             const aero_layer_geom* layers, char* err, int errlen);
void aero_model_free(aero_model_t m);

void aero_model_set_attn(aero_model_t m, aero_attn_fn fn, void* ctx);
void aero_model_set_ssm(aero_model_t m, aero_ssm_fn fn, void* ctx);
void aero_model_set_moe(aero_model_t m, aero_moe_fn fn, void* ctx);

// One pass. `tokens` n_tokens int32, `positions` n_tokens int64. Writes vocab
// floats into `logits`. Returns 1 on success, 0 on refusal (missing primitive /
// missing tensor), so the caller falls back rather than trusting a wrong result.
int aero_model_forward(aero_model_t m, const int32_t* tokens, int n_tokens,
                       const int64_t* positions, float* logits);

int aero_model_n_layer(aero_model_t m);
int aero_model_n_embd(aero_model_t m);
int aero_model_vocab(aero_model_t m);
const char* aero_model_arch(aero_model_t m);

}  // extern "C"
