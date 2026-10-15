// The native ATTENTION layer: a qwen35-style attention layer in one C entry point.
// Replaces the Python attention() hook hop. Includes projections (quantised,
// fused), per-head q/k rms-norms, PARTIAL rope, GQA, causal + windowed attention
// over the KV cache, the gate, and the output projection.
//
// CONTRACT (mirror of engine/transformer.py's attention(); a deviation is a bug):
//   * qgate_interleave: q_rows == 2*H*D as PER-HEAD PAIRS; q = [:, :, 0, :],
//     gate = [:, :, 1, :], gate applied as SIGMOID (the python default branch).
//   * q/k norms: rms over head_dim BEFORE rope.
//   * rope PARTIAL: rope_dims of head_dim rotate, the rest pass through.
//   * V := W_v x, or V := K when v_rows == 0 (gemma4 global layers).
//   * GQA: head h reads kv head h / (H/HK); causal; optional slide window.
//   * KV quantisers are the NATIVE ones (src/kv.cpp, byte-identical to python).
#pragma once

#include "aero/kernels.hpp"
#include "aero/kv.hpp"

#include <cmath>
#include <cstring>
#include <vector>

namespace aero {

enum class GateMode : int { NONE = 0, INTERLEAVED_SIGMOID = 1, SPLIT_SILU = 2 };

struct AttentionLayerArgs {
    const float* x;
    int64_t T;
    int64_t n_embd;
    const int32_t* qpos;

    DType q_dt;  const void* q_w;  int64_t q_rows;
    bool has_gate;
    DType k_dt;  const void* k_w;  int64_t k_rows;
    DType v_dt;  const void* v_w;  int64_t v_rows;
    DType o_dt;  const void* o_w;  int64_t o_rows;

    const float* q_norm;
    const float* k_norm;
    bool v_weightless_norm;
    float eps;

    float rope_base;
    int64_t rope_dims;
    const float* freq_factors;
    int64_t n_ff;

    int64_t n_head;
    int64_t n_head_kv;
    int64_t head_dim;
    int64_t value_dim;
    float scale;
    int window;

    GateMode gate;
    int threads;

    KvSlot* k_slot;
    KvSlot* v_slot;
    int64_t kv_phys;
};

inline bool attention_layer(const AttentionLayerArgs& a, float* out);

}  // namespace aero
