// Gated DeltaNet (SSM) inner loops, as a C ABI.
//
// Ported from engine/ssm.py. The Python in that file is the reference: it encodes
// seven conventions (per-head L2 of q/k with the scale on q, the silu*gate output
// gate, the causal depthwise conv tap order, the v-head -> k-head grouping
// direction) that are a CONVENTION, not data in the file. Every one of them is
// reproduced here verbatim so this can be dropped in for the numpy path without
// changing a single output.
//
// Weight LOADING stays in Python (it is I/O). What moves here is the per-token
// arithmetic: the projection matvecs, the conv, the recurrence scan, the gated
// RMSNorm and the output projection.
#pragma once

#include <cstdint>

extern "C" {

// softplus(x) = log1p(exp(-|x|)) + max(x, 0). Matches engine/ssm.py softplus().
float aero_ssm_softplus(float x);

// ONE step of the gated delta rule. q, k: (n_heads, d) ALREADY tiled (v-head h
// reads k-head h); v: (n_heads, d); alpha, beta, a_log, dt_bias: (n_heads,).
// `state` is (n_heads, d, d) and is UPDATED IN PLACE. `out` receives (n_heads, d).
// Mirrors engine/ssm.py delta_rule_step exactly.
void aero_ssm_delta_step(const float* q, const float* k, const float* v,
                         const float* alpha, const float* beta,
                         const float* a_log, const float* dt_bias,
                         float* state, int n_heads, int d, float scale, float eps,
                         float* out);

// The per-head scan over T tokens. qh, kh: (T, n_k, d); vh: (T, n_v, d);
// alpha, beta: (T, n_v). v-head h reads k-head h % n_k (the tile convention).
// `state` is (n_v, d, d), updated in place. `outs` receives (T, n_v, d).
void aero_ssm_scan(const float* qh, const float* kh, const float* vh,
                   const float* alpha, const float* beta,
                   const float* a_log, const float* dt_bias,
                   float* state, int T, int n_k, int n_v, int d,
                   float scale, float eps, float* outs);

// Causal depthwise conv1d, one step. x: (T, channels); w: (channels, kernel).
// `tail` holds the last `*tail_len` inputs (buffer must be >= (kernel-1)*channels
// floats); `*tail_len` is read and rewritten. `out` receives (T, channels).
// Tap order: w[:, 0] multiplies the OLDEST sample in the window.
void aero_ssm_conv(const float* x, const float* w, float* tail, int* tail_len,
                   float* out, int T, int channels, int kernel);

// Full SSM layer for T tokens. Returns 0 on success, -1 on a dimension
// combination it cannot honour (so the caller falls back to numpy).
//   x: (T, n_embd)                 w_qkv:  (qkv_width, n_embd)
//   w_gate: (inner_size, n_embd)   w_alpha/w_beta: (time_step_rank, n_embd)
//   a_log, dt_bias: (time_step_rank,)   conv1d: (qkv_width, conv_kernel)
//   norm: (state_size,)            w_out: (n_embd, inner_size)
//   conv_tail: >= (conv_kernel-1)*qkv_width floats, `*tail_len` read/rewritten
//   state: (n_v_heads, state_size, state_size) in/out    out: (T, n_embd)
int aero_ssm_forward(const float* x,
                     const float* w_qkv, const float* w_gate,
                     const float* w_alpha, const float* w_beta,
                     const float* a_log, const float* dt_bias,
                     const float* conv1d, const float* norm, const float* w_out,
                     int T, int n_embd, int inner_size, int state_size,
                     int n_group, int conv_kernel, int time_step_rank,
                     float eps, float norm_eps,
                     float* conv_tail, int* tail_len,
                     float* state, float* out);

}  // extern "C"
