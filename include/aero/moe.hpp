// Mixture-of-Experts FFN inner loops, as a C ABI.
//
// Ported from engine/moe.py. Weight loading stays in Python (the routed experts
// are read per-expert on demand through the row reader — that is the whole memory
// argument). What moves here is the per-token arithmetic: the top-k routing, the
// silu(gate)*up activation, the routing-weighted expert accumulation and the
// shared-expert path.
#pragma once

#include <cstdint>

extern "C" {

// Top-k routing, matching engine/moe.py route(). logits: (n_expert,).
// Writes idx_out (k,) and w_out (k,). Returns 0 on success, -1 if k is out of
// range. idx_out is int64 to match the numpy return type.
int aero_moe_route(const float* logits, int n_expert, int k, int norm_topk_prob,
                   int64_t* idx_out, float* w_out);

// One expert's weighted contribution:
//   acc += wt * ( down @ ( silu(gate @ h) * (up @ h) ) )
// h: (n_embd,); gate, up: (ffn_len, n_embd); down: (n_embd, ffn_len); acc: (n_embd,).
void aero_moe_expert_accum(const float* h, int n_embd,
                           const float* gate, const float* up, const float* down,
                           int ffn_len, float wt, float* acc);

// The shared expert over a whole block:
//   out += sigmoid(h @ gate_inp^T) * ( down @ ( silu(gate @ h^T) * (up @ h^T) ) )
// h: (T, n_embd); gate_inp: (1, n_embd); gate, up: (ffn_len, n_embd);
// down: (n_embd, ffn_len); out: (T, n_embd), added into.
void aero_moe_shared_accum(const float* h, int T, int n_embd,
                           const float* shexp_gate_inp, const float* shexp_gate,
                           const float* shexp_up, const float* shexp_down,
                           int ffn_len, float* out);

// Expert-row reader supplied by the caller (Python). `which`: 0 gate, 1 up,
// 2 down. Must fill `out` with the FULL expert matrix:
//   gate/up -> ffn_len * n_embd floats (row-major [ffn_len, n_embd])
//   down     -> n_embd * ffn_len floats (row-major [n_embd, ffn_len])
// Returns 0 on success, non-zero to abort (which makes the block return -1).
typedef int (*aero_moe_fetch_fn)(void* ctx, int which, int expert, float* out);

// Full MoE FFN block, matching engine/moe.py moe_ffn().
//   h: (T, n_embd)              router_logits: (T, n_expert)   out: (T, n_embd)
//   shexp_* may be NULL when has_shared is 0.
// Returns 0 on success, -1 on a fetch failure.
int aero_moe_ffn_block(const float* h, const float* router_logits, int T,
                       int n_embd, int n_expert, int n_expert_used, int ffn_len,
                       int norm_topk_prob, int has_shared,
                       const float* shexp_gate_inp, const float* shexp_gate,
                       const float* shexp_up, const float* shexp_down,
                       aero_moe_fetch_fn fetch, void* ctx, float* out);

}  // extern "C"
