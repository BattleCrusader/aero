// The native block loop, checked against an independently written scalar
// reference on a synthetic GGUF.
//
// WHY A SYNTHETIC FILE. The block loop's failure mode is not a crash: it is
// finite, plausible, wrong output (the wrong side of `post_attention_norm`, the
// wrong activation, the layer_output_scale applied to the deltas instead of the
// sum). None of those are visible in a smoke test, and none are visible by
// reading the code back. So the reference here is written out separately, as
// scalar loops over the same weights, and the two are compared numerically.
//
// The file is written out in full (F32 tensors, 8-wide, 2 layers), so this runs
// hermetically -- no real model, no Python -- and exercises the fp32 fallback
// path of the projection routine as well as the layouts.
#include "aero/model.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

static int g_checks = 0, g_fail = 0;
static void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) { ++g_fail; std::printf("  FAIL %s\n", what.c_str()); }
    else std::printf("  ok   %s\n", what.c_str());
}
static void banner(const char* s) { std::printf("\n%s\n", s); }



// ------------------------------------------------------------------ synthetic file
struct Buf {
    std::vector<uint8_t> d;
    void raw(const void* p, size_t n) {
        const uint8_t* q = (const uint8_t*)p;
        d.insert(d.end(), q, q + n);
    }
    void u32(uint32_t v) { raw(&v, 4); }
    void u64(uint64_t v) { raw(&v, 8); }
    void i64(int64_t v) { raw(&v, 8); }
    void str(const std::string& s) { u64(s.size()); raw(s.data(), s.size()); }
    void pad(size_t a) { while (d.size() % a) d.push_back(0); }
};

struct T { std::string name; std::vector<int64_t> dims; std::vector<float> data; };

// ---- ABLATION HOOK (temporary diagnostic) -----------------------------------
// Zeroes a named tensor family so a stage can be neutralised WITHOUT rebuilding.
// A numeric bug in a multi-stage pipeline is localised by switching stages off: the
// first ablation that makes the two sides agree identifies the broken stage. This
// beats reading the two implementations back and forth, which is how several turns
// were already spent failing to find it.
static void apply_ablation(std::unordered_map<std::string, T>& w, const char* which) {
    if (!which || !*which) return;
    const std::string z = which;
    for (auto& kv : w) {
        if (kv.first.find(z) != std::string::npos)
            std::fill(kv.second.data.begin(), kv.second.data.end(), 0.0f);
    }
}

// Weights are deterministic (an LCG), so a failure reproduces exactly.
static std::unordered_map<std::string, T> build_weights(int n_embd, int ffn,
                                                        int vocab, int n_layer) {
    uint64_t s = 12345;
    auto rnd = [&]() { s = s * 6364136223846793005ULL + 1442695040888963407ULL;
                       return ((int)((s >> 33) & 0xffff) / 32768.0f - 1.0f) * 0.7f; };
    std::unordered_map<std::string, T> w;
    auto add = [&](const std::string& n, std::vector<int64_t> gguf_dims, int64_t n_el) {
        T t; t.name = n; t.dims = std::move(gguf_dims);
        t.data.resize((size_t)n_el);
        for (auto& v : t.data) v = rnd();
        w[n] = std::move(t);
    };
    // GGUF dims are fastest-varying first: a (rows, cols) matrix is [cols, rows].
    add("token_embd.weight", {n_embd, vocab}, (int64_t)vocab * n_embd);
    add("output_norm.weight", {n_embd}, n_embd);
    for (int i = 0; i < n_layer; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        add(p + "attn_norm.weight", {n_embd}, n_embd);
        add(p + "post_attention_norm.weight", {n_embd}, n_embd);
        add(p + "ffn_norm.weight", {n_embd}, n_embd);
        add(p + "post_ffw_norm.weight", {n_embd}, n_embd);
        add(p + "layer_output_scale.weight", {1}, 1);
        add(p + "ffn_gate.weight", {n_embd, ffn}, (int64_t)ffn * n_embd);
        add(p + "ffn_up.weight", {n_embd, ffn}, (int64_t)ffn * n_embd);
        add(p + "ffn_down.weight", {ffn, n_embd}, (int64_t)n_embd * ffn);
    }
    return w;
}

static std::string write_gguf(const std::string& path,
                              std::unordered_map<std::string, T>& w) {
    const size_t ALIGN = 32;
    Buf b;
    b.raw("GGUF", 4);
    b.u32(3);
    b.u64(w.size());
    b.u64(2);
    // kv: general.architecture (string), general.alignment (uint32)
    b.str("general.architecture"); b.u32(8); b.str("test");
    b.str("general.alignment");    b.u32(4); b.u32((uint32_t)ALIGN);

    // offsets, each tensor aligned.
    //
    // BUG FOUND 2026-10-15: cursor advanced by data.size() — a FLOAT count — where
    // the tensor data below is written as data.size()*sizeof(float) bytes. Offsets
    // therefore OVERLAP: token_embd (64 floats = 256 B) at byte 192 runs to 448,
    // while the next tensor is placed at 256, inside token_embd's middle. Bigger
    // tensors clobbered their neighbours' data, so the reader saw partially zeroed
    // norms (e.g. blk.1.attn_norm 0/8 nonzero) and the "degenerate L1 is a no-op"
    // behaviour that sent me hunting for an engine bug that was never there.
    // Multiply by 4. The engine was computing on the data the file actually had;
    // the two sides disagreed because they read DIFFERENT BYTES.
    std::unordered_map<std::string, int64_t> off;
    int64_t cursor = 0;
    for (auto& kv : w) {
        off[kv.first] = cursor;
        cursor = ((cursor + (int64_t)kv.second.data.size() * 4 + (int64_t)ALIGN - 1) /
                  (int64_t)ALIGN) * (int64_t)ALIGN;
    }
    for (auto& kv : w) {
        const T& t = kv.second;
        b.str(t.name);
        b.u32((uint32_t)t.dims.size());
        for (int64_t d : t.dims) b.u64((uint64_t)d);
        b.u32(0);                    // GGMLType::F32
        b.i64(off[t.name]);
    }
    b.pad(ALIGN);
    // data section: append each tensor at its declared offset
    size_t written = 0;
    // write in the same order the offsets were assigned
    for (auto& kv : w) {
        const int64_t o = off[kv.first];
        while ((int64_t)written < o) { b.d.push_back(0); ++written; }
        b.raw(kv.second.data.data(), kv.second.data.size() * sizeof(float));
        written += kv.second.data.size() * sizeof(float);
    }
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) { std::printf("  cannot write %s\n", path.c_str()); return ""; }
    std::fwrite(b.d.data(), 1, b.d.size(), f);
    std::fclose(f);
    return path;
}

// ------------------------------------------------------------------ reference
// Written out separately from the block loop, deliberately: same maths, other
// code. gelu uses std::erf (exact) against the kernel's A&S form -- they agree
// to ~1e-7, far inside the tolerance.
static std::vector<float> ref_rmsnorm(const std::vector<float>& x,
                                      const std::vector<float>& w, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    const float inv = 1.0f / std::sqrt(ss / (float)n + eps);
    std::vector<float> o(n);
    for (int i = 0; i < n; ++i) o[i] = x[i] * inv * (w.empty() ? 1.0f : w[i]);
    return o;
}
static std::vector<float> ref_matvec(const T& W, const std::vector<float>& x,
                                     int rows, int cols) {
    std::vector<float> y(rows);
    for (int r = 0; r < rows; ++r) {
        float a = 0;
        for (int c = 0; c < cols; ++c) a += W.data[(size_t)r * cols + c] * x[c];
        y[r] = a;
    }
    return y;
}
static float ref_gelu(float z) {
    return 0.5f * z * (1.0f + std::erf(z * 0.7071067811865476f));
}
static float ref_silu(float z) { return z / (1.0f + std::exp(-z)); }

// stub attention: o = 0.5 * h. Deterministic and trivially reproducible.
static int stub_attn(void*, int, const float* h, const int64_t*, int T, int n, float* o) {
    for (int i = 0; i < T * n; ++i) o[i] = 0.5f * h[i];
    return 1;
}
static int stub_ssm(void*, int, const float* h, int T, int n, float* o) {
    for (int i = 0; i < T * n; ++i) o[i] = 0.25f * h[i];
    return 1;
}

struct Expect {
    std::vector<float> logits;
};

static Expect reference_forward(int arch, int gemma_scale_mode, float softcap,
                                float embed_scale, int n_embd, int ffn, int vocab,
                                int n_layer, const std::vector<int>& tokens,
                                std::unordered_map<std::string, T>& w) {
    std::vector<float> x(n_embd);
    const T& emb = w["token_embd.weight"];
    for (int i = 0; i < n_embd; ++i) x[i] = emb.data[(size_t)tokens[0] * n_embd + i];
    if (embed_scale > 0) for (auto& v : x) v *= embed_scale;

    for (int layer = 0; layer < n_layer; ++layer) {
        const std::string p = "blk." + std::to_string(layer) + ".";
        float ds = 1.0f, ss = 1.0f;
        if (gemma_scale_mode == 1) ds = w[p + "layer_output_scale.weight"].data[0];
        else if (gemma_scale_mode == 2) ss = w[p + "layer_output_scale.weight"].data[0];

        std::vector<float> h = ref_rmsnorm(x, w[p + "attn_norm.weight"].data, n_embd, 1e-6f);

        std::vector<float> o(n_embd);
        for (int i = 0; i < n_embd; ++i) o[i] = 0.5f * h[i];

        if (arch == 0) {
            std::vector<float> a = ref_rmsnorm(o, w[p + "post_attention_norm.weight"].data,
                                               n_embd, 1e-6f);
            std::vector<float> attn_out(n_embd);
            for (int i = 0; i < n_embd; ++i) attn_out[i] = x[i] + ds * a[i];
            std::vector<float> h2 = ref_rmsnorm(attn_out, w[p + "ffn_norm.weight"].data,
                                                n_embd, 1e-6f);
            std::vector<float> gt = ref_matvec(w[p + "ffn_gate.weight"], h2, ffn, n_embd);
            std::vector<float> up = ref_matvec(w[p + "ffn_up.weight"], h2, ffn, n_embd);
            std::vector<float> act(ffn);
            for (int i = 0; i < ffn; ++i) act[i] = ref_gelu(gt[i]) * up[i];
            std::vector<float> d = ref_matvec(w[p + "ffn_down.weight"], act, n_embd, ffn);

            d = ref_rmsnorm(d, w[p + "post_ffw_norm.weight"].data, n_embd, 1e-6f);
            for (int i = 0; i < n_embd; ++i) x[i] = (attn_out[i] + ds * d[i]) * ss;
        } else {
            // qwen, after_add: residual then norm-the-sum then FFN then residual
            std::vector<float> t1(n_embd);
            for (int i = 0; i < n_embd; ++i) t1[i] = x[i] + o[i];
            std::vector<float> h2 = ref_rmsnorm(t1, w[p + "post_attention_norm.weight"].data,
                                                n_embd, 1e-6f);
            std::vector<float> gt = ref_matvec(w[p + "ffn_gate.weight"], h2, ffn, n_embd);
            std::vector<float> up = ref_matvec(w[p + "ffn_up.weight"], h2, ffn, n_embd);
            std::vector<float> act(ffn);
            for (int i = 0; i < ffn; ++i) act[i] = ref_silu(gt[i]) * up[i];
            std::vector<float> d = ref_matvec(w[p + "ffn_down.weight"], act, n_embd, ffn);
            for (int i = 0; i < n_embd; ++i) x[i] = t1[i] + d[i];
        }
    }
    std::vector<float> hn = ref_rmsnorm(x, w["output_norm.weight"].data, n_embd, 1e-6f);
    std::vector<float> lg = ref_matvec(w["token_embd.weight"], hn, vocab, n_embd);
    if (softcap > 0) for (auto& v : lg) v = softcap * std::tanh(v / softcap);
    return Expect{lg};
}

// ------------------------------------------------------------------ C ABI decls
// The tensor primitives the fp32 fallback calls, declared here so the test can
// exercise the exact functions (they are extern "C" in src/tensor.cpp).
extern "C" {
int aero_tensor_matmul(const float* w, const float* X, float* Y, int64_t n_tokens,
                       int64_t n_rows, int64_t n_cols, int threads);
int aero_tensor_matvec(const float* w, const float* x, float* y, int64_t n_rows,
                       int64_t n_cols, int threads);
int aero_tensor_rmsnorm(const float* x, const float* weight, float* out, int64_t n,
                        float eps);
int aero_tensor_gelu(float* v, int64_t n);
int aero_tensor_swiglu(float* gate, const float* up, int64_t n);
}

// ------------------------------------------------------------------ main
int main() {
    std::printf("=== aero native model block loop ===\n");
    const int NE = 8, FF = 4, V = 8, NL = 2;
    auto w = build_weights(NE, FF, V, NL);
    const std::string path = "build/test_model_synth.gguf";
    if (write_gguf(path, w).empty()) { std::printf("cannot write synthetic gguf\n"); return 1; }

    std::vector<int> tokens = {3};
    std::vector<int64_t> pos = {0};

    // ---- [0] the primitives the fp32 fallback calls, against the reference
    banner("[0] fallback primitives agree with the scalar reference");
    {
        std::vector<float> x(NE), wv(NE), o(NE);
        for (int i = 0; i < NE; ++i) { x[i] = 0.3f * (i + 1) - 1.0f;
                                       wv[i] = 0.1f * i + 0.5f; }
        aero_tensor_rmsnorm(x.data(), wv.data(), o.data(), NE, 1e-6f);
        auto rr = ref_rmsnorm(x, wv, NE, 1e-6f);
        float d1 = 0;
        for (int i = 0; i < NE; ++i) d1 = std::max(d1, std::fabs(o[i] - rr[i]));
        std::printf("       rmsnorm  max |cpp-ref| = %.3e\n", d1);
        check(d1 < 1e-6f, "aero_tensor_rmsnorm matches ref_rmsnorm");

        const T& W = w["blk.0.ffn_gate.weight"];   // (FF, NE)
        std::vector<float> y(FF);
        aero_tensor_matmul(W.data.data(), x.data(), y.data(), 1, FF, NE, 1);
        auto ry = ref_matvec(W, x, FF, NE);
        float d2 = 0;
        for (int i = 0; i < FF; ++i) d2 = std::max(d2, std::fabs(y[i] - ry[i]));
        std::printf("       matmul   max |cpp-ref| = %.3e\n", d2);
        check(d2 < 1e-6f, "aero_tensor_matmul matches ref_matvec");

        std::vector<float> g{0.1f, -0.4f, 1.2f, 0.0f}, u{1.0f, 2.0f, -0.5f, 0.3f};
        std::vector<float> rg = g;
        aero_tensor_swiglu(g.data(), u.data(), (int64_t)g.size());
        for (size_t i = 0; i < g.size(); ++i)
            check(std::fabs(g[i] - ref_silu(rg[i]) * u[i]) < 1e-6f,
                  std::string("swiglu component ") + std::to_string(i));
    }

    // ---- [1] honesty: with no primitive registered, forward REFUSES
    banner("[1] forward refuses rather than computing without a primitive");
    {
        aero_geom g{};
        g.arch = 0; g.n_layer = NL; g.n_embd = NE; g.vocab = V;
        g.eps = 1e-6f; g.tied_output = 1; g.softcap = 0; g.embed_scale = 0;
        g.block_on_output = 0; g.gemma_out_scale = 2; g.threads = 1; g.block_rows = 64;
        std::vector<aero_layer_geom> lg(NL);
        for (int i = 0; i < NL; ++i) { lg[i].kind = 0; lg[i].is_moe = 0;
                                       lg[i].out_scale = 1.0f; }
        char err[256] = {0};
        aero_model_t m = aero_model_open(path.c_str(), &g, lg.data(), err, sizeof(err));
        check(m != nullptr, "model opens on the synthetic file");
        std::vector<float> logits(V, 0.0f);
        check(aero_model_forward(m, tokens.data(), 1, pos.data(), logits.data()) == 0,
              "forward returns 0 with no attention hook registered");
        aero_model_set_attn(m, stub_attn, nullptr);
        check(aero_model_forward(m, tokens.data(), 1, pos.data(), logits.data()) == 1,
              "forward succeeds once a hook is registered");
        check(aero_model_n_embd(m) == NE && aero_model_vocab(m) == V,
              "geometry accessors report the file's dims");
        aero_model_free(m);
    }

    // ---- [2] gemma4 layout
    banner("[2] gemma4 layout matches the scalar reference");
    {
        const float scale = 1.3f, softcap = 30.0f, escale = 2.0f;
        w["blk.0.layer_output_scale.weight"].data[0] = scale;
        w["blk.1.layer_output_scale.weight"].data[0] = scale;
        apply_ablation(w, std::getenv("AERO_ABLATE"));
        write_gguf(path, w);
        Expect e = reference_forward(0, 2, softcap, escale, NE, FF, V, NL, tokens, w);

        aero_geom g{};
        g.arch = 0; g.n_layer = NL; g.n_embd = NE; g.vocab = V;
        g.eps = 1e-6f; g.tied_output = 1; g.softcap = softcap; g.embed_scale = escale;
        g.block_on_output = 0; g.gemma_out_scale = 2; g.threads = 1; g.block_rows = 2;
        std::vector<aero_layer_geom> lg(NL);
        for (int i = 0; i < NL; ++i) { lg[i].kind = 0; lg[i].is_moe = 0;
                                       lg[i].out_scale = scale; }
        char err[256] = {0};
        aero_model_t m = aero_model_open(path.c_str(), &g, lg.data(), err, sizeof(err));
        aero_model_set_attn(m, stub_attn, nullptr);
        std::vector<float> logits(V, 0.0f);
        int ok = aero_model_forward(m, tokens.data(), 1, pos.data(), logits.data());
        check(ok == 1, "gemma forward returns 1");
        float maxd = 0;
        for (int i = 0; i < V; ++i) maxd = std::max(maxd, std::fabs(logits[i] - e.logits[i]));
        std::printf("       max |cpp - ref| = %.3e\n", maxd);
        for (int i = 0; i < V; ++i)
            std::printf("       [%d] cpp=% .6f ref=% .6f\n", i, logits[i], e.logits[i]);
        check(maxd < 1e-4f, "gemma logits match the reference (<1e-4)");
        aero_model_free(m);
    }

    // ---- [3] qwen layout, after_add, no soft-cap (the file did not ask)
    banner("[3] qwen layout (after_add, no soft-cap) matches the reference");
    {
        Expect e = reference_forward(1, 2, 0.0f, 0.0f, NE, FF, V, NL, tokens, w);
        aero_geom g{};
        g.arch = 1; g.n_layer = NL; g.n_embd = NE; g.vocab = V;
        g.eps = 1e-6f; g.tied_output = 1; g.softcap = 0; g.embed_scale = 0;
        g.block_on_output = 0; g.gemma_out_scale = 2; g.threads = 1; g.block_rows = 2;
        std::vector<aero_layer_geom> lg(NL);
        for (int i = 0; i < NL; ++i) { lg[i].kind = 0; lg[i].is_moe = 0;
                                       lg[i].out_scale = 1.0f; }
        char err[256] = {0};
        aero_model_t m = aero_model_open(path.c_str(), &g, lg.data(), err, sizeof(err));
        aero_model_set_attn(m, stub_attn, nullptr);
        std::vector<float> logits(V, 0.0f);
        check(aero_model_forward(m, tokens.data(), 1, pos.data(), logits.data()) == 1,
              "qwen forward returns 1");
        float maxd = 0;
        for (int i = 0; i < V; ++i) maxd = std::max(maxd, std::fabs(logits[i] - e.logits[i]));
        std::printf("       max |cpp - ref| = %.3e\n", maxd);
        check(maxd < 1e-4f, "qwen logits match the reference (<1e-4)");

        // the soft-cap must be applied when the file DOES ask for it, and must
        // change the result -- i.e. it is not silently dropped either
        std::vector<float> raw = logits;
        g.softcap = 30.0f;
        aero_model_t m2 = aero_model_open(path.c_str(), &g, lg.data(), err, sizeof(err));
        aero_model_set_attn(m2, stub_attn, nullptr);
        std::vector<float> capped(V, 0.0f);
        aero_model_forward(m2, tokens.data(), 1, pos.data(), capped.data());
        bool differs = false, all_bounded = true;
        for (int i = 0; i < V; ++i) {
            if (std::fabs(capped[i] - raw[i]) > 1e-6f) differs = true;
            if (std::fabs(capped[i]) > 30.0f + 1e-4f) all_bounded = false;
        }
        // 'raw' came from a softcap=0 geometry, 'capped' from softcap=30, so the two
        // must DIFFER; the previous assertion (!differs) demanded a softcap do
        // nothing -- it asserted the very bug it was written to catch.
        check(differs, "softcap=30 CHANGES the logits (not silently dropped)");
        check(all_bounded, "softcap=30 keeps every logit within +/-cap");
        aero_model_free(m2);
        aero_model_free(m);
    }


    // ---- [5] DEGENERATE WEIGHTS: the loop's wiring, with no arithmetic to hide in
    //
    // Zero every projection. Then attention contributes 0 (stub: 0.5*h), the FFN
    // contributes 0, and each layer reduces to:
    //
    //     attn_out = x + ds * post_attention_norm(0.5 * attn_norm(x))
    //     layer_out = (attn_out + ds * post_ffw_norm(0)) * ss
    //
    // with ffn_down zeroed, post_ffw_norm(0) is a fixed vector, so the expectation
    // is computable independently of any matvec. If the C++ disagrees HERE, the
    // block wiring is wrong; if it agrees, the fault is in the arithmetic path.
    banner("[5] degenerate weights isolate the loop's wiring");
    {
        auto w2 = build_weights(NE, FF, V, NL);
        for (auto& kv : w2) {
            if (kv.first.find("token_embd") == std::string::npos &&
                kv.first.find("output_norm") == std::string::npos &&
                kv.first.find("_norm.weight") == std::string::npos)
                std::fill(kv.second.data.begin(), kv.second.data.end(), 0.0f);
        }
        write_gguf(path, w2);

        const float escale = 0.0f;      // no embed scale: keep the expectation simple
        aero_geom g{};
        g.arch = 1; g.n_layer = NL; g.n_embd = NE; g.vocab = V;
        g.eps = 1e-6f; g.tied_output = 1; g.softcap = 0; g.embed_scale = escale;
        g.block_on_output = 0; g.gemma_out_scale = 0; g.threads = 1; g.block_rows = 2;
        std::vector<aero_layer_geom> lg2(NL);
        for (int i = 0; i < NL; ++i) { lg2[i].kind = 0; lg2[i].is_moe = 0; lg2[i].out_scale = 1.0f; }
        char err2[256] = {0};
        aero_model_t m2 = aero_model_open(path.c_str(), &g, lg2.data(), err2, sizeof(err2));
        aero_model_set_attn(m2, stub_attn, nullptr);
        std::vector<float> lg(V, 0.0f);
        const int okd = aero_model_forward(m2, tokens.data(), 1, pos.data(), lg.data());
        check(okd == 1, "degenerate forward returns 1");

        // independent expectation, computed here in scalar code with no matvec.
        // It must mirror the engine's ACTUAL after_add wiring, which is (and
        // matches engine/model.py's):
        //     x += attn_out(s);  h2 = post_attention_norm(x);  x += ffn(h2)
        // My first version of this expectation used a THIRD layout -- feeding the
        // FFN an un-normalised sum and post_ffw_norm'ing a zero FFN output --
        // which models no architecture and guaranteed a mismatch.
        std::vector<float> x(NE);
        { const T& e = w2["token_embd.weight"];
          for (int i = 0; i < NE; ++i) x[i] = e.data[(size_t)tokens[0] * NE + i]; }
        for (int l = 0; l < NL; ++l) {
            const std::string p2 = "blk." + std::to_string(l) + ".";
            std::vector<float> h = ref_rmsnorm(x, w2[p2 + "attn_norm.weight"].data, NE, 1e-6f);
            std::vector<float> o(NE);
            for (int i = 0; i < NE; ++i) o[i] = 0.5f * h[i];        // stub attention
            for (int i = 0; i < NE; ++i) x[i] += o[i];             // residual
            std::vector<float> h2 = ref_rmsnorm(x, w2[p2 + "post_attention_norm.weight"].data, NE, 1e-6f);
            // every FFN projection is zero: gv = 0, uv = 0, act = silu(0)*0 = 0,
            // ffn_down @ 0 = 0, so the FFN contributes exactly nothing.
            for (int i = 0; i < NE; ++i) x[i] += 0.0f;
        }
        std::vector<float> hn = ref_rmsnorm(x, w2["output_norm.weight"].data, NE, 1e-6f);
        std::vector<float> want = ref_matvec(w2["token_embd.weight"], hn, V, NE);
        float md = 0;
        for (int i = 0; i < V; ++i) md = std::max(md, std::fabs(lg[i] - want[i]));
        std::printf("       degenerate max |cpp - expect| = %.3e\n", md);
        for (int i = 0; i < V; ++i)
            std::printf("       [%d] cpp=% .6f expect=% .6f\n", i, lg[i], want[i]);
        check(md < 1e-5f, "degenerate-weights wiring matches the hand expectation");
        aero_model_free(m2);
    }

    // ---- [4] qwen SSM layer routes through the ssm hook
    banner("[4] an ssm layer routes through the ssm hook, not attention");
    {
        aero_geom g{};
        g.arch = 1; g.n_layer = 1; g.n_embd = NE; g.vocab = V;
        g.eps = 1e-6f; g.tied_output = 1; g.softcap = 0; g.embed_scale = 0;
        g.block_on_output = 0; g.gemma_out_scale = 2; g.threads = 1; g.block_rows = 2;
        std::vector<aero_layer_geom> lg(1);
        lg[0] = {1, 0, 1.0f};      // kind = ssm
        char err[256] = {0};
        aero_model_t m = aero_model_open(path.c_str(), &g, lg.data(), err, sizeof(err));
        std::vector<float> logits(V, 0.0f);
        aero_model_set_attn(m, stub_attn, nullptr);
        check(aero_model_forward(m, tokens.data(), 1, pos.data(), logits.data()) == 0,
              "ssm layer refuses with only an attention hook registered");
        aero_model_set_ssm(m, stub_ssm, nullptr);
        check(aero_model_forward(m, tokens.data(), 1, pos.data(), logits.data()) == 1,
              "ssm layer runs once the ssm hook is registered");
        aero_model_free(m);
    }

    std::printf("\n=== %d checks, %d failures ===\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
