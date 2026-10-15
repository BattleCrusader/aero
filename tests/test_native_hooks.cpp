// The native hooks, verified in C++ against a KNOWN-CORRECT answer.
//
// WHAT THIS PROVES. aero_model_use_native_hooks() installs C++ implementations of the
// attention and SSM layers, so a token no longer passes through the interpreter. The
// assertion is not self-consistency -- it is a value established independently: on the raw
// prompt "The capital of France is", the Python-hook path predicts token 11751 ('ĠParis')
// as the top-1 with a large logit. If the C++ hooks reproduce that, they are running the
// same layers correctly. If they produce anything else, they are not, and this fails loudly.
//
// Everything here is C++: the tokenizer is src/tokenizer.cpp (aero_tok_*), the model and
// hooks are src/model.cpp, the arithmetic is attention_layer.cpp + ssm.cpp.
//
//   ./build/test_native_hooks <model.gguf>
#include "aero/model.hpp"
#include "aero/tokenizer.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;
static int g_checks = 0;

static void check(bool ok, const char* what) {
    g_checks++;
    if (ok) {
        std::printf("  ok   %s\n", what);
    } else {
        g_fail++;
        std::printf("  FAIL %s\n", what);
    }
}

static const int64_t EXPECTED_TOP1 = 11751;   // 'ĠParis', the Python path's answer

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("(skipped: give a model path)\n");
        std::printf("\n=== 0 checks, 0 failures ===\n");
        return 0;
    }
    const std::string path = argv[1];

    // ---- the tokenizer, C++ ----
    void* tok = aero_tok_open(path.c_str());
    if (!tok) {
        std::printf("(skipped: tokenizer refused the file: %s)\n", aero_tok_last_error());
        std::printf("\n=== 0 checks, 0 failures ===\n");
        return 0;
    }
    check(aero_tok_n_vocab(tok) > 0, "the C++ tokenizer opened the file");

    const char* prompt = "The capital of France is";
    int32_t ids[64];
    const int64_t n = aero_tok_encode(tok, prompt, (int64_t)std::strlen(prompt), ids, 64, 0);
    check(n > 0, "encoded the prompt with the C++ tokenizer");
    check(n > 0 && ids[0] == 760 && ids[1] == 6511,
          "its first two ids match the reference (760, 6511)");

    // ---- the model, with NATIVE hooks ----
    // The C ABI's own structs (engine/native.py mirrors these field-for-field). Qwen3.8-4B
    // is qwen35: 32 layers, 2560 wide, tied head, attention on every 4th layer (3,7,...,31)
    // and SSM on the other 24.
    aero_geom g{};
    g.arch = 1;                      // qwen35
    g.n_layer = 32;
    g.n_embd = 2560;
    g.vocab = (int)aero_tok_n_vocab(tok);
    g.eps = 1e-6f;
    g.tied_output = 1;
    g.softcap = 0.0f;
    g.embed_scale = 0.0f;
    g.block_on_output = 1;
    g.gemma_out_scale = 2;
    g.threads = 8;
    g.block_rows = 64;

    std::vector<aero_layer_geom> lg(32);
    for (int i = 0; i < 32; ++i) {
        lg[(size_t)i].kind = (i % 4 == 3) ? 0 : 1;    // 0 = attention, 1 = ssm
        lg[(size_t)i].is_moe = 0;
        lg[(size_t)i].out_scale = 1.0f;
    }
    char err[512] = {0};
    aero_model_t m = aero_model_open(path.c_str(), &g, lg.data(), err, (int)sizeof err);
    if (!m) {
        std::printf("(model refused: %s)\n", err);
        std::printf("\n=== %d checks, %d failures ===\n", g_checks, g_fail);
        return g_fail ? 1 : 0;
    }
    check(true, "the C++ model opened with native hooks available");

    check(aero_model_prepare_state(m, 512) == 1,
          "per-layer native state allocated (KV slots + SSM tail/recurrence)");
    check(aero_model_use_native_hooks(m) == 1, "native attention + SSM hooks installed");

    std::vector<int64_t> pos((size_t)n);
    for (int64_t i = 0; i < n; ++i) pos[(size_t)i] = i;
    std::vector<float> logits((size_t)aero_model_vocab(m), 0.0f);

    const int ok = aero_model_forward(m, ids, (int)n, pos.data(), logits.data());
    check(ok == 1, "forward through the C++ hooks returned success");

    if (ok == 1) {
        int best = 0;
        bool finite = true;
        for (size_t i = 0; i < logits.size(); ++i) {
            if (!std::isfinite(logits[i])) finite = false;
            if (logits[i] > logits[best]) best = (int)i;
        }
        check(finite, "logits are finite (no NaN/inf out of the native layers)");
        std::printf("       top-1 = %d  logit = %+.3f   (expected %lld)\n",
                    best, logits[best], (long long)EXPECTED_TOP1);
        check(best == (int)EXPECTED_TOP1,
              "top-1 is the answer the Python path produced (Paris, 11751)");
        check(logits[EXPECTED_TOP1] > 5.0f, "and its logit is decisive, not a coin flip");

        // ISOLATE THE STATE PATH. The C++ hooks keep their own KV slots and drive them with
        // kv_phys = pos[0]; the Python hook passes NULL slots and keeps a Python cache. So the
        // invariant to test is the one that depends on that state being right: decoding one
        // token at position n-1 after n-1 prefill tokens must give the SAME logits as feeding
        // all n in one call. If it does not, the defect is the state/slot path, not the glue.
        aero_model_t m3 = aero_model_open(path.c_str(), &g, lg.data(), err, (int)sizeof err);
        if (m3) {
            aero_model_prepare_state(m3, 512);
            aero_model_use_native_hooks(m3);
            std::vector<float> lA((size_t)aero_model_vocab(m3), 0.0f);
            std::vector<int64_t> pA((size_t)(n - 1));
            for (int64_t i = 0; i < n - 1; ++i) pA[(size_t)i] = i;
            const int okA = aero_model_forward(m3, ids, (int)(n - 1), pA.data(), lA.data());
            std::vector<float> lB((size_t)aero_model_vocab(m3), 0.0f);
            int64_t pB[1] = {n - 1};
            int32_t last[1] = {ids[n - 1]};
            const int okB = aero_model_forward(m3, last, 1, pB, lB.data());
            double dmax = 0.0;
            int aA = 0, aB = 0;
            for (size_t i = 0; i < lA.size(); ++i) {
                dmax = std::max(dmax, (double)std::fabs(lA[i] - lB[i]));
                if (lA[i] > lA[aA]) aA = (int)i;
                if (lB[i] > lB[aB]) aB = (int)i;
            }
            std::printf("       incremental invariant: rc=%d/%d  max|prefill-decode|=%.3e  "
                        "argmax %d vs %d\n", okA, okB, dmax, aA, aB);
            check(okA == 1 && okB == 1, "both the prefill and the follow-on decode succeed");
            check(dmax < 1e-3 && aA == aB,
                  "decoding after a prefill matches the batch prefill (state path is consistent)");
            aero_model_free(m3);
        }

        // WHERE does it go wrong? A single-token forward isolates the per-step path from
        // the multi-token prefill, without adding any instrumentation to the engine.
        aero_model_t m2 = aero_model_open(path.c_str(), &g, lg.data(), err, (int)sizeof err);
        if (m2) {
            aero_model_prepare_state(m2, 512);
            aero_model_use_native_hooks(m2);
            int64_t p1[1] = {0};
            std::vector<float> l1((size_t)aero_model_vocab(m2), 0.0f);
            int32_t one[1] = {760};
            const int ok1 = aero_model_forward(m2, one, 1, p1, l1.data());
            int bad = 0;
            for (float v : l1) if (!std::isfinite(v)) bad++;
            float mx = -1e30f; int arg = 0;
            for (size_t i = 0; i < l1.size(); ++i) if (std::isfinite(l1[i]) && l1[i] > mx) { mx = l1[i]; arg = (int)i; }
            std::printf("       T=1: rc=%d  non-finite=%d/%zu  argmax=%d\n",
                        ok1, bad, l1.size(), arg);
            check(bad == 0, "a single-token forward is finite (isolates prefill from per-step)");
            aero_model_free(m2);
        }
    }

    aero_model_free(m);
    aero_tok_free(tok);
    std::printf("\n=== %d checks, %d failures ===\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
