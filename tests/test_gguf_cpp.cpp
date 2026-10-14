// GGUF reader self-test.
//
// The property that matters: this reader must agree with the PYTHON reader on the
// same file. They were written separately from the same format description, so
// agreeing on a real 2.6 GB model is meaningful evidence; agreeing with itself
// would not be.
//
// Prints a machine-readable summary of the parse so the Python side can diff it.
#include "aero/gguf.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace aero;

static int g_checks = 0, g_fails = 0;
static void check(bool ok, const char* what) {
    ++g_checks;
    std::printf(ok ? "  ok   %s\n" : "  FAIL %s\n", what);
    if (!ok) ++g_fails;
}
static void banner(const char* s) { std::printf("\n%s\n", s); }

int main(int argc, char** argv) {
    std::printf("=== aero GGUF reader ===\n");

    // ------------------------------------------------------------ type tables
    banner("[1] block geometry matches the format's own constants");
    struct { GGMLType t; int64_t bytes, values; } tbl[] = {
        {GGMLType::F32, 4, 1},      {GGMLType::F16, 2, 1},
        {GGMLType::BF16, 2, 1},    {GGMLType::Q4_0, 18, 32},
        {GGMLType::Q4_1, 20, 32},  {GGMLType::Q5_0, 22, 32},
        {GGMLType::Q5_1, 24, 32},  {GGMLType::Q8_0, 34, 32},
        {GGMLType::Q2_K, 84, 256}, {GGMLType::Q3_K, 110, 256},
        {GGMLType::Q4_K, 144, 256},{GGMLType::Q5_K, 176, 256},
        {GGMLType::Q6_K, 210, 256},{GGMLType::IQ4_XS, 136, 256},
    };
    for (auto& e : tbl) {
        char msg[96];
        std::snprintf(msg, sizeof(msg), "%s is %lld B / %lld values",
                      ggml_type_name(e.t), (long long)e.bytes, (long long)e.values);
        check(ggml_type_block_bytes(e.t) == e.bytes &&
              ggml_type_block_values(e.t) == e.values, msg);
    }
    check(ggml_type_supported(GGMLType::Q4_K) &&
          ggml_type_supported(GGMLType::Q6_K) &&
          ggml_type_supported(GGMLType::IQ4_XS),
          "the three K-quants the models use are supported");
    check(!ggml_type_supported(GGMLType::Q2_K),
          "an unsupported type reports unsupported rather than pretending");

    if (argc < 2) {
        std::printf("\n[2] (skipped: pass a .gguf path to exercise the parser)\n");
        std::printf("\n=== %d checks, %d failures ===\n", g_checks, g_fails);
        return g_fails == 0 ? 0 : 1;
    }

    // ------------------------------------------------------------ real file
    const std::string path = argv[1];
    banner("[2] parse a real model");
    GgufReader r(path);
    std::printf("PARSE arch=%s tensors=%zu size=%lld\n", r.arch().c_str(),
                r.tensors().size(), (long long)r.file_size());
    check(!r.arch().empty(), "architecture read from general.architecture");
    check(r.tensors().size() > 100, "tensor index populated");
    check(r.file_size() > 0, "file mapped");

    const int64_t n_layer = r.get_int("block_count", 0);
    const int64_t n_embd = r.get_int("embedding_length", 0);
    check(n_layer > 0, "block_count readable by bare name (arch prefix stripped)");
    check(n_embd > 0, "embedding_length readable by bare name");
    std::printf("       block_count=%lld embedding_length=%lld\n",
                (long long)n_layer, (long long)n_embd);

    // every tensor's byte range must lie INSIDE the file. A reader that reports a
    // tensor past EOF would read garbage (or segfault) later, so this is the single
    // most important structural check.
    int64_t worst = 0;
    const TensorInfo* worst_t = nullptr;
    for (auto& [name, t] : r.tensors()) {
        const int64_t end = t.offset + r.tensor_bytes(t);
        if (end > worst) { worst = end; worst_t = &t; }
    }
    std::printf("       furthest tensor ends %lld bytes into the data section\n",
                (long long)worst);
    check(worst_t != nullptr, "found a furthest tensor");
    // read the last byte of the furthest tensor: if the mapping were short this
    // would fault, so surviving it IS the check
    if (worst_t) {
        const uint8_t* p = r.tensor_data(*worst_t);
        const int64_t nb = r.tensor_bytes(*worst_t);
        volatile uint8_t last = p[nb - 1];
        (void)last;
        check(true, "the furthest tensor's last byte is inside the mapping (no fault)");
    }

    // ---- cross-check a known tensor against the Python reader
    banner("[3] the parsed facts the Python reader also reports");
    struct { const char* key; } keys[] = {
        {"block_count"}, {"embedding_length"}, {"attention.head_count"},
        {"attention.head_count_kv"}, {"attention.key_length"},
    };
    for (auto& k : keys) {
        const MetaValue* v = nullptr;
        (void)r.get_int(k.key, -1);
        std::printf("META %s %lld\n", k.key, (long long)r.get_int(k.key, -1));
        (void)v;
    }

    // ---- and every tensor the model assembly needs must be present
    banner("[4] the tensors the forward pass needs exist");
    std::vector<std::string> need = {
        "token_embd.weight", "output_norm.weight",
        "blk.0.attn_norm.weight", "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
    };
    for (auto& n : need) {
        char msg[128];
        std::snprintf(msg, sizeof(msg), "%s present", n.c_str());
        check(r.find(n) != nullptr, msg);
    }
    // and a missing one must be reportable rather than defaulting to something
    try {
        r.need("blk.99999.nope.weight");
        check(false, "need() throws for a missing tensor");
    } catch (const std::exception&) {
        check(true, "need() throws for a missing tensor");
    }

    // ---- dequantise one tensor and report its statistics, so Python can diff
    banner("[5] dequantise a real tensor");
    const TensorInfo& t = r.need("blk.0.attn_norm.weight");
    std::vector<float> buf((size_t)t.n_elements);
    r.dequantise(t, buf.data());
    double mn = buf[0], mx = buf[0], sum = 0.0, ss = 0.0;
    for (float f : buf) {
        mn = std::min(mn, (double)f); mx = std::max(mx, (double)f);
        sum += f; ss += (double)f * f;
    }
    const double mean = sum / (double)buf.size();
    const double var = ss / (double)buf.size() - mean * mean;
    std::printf("TENSOR %s n=%lld mean=%.8f std=%.8f min=%.8f max=%.8f\n",
                t.name.c_str(), (long long)t.n_elements, mean,
                var > 0 ? std::sqrt(var) : 0.0, mn, mx);
    check(std::isfinite(mean) && std::isfinite(mx), "dequantised values are finite");
    check(mx != mn, "the tensor is not degenerate (not all one value)");

    std::printf("\n=== %d checks, %d failures ===\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
