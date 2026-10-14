// GGUF reader: parse the header and tensor index, then mmap the weight data.
//
// WHY MMAP. The Python engine read each tensor with an explicit seek+read, and the
// profile showed 3.36 s per token just moving bytes (802 MB/s, which is the USB
// bus limit -- that part was optimal). But two things were wrong with it:
//
//   1. It forced a copy through a user buffer for every tensor, every token.
//   2. The memory guard had to reason about OUR allocations rather than about
//      what is actually resident.
//
// mmap fixes both: the kernel's page cache becomes the loader, a page already
// read is free to touch again, and the quantity the guard must bound is honestly
// the process's resident footprint -- which it already measures.
//
// The file is never copied in full. Only the pages a layer actually touches become
// resident, so a 6 GB model can be decoded with a working set far below its size.
//
// FORMAT (little-endian throughout):
//   magic "GGUF" | u32 version | u64 n_tensors | u64 n_kv
//   n_kv    x (string key, u32 type, value)
//   n_tensors x (string name, u32 n_dims, u64 dims[n_dims], u32 type, u64 offset)
//   pad to general.alignment
//   tensor data
// A string is u64 length + UTF-8. Tensor `offset` is relative to the data section.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace aero {

// ggml block formats this engine can compute. Anything else must be refused by the
// caller rather than silently misread -- a wrong dequant produces fluent nonsense,
// which is the failure mode this project keeps hitting.
enum class GGMLType : int {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14,
    IQ4_XS = 23, BF16 = 30,
};

const char* ggml_type_name(GGMLType t);
int64_t ggml_type_block_bytes(GGMLType t);      // bytes per block
int64_t ggml_type_block_values(GGMLType t);     // values per block
bool ggml_type_supported(GGMLType t);            // has a dequantiser

struct TensorInfo {
    std::string name;
    std::vector<int64_t> dims;   // GGUF order: fastest-varying first
    GGMLType type = GGMLType::F32;
    int64_t offset = 0;          // relative to the data section
    int64_t n_elements = 0;

    // GGUF stores dims reversed, so row-major shape is dims reversed.
    int64_t rows() const { return dims.size() <= 1 ? 1 : dims[dims.size() - 1]; }
    int64_t cols() const { return dims.empty() ? 0 : dims[0]; }
    bool is_2d() const { return dims.size() == 2; }
};

// Metadata values, kept as the few types the engine actually reads.
struct MetaValue {
    enum Kind { UINT, INT, FLOAT, STRING, ARRAY, BOOL, UINT64 } kind = UINT;
    uint64_t u = 0;
    int64_t i = 0;
    double f = 0.0;
    bool b = false;
    std::string s;
    std::vector<int64_t> arr_int;      // int/uint arrays (head_count_kv, sections)
    std::vector<std::string> arr_str;  // tokenizer vocab, merges
    std::vector<double> arr_flt;
};

class GgufReader {
public:
    explicit GgufReader(const std::string& path);
    ~GgufReader();
    GgufReader(const GgufReader&) = delete;
    GgufReader& operator=(const GgufReader&) = delete;

    // ---- metadata ----
    const std::string& arch() const { return arch_; }
    bool has(const std::string& key) const;
    // Scalar accessors. `suffix` matches after the "<arch>." prefix as well as
    // verbatim, because GGUF mixes both spellings ("qwen35.block_count" and
    // "tokenizer.ggml.tokens").
    int64_t get_int(const std::string& key, int64_t def = 0) const;
    double get_float(const std::string& key, double def = 0.0) const;
    std::string get_str(const std::string& key, const std::string& def = "") const;
    std::vector<int64_t> get_int_array(const std::string& key) const;
    std::vector<std::string> get_str_array(const std::string& key) const;
    // A key that may be either a scalar or an array, widened to one element per
    // layer. Gemma4 stores head_count_kv this way and a scalar read silently
    // applies layer 0's value to the whole model.
    std::vector<int64_t> get_int_either(const std::string& key, int64_t n) const;

    // ---- tensors ----
    const std::unordered_map<std::string, TensorInfo>& tensors() const { return tensors_; }
    const TensorInfo* find(const std::string& name) const;
    const TensorInfo& need(const std::string& name) const;   // throws if absent

    // Raw quantised bytes for a tensor (page-cache backed, no copy).
    const uint8_t* tensor_data(const TensorInfo& t) const;
    int64_t tensor_bytes(const TensorInfo& t) const;

    // Dequantise a whole tensor into `out` (n_elements floats).
    void dequantise(const TensorInfo& t, float* out) const;

    // Fused dequant+matvec / matmul for a supported type. Returns false if the type
    // has no kernel, so the caller can fall back rather than compute garbage.
    bool matvec(const TensorInfo& t, const float* x, float* y, int threads = 1) const;
    bool matmul(const TensorInfo& t, const float* X, float* Y,
                int64_t n_tokens, int threads = 1) const;

    int64_t file_size() const { return file_size_; }
    const std::string& path() const { return path_; }
    // Resident bytes touched so far -- informational; the guard measures the
    // process, which is the number that actually matters.
    int64_t mapped_bytes() const { return file_size_; }

private:
    void parse_header();
    void parse_metadata(const uint8_t*& p, const uint8_t* end);
    void parse_tensor_index(const uint8_t*& p, const uint8_t* end);
    bool lookup(const std::string& key, const MetaValue** out) const;

    std::string path_;
    int fd_ = -1;
    const uint8_t* base_ = nullptr;
    int64_t file_size_ = 0;
    int64_t data_offset_ = 0;
    std::string arch_;
    std::unordered_map<std::string, MetaValue> meta_;
    std::unordered_map<std::string, TensorInfo> tensors_;
    std::unordered_map<std::string, std::string> arch_keys_;  // "<arch>.<k>" -> k
};

}  // namespace aero
