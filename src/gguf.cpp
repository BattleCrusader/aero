// GGUF reader implementation. See gguf.hpp for the format and the mmap rationale.
#include "aero/gguf.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>

#include "aero/kernels.hpp"   // fused dequant matvec/matmul

namespace aero {
namespace {

// ---- little-endian scalar reads. The format is LE everywhere, and every field
// is read through these so a big-endian host is a compile error rather than a
// silent mis-parse.

struct Cur {
    const uint8_t* p;
    const uint8_t* end;
    void need(int64_t n) const {
        if (p + n > end) throw std::runtime_error("GGUF: truncated file");
    }
    uint32_t u32() { need(4); uint32_t v; std::memcpy(&v, p, 4); p += 4; return v; }
    uint64_t u64() { need(8); uint64_t v; std::memcpy(&v, p, 8); p += 8; return v; }
    int32_t i32() { need(4); int32_t v; std::memcpy(&v, p, 4); p += 4; return v; }
    int64_t i64() { need(8); int64_t v; std::memcpy(&v, p, 8); p += 8; return v; }
    float f32() { need(4); float v; std::memcpy(&v, p, 4); p += 4; return v; }
    double f64() { need(8); double v; std::memcpy(&v, p, 8); p += 8; return v; }
    uint8_t u8() { need(1); return *p++; }
    std::string str() {
        const uint64_t n = u64();
        need((int64_t)n);
        std::string s(reinterpret_cast<const char*>(p), (size_t)n);
        p += n;
        return s;
    }
};

// GGUF metadata value types
enum : uint32_t {
    KV_UINT8 = 0, KV_INT8, KV_UINT16, KV_INT16, KV_UINT32, KV_INT32,
    KV_FLOAT32, KV_BOOL, KV_STRING, KV_ARRAY, KV_UINT64, KV_INT64, KV_FLOAT64,
};

// Read one value into out. Arrays of strings (the vocab, the merges) are the big
// ones; everything else is small.
void read_value(Cur& c, uint32_t type, MetaValue& out) {
    switch (type) {
        case KV_UINT8:  out.kind = MetaValue::UINT;  out.u = c.u8(); break;
        case KV_INT8:   out.kind = MetaValue::INT;   out.i = (int8_t)c.u8(); break;
        case KV_UINT16: out.kind = MetaValue::UINT;  out.u = (uint16_t)c.u32(); break;
        case KV_INT16:  out.kind = MetaValue::INT;   out.i = (int16_t)c.u32(); break;
        case KV_UINT32: out.kind = MetaValue::UINT;  out.u = c.u32(); break;
        case KV_INT32:  out.kind = MetaValue::INT;   out.i = c.i32(); break;
        case KV_FLOAT32:out.kind = MetaValue::FLOAT; out.f = c.f32(); break;
        case KV_BOOL:   out.kind = MetaValue::BOOL;  out.b = c.u8() != 0; break;
        case KV_STRING: out.kind = MetaValue::STRING; out.s = c.str(); break;
        case KV_UINT64: out.kind = MetaValue::UINT64; out.u = c.u64(); break;
        case KV_INT64:  out.kind = MetaValue::INT;   out.i = c.i64(); break;
        case KV_FLOAT64:out.kind = MetaValue::FLOAT; out.f = c.f64(); break;
        case KV_ARRAY: {
            const uint32_t et = c.u32();
            const uint64_t n = c.u64();
            if (n > (uint64_t)1 << 32) throw std::runtime_error("GGUF: absurd array length");
            out.kind = MetaValue::ARRAY;
            for (uint64_t i = 0; i < n; ++i) {
                MetaValue e;
                read_value(c, et, e);
                switch (e.kind) {
                    case MetaValue::UINT: case MetaValue::UINT64:
                        out.arr_int.push_back((int64_t)e.u); break;
                    case MetaValue::INT: out.arr_int.push_back(e.i); break;
                    case MetaValue::FLOAT: out.arr_flt.push_back(e.f); break;
                    case MetaValue::STRING: out.arr_str.push_back(std::move(e.s)); break;
                    case MetaValue::BOOL: out.arr_int.push_back(e.b ? 1 : 0); break;
                    default: break;
                }
            }
            // An int array is also exposed as float for callers that want it that
            // way (rope sections are sometimes stored as floats).
            if (!out.arr_int.empty() && out.arr_flt.empty())
                for (int64_t v : out.arr_int) out.arr_flt.push_back((double)v);
            break;
        }
        default: throw std::runtime_error("GGUF: unknown metadata type");
    }
}

}  // namespace

// ------------------------------------------------------------------ type table

const char* ggml_type_name(GGMLType t) {
    switch (t) {
        case GGMLType::F32: return "F32";   case GGMLType::F16: return "F16";
        case GGMLType::Q4_0: return "Q4_0"; case GGMLType::Q4_1: return "Q4_1";
        case GGMLType::Q5_0: return "Q5_0"; case GGMLType::Q5_1: return "Q5_1";
        case GGMLType::Q8_0: return "Q8_0"; case GGMLType::Q2_K: return "Q2_K";
        case GGMLType::Q3_K: return "Q3_K"; case GGMLType::Q4_K: return "Q4_K";
        case GGMLType::Q5_K: return "Q5_K"; case GGMLType::Q6_K: return "Q6_K";
        case GGMLType::IQ4_XS: return "IQ4_XS";
        case GGMLType::BF16: return "BF16";
    }
    return "unknown";
}

int64_t ggml_type_block_values(GGMLType t) {
    switch (t) {
        case GGMLType::F32: case GGMLType::F16: case GGMLType::BF16: return 1;
        case GGMLType::Q4_0: case GGMLType::Q4_1:
        case GGMLType::Q5_0: case GGMLType::Q5_1: return 32;
        case GGMLType::Q8_0: return 32;
        case GGMLType::Q2_K: case GGMLType::Q3_K: case GGMLType::Q4_K:
        case GGMLType::Q5_K: case GGMLType::Q6_K: case GGMLType::IQ4_XS:
            return 256;
    }
    return 0;
}

int64_t ggml_type_block_bytes(GGMLType t) {
    switch (t) {
        case GGMLType::F32: return 4;   case GGMLType::F16: return 2;
        case GGMLType::BF16: return 2;
        case GGMLType::Q4_0: return 18; case GGMLType::Q4_1: return 20;
        case GGMLType::Q5_0: return 22; case GGMLType::Q5_1: return 24;
        case GGMLType::Q8_0: return 34;
        case GGMLType::Q2_K: return 84; case GGMLType::Q3_K: return 110;
        case GGMLType::Q4_K: return 144; case GGMLType::Q5_K: return 176;
        case GGMLType::Q6_K: return 210; case GGMLType::IQ4_XS: return 136;
    }
    return 0;
}

bool ggml_type_supported(GGMLType t) {
    switch (t) {
        case GGMLType::F32: case GGMLType::F16: case GGMLType::BF16:
        case GGMLType::Q4_K: case GGMLType::Q6_K: case GGMLType::IQ4_XS:
            return true;
        default:
            return false;
    }
}

// ------------------------------------------------------------------ reader

GgufReader::GgufReader(const std::string& path) : path_(path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("GGUF: cannot open " + path);
    struct stat st{};
    if (::fstat(fd_, &st) != 0) { ::close(fd_); throw std::runtime_error("GGUF: fstat failed"); }
    file_size_ = (int64_t)st.st_size;
    if (file_size_ <= 0) { ::close(fd_); throw std::runtime_error("GGUF: empty file"); }

    // MAP_PRIVATE: read-only, and we never write through it. The kernel pages in
    // only what is touched, which is what keeps a 6 GB model decodable in a small
    // working set.
    void* m = ::mmap(nullptr, (size_t)file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (m == MAP_FAILED) { ::close(fd_); throw std::runtime_error("GGUF: mmap failed"); }
    base_ = (const uint8_t*)m;

    parse_header();
}

GgufReader::~GgufReader() {
    if (base_ && base_ != MAP_FAILED) ::munmap((void*)base_, (size_t)file_size_);
    if (fd_ >= 0) ::close(fd_);
}

void GgufReader::parse_header() {
    const uint8_t* p = base_;
    const uint8_t* end = base_ + file_size_;
    if (std::memcmp(p, "GGUF", 4) != 0) throw std::runtime_error("GGUF: bad magic");
    p += 4;
    Cur c{p, end};
    const uint32_t version = c.u32();
    if (version < 2 || version > 3)
        throw std::runtime_error("GGUF: unsupported version " + std::to_string(version));
    (void)c.u64();   // n_tensors, re-read by parse_tensor_index
    (void)c.u64();   // n_kv,      re-read by parse_metadata
    p = c.p;

    // Each parser walks from the cursor it is given. They are separate rather than
    // one big pass so a malformed metadata blob cannot desynchronise the tensor
    // index and silently give every tensor the wrong offset.
    parse_metadata(p, end);
    parse_tensor_index(p, end);
}

void GgufReader::parse_metadata(const uint8_t*& pp, const uint8_t* end) {
    // pp already points just past n_kv.
    Cur c{pp, end};
    const uint64_t n_kv = [&] {
        const uint8_t* q = base_ + 4 + 4 + 8;
        uint64_t v;
        std::memcpy(&v, q, 8);
        return v;
    }();

    for (uint64_t i = 0; i < n_kv; ++i) {
        const std::string key = c.str();
        const uint32_t type = c.u32();
        MetaValue v;
        read_value(c, type, v);
        if (key == "general.architecture") arch_ = v.s;
        // Arch-scoped keys ("qwen35.block_count") are also reachable by their bare
        // suffix, because callers should not have to know the arch to ask. The
        // tokenizer namespace is left alone: "tokenizer.ggml.model" must not become
        // a bare "ggml.model" that shadows something else.
        const size_t dot = key.find('.');
        if (dot != std::string::npos && key.compare(0, 9, "tokenizer") != 0 &&
            key.compare(0, 8, "general.") != 0) {
            meta_.emplace(key.substr(dot + 1), v);
        }
        meta_.emplace(key, std::move(v));
    }
    pp = c.p;
}

void GgufReader::parse_tensor_index(const uint8_t*& pp, const uint8_t* end) {
    // pp already points at the first tensor record: parse_header consumed the
    // 4-byte magic, and parse_metadata consumed the version, both counts and every
    // metadata entry. Re-reading the header here is what made this throw "bad
    // magic" -- the cursor was already past it.
    Cur c{pp, end};
    const uint64_t n_tensors = [&] {
        const uint8_t* q = base_ + 4 + 4;
        uint64_t v;
        std::memcpy(&v, q, 8);
        return v;
    }();

    int64_t max_end = 0;
    for (uint64_t i = 0; i < n_tensors; ++i) {
        TensorInfo t;
        t.name = c.str();
        const uint32_t n_dims = c.u32();
        if (n_dims > 4) throw std::runtime_error("GGUF: bad n_dims");
        t.dims.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) t.dims[d] = (int64_t)c.u64();
        t.type = (GGMLType)c.u32();
        t.offset = c.i64();

        const int64_t bv = ggml_type_block_values(t.type);
        if (bv <= 0) throw std::runtime_error(
            std::string("GGUF: unknown tensor type ") + std::to_string((int)t.type)
            + " for " + t.name);
        int64_t n = 1;
        for (int64_t d : t.dims) n *= d;
        t.n_elements = n;
        if (n % bv != 0) throw std::runtime_error(
            "GGUF: " + t.name + " element count is not a whole number of blocks");
        max_end = std::max(max_end, t.offset + (n / bv) * ggml_type_block_bytes(t.type));
        tensors_.emplace(t.name, std::move(t));
    }

    // The data section starts at the next multiple of the alignment. Read
    // general.alignment when present, else use the format's documented default of
    // 32 -- older files omit the key, so defaulting is correct rather than lax.
    int64_t align = 32;
    auto ait = meta_.find("general.alignment");
    if (ait != meta_.end()) {
        const MetaValue& av = ait->second;
        if ((av.kind == MetaValue::UINT || av.kind == MetaValue::UINT64) && av.u > 0)
            align = (int64_t)av.u;
    }
    const int64_t here = (int64_t)(c.p - base_);
    data_offset_ = ((here + align - 1) / align) * align;
    if (data_offset_ + max_end > file_size_)
        throw std::runtime_error("GGUF: tensor data runs past end of file");
    pp = c.p;
}

bool GgufReader::lookup(const std::string& key, const MetaValue** out) const {
    auto it = meta_.find(key);
    if (it == meta_.end()) return false;
    *out = &it->second;
    return true;
}

bool GgufReader::has(const std::string& key) const { return meta_.count(key) > 0; }

int64_t GgufReader::get_int(const std::string& key, int64_t def) const {
    auto it = meta_.find(key);
    if (it == meta_.end()) return def;
    const MetaValue& v = it->second;
    if (v.kind == MetaValue::UINT || v.kind == MetaValue::UINT64) return (int64_t)v.u;
    if (v.kind == MetaValue::INT) return v.i;
    if (v.kind == MetaValue::BOOL) return v.b ? 1 : 0;
    return def;
}

double GgufReader::get_float(const std::string& key, double def) const {
    auto it = meta_.find(key);
    if (it == meta_.end()) return def;
    const MetaValue& v = it->second;
    if (v.kind == MetaValue::FLOAT) return v.f;
    if (v.kind == MetaValue::UINT || v.kind == MetaValue::UINT64) return (double)v.u;
    if (v.kind == MetaValue::INT) return (double)v.i;
    return def;
}

std::string GgufReader::get_str(const std::string& key, const std::string& def) const {
    auto it = meta_.find(key);
    if (it == meta_.end() || it->second.kind != MetaValue::STRING) return def;
    return it->second.s;
}

std::vector<int64_t> GgufReader::get_int_array(const std::string& key) const {
    auto it = meta_.find(key);
    if (it == meta_.end()) return {};
    return it->second.arr_int;
}

std::vector<std::string> GgufReader::get_str_array(const std::string& key) const {
    auto it = meta_.find(key);
    if (it == meta_.end()) return {};
    return it->second.arr_str;
}

std::vector<int64_t> GgufReader::get_int_either(const std::string& key, int64_t n) const {
    auto it = meta_.find(key);
    if (it == meta_.end()) return {};
    const MetaValue& v = it->second;
    if (v.kind == MetaValue::ARRAY) return v.arr_int;
    // scalar -> one value per layer. gemma4 stores head_count_kv as a list; reading
    // it as a scalar would apply layer 0's count to every layer, which is exactly
    // the class of silent mis-shape this project keeps finding.
    const int64_t s = get_int(key, 0);
    return std::vector<int64_t>((size_t)std::max<int64_t>(n, 1), s);
}

const TensorInfo* GgufReader::find(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

const TensorInfo& GgufReader::need(const std::string& name) const {
    const TensorInfo* t = find(name);
    if (!t) throw std::runtime_error("GGUF: no tensor named " + name);
    return *t;
}

const uint8_t* GgufReader::tensor_data(const TensorInfo& t) const {
    return base_ + data_offset_ + t.offset;
}

int64_t GgufReader::tensor_bytes(const TensorInfo& t) const {
    const int64_t bv = ggml_type_block_values(t.type);
    return (t.n_elements / bv) * ggml_type_block_bytes(t.type);
}

void GgufReader::dequantise(const TensorInfo& t, float* out) const {
    if (!ggml_type_supported(t.type))
        throw std::runtime_error(std::string("GGUF: no dequantiser for ") +
                                 ggml_type_name(t.type));
    dequant_ref((DType)(int)t.type, tensor_data(t), out, t.n_elements);
}

bool GgufReader::matvec(const TensorInfo& t, const float* x, float* y, int threads) const {
    if (!t.is_2d()) return false;
    const int64_t R = t.rows(), C = t.cols();
    if (t.n_elements != R * C) return false;
    return aero::matvec((DType)(int)t.type, tensor_data(t), x, y, R, C, threads);
}

bool GgufReader::matmul(const TensorInfo& t, const float* X, float* Y,
                        int64_t n_tokens, int threads) const {
    if (!t.is_2d()) return false;
    const int64_t R = t.rows(), C = t.cols();
    if (t.n_elements != R * C) return false;
    return aero::matmul((DType)(int)t.type, tensor_data(t), X, Y,
                        n_tokens, R, C, threads);
}

}  // namespace aero
