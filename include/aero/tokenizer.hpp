// The tokenizer, native. Port of engine/tokenizer.py's Tokenizer.
//
// WHY THIS IS WORTH PORTING: tokenization sat in Python on the inference path, so
// every prompt paid an interpreter loop over the vocabulary before the model could
// run. It is also the one piece whose correctness the whole engine depends on and
// which no amount of C++ downstream can repair: a wrong id is a wrong token.
//
// TWO VOCAB FAMILIES, and they are NOT interchangeable (feeding qwen's vocab through
// the sentencepiece path yields ids from the wrong vocabulary):
//   * "gpt2" / "qwen2"  byte-level BPE. Each byte of a pre-tokeniser unit is first
//                       remapped onto a printable codepoint (the GPT-2 byte->unicode
//                       table, which is why a space shows up as U+0120). A regex then
//                       splits the RAW text into units, and lowest-rank merges run
//                       inside each unit.
//   * "gemma*" / "llama" / "spm"  sentencepiece. Text stays UTF-8, a space is written
//                       as U+2581, there is no byte remap, and merges run over whole
//                       newline-delimited runs.
//
// MATCHED TO llama.cpp, NOT to our Python. The Python reference had the pre-tokeniser
// regex slightly wrong (its letter branch allowed only a leading SPACE, where the
// published Qwen3.5 rule allows any single non-letter/non-digit prefix, so "x\ty" split
// as 'x' | tab | 'y' instead of 'x' | <tab,y>). Reading llama.cpp's vocab/unicode code
// for the published regexes and split semantics (never copying it) is how this was
// pinned; tests/test_tokenizer_cpp.py then proves the result against `llama-tokenize`
// on a battery, and prints the exact ids for any mismatch.
//
// Layout: the C ABI below is pointer+lengths only, so ctypes never sees a C++ type
// (see src/kv.cpp for the same pattern).
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "aero/gguf.hpp"

namespace aero {

enum class VocabFamily { GPT2 = 0, SPM = 1 };

// Which pre-tokeniser regex family. Chosen from `tokenizer.ggml.pre`, or inferred from
// the tokenizer model when that key is absent (gemma's GGUF carries no `pre`).
enum class PreType { GPT2 = 0, LLAMA3, QWEN2, QWEN35, NEWLINES };

class Tokenizer {
public:
    // Load tokens / merges / types / specials and the family flags from an already-open
    // GGUF. Throws std::runtime_error (with a reason) on anything it cannot do honestly:
    // no tokenizer, an empty vocab, a malformed merge rule, an unsupported model.
    void load(GgufReader& reader);

    // Build a tokenizer straight from arrays (used by tests and fixtures).
    void load_arrays(const std::vector<std::string>& tokens,
                     const std::vector<std::string>& merges,
                     const std::vector<int64_t>& token_types,
                     VocabFamily family, PreType pre, bool add_bos,
                     bool add_space_prefix,
                     int32_t eos_id, int32_t bos_id, int32_t pad_id);

    // text -> ids. `add_bos_override`: -1 = use the file's flag, 0 = never, 1 = force.
    // Special tokens present in the input are matched whole, never through BPE.
    std::vector<int32_t> encode(const std::string& text, int add_bos_override = -1) const;

    // ids -> text. `skip_special` drops special-token text (chat framing must not leak).
    // Throws std::out_of_range on an id outside the vocabulary (never silently wraps).
    std::string decode(const std::vector<int32_t>& ids, bool skip_special = true) const;

    // ---- read-only accessors ----
    int64_t n_vocab() const { return (int64_t)tokens_.size(); }
    int64_t n_merges() const { return (int64_t)n_merges_; }
    int64_t n_special() const { return (int64_t)special_ids_.size(); }
    VocabFamily family() const { return family_; }
    PreType pre() const { return pre_; }
    bool add_bos() const { return add_bos_; }
    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }
    int32_t pad_id() const { return pad_id_; }
    const std::string& token(int32_t id) const { return tokens_.at((size_t)id); }
    bool is_special_id(int32_t id) const { return special_ids_.count(id) != 0; }

private:
    std::vector<int32_t> encode_plain(const std::string& text) const;
    std::vector<std::string> bpe(const std::string& word) const;
    void emit_symbol(const std::string& sym, std::vector<int32_t>& out, bool spm) const;

    void build_byte_table();

    std::vector<std::string> tokens_;
    std::unordered_map<std::string, int32_t> encoder_;   // token text -> id (first wins)
    std::unordered_map<std::string, int32_t> ranks_;     // left '\0' right -> rank
    std::unordered_map<std::string, int32_t> special_map_;
    std::unordered_set<int32_t> special_ids_;
    // specials as (id, text), sorted longest-text-first then by id
    std::vector<std::pair<int32_t, std::string>> special_list_;
    // byte tokens (type 6, "<0xHH>") -> the byte they stand for, so decode returns the
    // raw byte rather than the literal, exactly as llama.cpp's token_to_piece does
    std::unordered_map<int32_t, uint8_t> byte_tokens_;

    VocabFamily family_ = VocabFamily::GPT2;
    PreType pre_ = PreType::GPT2;
    bool byte_encode_ = true;
    bool add_bos_ = false;
    bool add_space_prefix_ = false;
    int32_t bos_id_ = -1, eos_id_ = -1, pad_id_ = -1;
    int64_t n_merges_ = 0;

    // byte -> its GPT-2 printable-codepoint UTF-8 string (256 entries), and the reverse.
    std::string b2u_[256];
    std::unordered_map<uint32_t, int> u2b_;
};

}  // namespace aero

// ---------------------------------------------------------------------------
// C ABI. Opaque handle; every call takes pointer+lengths so ctypes sees only
// ints, pointers and char buffers.
// ---------------------------------------------------------------------------
extern "C" {

// Open a GGUF and load its tokenizer. Returns nullptr on any failure (missing file,
// no tokenizer, unsupported model); the reason is available from aero_tok_last_error().
void* aero_tok_open(const char* path);
void  aero_tok_free(void* h);
const char* aero_tok_last_error(void);

int64_t aero_tok_n_vocab(void* h);
int64_t aero_tok_n_merges(void* h);
int64_t aero_tok_n_special(void* h);
int64_t aero_tok_bos_id(void* h);      // -1 when absent
int64_t aero_tok_eos_id(void* h);
int64_t aero_tok_pad_id(void* h);
int     aero_tok_add_bos(void* h);     // 1/0
int     aero_tok_family(void* h);      // 0 = gpt2 (byte-level), 1 = spm
int     aero_tok_pre(void* h);         // PreType enum as int
int     aero_tok_is_special_id(void* h, int32_t id);

// Token text for an id: writes up to `cap` bytes, returns the FULL byte length of the
// token (which may exceed cap), or -1 for an out-of-range id. No NUL is appended.
int64_t aero_tok_token(void* h, int32_t id, char* out, int64_t cap);

// text -> ids. Returns the number of ids the text encodes to. If `out` is non-null it
// must have room for all of them (`cap >= n`) or the call REFUSES with -1 -- it never
// writes a truncated token list. Pass out=null to query the count. Returns -1 on a bad
// handle, negative length, or null text with a positive length.
// `add_bos`: -1 = file default, 0 = off, 1 = on.
int64_t aero_tok_encode(void* h, const char* text, int64_t len,
                        int32_t* out, int64_t cap, int add_bos);

// ids -> text bytes. Same contract as encode: returns the byte count, refuses (-1) if
// `cap` is too small, and returns -1 for an out-of-range id. `skip_special` (1/0).
int64_t aero_tok_decode(void* h, const int32_t* ids, int64_t n,
                        char* out, int64_t cap, int skip_special);

}  // extern "C"
