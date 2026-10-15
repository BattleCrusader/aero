// Tokenizer implementation. See include/aero/tokenizer.hpp for the contract and the
// two vocab families.
//
// The two pre-tokeniser rules implemented here are the PUBLISHED tokenizer.json
// patterns, matched to llama.cpp's own split semantics (read from its src/unicode.cpp
// and src/llama-vocab.cpp for the rules -- never copied):
//
//   Qwen3.5  "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
//            "|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}"
//            "| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
//   Gemma4   "[^\\n]+|[\\n]+"  with byte_encode = false (raw UTF-8, space -> U+2581)
//
// The letter branch's prefix is `[^\r\n\p{L}\p{N}]?` -- any SINGLE non-letter,
// non-digit character, not just a space. That is the whole reason "x<TAB>y" must
// tokenize as 'x' | <TAB,y> and not 'x' | <TAB> | 'y'; our Python reference used
// ` ?` and got the latter, which `llama-tokenize` refutes.
#include "aero/tokenizer.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace aero {
namespace {

// Unicode category ranges, generated (see the file header). \p{L} \p{M} \p{N}
// \p{P} \p{S} as sorted [lo,hi] codepoint pairs.
#include "tokenizer_unicode_tables.inc"

bool in_ranges(const uint32_t (*r)[2], size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        const size_t mid = (lo + hi) >> 1;
        if (cp < r[mid][0]) hi = mid;
        else if (cp > r[mid][1]) lo = mid + 1;
        else return true;
    }
    return false;
}

bool is_letter(uint32_t c) { return in_ranges(kRangesLetter, kRangesLetter_n, c); }
bool is_mark(uint32_t c)   { return in_ranges(kRangesMark,   kRangesMark_n,   c); }
bool is_number(uint32_t c) { return in_ranges(kRangesNumber, kRangesNumber_n, c); }
bool is_punct(uint32_t c)  { return in_ranges(kRangesPunct,  kRangesPunct_n,  c); }
bool is_symbol(uint32_t c) { return in_ranges(kRangesSymbol, kRangesSymbol_n, c); }

// The Unicode White_Space property (the set llama.cpp flags as whitespace, so `\s`).
bool is_space(uint32_t c) {
    return (c >= 0x09 && c <= 0x0D) || c == 0x20 || c == 0x85 || c == 0xA0 ||
           c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 ||
           c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

// llama.cpp gates a split branch on `flags.as_uint() != 0` -- "this codepoint has at
// least one category bit". Reproduced so a NUL or other flagless control char splits
// the same way (it must NOT be swept into a punctuation run).
bool has_flags(uint32_t c) {
    return is_space(c) || is_letter(c) || is_mark(c) || is_number(c) ||
           is_punct(c) || is_symbol(c);
}

// ---- UTF-8 ----------------------------------------------------------------
size_t utf8_seq_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c >> 5) == 0x6) return 2;
    if ((c >> 4) == 0xE) return 3;
    if ((c >> 3) == 0x1E) return 4;
    return 1;   // invalid lead byte: consume one byte
}

uint32_t utf8_decode(const std::string& s, size_t i, size_t& len) {
    const unsigned char c = (unsigned char)s[i];
    const size_t need = utf8_seq_len(c);
    if (need == 1 || i + need > s.size()) { len = 1; return c; }
    uint32_t cp = 0;
    switch (need) {
        case 2: cp = c & 0x1F; break;
        case 3: cp = c & 0x0F; break;
        default: cp = c & 0x07; break;
    }
    for (size_t k = 1; k < need; ++k) {
        const unsigned char cc = (unsigned char)s[i + k];
        if ((cc & 0xC0) != 0x80) { len = 1; return c; }   // malformed; treat as one byte
        cp = (cp << 6) | (cc & 0x3F);
    }
    len = need;
    return cp;
}

void utf8_append(std::string& out, uint32_t cp) {
    if (cp < 0x80) { out.push_back((char)cp); return; }
    if (cp < 0x800) {
        out.push_back((char)(0xC0 | (cp >> 6)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
        return;
    }
    if (cp < 0x10000) {
        out.push_back((char)(0xE0 | (cp >> 12)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
        return;
    }
    out.push_back((char)(0xF0 | (cp >> 18)));
    out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back((char)(0x80 | (cp & 0x3F)));
}

// Decode UTF-8 to text, replacing each malformed byte with U+FFFD (like Python's
// bytes.decode(errors="replace")). Well-formed input is passed through untouched.
std::string utf8_replace(const std::string& bytes) {
    std::string out;
    out.reserve(bytes.size());
    size_t i = 0;
    while (i < bytes.size()) {
        const unsigned char c = (unsigned char)bytes[i];
        if (c < 0x80) { out.push_back((char)c); ++i; continue; }
        const size_t need = utf8_seq_len(c);
        bool ok = need > 1 && i + need <= bytes.size();
        uint32_t cp = 0;
        if (ok) {
            cp = (need == 2 ? c & 0x1F : need == 3 ? c & 0x0F : c & 0x07);
            for (size_t k = 1; k < need; ++k) {
                const unsigned char cc = (unsigned char)bytes[i + k];
                if ((cc & 0xC0) != 0x80) { ok = false; break; }
                cp = (cp << 6) | (cc & 0x3F);
            }
            // reject overlongs, surrogates and out-of-range
            if (ok) {
                const uint32_t min = (need == 2 ? 0x80 : need == 3 ? 0x800 : 0x10000);
                if (cp < min || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) ok = false;
            }
        }
        if (ok) { out.append(bytes, i, need); i += need; }
        else { out += "\xEF\xBF\xBD"; ++i; }
    }
    return out;
}

// Split into UTF-8 codepoints, keeping each codepoint's bytes as its own string.
std::vector<std::string> split_codepoints(const std::string& s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t len;
        utf8_decode(s, i, len);
        out.push_back(s.substr(i, len));
        i += len;
    }
    return out;
}

// A codepoint sequence plus the byte offset of each codepoint (and one past the end),
// so a splitter can work in codepoint indices and emit byte ranges.
struct Cps {
    std::vector<uint32_t> cp;
    std::vector<size_t> off;
};

Cps to_cps(const std::string& s) {
    Cps r;
    size_t i = 0;
    while (i < s.size()) {
        r.off.push_back(i);
        size_t len;
        r.cp.push_back(utf8_decode(s, i, len));
        i += len;
    }
    r.off.push_back(s.size());
    return r;
}

using Range = std::pair<size_t, size_t>;   // byte offsets [begin,end)

// ---- pre-tokeniser split functions ---------------------------------------
// Each returns byte ranges covering the WHOLE text, in order, with no gaps (they
// cover every position, exactly as llama.cpp's regexes do).

// The Qwen/llama3 family. `with_mark` admits \p{M} into the letter run (Qwen3.5);
// `digit_max` is the run length for the \p{N} branch (1 for Qwen, 3 for llama3).
std::vector<Range> split_qwen_like(const std::string& s, bool with_mark, int digit_max) {
    const Cps c = to_cps(s);
    const size_t n = c.cp.size();
    std::vector<Range> out;
    auto lt = [&](size_t p) { return p < n && (is_letter(c.cp[p]) || (with_mark && is_mark(c.cp[p]))); };
    auto emit = [&](size_t b, size_t e) { out.emplace_back(c.off[b], c.off[e]); };

    size_t pos = 0;
    while (pos < n) {
        const uint32_t cpt = c.cp[pos];

        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (cpt == '\'' && pos + 1 < n) {
            uint32_t n1 = c.cp[pos + 1];
            if (n1 >= 'A' && n1 <= 'Z') n1 += 32;
            if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') { emit(pos, pos + 2); pos += 2; continue; }
            if (pos + 2 < n) {
                uint32_t n2 = c.cp[pos + 2];
                if (n2 >= 'A' && n2 <= 'Z') n2 += 32;
                if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') || (n1 == 'l' && n2 == 'l')) {
                    emit(pos, pos + 3); pos += 3; continue;
                }
            }
        }

        // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        if (!(cpt == '\r' || cpt == '\n' || is_number(cpt))) {
            if (lt(pos) || is_mark(cpt) || lt(pos + 1) || (pos + 1 < n && is_mark(c.cp[pos + 1]))) {
                size_t p = pos + 1;
                while (lt(p) || (p < n && is_mark(c.cp[p]))) ++p;
                emit(pos, p); pos = p; continue;
            }
        }

        // \p{N}   (single digit for Qwen; \p{N}{1,3} for llama3)
        if (is_number(cpt)) {
            size_t p = pos + 1, k = 1;
            while (k < (size_t)digit_max && p < n && is_number(c.cp[p])) { ++p; ++k; }
            emit(pos, p); pos = p; continue;
        }

        // <space>?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        {
            auto nz = [&](size_t p) { return p < n && !(is_space(c.cp[p]) || is_letter(c.cp[p]) ||
                                                         (with_mark && is_mark(c.cp[p])) || is_number(c.cp[p])); };
            auto inrange = [&](size_t p) { return p < n && has_flags(c.cp[p]); };
            const bool f2 = (cpt == ' ') ? nz(pos + 1) : nz(pos);
            if (f2 && inrange(pos)) {
                size_t p = pos + (cpt == ' ' ? 1 : 0);
                while (nz(p) && inrange(p)) ++p;
                while (p < n && (c.cp[p] == '\r' || c.cp[p] == '\n')) ++p;
                emit(pos, p); pos = p; continue;
            }
        }

        // \s*[\r\n]+  |  \s+(?!\S)  |  \s+
        size_t nws = 0, last_rn = 0;
        while (pos + nws < n && is_space(c.cp[pos + nws])) {
            if (c.cp[pos + nws] == '\r' || c.cp[pos + nws] == '\n') last_rn = pos + nws + 1;
            ++nws;
        }
        if (last_rn > 0) { emit(pos, last_rn); pos = last_rn; continue; }
        if (nws > 1 && pos + nws < n) { emit(pos, pos + nws - 1); pos += nws - 1; continue; }
        if (nws > 0) { emit(pos, pos + nws); pos += nws; continue; }

        emit(pos, pos + 1); pos += 1;   // no match: one codepoint
    }
    return out;
}

// "[^\n]+|[\n]+"  -- runs of non-newline characters and runs of newline characters.
std::vector<Range> split_newlines(const std::string& s) {
    const Cps c = to_cps(s);
    const size_t n = c.cp.size();
    std::vector<Range> out;
    size_t pos = 0;
    while (pos < n) {
        const bool nl = (c.cp[pos] == '\n');
        size_t p = pos;
        while (p < n && ((c.cp[p] == '\n') == nl)) ++p;
        out.emplace_back(c.off[pos], c.off[p]);
        pos = p;
    }
    return out;
}

// The classic GPT-2 rule: "'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)".
// (No model with this pre-type is available to verify against, so it is implemented
// from the published pattern rather than pinned to the oracle.)
std::vector<Range> split_gpt2(const std::string& s) {
    const Cps c = to_cps(s);
    const size_t n = c.cp.size();
    std::vector<Range> out;
    auto emit = [&](size_t b, size_t e) { out.emplace_back(c.off[b], c.off[e]); };
    auto other = [&](size_t p) { return !(is_space(c.cp[p]) || is_letter(c.cp[p]) || is_number(c.cp[p])); };

    size_t pos = 0;
    while (pos < n) {
        const uint32_t cpt = c.cp[pos];
        if (cpt == '\'' && pos + 1 < n) {
            const uint32_t n1 = c.cp[pos + 1];
            if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') { emit(pos, pos + 2); pos += 2; continue; }
            if (pos + 2 < n) {
                const uint32_t n2 = c.cp[pos + 2];
                if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') || (n1 == 'l' && n2 == 'l')) {
                    emit(pos, pos + 3); pos += 3; continue;
                }
            }
        }
        // ' ?\p{L}+'
        if (is_letter(cpt) || (cpt == ' ' && pos + 1 < n && is_letter(c.cp[pos + 1]))) {
            size_t p = pos + (cpt == ' ' ? 1 : 0) + 1;
            while (p < n && is_letter(c.cp[p])) ++p;
            emit(pos, p); pos = p; continue;
        }
        // ' ?\p{N}+'
        if (is_number(cpt) || (cpt == ' ' && pos + 1 < n && is_number(c.cp[pos + 1]))) {
            size_t p = pos + (cpt == ' ' ? 1 : 0) + 1;
            while (p < n && is_number(c.cp[p])) ++p;
            emit(pos, p); pos = p; continue;
        }
        // ' ?[^\s\p{L}\p{N}]+'
        if (other(pos) || (cpt == ' ' && pos + 1 < n && other(pos + 1))) {
            size_t p = pos + (cpt == ' ' ? 1 : 0) + 1;
            while (p < n && other(p)) ++p;
            emit(pos, p); pos = p; continue;
        }
        // '\s+(?!\S)'
        if (is_space(cpt)) {
            size_t p = pos;
            while (p < n && is_space(c.cp[p])) ++p;
            if (p < n && p > pos) --p;      // greedy then backtrack off the last space
            if (p <= pos) p = pos + 1;
            emit(pos, p); pos = p; continue;
        }
        emit(pos, pos + 1); pos += 1;
    }
    return out;
}

std::vector<Range> split_for(PreType pre, const std::string& s) {
    switch (pre) {
        case PreType::QWEN35: return split_qwen_like(s, true, 1);
        case PreType::QWEN2:  return split_qwen_like(s, false, 1);
        case PreType::LLAMA3: return split_qwen_like(s, false, 3);
        case PreType::NEWLINES: return split_newlines(s);
        case PreType::GPT2:
        default: return split_gpt2(s);
    }
}

PreType resolve_pre(const std::string& model, const std::string& pre) {
    // `tokenizer.ggml.pre` when present, else inferred from the model name (gemma's
    // GGUF omits it). Unknown values fall back to the family's sane default rather
    // than guessing a wrong vocabulary split.
    std::string p = pre;
    if (p.empty()) {
        if (model.rfind("gemma", 0) == 0) p = "gemma4";
        else if (model == "qwen2" || model == "qwen35") p = model;
        else if (model == "gpt2") p = "gpt2";
        else if (model == "llama") p = "llama";
    }
    if (p == "qwen35") return PreType::QWEN35;
    if (p == "qwen2" || p == "qwen" || p == "deepseek-r1-qwen" || p == "megrez" ||
        p == "kormo" || p == "f2llmv2" || p == "stablelm2" || p == "hunyuan" ||
        p == "solar-open") return PreType::QWEN2;
    if (p == "gemma4" || p == "granite-embed-multi-311m" || p == "sarvam-moe" ||
        p == "whitespace") return PreType::NEWLINES;
    if (p == "llama3" || p == "llama-bpe" || p == "falcon" || p == "jais" ||
        p == "jais2") return PreType::LLAMA3;
    return PreType::GPT2;
}

}  // namespace

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

void Tokenizer::build_byte_table() {
    // GPT-2's byte -> unicode table: printable ASCII maps to itself; every remaining
    // byte is shifted into a contiguous range starting at U+0100, in byte order, so no
    // byte maps to whitespace or a control character. Must match exactly or every id
    // shifts.
    bool seen[256] = {false};
    std::vector<int> bs, cs;
    for (int b = '!'; b <= '~'; ++b) { bs.push_back(b); }
    for (int b = 0xA1; b <= 0xAC; ++b) { bs.push_back(b); }
    for (int b = 0xAE; b <= 0xFF; ++b) { bs.push_back(b); }
    for (int b : bs) seen[b] = true;
    cs = bs;
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
        if (!seen[b]) { bs.push_back(b); cs.push_back(256 + extra); ++extra; }
    }
    for (size_t i = 0; i < bs.size(); ++i) {
        std::string u;
        utf8_append(u, (uint32_t)cs[i]);
        b2u_[bs[i]] = u;
        u2b_[(uint32_t)cs[i]] = bs[i];
    }
}

void Tokenizer::load_arrays(const std::vector<std::string>& tokens,
                            const std::vector<std::string>& merges,
                            const std::vector<int64_t>& token_types,
                            VocabFamily family, PreType pre, bool add_bos,
                            bool add_space_prefix,
                            int32_t eos_id, int32_t bos_id, int32_t pad_id) {
    if (tokens.empty()) throw std::runtime_error("empty vocabulary");
    family_ = family;
    pre_ = pre;
    byte_encode_ = (family == VocabFamily::GPT2);
    add_bos_ = add_bos;
    add_space_prefix_ = add_space_prefix;
    bos_id_ = bos_id;
    eos_id_ = eos_id;
    pad_id_ = pad_id;

    tokens_ = tokens;
    encoder_.clear();
    encoder_.reserve(tokens_.size() * 2);
    for (size_t i = 0; i < tokens_.size(); ++i) {
        // first occurrence wins: a duplicate would otherwise shift every token
        encoder_.emplace(tokens_[i], (int32_t)i);
    }

    ranks_.clear();
    ranks_.reserve(merges.size() * 2);
    for (size_t rank = 0; rank < merges.size(); ++rank) {
        const std::string& m = merges[rank];
        const size_t sp = m.find(' ');
        if (sp == std::string::npos) throw std::runtime_error("merge " + std::to_string(rank) +
            " is not a pair: " + m);
        const std::string a = m.substr(0, sp), b = m.substr(sp + 1);
        if (b.find(' ') != std::string::npos) throw std::runtime_error("merge " + std::to_string(rank) +
            " is not a pair: " + m);
        // the rank key joins the two sides with NUL, which no token may contain
        if (a.find('\0') != std::string::npos || b.find('\0') != std::string::npos)
            throw std::runtime_error("merge " + std::to_string(rank) + " contains a NUL byte");
        ranks_.emplace(a + '\0' + b, (int32_t)rank);
    }
    n_merges_ = (int64_t)merges.size();

    // specials: the token types llama.cpp treats as special (control, unknown,
    // user-defined). type 5 (unused) and type 6 (byte) are NOT special -- llama.cpp
    // refutes treating them as such (it BPEs "[PAD248077]" and "<0x41>").
    special_map_.clear();
    special_ids_.clear();
    special_list_.clear();
    for (size_t i = 0; i < token_types.size() && i < tokens_.size(); ++i) {
        const int t = (int)token_types[i];
        if (t == 2 || t == 3 || t == 4) {
            special_map_.emplace(tokens_[i], (int32_t)i);
            special_ids_.insert((int32_t)i);
        }
    }
    for (const auto& kv : special_map_) special_list_.push_back({kv.second, kv.first});
    // longest text first, then lowest id (deterministic; matches Python's stable sort)
    std::sort(special_list_.begin(), special_list_.end(),
              [](const std::pair<int32_t, std::string>& x,
                 const std::pair<int32_t, std::string>& y) {
                  if (x.second.size() != y.second.size()) return x.second.size() > y.second.size();
                  return x.first < y.first;
              });

    // byte tokens: type 6 with the "<0xHH>" shape. Decode maps them back to the byte.
    byte_tokens_.clear();
    for (size_t i = 0; i < token_types.size() && i < tokens_.size(); ++i) {
        if ((int)token_types[i] != 6) continue;
        const std::string& t = tokens_[i];
        if (t.size() != 6 || t.compare(0, 3, "<0x") != 0 || t[5] != '>') continue;
        auto hexval = [](char c) -> int {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            return -1;
        };
        const int hi = hexval(t[3]), lo = hexval(t[4]);
        if (hi < 0 || lo < 0) continue;
        byte_tokens_[(int32_t)i] = (uint8_t)((hi << 4) | lo);
    }

    build_byte_table();
}

void Tokenizer::load(GgufReader& reader) {
    if (!reader.has("tokenizer.ggml.tokens") || !reader.has("tokenizer.ggml.merges"))
        throw std::runtime_error(
            "GGUF has no tokenizer (tokenizer.ggml.tokens/merges absent); this file "
            "was converted without one and cannot produce text");

    const std::string model = reader.get_str("tokenizer.ggml.model");
    VocabFamily family;
    if (model == "gpt2" || model == "qwen2") family = VocabFamily::GPT2;
    else if (model == "llama" || model == "spm" || model == "t5" || model == "umt5" ||
             model.rfind("gemma", 0) == 0) family = VocabFamily::SPM;
    else throw std::runtime_error("unsupported tokenizer model '" + model + "'; supported: gpt2 "
                                  "(byte-level BPE) and llama/spm/gemma* (sentencepiece)");

    const std::string pre = reader.get_str("tokenizer.ggml.pre");
    PreType pt = resolve_pre(model, pre);
    // gemma's GGUF carries no `pre`; its SPM family always splits on newlines
    if (family == VocabFamily::SPM && pt == PreType::GPT2) pt = PreType::NEWLINES;

    bool add_bos = reader.get_int("tokenizer.ggml.add_bos_token", 0) != 0;
    // llama.cpp forces add_bos=true for the gemma4 pre-type regardless of the file
    if (pt == PreType::NEWLINES && model.rfind("gemma", 0) == 0) add_bos = true;

    load_arrays(reader.get_str_array("tokenizer.ggml.tokens"),
                reader.get_str_array("tokenizer.ggml.merges"),
                reader.get_int_array("tokenizer.ggml.token_type"),
                family, pt, add_bos,
                reader.get_int("tokenizer.ggml.add_space_prefix", 0) != 0,
                (int32_t)reader.get_int("tokenizer.ggml.eos_token_id", -1),
                (int32_t)reader.get_int("tokenizer.ggml.bos_token_id", -1),
                (int32_t)reader.get_int("tokenizer.ggml.padding_token_id", -1));
}

std::vector<std::string> Tokenizer::bpe(const std::string& word) const {
    std::vector<std::string> syms = split_codepoints(word);
    if (syms.size() < 2) return syms;
    while (syms.size() > 1) {
        size_t best = (size_t)-1;
        int32_t best_rank = 0;
        for (size_t i = 0; i + 1 < syms.size(); ++i) {
            auto it = ranks_.find(syms[i] + '\0' + syms[i + 1]);
            if (it != ranks_.end() && (best == (size_t)-1 || it->second < best_rank)) {
                best = i;
                best_rank = it->second;
            }
        }
        if (best == (size_t)-1) break;
        syms[best] += syms[best + 1];
        syms.erase(syms.begin() + (long)best + 1);
    }
    return syms;
}

void Tokenizer::emit_symbol(const std::string& sym, std::vector<int32_t>& out, bool spm) const {
    auto it = encoder_.find(sym);
    if (it != encoder_.end()) { out.push_back(it->second); return; }
    // Unreachable for a well-formed vocab; never silently drop input.
    if (spm) {
        // sentencepiece fallback: per BYTE, the "<0xXX>" byte token (llama.cpp's rule)
        static const char* hex = "0123456789ABCDEF";
        for (unsigned char b : sym) {
            char buf[7] = {'<', '0', 'x', hex[b >> 4], hex[b & 15], '>', 0};
            auto t = encoder_.find(std::string(buf, 6));
            if (t != encoder_.end()) out.push_back(t->second);
        }
    } else {
        for (const std::string& cp : split_codepoints(sym)) {
            auto t = encoder_.find(cp);
            if (t != encoder_.end()) out.push_back(t->second);
        }
    }
}

std::vector<int32_t> Tokenizer::encode_plain(const std::string& text) const {
    std::vector<int32_t> out;
    if (family_ == VocabFamily::SPM) {
        // space -> U+2581 (no byte remap), then a newline split, then BPE per run.
        std::string s;
        s.reserve(text.size() + 1);
        if (add_space_prefix_) s += "\xE2\x96\x81";
        for (char ch : text) {
            if (ch == ' ') s += "\xE2\x96\x81";
            else s.push_back(ch);
        }
        const Cps c = to_cps(s);
        size_t i = 0;
        while (i < c.cp.size()) {
            const bool nl = (c.cp[i] == '\n');
            size_t j = i;
            while (j < c.cp.size() && ((c.cp[j] == '\n') == nl)) ++j;
            const std::string word = s.substr(c.off[i], c.off[j] - c.off[i]);
            // gemma4: a word of nothing but newlines that is itself a token is taken whole
            if (!word.empty() && word.find_first_not_of('\n') == std::string::npos) {
                auto it = encoder_.find(word);
                if (it != encoder_.end()) { out.push_back(it->second); i = j; continue; }
            }
            for (const std::string& sym : bpe(word)) emit_symbol(sym, out, true);
            i = j;
        }
        return out;
    }

    for (const Range& r : split_for(pre_, text)) {
        const std::string unit = text.substr(r.first, r.second - r.first);
        std::string mapped;
        mapped.reserve(unit.size());
        for (unsigned char b : unit) mapped += b2u_[b];
        for (const std::string& sym : bpe(mapped)) emit_symbol(sym, out, false);
    }
    return out;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, int add_bos_override) const {
    std::vector<int32_t> ids;
    const bool want_bos = (add_bos_override < 0) ? add_bos_ : (add_bos_override != 0);
    // llama.cpp appends BOS unconditionally when the model asks for it -- it does NOT
    // suppress it for a prompt that already starts with the BOS literal. Matched here
    // deliberately (our Python de-duplicated; llama.cpp's own tokenizer does not).
    if (want_bos && bos_id_ >= 0) ids.push_back(bos_id_);
    if (text.empty()) return ids;

    if (special_list_.empty()) {
        std::vector<int32_t> plain = encode_plain(text);
        ids.insert(ids.end(), plain.begin(), plain.end());
        return ids;
    }

    // split out special tokens first (longest text first at each position), so their
    // characters never reach the byte mapping or BPE
    std::string pending;
    size_t i = 0;
    while (i < text.size()) {
        int32_t sid = -1;
        size_t slen = 0;
        for (const auto& sp : special_list_) {
            if (text.compare(i, sp.second.size(), sp.second) == 0) { sid = sp.first; slen = sp.second.size(); break; }
        }
        if (slen > 0) {
            if (!pending.empty()) {
                std::vector<int32_t> plain = encode_plain(pending);
                ids.insert(ids.end(), plain.begin(), plain.end());
                pending.clear();
            }
            ids.push_back(sid);
            i += slen;
        } else {
            pending.push_back(text[i]);
            ++i;
        }
    }
    if (!pending.empty()) {
        std::vector<int32_t> plain = encode_plain(pending);
        ids.insert(ids.end(), plain.begin(), plain.end());
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids, bool skip_special) const {
    std::string joined;
    for (int32_t id : ids) {
        if (id < 0 || (size_t)id >= tokens_.size())
            throw std::out_of_range("token id " + std::to_string(id) +
                                    " out of range (vocab " + std::to_string(tokens_.size()) + ")");
        if (skip_special && special_ids_.count(id)) continue;
        // a byte token stands for a raw byte (a control char the vocab has no token
        // for); emitting the literal "<0x0D>" instead would corrupt the text
        auto bt = byte_tokens_.find(id);
        if (bt != byte_tokens_.end()) joined.push_back((char)bt->second);
        else joined += tokens_[(size_t)id];
    }
    if (family_ == VocabFamily::SPM) {
        std::string out;
        out.reserve(joined.size());
        for (size_t i = 0; i < joined.size();) {
            if (i + 3 <= joined.size() && (unsigned char)joined[i] == 0xE2 &&
                (unsigned char)joined[i + 1] == 0x96 && (unsigned char)joined[i + 2] == 0x81) {
                out.push_back(' ');
                i += 3;
            } else {
                out.push_back(joined[i]);
                ++i;
            }
        }
        return out;
    }
    std::string bytes;
    bytes.reserve(joined.size());
    size_t i = 0;
    while (i < joined.size()) {
        size_t len;
        const uint32_t cp = utf8_decode(joined, i, len);
        auto it = u2b_.find(cp);
        if (it != u2b_.end()) bytes.push_back((char)it->second);
        else bytes.append(joined, i, len);   // not from the byte table: pass UTF-8 through
        i += len;
    }
    return utf8_replace(bytes);
}

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------
namespace {
struct TokHandle {
    aero::Tokenizer tk;
};
std::string g_tok_error;
}  // namespace

extern "C" {

const char* aero_tok_last_error(void) { return g_tok_error.c_str(); }

void* aero_tok_open(const char* path) {
    g_tok_error.clear();
    if (!path) { g_tok_error = "null path"; return nullptr; }
    try {
        // The reader is scoped: the tokenizer copies everything it needs, so the mmap
        // and the metadata arrays are released immediately (the tokenizer alone holds
        // the vocab, not both copies).
        aero::GgufReader reader(path);
        auto* h = new TokHandle();
        h->tk.load(reader);
        return h;
    } catch (const std::exception& e) {
        g_tok_error = e.what();
        return nullptr;
    }
}

void aero_tok_free(void* h) { delete static_cast<TokHandle*>(h); }

int64_t aero_tok_n_vocab(void* h) {
    auto* t = static_cast<TokHandle*>(h);
    return t ? t->tk.n_vocab() : -1;
}
int64_t aero_tok_n_merges(void* h) {
    auto* t = static_cast<TokHandle*>(h);
    return t ? t->tk.n_merges() : -1;
}
int64_t aero_tok_n_special(void* h) {
    auto* t = static_cast<TokHandle*>(h);
    return t ? t->tk.n_special() : -1;
}
int64_t aero_tok_bos_id(void* h) {
    auto* t = static_cast<TokHandle*>(h);
    return t ? t->tk.bos_id() : -1;
}
int64_t aero_tok_eos_id(void* h) {
    auto* t = static_cast<TokHandle*>(h);
    return t ? t->tk.eos_id() : -1;
}
int64_t aero_tok_pad_id(void* h) {
    auto* t = static_cast<TokHandle*>(h);
    return t ? t->tk.pad_id() : -1;
}
int aero_tok_add_bos(void* h) {
    auto* t = static_cast<TokHandle*>(h);
    return t ? (t->tk.add_bos() ? 1 : 0) : -1;
}
int aero_tok_family(void* h) {
    auto* t = static_cast<TokHandle*>(h);
    if (!t) return -1;
    return t->tk.family() == aero::VocabFamily::SPM ? 1 : 0;
}
int aero_tok_pre(void* h) {
    auto* t = static_cast<TokHandle*>(h);
    return t ? (int)t->tk.pre() : -1;
}
int aero_tok_is_special_id(void* h, int32_t id) {
    auto* t = static_cast<TokHandle*>(h);
    return t ? (t->tk.is_special_id(id) ? 1 : 0) : -1;
}

int64_t aero_tok_token(void* h, int32_t id, char* out, int64_t cap) {
    auto* t = static_cast<TokHandle*>(h);
    if (!t || id < 0 || (int64_t)id >= t->tk.n_vocab()) return -1;
    const std::string& s = t->tk.token(id);
    if (out && cap > 0) std::memcpy(out, s.data(), (size_t)std::min<int64_t>((int64_t)s.size(), cap));
    return (int64_t)s.size();
}

int64_t aero_tok_encode(void* h, const char* text, int64_t len,
                        int32_t* out, int64_t cap, int add_bos) {
    auto* t = static_cast<TokHandle*>(h);
    if (!t || len < 0 || (!text && len > 0)) return -1;
    std::vector<int32_t> ids;
    try {
        ids = t->tk.encode(std::string(text ? text : "", (size_t)len), add_bos);
    } catch (const std::exception&) {
        return -1;
    }
    const int64_t n = (int64_t)ids.size();
    if (out) {
        if (cap < n) return -1;                       // refuse rather than truncate
        if (n) std::memcpy(out, ids.data(), (size_t)n * sizeof(int32_t));
    }
    return n;
}

int64_t aero_tok_decode(void* h, const int32_t* ids, int64_t n,
                        char* out, int64_t cap, int skip_special) {
    auto* t = static_cast<TokHandle*>(h);
    if (!t || n < 0 || (!ids && n > 0)) return -1;
    std::string s;
    try {
        s = t->tk.decode(std::vector<int32_t>(ids, ids + n), skip_special != 0);
    } catch (const std::exception&) {
        return -1;                                    // out-of-range id: refuse, do not wrap
    }
    const int64_t len = (int64_t)s.size();
    if (out) {
        if (cap < len) return -1;
        if (len) std::memcpy(out, s.data(), (size_t)len);
    }
    return len;
}

}  // extern "C"

}  // namespace aero
