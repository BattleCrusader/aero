// Chat-envelope rendering, native. See include/aero/chat_template.hpp for why.
//
// The literals below are transcribed from engine/chat_template.py, verified
// against its rendered output byte-for-byte (repr compared) rather than against
// a reading of the Jinja: the Python renderer is the oracle this port must
// match. Note the leading SPACE in chatml's " thinking": it is in the shipped
// renderer and in the GGUF's own template, and dropping it would change the
// prompt by one byte.

#include "aero/chat_template.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace aero {

namespace {

struct EnvelopeSpec {
    Envelope env;
    const char* name;
    const char* marker;
};

// Order is load-bearing: gemma4 first, exactly as engine.chat_template.ENVELOPES.
const EnvelopeSpec kEnvelopes[] = {
    {ENV_GEMMA4, "gemma4", "<|turn>"},
    {ENV_CHATML, "chatml", "<|im_start|>"},
};

// ---- gemma4 ------------------------------------------------------------------
// bos is emitted by the template itself; the assistant role is 'model'.
std::string render_gemma4(const std::vector<ChatMessage>& messages,
                          bool add_generation_prompt, bool enable_thinking) {
    std::string out = "<bos>";
    for (const ChatMessage& m : messages) {
        const std::string& role = m.role;
        out += "<|turn>";
        out += (role == "assistant") ? "model" : role;
        out += "\n";
        out += m.content;
        out += "<turn|>\n";
    }
    if (add_generation_prompt) {
        out += "<|turn>model\n";
        // INVERTED on purpose: thinking OFF closes the channel immediately,
        // which is what forces a direct answer.
        if (!enable_thinking) out += "<|channel>thought\n<channel|>";
    }
    return out;
}

// ---- chatml (qwen) -----------------------------------------------------------
std::string render_chatml(const std::vector<ChatMessage>& messages,
                          bool add_generation_prompt, bool enable_thinking) {
    std::string out;
    for (const ChatMessage& m : messages) {
        out += "<|im_start|>";
        out += m.role;
        out += "\n";
        out += m.content;
        out += "<|im_end|>\n";
    }
    if (add_generation_prompt) {
        out += "<|im_start|>assistant\n";
        out += enable_thinking ? " thinking\n" : " thinking\n\n</think>\n\n";
    }
    return out;
}

const EnvelopeSpec* spec_for(Envelope e) {
    for (const EnvelopeSpec& s : kEnvelopes)
        if (s.env == e) return &s;
    return nullptr;
}

}  // namespace

bool thinking_from_env() {
    const char* v = std::getenv("AERO_THINKING");
    return v != nullptr && std::strcmp(v, "1") == 0;
}

Envelope detect_envelope(const std::vector<std::string>& specials) {
    for (const EnvelopeSpec& s : kEnvelopes)
        for (const std::string& tok : specials)
            if (tok == s.marker) return s.env;
    return ENV_UNKNOWN;
}

std::string render_chat(const std::vector<ChatMessage>& messages, Envelope env,
                        bool add_generation_prompt, int enable_thinking) {
    if (enable_thinking < 0) enable_thinking = thinking_from_env() ? 1 : 0;
    switch (env) {
        case ENV_GEMMA4:
            return render_gemma4(messages, add_generation_prompt, enable_thinking != 0);
        case ENV_CHATML:
            return render_chatml(messages, add_generation_prompt, enable_thinking != 0);
        default:
            throw std::invalid_argument(
                "unrecognised chat template: this file has neither '<|turn>' "
                "(gemma4) nor '<|im_start|>' (chatml). Refusing to emit unframed text.");
    }
}

const char* envelope_name(Envelope e) {
    const EnvelopeSpec* s = spec_for(e);
    return s ? s->name : "";
}

const char* envelope_marker(Envelope e) {
    const EnvelopeSpec* s = spec_for(e);
    return s ? s->marker : "";
}

}  // namespace aero

// ---------------------------------------------------------------------- C ABI
// The helpers live OUTSIDE the extern "C" block on purpose: a namespace declared
// inside `extern "C"` gives its members C linkage, which both exports internal
// helpers as global symbols (nm shows _gather/_render_impl) and clashes with
// their internal linkage. Keep them at file scope; the ABI functions below call
// them.
namespace {

// Collect a `const char* const*` array into a vector<string>, skipping NULLs
// (a NULL entry is not a special token; Python's dict cannot hold one).
std::vector<std::string> gather(const char* const* arr, int n) {
    std::vector<std::string> v;
    if (arr == nullptr || n <= 0) return v;
    v.reserve((size_t)n);
    for (int i = 0; i < n; ++i)
        if (arr[i] != nullptr) v.emplace_back(arr[i]);
    return v;
}

// Translate the C arguments into a rendered string. Returns a negative
// AERO_CHAT_ERR_* on refusal instead of throwing across the boundary.
int render_impl(const char* const* roles, const char* const* contents,
                int n_messages, const char* const* specials, int n_specials,
                int add_generation_prompt, int enable_thinking, std::string* out) {
    if (specials == nullptr) return AERO_CHAT_ERR_NO_TOKENIZER;
    if (n_messages < 0) return AERO_CHAT_ERR_BAD_ARGS;
    if (n_messages > 0 && (roles == nullptr || contents == nullptr))
        return AERO_CHAT_ERR_BAD_ARGS;

    std::vector<std::string> toks = gather(specials, n_specials);
    aero::Envelope env = aero::detect_envelope(toks);
    if (env == aero::ENV_UNKNOWN) return AERO_CHAT_ERR_UNKNOWN_TEMPLATE;

    std::vector<aero::ChatMessage> msgs;
    msgs.reserve((size_t)n_messages);
    for (int i = 0; i < n_messages; ++i) {
        aero::ChatMessage m;
        // Python: m.get("role", "user") / m.get("content", "")
        m.role = (roles[i] != nullptr) ? roles[i] : "user";
        m.content = (contents[i] != nullptr) ? contents[i] : "";
        msgs.push_back(std::move(m));
    }

    try {
        *out = aero::render_chat(msgs, env, add_generation_prompt != 0, enable_thinking);
    } catch (const std::exception&) {
        return AERO_CHAT_ERR_UNKNOWN_TEMPLATE;
    }
    return (int)out->size();
}

}  // namespace

extern "C" {

int aero_chat_detect(const char* const* specials, int n_specials) {
    if (specials == nullptr) return AERO_CHAT_ERR_NO_TOKENIZER;
    return (int)aero::detect_envelope(gather(specials, n_specials));
}

int aero_chat_render_size(const char* const* roles, const char* const* contents,
                          int n_messages, const char* const* specials,
                          int n_specials, int add_generation_prompt,
                          int enable_thinking) {
    std::string s;
    int rc = render_impl(roles, contents, n_messages, specials, n_specials,
                         add_generation_prompt, enable_thinking, &s);
    if (rc < 0) return rc;
    return (int)s.size() + 1;   // + NUL
}

int aero_chat_render_into(const char* const* roles, const char* const* contents,
                          int n_messages, const char* const* specials,
                          int n_specials, int add_generation_prompt,
                          int enable_thinking, char* out, int out_len) {
    if (out == nullptr || out_len <= 0) return AERO_CHAT_ERR_BAD_ARGS;
    // Fail closed: make the buffer empty before anything can read it.
    out[0] = '\0';
    std::string s;
    int rc = render_impl(roles, contents, n_messages, specials, n_specials,
                         add_generation_prompt, enable_thinking, &s);
    if (rc < 0) return rc;
    if ((int)s.size() + 1 > out_len) return AERO_CHAT_ERR_TOO_SMALL;
    std::memcpy(out, s.data(), s.size());
    out[s.size()] = '\0';
    return (int)s.size();
}

char* aero_chat_render(const char* const* roles, const char* const* contents,
                       int n_messages, const char* const* specials,
                       int n_specials, int add_generation_prompt,
                       int enable_thinking) {
    std::string s;
    int rc = render_impl(roles, contents, n_messages, specials, n_specials,
                         add_generation_prompt, enable_thinking, &s);
    if (rc < 0) return nullptr;
    char* p = (char*)std::malloc(s.size() + 1);
    if (p == nullptr) return nullptr;
    std::memcpy(p, s.data(), s.size());
    p[s.size()] = '\0';
    return p;
}

void aero_chat_free(char* p) { std::free(p); }

const char* aero_chat_envelope_name(int env) {
    return aero::envelope_name((aero::Envelope)env);
}

const char* aero_chat_marker(int env) { return aero::envelope_marker((aero::Envelope)env); }

int aero_chat_thinking_from_env(void) { return aero::thinking_from_env() ? 1 : 0; }

}  // extern "C"
