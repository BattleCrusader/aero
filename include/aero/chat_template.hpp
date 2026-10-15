// Chat-envelope rendering, native. A port of engine/chat_template.py.
//
// WHY THIS EXISTS AT ALL. The interactive runtime and every check tool MUST
// produce byte-identical input for the same conversation. A tool that renders its
// own prompt "verifies" something the engine never sends -- which is exactly how
// the BOS bug hid (the engine dropped BOS, the tool did not test through the
// engine, and the model's garbage looked like an architecture bug). So the
// renderer is one piece of code, and the Python one is the ORACLE: this port is
// accepted only when its bytes match engine/chat_template.py's render_chat()
// exactly, and tests/test_chat_template_cpp.py proves that.
//
// WHAT EACH ENVELOPE IS (transcribed from `tokenizer.chat_template` in the GGUF;
// the real gemma-4 template is the 18922-char Jinja string in the file and the
// qwen one is 7756 chars -- see the test's real-file section):
//
//   gemma4:  <bos>                         (the template emits bos_token ITSELF,
//                                           so the text is tokenized add_bos=false
//                                           and the tokenizer must not add a second)
//            per message: <|turn>{role}\n{content}<turn|>\n
//            gen prompt:  <|turn>model\n  (+ "<|channel>thought\n<channel|>"
//                                          when thinking is DISABLED)
//   chatml:  per message: <|im_start|>{role}\n{content}<|im_end|>\n
//            gen prompt:  <|im_start|>assistant\n
//                         thinking ON : " thinking\n"
//                         thinking OFF: " thinking\n\n</think>\n\n"
//
// TWO DETAILS THAT ARE EASY TO GET BACKWARDS, both pinned below:
//   * gemma4 spells the assistant role 'model', not 'assistant';
//   * gemma4's thinking flag is INVERTED from intuition -- thinking DISABLED
//     emits an immediately-closed thought channel, and that is what forces a
//     direct answer; thinking ENABLED emits nothing, leaving the model inside
//     the channel so it thinks out loud.
//
// SCOPE. This mirrors the PYTHON renderer exactly, down to its simplifications.
// The real gemma4 Jinja also trims user content, merges consecutive assistant
// turns and gates the thinking channel on tool state; the shipped engine does
// none of that and the C++ must not either, or the two prompts would differ by
// a space and this port would be worthless. Where the Python renderer is silent
// (unknown role, missing role/content key) this port is silent the same way.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace aero {

// Mirrors engine.chat_template.ENVELOPES, in the SAME order: gemma4 is probed
// first, so a tokenizer that carries both markers resolves to gemma4.
enum Envelope : int {
    ENV_UNKNOWN = -1,
    ENV_GEMMA4  = 0,
    ENV_CHATML  = 1,
};

struct ChatMessage {
    std::string role;
    std::string content;
};

// Name of the envelope this file's own special-token table calls for, or
// ENV_UNKNOWN. Detection is driven by the file's tokens, never by a filename or
// a guess -- the same rule detect_envelope() applies to tokenizer._special.
Envelope detect_envelope(const std::vector<std::string>& specials);

// Render `messages` the way that envelope does.
//
// `enable_thinking`: 1 on, 0 off, ANY NEGATIVE -> read AERO_THINKING (that is
// Python's `enable_thinking=None` path; "1" means on, anything else off).
//
// Throws std::invalid_argument for ENV_UNKNOWN -- refusing to emit unframed
// text, which is the whole point of the ENVELOPES table. The C ABI converts
// that into AERO_CHAT_ERR_UNKNOWN_TEMPLATE rather than letting an exception
// cross the boundary.
std::string render_chat(const std::vector<ChatMessage>& messages, Envelope env,
                        bool add_generation_prompt, int enable_thinking);

// AERO_THINKING resolved the way the Python renderer resolves it.
bool thinking_from_env();

const char* envelope_name(Envelope e);    // "gemma4" / "chatml" / ""
const char* envelope_marker(Envelope e);  // "<|turn>" / "<|im_start|>" / ""

}  // namespace aero

// ---------------------------------------------------------------------- C ABI
// The engine drives this through ctypes (and, later, directly). Rules:
//   * no C++ type, no exception and no allocation lifetime crosses the boundary;
//   * role/content arrays are parallel `const char*` arrays; a NULL role is
//     "user" and a NULL content is "" (Python's m.get(role, "user")/m.get(
//     "content", "")), and a NULL `specials` means "no tokenizer at all".
//   * negative return codes are errors and are never a length.
extern "C" {

enum {
    AERO_CHAT_ERR_UNKNOWN_TEMPLATE = -1,  // neither marker present
    AERO_CHAT_ERR_NO_TOKENIZER     = -2,  // specials == NULL
    AERO_CHAT_ERR_BAD_ARGS         = -3,  // n_messages < 0, NULL arrays, ...
    AERO_CHAT_ERR_TOO_SMALL        = -4,  // out buffer cannot hold the prompt
};

// Envelope code for `specials` (see aero::Envelope / ENV_*), or a negative
// AERO_CHAT_ERR_* code.
int aero_chat_detect(const char* const* specials, int n_specials);

// Bytes needed INCLUDING the NUL terminator, or a negative AERO_CHAT_ERR_*.
int aero_chat_render_size(const char* const* roles, const char* const* contents,
                          int n_messages, const char* const* specials,
                          int n_specials, int add_generation_prompt,
                          int enable_thinking);

// Render into `out`. Returns the number of bytes written EXCLUDING the NUL, or
// a negative AERO_CHAT_ERR_*. Never writes past out_len, and `out` is always
// NUL-terminated on success.
int aero_chat_render_into(const char* const* roles, const char* const* contents,
                          int n_messages, const char* const* specials,
                          int n_specials, int add_generation_prompt,
                          int enable_thinking, char* out, int out_len);

// Convenience: a malloc'd copy the caller frees with aero_chat_free(). NULL on
// any error (the render then refuses rather than emitting unframed text).
char* aero_chat_render(const char* const* roles, const char* const* contents,
                       int n_messages, const char* const* specials,
                       int n_specials, int add_generation_prompt,
                       int enable_thinking);
void aero_chat_free(char* p);

const char* aero_chat_envelope_name(int env);
const char* aero_chat_marker(int env);
int aero_chat_thinking_from_env(void);

}  // extern "C"
