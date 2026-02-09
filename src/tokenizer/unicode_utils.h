#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace UnicodeUtils {

// Decode one UTF-8 codepoint from a byte sequence.
// Returns the codepoint and advances pos past the consumed bytes.
// Returns 0xFFFD (replacement char) on invalid sequences.
uint32_t decodeUtf8(const char* data, size_t len, size_t& pos);

// Decode entire UTF-8 string into codepoints.
std::vector<uint32_t> toCodepoints(const std::string& utf8);

// Encode a single codepoint to UTF-8 bytes.
std::string encodeUtf8(uint32_t cp);

// Encode codepoints to UTF-8 string.
std::string fromCodepoints(const std::vector<uint32_t>& cps);

// Unicode category checks (broad categories for pre-tokenization).
bool isLetter(uint32_t cp);
bool isNumber(uint32_t cp);
bool isWhitespace(uint32_t cp);
bool isPunctuation(uint32_t cp);

// Simple NFC normalization (pass-through for ASCII; for non-ASCII
// this is a best-effort implementation covering common cases).
std::string nfcNormalize(const std::string& input);

} // namespace UnicodeUtils
