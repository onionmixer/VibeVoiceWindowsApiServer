#include "tokenizer/bpe_tokenizer.h"
#include "tokenizer/special_tokens.h"
#include "tokenizer/unicode_utils.h"
#include <json.hpp>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>

using json = nlohmann::json;

const std::string BPETokenizer::emptyString_;

// ── GPT-2 / Qwen2 byte-level BPE encoding table ──
// Maps byte values 0-255 to unicode characters used in the BPE vocab.
// Printable ASCII bytes map to themselves; non-printable bytes map to
// unicode chars starting at U+0100 (Ā, ā, Ă, ...).

static void buildByteToUnicode(std::string byteToChar[256],
                                std::unordered_map<uint32_t, uint8_t>& charToByte) {
    // Bytes that map directly to their unicode codepoint
    // (printable ASCII range + Latin-1 Supplement printable)
    int n = 0;
    uint32_t mapping[256];

    for (int b = 0; b < 256; ++b) {
        if ((b >= 0x21 && b <= 0x7E) ||    // ASCII printable (! to ~)
            (b >= 0xA1 && b <= 0xAC) ||    // Latin-1 Supplement ¡ to ¬
            (b >= 0xAE && b <= 0xFF)) {    // Latin-1 Supplement ® to ÿ
            mapping[b] = (uint32_t)b;
        } else {
            mapping[b] = 256 + n;
            ++n;
        }
    }

    for (int b = 0; b < 256; ++b) {
        byteToChar[b] = UnicodeUtils::encodeUtf8(mapping[b]);
        charToByte[mapping[b]] = (uint8_t)b;
    }
}

// ── Constructor / Destructor ──

BPETokenizer::BPETokenizer() {
    buildByteToUnicode(byteToChar_, charToByte_);
}

BPETokenizer::~BPETokenizer() = default;

// ── Loading ──

bool BPETokenizer::load(const std::string& tokenizerJsonPath) {
    std::ifstream f(tokenizerJsonPath);
    if (!f.is_open()) {
        fprintf(stderr, "Cannot open tokenizer: %s\n", tokenizerJsonPath.c_str());
        return false;
    }

    json j;
    try {
        f >> j;
    } catch (const json::parse_error& e) {
        fprintf(stderr, "JSON parse error in tokenizer: %s\n", e.what());
        return false;
    }

    // Parse added tokens
    if (j.contains("added_tokens")) {
        for (auto& at : j["added_tokens"]) {
            AddedToken tok;
            tok.content = at.value("content", "");
            tok.id = at.value("id", -1);
            tok.special = at.value("special", false);
            addedTokens_.push_back(tok);
        }
    }

    // Parse vocab
    if (!j.contains("model") || !j["model"].contains("vocab")) {
        fprintf(stderr, "No model.vocab in tokenizer.json\n");
        return false;
    }

    auto& vocab = j["model"]["vocab"];
    int maxId = -1;
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        std::string token = it.key();
        int32_t id = it.value().get<int32_t>();
        tokenToId_[token] = id;
        if (id > maxId) maxId = id;
    }

    // Also add added_tokens to vocab map
    for (auto& at : addedTokens_) {
        tokenToId_[at.content] = at.id;
        if (at.id > maxId) maxId = at.id;
    }

    // Build id -> token lookup
    idToToken_.resize(maxId + 1);
    for (auto& kv : tokenToId_) {
        if (kv.second >= 0 && kv.second <= maxId) {
            idToToken_[kv.second] = kv.first;
        }
    }

    // Parse merges
    if (j["model"].contains("merges")) {
        auto& merges = j["model"]["merges"];
        for (size_t i = 0; i < merges.size(); ++i) {
            auto& merge = merges[i];
            std::string first, second;
            if (merge.is_array() && merge.size() == 2) {
                first = merge[0].get<std::string>();
                second = merge[1].get<std::string>();
            } else if (merge.is_string()) {
                // "token1 token2" format
                std::string s = merge.get<std::string>();
                size_t sp = s.find(' ');
                if (sp == std::string::npos) continue;
                first = s.substr(0, sp);
                second = s.substr(sp + 1);
            } else {
                continue;
            }
            std::string key = first + " " + second;
            mergeRanks_[key] = (int)i;
        }
    }

    loaded_ = true;
    fprintf(stderr, "Tokenizer loaded: vocab=%zu, merges=%zu, added_tokens=%zu\n",
            tokenToId_.size(), mergeRanks_.size(), addedTokens_.size());
    return true;
}

// ── Pre-tokenization (Qwen2 regex pattern) ──
// Pattern: (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
//          ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+

std::vector<std::string> BPETokenizer::preTokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    auto cps = UnicodeUtils::toCodepoints(text);
    size_t n = cps.size();
    size_t i = 0;

    // Helper to build a UTF-8 string from a range of codepoints
    auto cpSlice = [&](size_t from, size_t to) -> std::string {
        std::string s;
        for (size_t k = from; k < to; ++k) {
            s += UnicodeUtils::encodeUtf8(cps[k]);
        }
        return s;
    };

    // Check for contraction suffix starting at position i
    // Returns length in codepoints if matched, 0 otherwise
    auto matchContraction = [&](size_t pos) -> size_t {
        if (pos >= n) return 0;
        // Must start with ' (apostrophe U+0027 or U+2019)
        if (cps[pos] != '\'' && cps[pos] != 0x2019) return 0;
        if (pos + 1 >= n) return 0;

        uint32_t next = cps[pos + 1] | 0x20; // lowercase
        // 's, 't, 'm, 'd (2 chars)
        if (next == 's' || next == 't' || next == 'm' || next == 'd') return 2;
        // 're, 've, 'll (3 chars)
        if (pos + 2 < n) {
            uint32_t next2 = cps[pos + 2] | 0x20;
            if (next == 'r' && next2 == 'e') return 3;
            if (next == 'v' && next2 == 'e') return 3;
            if (next == 'l' && next2 == 'l') return 3;
        }
        return 0;
    };

    while (i < n) {
        // Try contraction: (?i:'s|'t|'re|'ve|'m|'ll|'d)
        size_t contrLen = matchContraction(i);
        if (contrLen > 0) {
            tokens.push_back(cpSlice(i, i + contrLen));
            i += contrLen;
            continue;
        }

        // Try: [^\r\n\p{L}\p{N}]?\p{L}+
        {
            size_t start = i;
            size_t j = i;

            // Optional non-letter, non-number, non-newline char
            if (j < n && !UnicodeUtils::isLetter(cps[j]) &&
                !UnicodeUtils::isNumber(cps[j]) &&
                cps[j] != '\r' && cps[j] != '\n') {
                // Check if followed by at least one letter
                if (j + 1 < n && UnicodeUtils::isLetter(cps[j + 1])) {
                    j++;
                }
            }

            // One or more letters
            if (j < n && UnicodeUtils::isLetter(cps[j])) {
                while (j < n && UnicodeUtils::isLetter(cps[j])) j++;
                tokens.push_back(cpSlice(start, j));
                i = j;
                continue;
            }
        }

        // Try: \p{N}
        if (UnicodeUtils::isNumber(cps[i])) {
            tokens.push_back(cpSlice(i, i + 1));
            i++;
            continue;
        }

        // Try: ?[^\s\p{L}\p{N}]+[\r\n]*
        {
            size_t start = i;
            size_t j = i;

            // Optional space
            if (j < n && cps[j] == ' ') j++;

            size_t symStart = j;
            // One or more non-whitespace, non-letter, non-number
            while (j < n && !UnicodeUtils::isWhitespace(cps[j]) &&
                   !UnicodeUtils::isLetter(cps[j]) &&
                   !UnicodeUtils::isNumber(cps[j])) {
                j++;
            }

            if (j > symStart) {
                // Consume trailing \r\n
                while (j < n && (cps[j] == '\r' || cps[j] == '\n')) j++;
                tokens.push_back(cpSlice(start, j));
                i = j;
                continue;
            }

            // If we consumed only a space but no symbols, don't match this rule
            if (j > start && j == symStart) {
                // Fall through to whitespace rules
            }
        }

        // Try: \s*[\r\n]+
        {
            size_t j = i;
            while (j < n && UnicodeUtils::isWhitespace(cps[j]) &&
                   cps[j] != '\r' && cps[j] != '\n') {
                j++;
            }
            if (j < n && (cps[j] == '\r' || cps[j] == '\n')) {
                while (j < n && (cps[j] == '\r' || cps[j] == '\n')) j++;
                tokens.push_back(cpSlice(i, j));
                i = j;
                continue;
            }
        }

        // Try: \s+(?!\S) — whitespace not followed by non-whitespace (trailing spaces)
        // Try: \s+ — any whitespace sequence
        if (UnicodeUtils::isWhitespace(cps[i])) {
            size_t j = i;
            while (j < n && UnicodeUtils::isWhitespace(cps[j])) j++;

            // \s+(?!\S): trailing whitespace (followed by nothing or more whitespace)
            // In practice, emit all but the last space if followed by non-ws
            if (j < n) {
                // There's a non-ws char after — this is \s+ case
                // but Qwen2 splits: all-but-last as (?!\S) match, last as part of next
                // Actually the Qwen2 regex is: \s+(?!\S)|\s+
                // (?!\S) means "not followed by non-whitespace" = "followed by whitespace or end"
                // So if followed by non-whitespace, the (?!\S) fails, and \s+ matches all
                tokens.push_back(cpSlice(i, j));
                i = j;
            } else {
                // End of string — (?!\S) succeeds
                tokens.push_back(cpSlice(i, j));
                i = j;
            }
            continue;
        }

        // Fallback: single character
        tokens.push_back(cpSlice(i, i + 1));
        i++;
    }

    return tokens;
}

// ── Byte-level encoding ──

std::string BPETokenizer::bytesToBpeChars(const std::string& word) const {
    std::string result;
    for (uint8_t b : word) {
        result += byteToChar_[b];
    }
    return result;
}

std::string BPETokenizer::bpeCharsToBytes(const std::string& token) const {
    std::string result;
    auto cps = UnicodeUtils::toCodepoints(token);
    for (uint32_t cp : cps) {
        auto it = charToByte_.find(cp);
        if (it != charToByte_.end()) {
            result += (char)it->second;
        }
    }
    return result;
}

// ── BPE merge algorithm ──

static std::string makePairKey(const std::string& a, const std::string& b) {
    return a + " " + b;
}

std::vector<int32_t> BPETokenizer::bpeEncode(const std::string& bpeWord) const {
    // Split into individual unicode characters
    auto cps = UnicodeUtils::toCodepoints(bpeWord);
    std::vector<std::string> symbols;
    for (uint32_t cp : cps) {
        symbols.push_back(UnicodeUtils::encodeUtf8(cp));
    }

    if (symbols.size() <= 1) {
        // Single char or empty — look up directly
        auto it = tokenToId_.find(bpeWord);
        if (it != tokenToId_.end()) {
            return {it->second};
        }
        // Unknown single char — should not happen with byte-level BPE
        return {};
    }

    // Iteratively merge the highest-priority pair
    while (symbols.size() > 1) {
        // Find the pair with the lowest merge rank
        int bestRank = std::numeric_limits<int>::max();
        size_t bestIdx = (size_t)-1;

        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            std::string key = makePairKey(symbols[i], symbols[i + 1]);
            auto it = mergeRanks_.find(key);
            if (it != mergeRanks_.end() && it->second < bestRank) {
                bestRank = it->second;
                bestIdx = i;
            }
        }

        if (bestIdx == (size_t)-1) break; // No more merges possible

        // Merge the pair at bestIdx
        std::string merged = symbols[bestIdx] + symbols[bestIdx + 1];
        symbols[bestIdx] = merged;
        symbols.erase(symbols.begin() + bestIdx + 1);
    }

    // Convert symbols to token IDs
    std::vector<int32_t> ids;
    for (auto& sym : symbols) {
        auto it = tokenToId_.find(sym);
        if (it != tokenToId_.end()) {
            ids.push_back(it->second);
        } else {
            // Unknown token — encode individual bytes as fallback
            for (uint8_t b : sym) {
                auto bit = tokenToId_.find(byteToChar_[b]);
                if (bit != tokenToId_.end()) {
                    ids.push_back(bit->second);
                }
            }
        }
    }

    return ids;
}

// ── Public encode/decode ──

std::vector<int32_t> BPETokenizer::encode(const std::string& text) const {
    if (!loaded_) return {};

    // First, handle added/special tokens by splitting text around them
    // Sort by length (longest first) to avoid partial matches
    struct ATMatch { size_t pos; size_t len; int32_t id; };
    std::vector<ATMatch> matches;

    // Find all added token occurrences
    std::string remaining = text;
    std::vector<std::pair<std::string, int32_t>> sortedAdded;
    for (auto& at : addedTokens_) {
        sortedAdded.push_back({at.content, at.id});
    }
    // Sort by length descending for greedy matching
    std::sort(sortedAdded.begin(), sortedAdded.end(),
              [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

    // Split text into segments: (text_chunk, special_token_id)*
    struct Segment {
        std::string text;
        int32_t specialId; // -1 if regular text
    };
    std::vector<Segment> segments;

    size_t searchStart = 0;
    while (searchStart < text.size()) {
        // Find earliest added token match
        size_t bestPos = std::string::npos;
        size_t bestLen = 0;
        int32_t bestId = -1;

        for (auto& [tok, id] : sortedAdded) {
            size_t pos = text.find(tok, searchStart);
            if (pos != std::string::npos && (pos < bestPos || (pos == bestPos && tok.size() > bestLen))) {
                bestPos = pos;
                bestLen = tok.size();
                bestId = id;
            }
        }

        if (bestPos == std::string::npos) {
            // No more special tokens — rest is regular text
            segments.push_back({text.substr(searchStart), -1});
            break;
        }

        // Text before the special token
        if (bestPos > searchStart) {
            segments.push_back({text.substr(searchStart, bestPos - searchStart), -1});
        }

        // The special token itself
        segments.push_back({"", bestId});
        searchStart = bestPos + bestLen;
    }

    // Encode each segment
    std::vector<int32_t> allIds;

    for (auto& seg : segments) {
        if (seg.specialId >= 0) {
            allIds.push_back(seg.specialId);
            continue;
        }

        // Pre-tokenize
        auto words = preTokenize(seg.text);

        // BPE encode each word
        for (auto& word : words) {
            std::string bpeChars = bytesToBpeChars(word);
            auto ids = bpeEncode(bpeChars);
            allIds.insert(allIds.end(), ids.begin(), ids.end());
        }
    }

    return allIds;
}

std::string BPETokenizer::decode(const std::vector<int32_t>& ids) const {
    if (!loaded_) return "";

    std::string bpeStr;
    for (int32_t id : ids) {
        if (id >= 0 && id < (int32_t)idToToken_.size()) {
            bpeStr += idToToken_[id];
        }
    }

    // Convert BPE chars back to bytes
    return bpeCharsToBytes(bpeStr);
}

int32_t BPETokenizer::tokenToId(const std::string& token) const {
    auto it = tokenToId_.find(token);
    return it != tokenToId_.end() ? it->second : -1;
}

const std::string& BPETokenizer::idToToken(int32_t id) const {
    if (id >= 0 && id < (int32_t)idToToken_.size()) {
        return idToToken_[id];
    }
    return emptyString_;
}

// ── Special token loading ──

bool loadTTSSpecialTokens(const std::string& jsonPath, TTSSpecialTokens& out) {
    std::ifstream f(jsonPath);
    if (!f.is_open()) {
        fprintf(stderr, "Cannot open special tokens: %s\n", jsonPath.c_str());
        return false;
    }

    json j;
    try {
        f >> j;
    } catch (const json::parse_error& e) {
        fprintf(stderr, "JSON parse error: %s\n", e.what());
        return false;
    }

    if (j.contains("speech_start")) out.speech_start = j["speech_start"]["id"].get<int32_t>();
    if (j.contains("speech_end"))   out.speech_end = j["speech_end"]["id"].get<int32_t>();
    if (j.contains("speech_diffusion")) out.speech_diffusion = j["speech_diffusion"]["id"].get<int32_t>();
    if (j.contains("eos"))         out.eos = j["eos"]["id"].get<int32_t>();
    if (j.contains("pad"))         out.pad = j["pad"]["id"].get<int32_t>();

    return true;
}

bool loadASRSpecialTokens(const std::string& jsonPath, ASRSpecialTokens& out) {
    std::ifstream f(jsonPath);
    if (!f.is_open()) {
        fprintf(stderr, "Cannot open special tokens: %s\n", jsonPath.c_str());
        return false;
    }

    json j;
    try {
        f >> j;
    } catch (const json::parse_error& e) {
        fprintf(stderr, "JSON parse error: %s\n", e.what());
        return false;
    }

    if (j.contains("speech_start")) out.speech_start = j["speech_start"]["id"].get<int32_t>();
    if (j.contains("speech_end"))   out.speech_end = j["speech_end"]["id"].get<int32_t>();
    if (j.contains("speech_pad"))   out.speech_pad = j["speech_pad"]["id"].get<int32_t>();
    if (j.contains("eos"))         out.eos = j["eos"]["id"].get<int32_t>();
    if (j.contains("pad"))         out.pad = j["pad"]["id"].get<int32_t>();

    return true;
}
