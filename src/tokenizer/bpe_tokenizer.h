#pragma once
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

class BPETokenizer {
public:
    BPETokenizer();
    ~BPETokenizer();

    // Load tokenizer from tokenizer.json (Qwen2 format).
    bool load(const std::string& tokenizerJsonPath);

    // Encode text to token IDs.
    std::vector<int32_t> encode(const std::string& text) const;

    // Decode token IDs to text.
    std::string decode(const std::vector<int32_t>& ids) const;

    // Get vocab size.
    size_t vocabSize() const { return tokenToId_.size(); }

    // Look up a single token string -> id. Returns -1 if not found.
    int32_t tokenToId(const std::string& token) const;

    // Look up id -> token string. Returns "" if not found.
    const std::string& idToToken(int32_t id) const;

    bool isLoaded() const { return loaded_; }

private:
    // Pre-tokenize text into word chunks using Qwen2 regex pattern.
    std::vector<std::string> preTokenize(const std::string& text) const;

    // Apply byte-level encoding: convert raw bytes to GPT-2/Qwen2 unicode chars.
    std::string bytesToBpeChars(const std::string& word) const;

    // Reverse byte-level encoding.
    std::string bpeCharsToBytes(const std::string& token) const;

    // Apply BPE merges to a single pre-tokenized word.
    std::vector<int32_t> bpeEncode(const std::string& bpeWord) const;

    bool loaded_ = false;

    // Vocab: token string <-> id
    std::unordered_map<std::string, int32_t> tokenToId_;
    std::vector<std::string> idToToken_;

    // Merge rules: (pair of token strings) -> merge priority (lower = higher priority)
    // Stored as pair_key -> rank
    std::unordered_map<std::string, int> mergeRanks_;

    // GPT-2 byte-level BPE tables
    // byte value (0-255) -> unicode char used in BPE vocab
    std::string byteToChar_[256];
    // unicode char (codepoint) -> byte value
    std::unordered_map<uint32_t, uint8_t> charToByte_;

    // Added tokens (special tokens that bypass BPE)
    struct AddedToken {
        std::string content;
        int32_t id;
        bool special;
    };
    std::vector<AddedToken> addedTokens_;

    static const std::string emptyString_;
};
