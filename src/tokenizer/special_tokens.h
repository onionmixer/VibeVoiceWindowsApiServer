#pragma once
#include <cstdint>
#include <string>

// TTS special token IDs (from special_tokens.json)
struct TTSSpecialTokens {
    int32_t speech_start = -1;      // <|vision_start|>
    int32_t speech_end = -1;        // <|vision_end|>
    int32_t speech_diffusion = -1;  // <|vision_pad|>
    int32_t eos = -1;              // <|endoftext|>
    int32_t pad = -1;              // <|image_pad|>
};

// ASR special token IDs (from special_tokens.json)
struct ASRSpecialTokens {
    int32_t speech_start = -1;      // <|object_ref_start|>
    int32_t speech_end = -1;        // <|object_ref_end|>
    int32_t speech_pad = -1;        // <|box_start|>
    int32_t eos = -1;              // <|endoftext|>
    int32_t pad = -1;              // <|image_pad|>
};

// Load from special_tokens.json
bool loadTTSSpecialTokens(const std::string& jsonPath, TTSSpecialTokens& out);
bool loadASRSpecialTokens(const std::string& jsonPath, ASRSpecialTokens& out);
