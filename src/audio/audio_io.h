#pragma once
#include <cstdint>
#include <string>
#include <vector>

enum class AudioFormat {
    Unknown,
    WAV,
    MP3,
    FLAC,
    OGG,
};

namespace AudioIO {

// Detect format from magic bytes.
AudioFormat detectFormat(const uint8_t* data, size_t size);
AudioFormat detectFormat(const std::string& path);

// Load audio from file into float32 samples (interleaved if multi-channel).
// Returns sample rate and channel count.
bool load(const std::string& path, std::vector<float>& samples,
          uint32_t& sampleRate, uint32_t& channels);

// Load from memory buffer.
bool loadFromMemory(const uint8_t* data, size_t size, AudioFormat fmt,
                    std::vector<float>& samples,
                    uint32_t& sampleRate, uint32_t& channels);

// Write mono int16 WAV file.
bool writeWav(const std::string& path, const std::vector<float>& samples,
              uint32_t sampleRate);

// Convert multi-channel to mono (average).
void toMono(const std::vector<float>& input, uint32_t channels,
            std::vector<float>& output);

// Linear interpolation resample.
void resample(const std::vector<float>& input, uint32_t srcRate,
              uint32_t dstRate, std::vector<float>& output);

// RMS-based normalization to target dBFS (default -25 dBFS).
void normalizeDb(std::vector<float>& samples, float targetDb = -25.0f);

// Full pipeline: load → mono → resample to targetRate → normalize.
bool loadAndPrepare(const std::string& path, std::vector<float>& samples,
                    uint32_t targetRate = 24000, float targetDb = -25.0f);

// Full pipeline from memory buffer.
bool loadAndPrepareFromMemory(const uint8_t* data, size_t size,
                              std::vector<float>& samples,
                              uint32_t targetRate = 24000,
                              float targetDb = -25.0f);

} // namespace AudioIO
