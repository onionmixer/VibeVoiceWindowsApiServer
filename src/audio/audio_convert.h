#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace AudioConvert {

enum class OutputFormat {
    MP3,
    OPUS,
    AAC,
    FLAC,
};

// Convert WAV data (in memory) to target format via ffmpeg subprocess.
// Returns the encoded output bytes.
bool convert(const std::vector<uint8_t>& wavData, OutputFormat format,
             std::vector<uint8_t>& output);

// Convert a WAV file on disk to target format.
bool convertFile(const std::string& inputPath, const std::string& outputPath,
                 OutputFormat format);

// Get file extension for an output format.
const char* formatExtension(OutputFormat format);

// Get ffmpeg codec name for an output format.
const char* formatCodec(OutputFormat format);

} // namespace AudioConvert
