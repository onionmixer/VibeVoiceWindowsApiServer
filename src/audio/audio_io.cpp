#include "audio/audio_io.h"
#include <dr_wav.h>
#include <dr_mp3.h>
#include <dr_flac.h>

// Include stb_vorbis declarations only (implementation in audio_libs_impl.cpp)
#define STB_VORBIS_HEADER_ONLY
extern "C" {
#include "stb_vorbis.c"
}

#include <cmath>
#include <cstdio>
#include <fstream>
#include <algorithm>

namespace AudioIO {

AudioFormat detectFormat(const uint8_t* data, size_t size) {
    if (size < 4) return AudioFormat::Unknown;

    // WAV: "RIFF"
    if (data[0] == 'R' && data[1] == 'I' && data[2] == 'F' && data[3] == 'F') {
        return AudioFormat::WAV;
    }
    // FLAC: "fLaC"
    if (data[0] == 'f' && data[1] == 'L' && data[2] == 'a' && data[3] == 'C') {
        return AudioFormat::FLAC;
    }
    // OGG: "OggS"
    if (data[0] == 'O' && data[1] == 'g' && data[2] == 'g' && data[3] == 'S') {
        return AudioFormat::OGG;
    }
    // MP3: sync word 0xFFE0 or ID3 tag
    if ((data[0] == 0xFF && (data[1] & 0xE0) == 0xE0) ||
        (data[0] == 'I' && data[1] == 'D' && data[2] == '3')) {
        return AudioFormat::MP3;
    }

    return AudioFormat::Unknown;
}

AudioFormat detectFormat(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return AudioFormat::Unknown;
    uint8_t header[4] = {};
    f.read(reinterpret_cast<char*>(header), 4);
    return detectFormat(header, (size_t)f.gcount());
}

// ── Loaders ──

static bool loadWav(const std::string& path, std::vector<float>& samples,
                    uint32_t& sampleRate, uint32_t& channels) {
    drwav wav;
    if (!drwav_init_file(&wav, path.c_str(), nullptr)) {
        fprintf(stderr, "Failed to open WAV: %s\n", path.c_str());
        return false;
    }
    uint64_t totalFrames = wav.totalPCMFrameCount;
    channels = wav.channels;
    sampleRate = wav.sampleRate;
    samples.resize((size_t)(totalFrames * channels));
    drwav_read_pcm_frames_f32(&wav, totalFrames, samples.data());
    drwav_uninit(&wav);
    return true;
}

static bool loadMp3(const std::string& path, std::vector<float>& samples,
                    uint32_t& sampleRate, uint32_t& channels) {
    drmp3 mp3;
    if (!drmp3_init_file(&mp3, path.c_str(), nullptr)) {
        fprintf(stderr, "Failed to open MP3: %s\n", path.c_str());
        return false;
    }
    uint64_t totalFrames = drmp3_get_pcm_frame_count(&mp3);
    channels = mp3.channels;
    sampleRate = mp3.sampleRate;
    samples.resize((size_t)(totalFrames * channels));
    drmp3_read_pcm_frames_f32(&mp3, totalFrames, samples.data());
    drmp3_uninit(&mp3);
    return true;
}

static bool loadFlac(const std::string& path, std::vector<float>& samples,
                     uint32_t& sampleRate, uint32_t& channels) {
    drflac* flac = drflac_open_file(path.c_str(), nullptr);
    if (!flac) {
        fprintf(stderr, "Failed to open FLAC: %s\n", path.c_str());
        return false;
    }
    uint64_t totalFrames = flac->totalPCMFrameCount;
    channels = flac->channels;
    sampleRate = flac->sampleRate;
    samples.resize((size_t)(totalFrames * channels));
    drflac_read_pcm_frames_f32(flac, totalFrames, samples.data());
    drflac_close(flac);
    return true;
}

static bool loadOgg(const std::string& path, std::vector<float>& samples,
                    uint32_t& sampleRate, uint32_t& channels) {
    int ch = 0, sr = 0;
    short* decoded = nullptr;
    int numSamples = stb_vorbis_decode_filename(path.c_str(), &ch, &sr, &decoded);
    if (numSamples <= 0 || !decoded) {
        fprintf(stderr, "Failed to open OGG: %s\n", path.c_str());
        return false;
    }
    channels = (uint32_t)ch;
    sampleRate = (uint32_t)sr;
    size_t totalSamples = (size_t)numSamples * channels;
    samples.resize(totalSamples);
    for (size_t i = 0; i < totalSamples; ++i) {
        samples[i] = (float)decoded[i] / 32768.0f;
    }
    free(decoded);
    return true;
}

// ── Memory loaders ──

static bool loadWavMem(const uint8_t* data, size_t size, std::vector<float>& samples,
                       uint32_t& sampleRate, uint32_t& channels) {
    drwav wav;
    if (!drwav_init_memory(&wav, data, size, nullptr)) return false;
    uint64_t totalFrames = wav.totalPCMFrameCount;
    channels = wav.channels;
    sampleRate = wav.sampleRate;
    samples.resize((size_t)(totalFrames * channels));
    drwav_read_pcm_frames_f32(&wav, totalFrames, samples.data());
    drwav_uninit(&wav);
    return true;
}

static bool loadMp3Mem(const uint8_t* data, size_t size, std::vector<float>& samples,
                       uint32_t& sampleRate, uint32_t& channels) {
    drmp3 mp3;
    if (!drmp3_init_memory(&mp3, data, size, nullptr)) return false;
    uint64_t totalFrames = drmp3_get_pcm_frame_count(&mp3);
    channels = mp3.channels;
    sampleRate = mp3.sampleRate;
    samples.resize((size_t)(totalFrames * channels));
    drmp3_read_pcm_frames_f32(&mp3, totalFrames, samples.data());
    drmp3_uninit(&mp3);
    return true;
}

static bool loadFlacMem(const uint8_t* data, size_t size, std::vector<float>& samples,
                        uint32_t& sampleRate, uint32_t& channels) {
    unsigned int ch = 0, sr = 0;
    drflac_uint64 totalFrames = 0;
    float* decoded = drflac_open_memory_and_read_pcm_frames_f32(
        data, size, &ch, &sr, &totalFrames, nullptr);
    if (!decoded) return false;
    channels = ch;
    sampleRate = sr;
    samples.assign(decoded, decoded + totalFrames * ch);
    drflac_free(decoded, nullptr);
    return true;
}

static bool loadOggMem(const uint8_t* data, size_t size, std::vector<float>& samples,
                       uint32_t& sampleRate, uint32_t& channels) {
    int ch = 0, sr = 0;
    short* decoded = nullptr;
    int numSamples = stb_vorbis_decode_memory(data, (int)size, &ch, &sr, &decoded);
    if (numSamples <= 0 || !decoded) return false;
    channels = (uint32_t)ch;
    sampleRate = (uint32_t)sr;
    size_t totalSamples = (size_t)numSamples * channels;
    samples.resize(totalSamples);
    for (size_t i = 0; i < totalSamples; ++i) {
        samples[i] = (float)decoded[i] / 32768.0f;
    }
    free(decoded);
    return true;
}

// ── Public API ──

bool load(const std::string& path, std::vector<float>& samples,
          uint32_t& sampleRate, uint32_t& channels) {
    AudioFormat fmt = detectFormat(path);
    switch (fmt) {
        case AudioFormat::WAV:  return loadWav(path, samples, sampleRate, channels);
        case AudioFormat::MP3:  return loadMp3(path, samples, sampleRate, channels);
        case AudioFormat::FLAC: return loadFlac(path, samples, sampleRate, channels);
        case AudioFormat::OGG:  return loadOgg(path, samples, sampleRate, channels);
        default:
            fprintf(stderr, "Unknown audio format: %s\n", path.c_str());
            return false;
    }
}

bool loadFromMemory(const uint8_t* data, size_t size, AudioFormat fmt,
                    std::vector<float>& samples,
                    uint32_t& sampleRate, uint32_t& channels) {
    if (fmt == AudioFormat::Unknown) {
        fmt = detectFormat(data, size);
    }
    switch (fmt) {
        case AudioFormat::WAV:  return loadWavMem(data, size, samples, sampleRate, channels);
        case AudioFormat::MP3:  return loadMp3Mem(data, size, samples, sampleRate, channels);
        case AudioFormat::FLAC: return loadFlacMem(data, size, samples, sampleRate, channels);
        case AudioFormat::OGG:  return loadOggMem(data, size, samples, sampleRate, channels);
        default:
            fprintf(stderr, "Unknown audio format from memory\n");
            return false;
    }
}

bool writeWav(const std::string& path, const std::vector<float>& samples,
              uint32_t sampleRate) {
    drwav wav;
    drwav_data_format format = {};
    format.container = drwav_container_riff;
    format.format = DR_WAVE_FORMAT_PCM;
    format.channels = 1;
    format.sampleRate = sampleRate;
    format.bitsPerSample = 16;

    if (!drwav_init_file_write(&wav, path.c_str(), &format, nullptr)) {
        fprintf(stderr, "Failed to create WAV: %s\n", path.c_str());
        return false;
    }

    // Convert float32 to int16
    std::vector<int16_t> pcm(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        float s = samples[i];
        s = (std::max)(-1.0f, (std::min)(1.0f, s));
        pcm[i] = (int16_t)(s * 32767.0f);
    }

    drwav_write_pcm_frames(&wav, samples.size(), pcm.data());
    drwav_uninit(&wav);
    return true;
}

void toMono(const std::vector<float>& input, uint32_t channels,
            std::vector<float>& output) {
    if (channels <= 1) {
        output = input;
        return;
    }
    size_t frames = input.size() / channels;
    output.resize(frames);
    for (size_t i = 0; i < frames; ++i) {
        float sum = 0.0f;
        for (uint32_t c = 0; c < channels; ++c) {
            sum += input[i * channels + c];
        }
        output[i] = sum / (float)channels;
    }
}

void resample(const std::vector<float>& input, uint32_t srcRate,
              uint32_t dstRate, std::vector<float>& output) {
    if (srcRate == dstRate) {
        output = input;
        return;
    }
    double ratio = (double)srcRate / (double)dstRate;
    size_t outLen = (size_t)((double)input.size() / ratio);
    output.resize(outLen);
    for (size_t i = 0; i < outLen; ++i) {
        double srcIdx = (double)i * ratio;
        size_t idx0 = (size_t)srcIdx;
        double frac = srcIdx - (double)idx0;
        size_t idx1 = idx0 + 1;
        if (idx1 >= input.size()) idx1 = input.size() - 1;
        output[i] = (float)((1.0 - frac) * input[idx0] + frac * input[idx1]);
    }
}

void normalizeDb(std::vector<float>& samples, float targetDb) {
    if (samples.empty()) return;

    // Compute RMS
    double sumSq = 0.0;
    for (float s : samples) {
        sumSq += (double)s * s;
    }
    double rms = sqrt(sumSq / (double)samples.size());

    if (rms < 1e-10) return; // silence

    double currentDb = 20.0 * log10(rms);
    double gain = pow(10.0, ((double)targetDb - currentDb) / 20.0);

    for (float& s : samples) {
        s = (float)((double)s * gain);
    }
}

bool loadAndPrepare(const std::string& path, std::vector<float>& samples,
                    uint32_t targetRate, float targetDb) {
    std::vector<float> raw;
    uint32_t sr = 0, ch = 0;
    if (!load(path, raw, sr, ch)) return false;

    // Mono
    std::vector<float> mono;
    toMono(raw, ch, mono);

    // Resample
    std::vector<float> resampled;
    resample(mono, sr, targetRate, resampled);

    // Normalize
    normalizeDb(resampled, targetDb);

    samples = std::move(resampled);
    return true;
}

bool loadAndPrepareFromMemory(const uint8_t* data, size_t size,
                              std::vector<float>& samples,
                              uint32_t targetRate, float targetDb) {
    std::vector<float> raw;
    uint32_t sr = 0, ch = 0;
    if (!loadFromMemory(data, size, AudioFormat::Unknown, raw, sr, ch)) return false;

    std::vector<float> mono;
    toMono(raw, ch, mono);

    std::vector<float> resampled;
    resample(mono, sr, targetRate, resampled);

    normalizeDb(resampled, targetDb);

    samples = std::move(resampled);
    return true;
}

} // namespace AudioIO
