#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <NvInfer.h>

#include "utils/cuda_buffer.h"
#include "inference/model_config.h"
#include "inference/trt_engine.h"
#include "audio/audio_io.h"
#include "audio/audio_convert.h"
#include "tokenizer/bpe_tokenizer.h"
#include "tokenizer/special_tokens.h"

// ── Test helpers ──

static int testsPassed = 0;
static int testsFailed = 0;

static void check(bool ok, const char* name) {
    if (ok) {
        fprintf(stderr, "  [PASS] %s\n", name);
        ++testsPassed;
    } else {
        fprintf(stderr, "  [FAIL] %s\n", name);
        ++testsFailed;
    }
}

// ── Test: CudaBuffer ──

static void testCudaBuffer() {
    fprintf(stderr, "\n--- CudaBuffer ---\n");

    CudaBuffer buf;
    check(buf.data() == nullptr, "initial null");

    // Upload a vector
    std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f};
    check(buf.upload(src), "upload");
    check(buf.size() == src.size() * sizeof(float), "size after upload");

    // Download and verify
    std::vector<float> dst;
    check(buf.download(dst), "download");
    check(dst.size() == src.size(), "download size");

    bool match = true;
    for (size_t i = 0; i < src.size(); ++i) {
        if (dst[i] != src[i]) { match = false; break; }
    }
    check(match, "upload/download round-trip values");

    // Resize (smaller — should reuse)
    size_t oldCap = buf.capacity();
    check(buf.resize(4), "resize smaller");
    check(buf.capacity() == oldCap, "capacity preserved on shrink");

    // Move
    CudaBuffer buf2 = std::move(buf);
    check(buf.data() == nullptr, "moved-from is null");
    check(buf2.data() != nullptr, "moved-to is valid");

    buf2.free();
    check(buf2.data() == nullptr, "free");
}

// ── Test: ModelConfig ──

static void testModelConfig() {
    fprintf(stderr, "\n--- ModelConfig ---\n");

    ModelMetadata meta;
    bool ok = loadModelMetadata("onnx/tts_0.5b/model_metadata.json", meta);
    check(ok, "load model_metadata.json");
    if (ok) {
        check(meta.model_type == "tts_0.5b", "model_type");
        check(meta.hidden_size == 896, "hidden_size");
        check(meta.num_hidden_layers == 24, "num_hidden_layers");
        check(meta.num_key_value_heads == 2, "num_key_value_heads");
        check(meta.vocab_size == 151936, "vocab_size");
        check(meta.head_dim == 64, "head_dim");
        check(meta.sample_rate == 24000, "sample_rate");
        check(meta.has_diffusion, "has_diffusion");
        check(meta.diffusion.latent_size == 64, "diffusion.latent_size");
        check(meta.base_lm_layers == 4, "base_lm_layers");
        check(meta.tts_lm_layers == 20, "tts_lm_layers");
    }

    // Connector weights
    ConnectorWeights conn;
    ok = loadConnectorWeights("onnx/tts_0.5b/weights/acoustic_connector.bin", conn);
    check(ok, "load acoustic_connector.bin");
    if (ok) {
        check(conn.input_dim > 0, "connector input_dim > 0");
        check(conn.output_dim > 0, "connector output_dim > 0");
        check(conn.fc1_weight.size() == (size_t)conn.input_dim * conn.output_dim,
              "connector fc1_weight size");
    }

    // Embedding weights
    EmbeddingWeights emb;
    ok = loadEmbeddingWeights("onnx/tts_0.5b/weights/embed_tokens.bin", emb);
    check(ok, "load embed_tokens.bin");
    if (ok) {
        check(emb.num_embeddings > 0, "embedding num_embeddings > 0");
        check(emb.embedding_dim > 0, "embedding embedding_dim > 0");
        check(emb.data.size() == (size_t)emb.num_embeddings * emb.embedding_dim,
              "embedding data size");
    }

    // Scalar
    float scalingFactor = 0.0f;
    ok = loadScalar("onnx/tts_0.5b/weights/speech_scaling_factor.bin", scalingFactor);
    check(ok, "load speech_scaling_factor.bin");
    if (ok) {
        check(scalingFactor != 0.0f, "scaling_factor non-zero");
        fprintf(stderr, "    speech_scaling_factor = %f\n", scalingFactor);
    }

    // Voice preset (streaming model)
    VoicePreset vp;
    ok = loadVoicePreset("voices/streaming_model/en-Carter_man.bin", vp);
    check(ok, "load voice preset");
    if (ok) {
        check(vp.hidden_size == 896, "voice preset hidden_size");
        check(vp.head_dim == 64, "voice preset head_dim");
        check(vp.num_groups == 2, "voice preset num_groups");
        check(vp.groups.size() == 2, "voice preset groups count");
        fprintf(stderr, "    group[0]: layers=%u, kv_heads=%u, seq_len=%u\n",
                vp.groups[0].num_layers, vp.groups[0].num_kv_heads, vp.groups[0].seq_len);
        fprintf(stderr, "    group[1]: layers=%u, kv_heads=%u, seq_len=%u\n",
                vp.groups[1].num_layers, vp.groups[1].num_kv_heads, vp.groups[1].seq_len);
    }
}

// ── Test: AudioIO ──

static void testAudioIO() {
    fprintf(stderr, "\n--- AudioIO ---\n");

    // Try loading a voice preset WAV file
    std::string wavPath = "voices/full_model/en-Alice_woman.wav";
    std::vector<float> samples;
    uint32_t sr = 0, ch = 0;
    bool ok = AudioIO::load(wavPath, samples, sr, ch);
    check(ok, "load WAV");
    if (ok) {
        fprintf(stderr, "    sr=%u, ch=%u, samples=%zu\n", sr, ch, samples.size());
        check(sr > 0, "sample rate > 0");
        check(samples.size() > 0, "samples non-empty");

        // Write back to temp file
        std::vector<float> mono;
        AudioIO::toMono(samples, ch, mono);

        std::string tmpPath = "test_output.wav";
        ok = AudioIO::writeWav(tmpPath, mono, sr);
        check(ok, "write WAV");

        // Read back
        if (ok) {
            std::vector<float> readBack;
            uint32_t sr2 = 0, ch2 = 0;
            ok = AudioIO::load(tmpPath, readBack, sr2, ch2);
            check(ok, "read back WAV");
            check(sr2 == sr, "sample rate preserved");
            check(ch2 == 1, "mono output");
            // Allow 1-sample difference due to float->int16->float conversion
            check(readBack.size() == mono.size() || readBack.size() == mono.size() + 1 ||
                  readBack.size() + 1 == mono.size(), "sample count preserved");

            // Cleanup
            remove(tmpPath.c_str());
        }
    }

    // Test loadAndPrepare
    ok = AudioIO::loadAndPrepare(wavPath, samples, 24000, -25.0f);
    check(ok, "loadAndPrepare");
    if (ok) {
        fprintf(stderr, "    prepared samples: %zu (%.2f sec at 24kHz)\n",
                samples.size(), (float)samples.size() / 24000.0f);
    }

    // Test resample
    {
        std::vector<float> in(48000, 0.5f); // 1 sec at 48kHz
        std::vector<float> out;
        AudioIO::resample(in, 48000, 24000, out);
        // Should be approximately 24000 samples
        check(out.size() >= 23900 && out.size() <= 24100, "resample 48k->24k length");
    }
}

// ── Test: AudioConvert ──

static void testAudioConvert() {
    fprintf(stderr, "\n--- AudioConvert ---\n");
    fprintf(stderr, "  (requires ffmpeg.exe in PATH)\n");

    // Create a tiny WAV in memory
    std::vector<float> sine(24000);
    for (size_t i = 0; i < sine.size(); ++i) {
        sine[i] = 0.5f * sinf(2.0f * 3.14159265f * 440.0f * (float)i / 24000.0f);
    }

    std::string tmpWav = "test_convert_input.wav";
    bool ok = AudioIO::writeWav(tmpWav, sine, 24000);
    check(ok, "write test WAV for conversion");

    if (ok) {
        std::string tmpMp3 = "test_convert_output.mp3";
        ok = AudioConvert::convertFile(tmpWav, tmpMp3, AudioConvert::OutputFormat::MP3);
        check(ok, "convert WAV -> MP3");

        if (ok) {
            // Verify output exists and is not empty
            std::vector<float> mp3Samples;
            uint32_t sr2 = 0, ch2 = 0;
            ok = AudioIO::load(tmpMp3, mp3Samples, sr2, ch2);
            check(ok, "read back MP3");
            check(mp3Samples.size() > 0, "MP3 non-empty");
            remove(tmpMp3.c_str());
        }

        remove(tmpWav.c_str());
    }
}

// ── Test: BPETokenizer ──

static void testBPETokenizer() {
    fprintf(stderr, "\n--- BPETokenizer ---\n");

    BPETokenizer tok;
    bool ok = tok.load("tokenizer/qwen2.5-0.5b/tokenizer.json");
    check(ok, "load tokenizer.json");
    if (!ok) return;

    check(tok.vocabSize() > 150000, "vocab size > 150k");
    fprintf(stderr, "    vocab size: %zu\n", tok.vocabSize());

    // Test basic encoding
    {
        auto ids = tok.encode("Hello, world!");
        check(!ids.empty(), "encode 'Hello, world!' non-empty");
        fprintf(stderr, "    'Hello, world!' -> [");
        for (size_t i = 0; i < ids.size(); ++i) {
            if (i > 0) fprintf(stderr, ", ");
            fprintf(stderr, "%d", ids[i]);
        }
        fprintf(stderr, "]\n");

        // Round-trip
        std::string decoded = tok.decode(ids);
        check(decoded == "Hello, world!", "encode/decode round-trip ASCII");
        fprintf(stderr, "    decoded: '%s'\n", decoded.c_str());
    }

    // Test Korean text
    {
        std::string korean = "\xec\x95\x88\xeb\x85\x95\xed\x95\x98\xec\x84\xb8\xec\x9a\x94"; // annyeonghaseyo
        auto ids = tok.encode(korean);
        check(!ids.empty(), "encode Korean non-empty");
        std::string decoded = tok.decode(ids);
        check(decoded == korean, "encode/decode round-trip Korean");
        fprintf(stderr, "    Korean round-trip: %s\n", decoded == korean ? "OK" : "MISMATCH");
    }

    // Test CJK
    {
        std::string cjk = "\xe4\xbd\xa0\xe5\xa5\xbd\xe4\xb8\x96\xe7\x95\x8c"; // nihao shijie
        auto ids = tok.encode(cjk);
        check(!ids.empty(), "encode CJK non-empty");
        std::string decoded = tok.decode(ids);
        check(decoded == cjk, "encode/decode round-trip CJK");
    }

    // Special tokens
    TTSSpecialTokens tts;
    ok = loadTTSSpecialTokens("tokenizer/qwen2.5-0.5b/special_tokens.json", tts);
    check(ok, "load special_tokens.json");
    if (ok) {
        check(tts.speech_start == 151652, "speech_start id");
        check(tts.speech_end == 151653, "speech_end id");
        check(tts.eos == 151643, "eos id");
        fprintf(stderr, "    speech_start=%d, speech_end=%d, eos=%d\n",
                tts.speech_start, tts.speech_end, tts.eos);
    }
}

// ── Test: TRTEngine ──

static void testTRTEngine() {
    fprintf(stderr, "\n--- TRTEngine ---\n");
    fprintf(stderr, "  (requires pre-built .trt engines from build_engines.py)\n");

    // Just test that the class can be instantiated
    TRTEngine engine;
    check(!engine.isLoaded(), "initial not loaded");

    // Try loading a .trt file if it exists
    // This would only work after running build_engines.py
    // For now, just verify the interface compiles and links
    fprintf(stderr, "  TRTEngine interface verified (no .trt file to test)\n");
}

// ── Main ──

int main(int /*argc*/, char* /*argv*/[]) {
    std::cout << "=== VibeVoice Windows API Server ===" << std::endl;

    // CUDA version
    int cuda_runtime_ver = 0;
    cudaRuntimeGetVersion(&cuda_runtime_ver);
    int cuda_major = cuda_runtime_ver / 1000;
    int cuda_minor = (cuda_runtime_ver % 1000) / 10;
    std::cout << "CUDA Runtime: " << cuda_major << "." << cuda_minor << std::endl;

    // CUDA device info
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    std::cout << "CUDA Devices:  " << device_count << std::endl;
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "  [" << i << "] " << prop.name
                  << " (SM " << prop.major << "." << prop.minor
                  << ", " << (prop.totalGlobalMem / (1024 * 1024)) << " MB)" << std::endl;
    }

    // TensorRT version
    int trt_ver = getInferLibVersion();
    int trt_major = trt_ver / 1000;
    int trt_minor = (trt_ver % 1000) / 100;
    int trt_patch = trt_ver % 100;
    std::cout << "TensorRT:      " << trt_major << "." << trt_minor << "." << trt_patch << std::endl;

    std::cout << "\n=== Phase 2 Module Tests ===" << std::endl;

    testCudaBuffer();
    testModelConfig();
    testAudioIO();
    testAudioConvert();
    testBPETokenizer();
    testTRTEngine();

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Passed: " << testsPassed << std::endl;
    std::cout << "Failed: " << testsFailed << std::endl;

    return testsFailed > 0 ? 1 : 0;
}
