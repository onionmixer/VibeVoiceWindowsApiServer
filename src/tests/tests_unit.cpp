#include "tests/test_common.h"

#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <NvInfer.h>

#include "utils/cuda_buffer.h"
#include "inference/model_config.h"
#include "inference/trt_engine.h"
#include "inference/dpm_solver.h"
#include "inference/kv_cache.h"
#include "inference/gpu_ops.h"
#include "inference/tts_pipeline.h"
#include "inference/stt_pipeline.h"
#include "server/http_server.h"
#include "audio/audio_io.h"
#include "audio/audio_convert.h"
#include "tokenizer/bpe_tokenizer.h"
#include "tokenizer/special_tokens.h"

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

// ── Test: DPMSolver ──

static void testDPMSolver() {
    fprintf(stderr, "\n--- DPMSolver ---\n");

    DPMSolver dpm(1000, 5, "cosine", "v_prediction", 2);

    // Verify timestep count
    check(dpm.numSteps() == 5, "numSteps == 5");

    // Timesteps should be valid (0..999)
    bool validTimesteps = true;
    for (int i = 0; i < dpm.numSteps(); ++i) {
        int t = dpm.timestep(i);
        if (t < 0 || t > 999) { validTimesteps = false; break; }
    }
    check(validTimesteps, "timesteps in range [0, 999]");

    // Timesteps should be decreasing (from high noise to low noise)
    bool decreasing = true;
    for (int i = 1; i < dpm.numSteps(); ++i) {
        if (dpm.timestep(i) >= dpm.timestep(i - 1)) { decreasing = false; break; }
    }
    check(decreasing, "timesteps are decreasing");

    // Test step function with dummy data
    int n = 64;
    std::vector<float> modelOut(n, 0.1f);
    std::vector<float> sample(n, 1.0f);
    std::vector<float> prevSample(n, 0.0f);

    dpm.reset();
    dpm.step(modelOut.data(), 0, sample.data(), prevSample.data(), n);

    // After one step, output should differ from input
    bool changed = false;
    for (int i = 0; i < n; ++i) {
        if (fabsf(prevSample[i] - sample[i]) > 1e-6f) { changed = true; break; }
    }
    check(changed, "step() produces different output");

    // Full solver loop should converge
    dpm.reset();
    std::vector<float> noise(n, 0.5f);
    for (int step = 0; step < dpm.numSteps(); ++step) {
        std::vector<float> vPred(n, 0.0f); // zero v_prediction
        dpm.step(vPred.data(), step, noise.data(), prevSample.data(), n);
        noise = prevSample;
    }
    // After solver with zero v_pred, result should be finite
    bool finite = true;
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(noise[i])) { finite = false; break; }
    }
    check(finite, "full solver loop produces finite output");

    // Test setTimesteps
    dpm.setTimesteps(20);
    check(dpm.numSteps() == 20, "setTimesteps(20)");

    fprintf(stderr, "    timesteps(5): [");
    dpm.setTimesteps(5);
    for (int i = 0; i < dpm.numSteps(); ++i) {
        if (i > 0) fprintf(stderr, ", ");
        fprintf(stderr, "%d", dpm.timestep(i));
    }
    fprintf(stderr, "]\n");
}

// ── Test: KVCache ──

static void testKVCache() {
    fprintf(stderr, "\n--- KVCache ---\n");

    KVCache kv;
    bool ok = kv.init(4, 2, 64, 512);
    check(ok, "init(4 layers, 2 heads, 64 dim, 512 max)");
    if (!ok) return;

    check(kv.numLayers() == 4, "numLayers == 4");
    check(kv.numKVHeads() == 2, "numKVHeads == 2");
    check(kv.headDim() == 64, "headDim == 64");
    check(kv.seqLen() == 0, "initial seqLen == 0");

    // Pointers should be valid and different for past vs present
    void* pk0 = kv.pastKeyPtr(0);
    void* prk0 = kv.presentKeyPtr(0);
    check(pk0 != nullptr, "pastKeyPtr not null");
    check(prk0 != nullptr, "presentKeyPtr not null");
    check(pk0 != prk0, "past != present (double-buffer)");

    // Advance and verify swap
    kv.advance();
    check(kv.seqLen() == 1, "seqLen after advance");
    // After advance, past and present should swap
    void* pk0After = kv.pastKeyPtr(0);
    check(pk0After == prk0, "past after advance == old present");

    // Reset
    kv.reset();
    check(kv.seqLen() == 0, "seqLen after reset");

    // Test loading from preset
    VoicePreset vp;
    ok = loadVoicePreset("voices/streaming_model/en-Carter_man.bin", vp);
    if (ok && vp.num_groups >= 1) {
        KVCache kvPreset;
        ok = kvPreset.loadFromPreset(vp.groups[0], vp.hidden_size, vp.head_dim);
        check(ok, "loadFromPreset");
        if (ok) {
            check(kvPreset.seqLen() == (int)vp.groups[0].seq_len, "preset seqLen matches");
            check(kvPreset.numLayers() == (int)vp.groups[0].num_layers, "preset numLayers matches");
            fprintf(stderr, "    preset: %d layers, seq_len=%d\n",
                    kvPreset.numLayers(), kvPreset.seqLen());
        }
    } else {
        fprintf(stderr, "  (skipping loadFromPreset test - voice file not available)\n");
    }
}

// ── Test: GpuOps ──

static void testGpuOps() {
    fprintf(stderr, "\n--- GpuOps ---\n");

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    cublasHandle_t cublas = nullptr;
    cublasCreate(&cublas);
    cublasSetStream(cublas, stream);

    // Test vectorAdd
    {
        int n = 4;
        std::vector<uint16_t> aHost(n), bHost(n);
        for (int i = 0; i < n; ++i) {
            aHost[i] = 0x3C00; // 1.0 in fp16
            bHost[i] = 0x4000; // 2.0 in fp16
        }
        CudaBuffer aGpu, bGpu, outGpu;
        aGpu.upload(aHost);
        bGpu.upload(bHost);
        outGpu.resize(n * sizeof(uint16_t));

        GpuOps::vectorAdd(aGpu.as<__half>(), bGpu.as<__half>(),
                          outGpu.as<__half>(), n, stream);

        // Convert result to float
        CudaBuffer outF32;
        outF32.resize(n * sizeof(float));
        GpuOps::halfToFloat(outGpu.as<__half>(), outF32.as<float>(), n, stream);
        cudaStreamSynchronize(stream);

        std::vector<float> result(n);
        outF32.copyToHost(result.data(), n * sizeof(float));
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (fabsf(result[i] - 3.0f) > 0.01f) { ok = false; break; }
        }
        check(ok, "vectorAdd: 1.0 + 2.0 = 3.0");
    }

    // Test halfToFloat / floatToHalf round-trip
    {
        int n = 8;
        std::vector<float> src = {0.0f, 1.0f, -1.0f, 0.5f, 2.0f, -2.0f, 100.0f, -0.001f};
        CudaBuffer srcGpu, halfGpu, dstGpu;
        srcGpu.upload(src);
        halfGpu.resize(n * sizeof(uint16_t));
        dstGpu.resize(n * sizeof(float));

        GpuOps::floatToHalf(srcGpu.as<float>(), halfGpu.as<__half>(), n, stream);
        GpuOps::halfToFloat(halfGpu.as<__half>(), dstGpu.as<float>(), n, stream);
        cudaStreamSynchronize(stream);

        std::vector<float> dst(n);
        dstGpu.copyToHost(dst.data(), n * sizeof(float));
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (fabsf(dst[i] - src[i]) > 0.01f * fabsf(src[i]) + 0.001f) { ok = false; break; }
        }
        check(ok, "floatToHalf/halfToFloat round-trip");
    }

    // Test cfgBlend
    {
        int n = 4;
        std::vector<uint16_t> condH(n), uncondH(n);
        for (int i = 0; i < n; ++i) {
            condH[i] = 0x4000;  // 2.0
            uncondH[i] = 0x3C00; // 1.0
        }
        CudaBuffer condGpu, uncondGpu, outGpu, outF32;
        condGpu.upload(condH);
        uncondGpu.upload(uncondH);
        outGpu.resize(n * sizeof(uint16_t));
        outF32.resize(n * sizeof(float));

        float scale = 1.5f;
        GpuOps::cfgBlend(condGpu.as<__half>(), uncondGpu.as<__half>(), scale,
                          outGpu.as<__half>(), n, stream);
        GpuOps::halfToFloat(outGpu.as<__half>(), outF32.as<float>(), n, stream);
        cudaStreamSynchronize(stream);

        std::vector<float> result(n);
        outF32.copyToHost(result.data(), n * sizeof(float));
        // Expected: 1.0 + 1.5 * (2.0 - 1.0) = 2.5
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            if (fabsf(result[i] - 2.5f) > 0.05f) { ok = false; break; }
        }
        check(ok, "cfgBlend: 1.0 + 1.5*(2.0-1.0) = 2.5");
    }

    // Test replaceMaskedEmbeds
    {
        int totalTokens = 8, H = 4, T = 3;
        // Create embeds [8, 4] all zeros
        CudaBuffer embedsGpu;
        embedsGpu.resize(totalTokens * H * sizeof(uint16_t));
        cudaMemset(embedsGpu.data(), 0, totalTokens * H * sizeof(uint16_t));

        // Speech features [3, 4] all 1.0
        std::vector<uint16_t> featHost(T * H, 0x3C00);  // 1.0
        CudaBuffer featGpu;
        featGpu.upload(featHost);

        // Mask indices: positions 2, 4, 6
        std::vector<int32_t> maskHost = {2, 4, 6};
        CudaBuffer maskGpu;
        maskGpu.upload(maskHost);

        GpuOps::replaceMaskedEmbeds(embedsGpu.as<__half>(), featGpu.as<__half>(),
                                     maskGpu.as<int32_t>(), T, H, stream);

        // Verify: rows 2,4,6 should be 1.0, others 0.0
        CudaBuffer resultF32;
        resultF32.resize(totalTokens * H * sizeof(float));
        GpuOps::halfToFloat(embedsGpu.as<__half>(), resultF32.as<float>(), totalTokens * H, stream);
        cudaStreamSynchronize(stream);

        std::vector<float> result(totalTokens * H);
        resultF32.copyToHost(result.data(), result.size() * sizeof(float));

        bool ok = true;
        for (int r = 0; r < totalTokens; ++r) {
            float expected = (r == 2 || r == 4 || r == 6) ? 1.0f : 0.0f;
            for (int c = 0; c < H; ++c) {
                if (fabsf(result[r * H + c] - expected) > 0.01f) { ok = false; break; }
            }
            if (!ok) break;
        }
        check(ok, "replaceMaskedEmbeds: correct rows replaced");
    }

    // Test linearForwardBatch
    {
        int M = 2, N = 3, K = 4;
        // A: [2,4] = [[1,0,0,0],[0,1,0,0]]  (identity-like)
        // B: [3,4] = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
        // C = A @ B^T = [[1,5,9],[2,6,10]]
        std::vector<float> aF = {1,0,0,0, 0,1,0,0};
        std::vector<float> bF = {1,2,3,4, 5,6,7,8, 9,10,11,12};
        CudaBuffer aF32, bF32, aGpu, bGpu, cGpu, cF32;
        aF32.upload(aF); bF32.upload(bF);
        aGpu.resize(M*K*sizeof(uint16_t));
        bGpu.resize(N*K*sizeof(uint16_t));
        cGpu.resize(M*N*sizeof(uint16_t));
        cF32.resize(M*N*sizeof(float));

        GpuOps::floatToHalf(aF32.as<float>(), aGpu.as<__half>(), M*K, stream);
        GpuOps::floatToHalf(bF32.as<float>(), bGpu.as<__half>(), N*K, stream);

        GpuOps::linearForwardBatch(cublas, aGpu.as<__half>(), bGpu.as<__half>(),
                                    nullptr, cGpu.as<__half>(), M, N, K, stream);

        GpuOps::halfToFloat(cGpu.as<__half>(), cF32.as<float>(), M*N, stream);
        cudaStreamSynchronize(stream);

        std::vector<float> result(M*N);
        cF32.copyToHost(result.data(), M*N*sizeof(float));
        // Expected: [1,5,9, 2,6,10]
        float expected[] = {1,5,9, 2,6,10};
        bool ok = true;
        for (int i = 0; i < M*N; ++i) {
            if (fabsf(result[i] - expected[i]) > 0.1f) { ok = false; break; }
        }
        check(ok, "linearForwardBatch: A@B^T correct");
    }

    // Test rmsNormBatched
    {
        int M = 2, N = 4;
        // input: [[2,2,2,2],[4,4,4,4]], weight: [1,1,1,1]
        // RMS of [2,2,2,2] = sqrt(4) = 2, normalized = [1,1,1,1]
        // RMS of [4,4,4,4] = sqrt(16) = 4, normalized = [1,1,1,1]
        std::vector<float> inF = {2,2,2,2, 4,4,4,4};
        std::vector<float> wF = {1,1,1,1};
        CudaBuffer inF32, wF32, inGpu, wGpu, outGpu, outF32;
        inF32.upload(inF); wF32.upload(wF);
        inGpu.resize(M*N*sizeof(uint16_t));
        wGpu.resize(N*sizeof(uint16_t));
        outGpu.resize(M*N*sizeof(uint16_t));
        outF32.resize(M*N*sizeof(float));

        GpuOps::floatToHalf(inF32.as<float>(), inGpu.as<__half>(), M*N, stream);
        GpuOps::floatToHalf(wF32.as<float>(), wGpu.as<__half>(), N, stream);

        GpuOps::rmsNormBatched(inGpu.as<__half>(), wGpu.as<__half>(),
                               outGpu.as<__half>(), M, N, 1e-6f, stream);

        GpuOps::halfToFloat(outGpu.as<__half>(), outF32.as<float>(), M*N, stream);
        cudaStreamSynchronize(stream);

        std::vector<float> result(M*N);
        outF32.copyToHost(result.data(), M*N*sizeof(float));
        // All values should be ~1.0
        bool ok = true;
        for (int i = 0; i < M*N; ++i) {
            if (fabsf(result[i] - 1.0f) > 0.05f) { ok = false; break; }
        }
        check(ok, "rmsNormBatched: uniform rows normalize to 1.0");
    }

    cublasDestroy(cublas);
    cudaStreamDestroy(stream);
}

// ── Test: TTSPipeline (load check only) ──

static void testTTSPipeline() {
    fprintf(stderr, "\n--- TTSPipeline ---\n");

    TTSPipeline pipeline;
    check(!pipeline.isLoaded(), "initial not loaded");

    // Just verify the class can be instantiated and interface works
    auto voices = pipeline.availableVoices();
    check(voices.empty(), "no voices before load");

    fprintf(stderr, "  TTSPipeline interface verified\n");
    fprintf(stderr, "  (full pipeline test requires TRT engines)\n");
}

// ── Test: STTPipeline ──

static void testSTTPipeline() {
    fprintf(stderr, "\n--- STTPipeline ---\n");

    // ASR metadata
    {
        ModelMetadata meta;
        bool ok = loadModelMetadata("onnx/asr/model_metadata.json", meta);
        check(ok, "ASR: load model_metadata.json");
        if (ok) {
            check(meta.hidden_size == 3584, "ASR: hidden_size == 3584");
            check(meta.num_hidden_layers == 28, "ASR: num_hidden_layers == 28");
            check(meta.num_key_value_heads == 4, "ASR: num_key_value_heads == 4");
            check(meta.head_dim == 128, "ASR: head_dim == 128");
            check(meta.vocab_size == 152064, "ASR: vocab_size == 152064");
            fprintf(stderr, "    ASR: type=%s, hidden=%d, layers=%d, kv_heads=%d\n",
                    meta.model_type.c_str(), meta.hidden_size,
                    meta.num_hidden_layers, meta.num_key_value_heads);
        }
    }

    // ASR special tokens
    {
        ASRSpecialTokens tokens;
        bool ok = loadASRSpecialTokens("tokenizer/qwen2.5-7b/special_tokens.json", tokens);
        check(ok, "ASR: load special_tokens.json");
        if (ok) {
            check(tokens.speech_start == 151646, "ASR: speech_start == 151646");
            check(tokens.speech_end == 151647, "ASR: speech_end == 151647");
            check(tokens.speech_pad == 151648, "ASR: speech_pad == 151648");
            check(tokens.eos == 151643, "ASR: eos == 151643");
            fprintf(stderr, "    ASR: speech_start=%d, speech_end=%d, speech_pad=%d, eos=%d\n",
                    tokens.speech_start, tokens.speech_end, tokens.speech_pad, tokens.eos);
        }
    }

    // ASR connector weights
    {
        ConnectorWeights acConn;
        bool ok = loadConnectorWeights("onnx/asr/weights/acoustic_connector.bin", acConn);
        check(ok, "ASR: load acoustic_connector.bin");
        if (ok) {
            check(acConn.input_dim == 64, "ASR: acoustic input_dim == 64");
            check(acConn.output_dim == 3584, "ASR: acoustic output_dim == 3584");
            fprintf(stderr, "    ASR acoustic connector: %u -> %u\n", acConn.input_dim, acConn.output_dim);
        }

        ConnectorWeights semConn;
        ok = loadConnectorWeights("onnx/asr/weights/semantic_connector.bin", semConn);
        check(ok, "ASR: load semantic_connector.bin");
        if (ok) {
            check(semConn.input_dim == 128, "ASR: semantic input_dim == 128");
            check(semConn.output_dim == 3584, "ASR: semantic output_dim == 3584");
            fprintf(stderr, "    ASR semantic connector: %u -> %u\n", semConn.input_dim, semConn.output_dim);
        }
    }

    // ASR tokenizer (7B)
    {
        BPETokenizer tok;
        bool ok = tok.load("tokenizer/qwen2.5-7b/tokenizer.json");
        check(ok, "ASR: load 7B tokenizer");
        if (ok) {
            check(tok.vocabSize() > 150000, "ASR: vocab > 150k");
            fprintf(stderr, "    ASR tokenizer: vocab=%zu\n", tok.vocabSize());
        }
    }

    // STTPipeline interface
    {
        STTPipeline stt;
        check(!stt.isLoaded(), "ASR: initial not loaded");
        fprintf(stderr, "  STTPipeline interface verified\n");
        fprintf(stderr, "  (full pipeline test requires TRT engines)\n");
    }

    // parseTranscription test
    {
        std::string testJson = R"(```json
[{"Start time": "0.00s", "End time": "3.50s", "Speaker ID": "1", "Content": "Hello world"},
 {"Start time": "3.50s", "End time": "7.20s", "Speaker ID": "2", "Content": "How are you?"}]
```)";
        auto segments = STTPipeline::parseTranscription(testJson);
        check(segments.size() == 2, "parseTranscription: 2 segments");
        if (segments.size() == 2) {
            check(fabsf(segments[0].startTime - 0.0f) < 0.01f, "parseTranscription: seg0 start");
            check(fabsf(segments[0].endTime - 3.5f) < 0.01f, "parseTranscription: seg0 end");
            check(segments[0].text == "Hello world", "parseTranscription: seg0 text");
            check(segments[0].speakerId == "1", "parseTranscription: seg0 speaker");
            check(fabsf(segments[1].startTime - 3.5f) < 0.01f, "parseTranscription: seg1 start");
            check(segments[1].text == "How are you?", "parseTranscription: seg1 text");
        }
    }

    // SRT formatting test
    {
        std::vector<STTPipeline::Segment> segs = {
            {0.0f, 3.5f, "1", "Hello world"},
            {3.5f, 7.2f, "2", "How are you?"},
        };
        std::string srt = STTPipeline::formatSRT(segs);
        check(srt.find("00:00:00,000 --> 00:00:03,500") != std::string::npos, "SRT: timestamp format");
        check(srt.find("Hello world") != std::string::npos, "SRT: text present");
        check(srt.find("1\n") != std::string::npos, "SRT: sequence number");
    }

    // VTT formatting test
    {
        std::vector<STTPipeline::Segment> segs = {
            {0.0f, 3.5f, "1", "Hello world"},
        };
        std::string vtt = STTPipeline::formatVTT(segs);
        check(vtt.find("WEBVTT") != std::string::npos, "VTT: header present");
        check(vtt.find("00:00:00.000 --> 00:00:03.500") != std::string::npos, "VTT: timestamp format");
    }
}

// ── Test: HttpServer config ──

static void testHttpServer() {
    fprintf(stderr, "\n--- HttpServer ---\n");

    // Test config loading (if config.json exists)
    {
        ServerConfig cfg;
        // Create a minimal test config in memory
        std::string testConfigPath = "test_config_temp.json";
        {
            FILE* f = fopen(testConfigPath.c_str(), "w");
            if (f) {
                fprintf(f, R"({
  "server": { "host": "127.0.0.1", "port": 9090 },
  "models": {
    "tts_0.5b": { "enabled": false },
    "asr": { "enabled": false }
  },
  "defaults": { "tts_voice": "en-Alice_woman" }
})");
                fclose(f);
            }
        }

        bool ok = loadServerConfig(testConfigPath, cfg);
        check(ok, "loadServerConfig");
        if (ok) {
            check(cfg.host == "127.0.0.1", "config: host");
            check(cfg.port == 9090, "config: port");
            check(cfg.defaultTTSVoice == "en-Alice_woman", "config: default voice");
            check(!cfg.tts05b.enabled, "config: tts_0.5b disabled");
            check(!cfg.asr.enabled, "config: asr disabled");
        }
        remove(testConfigPath.c_str());
    }

    fprintf(stderr, "  HttpServer config test done\n");
}

// ── Unit Test Runner ──

int runUnitTests() {
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

    std::cout << "\n=== Phase 3 Module Tests ===" << std::endl;

    testDPMSolver();
    testKVCache();
    testGpuOps();
    testTTSPipeline();

    std::cout << "\n=== Phase 4+5 Module Tests ===" << std::endl;

    testSTTPipeline();
    testHttpServer();

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Passed: " << testsPassed << std::endl;
    std::cout << "Failed: " << testsFailed << std::endl;

    return testsFailed > 0 ? 1 : 0;
}
