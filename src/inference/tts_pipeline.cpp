#include "inference/tts_pipeline.h"
#include <algorithm>
#include "utils/logger.h"
#include <cstring>
#include <cmath>
#include <random>
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

static std::string toLower(const std::string& s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return result;
}

// Diagnostic: download a GPU fp16 buffer and log value range
static void diagLogFp16(const char* label, const __half* gpuPtr, int n, cudaStream_t stream) {
    cudaStreamSynchronize(stream);
    std::vector<uint16_t> tmp(n);
    cudaMemcpy(tmp.data(), gpuPtr, n * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    float minV = 1e30f, maxV = -1e30f, sumV = 0;
    int nanCount = 0;
    for (int i = 0; i < n; ++i) {
        // fp16 -> fp32 conversion
        uint16_t h = tmp[i];
        uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
        uint32_t exp = (h >> 10) & 0x1fu;
        uint32_t man = h & 0x3ffu;
        uint32_t result;
        if (exp == 0) { result = (man == 0) ? sign : (sign | ((112+1) << 23) | (man << 13)); }
        else if (exp == 31) { result = sign | 0x7f800000u | (man << 13); }
        else { result = sign | ((exp + 112) << 23) | (man << 13); }
        float f; std::memcpy(&f, &result, sizeof(float));
        if (std::isnan(f) || std::isinf(f)) { nanCount++; continue; }
        if (f < minV) minV = f;
        if (f > maxV) maxV = f;
        sumV += f;
    }
    LOG_DEBUG("DIAG", "%s [%d]: min=%.4f max=%.4f avg=%.4f nan=%d",
             label, n, minV, maxV, sumV / std::max(1, n - nanCount), nanCount);
}

// Diagnostic: download a GPU fp32 buffer and log value range
static void diagLogFp32(const char* label, const float* gpuPtr, int n, cudaStream_t stream) {
    cudaStreamSynchronize(stream);
    std::vector<float> tmp(n);
    cudaMemcpy(tmp.data(), gpuPtr, n * sizeof(float), cudaMemcpyDeviceToHost);
    float minV = 1e30f, maxV = -1e30f, sumV = 0;
    int nanCount = 0;
    for (int i = 0; i < n; ++i) {
        if (std::isnan(tmp[i]) || std::isinf(tmp[i])) { nanCount++; continue; }
        if (tmp[i] < minV) minV = tmp[i];
        if (tmp[i] > maxV) maxV = tmp[i];
        sumV += tmp[i];
    }
    LOG_DEBUG("DIAG", "%s [%d]: min=%.4f max=%.4f avg=%.4f nan=%d",
             label, n, minV, maxV, sumV / std::max(1, n - nanCount), nanCount);
}

TTSPipeline::TTSPipeline()
    : dpm_(1000, 5, "cosine", "v_prediction", 2)
{}

TTSPipeline::~TTSPipeline() {
    if (cublas_) {
        cublasDestroy(cublas_);
        cublas_ = nullptr;
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

// ── Loading ──

bool TTSPipeline::load(const Config& cfg) {
    cfg_ = cfg;
    type_ = cfg.type;

    // Load metadata
    if (!loadModelMetadata(cfg.metadataPath, meta_)) {
        LOG_ERROR("TTS", "failed to load metadata from %s", cfg.metadataPath.c_str());
        return false;
    }
    LOG_INFO("TTS", "model_type=%s, hidden=%d, diffusion=%s",
             meta_.model_type.c_str(), meta_.hidden_size,
             meta_.has_diffusion ? "yes" : "no");

    // Create CUDA stream and cuBLAS handle
    cudaError_t cerr = cudaStreamCreate(&stream_);
    if (cerr != cudaSuccess) {
        LOG_ERROR("TTS", "failed to create CUDA stream: %s", cudaGetErrorString(cerr));
        return false;
    }
    cublasStatus_t bstat = cublasCreate(&cublas_);
    if (bstat != CUBLAS_STATUS_SUCCESS) {
        LOG_ERROR("TTS", "failed to create cuBLAS handle");
        return false;
    }
    cublasSetStream(cublas_, stream_);

    // Load tokenizer
    if (!tokenizer_.load(cfg.tokenizerPath)) {
        LOG_ERROR("TTS", "failed to load tokenizer from %s", cfg.tokenizerPath.c_str());
        return false;
    }
    if (!loadTTSSpecialTokens(cfg.specialTokensPath, specialTokens_)) {
        LOG_ERROR("TTS", "failed to load special tokens from %s", cfg.specialTokensPath.c_str());
        return false;
    }
    LOG_INFO("TTS", "tokenizer loaded (vocab=%zu)", tokenizer_.vocabSize());

    // Load TRT engines
    if (type_ == ModelType::STREAMING_0_5B) {
        std::string dir = cfg.engineDir + "/";
        if (!baseLmPrefill_.loadFromFile(dir + "base_lm_prefill.trt")) return false;
        if (!baseLmDecode_.loadFromFile(dir + "base_lm_decode.trt")) return false;
        if (!ttsLmPrefill_.loadFromFile(dir + "tts_lm_prefill.trt")) return false;
        if (!ttsLmDecode_.loadFromFile(dir + "tts_lm_decode.trt")) return false;
        if (!diffusionHead_.loadFromFile(dir + "diffusion_head.trt")) return false;
        if (!acousticDecoder_.loadFromFile(dir + "acoustic_decoder.trt")) return false;
        LOG_INFO("TTS", "loaded 6 TRT engines (0.5B)");

        // Optional: streaming acoustic decoder (matches Python's per-frame decode with cache)
        if (streaming05bAcDecoder_.loadFromFile(dir + "streaming_acoustic_decoder.trt")) {
            std::string onnxDir = cfg.metadataPath.substr(0, cfg.metadataPath.find_last_of("/\\"));
            auto loadCacheSize = [](const std::string& path, int defaultSize) -> int {
                std::ifstream f(path);
                if (!f.is_open()) return defaultSize;
                json j = json::parse(f);
                return j.value("total_cache_size", defaultSize);
            };
            acDec05bCacheSize_ = loadCacheSize(onnxDir + "/streaming_acoustic_decoder_cache.json", 182080);
            acDec05bCacheGpu_.resize(acDec05bCacheSize_ * sizeof(float));
            acDec05bCacheOutGpu_.resize(acDec05bCacheSize_ * sizeof(float));
            LOG_INFO("TTS", "loaded streaming_acoustic_decoder for 0.5B (cache=%d)", acDec05bCacheSize_);
        } else {
            LOG_WARN("TTS", "streaming_acoustic_decoder not found for 0.5B, will use batch decode");
        }
    } else {
        std::string dir = cfg.engineDir + "/";
        if (!lmPrefill_.loadFromFile(dir + "language_model_prefill.trt")) return false;
        if (!lmDecode_.loadFromFile(dir + "language_model_decode.trt")) return false;
        if (!acousticEncoder_.loadFromFile(dir + "acoustic_encoder.trt")) return false;
        if (!semanticEncoder_.loadFromFile(dir + "semantic_encoder.trt")) return false;
        if (!diffusionHead_.loadFromFile(dir + "diffusion_head.trt")) return false;
        if (!acousticDecoder_.loadFromFile(dir + "acoustic_decoder.trt")) return false;

        // Load streaming engines for autoregressive cache-based inference
        if (!streamingSemEncoder_.loadFromFile(dir + "streaming_semantic_encoder.trt")) return false;
        if (!streamingAcDecoder_.loadFromFile(dir + "streaming_acoustic_decoder.trt")) return false;
        LOG_INFO("TTS", "loaded 8 TRT engines (1.5B, including streaming)");

        // Load streaming cache sizes from metadata JSONs
        {
            std::string onnxDir = cfg.metadataPath.substr(0, cfg.metadataPath.find_last_of("/\\"));
            auto loadCacheSize = [](const std::string& path, int defaultSize) -> int {
                std::ifstream f(path);
                if (!f.is_open()) return defaultSize;
                json j = json::parse(f);
                return j.value("total_cache_size", defaultSize);
            };
            semCacheSize_ = loadCacheSize(onnxDir + "/streaming_semantic_encoder_cache.json", 159622);
            acDecCacheSize_ = loadCacheSize(onnxDir + "/streaming_acoustic_decoder_cache.json", 182080);
            LOG_INFO("TTS", "streaming cache sizes: semantic=%d, acoustic_decoder=%d",
                     semCacheSize_, acDecCacheSize_);

            // Allocate double-buffered cache GPU buffers
            semCacheGpu_.resize(semCacheSize_ * sizeof(float));
            semCacheOutGpu_.resize(semCacheSize_ * sizeof(float));
            acDecCacheGpu_.resize(acDecCacheSize_ * sizeof(float));
            acDecCacheOutGpu_.resize(acDecCacheSize_ * sizeof(float));
        }
    }

    // Load weights
    if (!loadWeights()) return false;

    // Load voice presets
    if (!loadVoicePresets()) return false;

    // Configure DPM solver from metadata
    if (meta_.has_diffusion) {
        dpm_ = DPMSolver(meta_.diffusion.num_train_timesteps,
                         cfg.inferenceSteps,
                         meta_.diffusion.beta_schedule,
                         meta_.diffusion.prediction_type,
                         2);
    }

    // Allocate scratch buffers
    int H = meta_.hidden_size;
    int latentDim = meta_.has_diffusion ? meta_.diffusion.latent_size : 64;

    // fp16 scratch buffers for GPU-side operations (embedding, vectorAdd, etc.)
    scratchEmbed_.resize(H * sizeof(uint16_t));
    // fp32 scratch for LM hidden states (1.5B FP32 engines)
    scratchHidden_.resize(H * sizeof(float));
    scratchHidden2_.resize(H * sizeof(float));
    scratchEmbedF32_.resize(H * sizeof(float));
    scratchLatent_.resize(latentDim * sizeof(uint16_t));
    scratchLatentF32_.resize(latentDim * sizeof(float));
    scratchAudio_.resize(1 * 1 * 32000 * sizeof(uint16_t));
    scratchTimesteps_.resize(2 * sizeof(int32_t));
    positionIdsBuf_.resize(sizeof(int32_t));
    inputIdBuf_.resize(sizeof(int32_t));
    connScratch_.resize(H * sizeof(uint16_t));

    // fp32 staging buffers for TRT engine I/O (engines use fp32 I/O from ONNX)
    stagingEmbedsF32_.resize(H * sizeof(float));
    stagingHiddenF32_.resize(H * sizeof(float));
    stagingCondF32_.resize(2 * H * sizeof(float));
    stagingDecoderInF32_.resize(latentDim * sizeof(float));

    // Per-request scratch buffers
    int hopLen = meta_.hop_length > 0 ? meta_.hop_length : 3200;
    scratchDenormLatent_.resize(latentDim * sizeof(uint16_t));
    scratchAudioF32_.resize(hopLen * sizeof(float));
    scratchSampleF32_.resize(2 * latentDim * sizeof(float));
    scratchVpredF32_.resize(2 * latentDim * sizeof(float));
    if (eosHiddenSize_ > 0) {
        scratchEosHidden_.resize(eosHiddenSize_ * sizeof(uint16_t));
        scratchEosOut_.resize(1 * sizeof(uint16_t));
        scratchEosSigmoid_.resize(sizeof(float));
    }

    // 0.5B streaming acoustic decoder loop buffers
    if (type_ == ModelType::STREAMING_0_5B && streaming05bAcDecoder_.isLoaded()) {
        loop05bSingleLatGpu_.resize(latentDim * sizeof(float));
        loop05bSingleAudioGpu_.resize(hopLen * sizeof(float));
    }

    // 1.5B-specific scratch buffers
    if (type_ == ModelType::FULL_1_5B) {
        int vocabSize = meta_.vocab_size;
        stagingLogitsF32_.resize(vocabSize * sizeof(float));
        // Prefill buffers: max prefill seq ~512 tokens
        int maxPrefill = 512;
        stagingPrefillEmbedsF32_.resize((size_t)maxPrefill * H * sizeof(float));
        stagingPrefillLogitsF32_.resize((size_t)maxPrefill * vocabSize * sizeof(float));
        stagingPrefillHiddenF32_.resize((size_t)maxPrefill * H * sizeof(float));
        int semDim = meta_.semantic_vae_dim > 0 ? meta_.semantic_vae_dim : 128;
        scratchSemanticLatent_.resize(semDim * sizeof(uint16_t));
        scratchSemanticEmbed_.resize(H * sizeof(uint16_t));
        scratchVoiceEmbedF32_.resize((size_t)256 * H * sizeof(float)); // max 256 voice tokens

        // Per-iteration reusable buffers (avoid cudaMalloc/Free per step in autoregressive loop)
        loopSingleLatGpu_.resize(latentDim * sizeof(float));
        loopSingleAudioGpu_.resize(hopLen * sizeof(float));
        loopSemOutGpu_.resize(semDim * sizeof(float));
        loopSemFrameGpu_.resize(semDim * sizeof(float));
        loopSemLatFp16_.resize(semDim * sizeof(uint16_t));
        loopNegLogitsF32_.resize(vocabSize * sizeof(float));
        LOG_INFO("TTS", "allocated 1.5B scratch: vocab=%d, semDim=%d", vocabSize, semDim);
    }

    loaded_ = true;
    LOG_INFO("TTS", "ready (%zu voices available)", availableVoices().size());
    return true;
}

bool TTSPipeline::loadWeights() {
    std::string dir = cfg_.weightsDir + "/";

    // Embedding tokens
    EmbeddingWeights emb;
    if (!loadEmbeddingWeights(dir + "embed_tokens.bin", emb)) return false;
    if (!embedTokensGpu_.upload(emb.data)) return false;
    LOG_INFO("TTS", "embed_tokens [%u, %u] uploaded", emb.num_embeddings, emb.embedding_dim);

    // Acoustic connector
    ConnectorWeights conn;
    if (!loadConnectorWeights(dir + "acoustic_connector.bin", conn)) return false;
    connInputDim_ = conn.input_dim;
    connOutputDim_ = conn.output_dim;
    if (!connFc1W_.upload(conn.fc1_weight)) return false;
    if (!connFc1B_.upload(conn.fc1_bias)) return false;
    if (!connNormW_.upload(conn.norm_weight)) return false;
    if (!connFc2W_.upload(conn.fc2_weight)) return false;
    if (!connFc2B_.upload(conn.fc2_bias)) return false;
    LOG_INFO("TTS", "acoustic_connector [%u -> %u] uploaded", conn.input_dim, conn.output_dim);

    // Scaling factors
    if (!loadScalar(dir + "speech_scaling_factor.bin", speechScalingFactor_)) return false;
    if (!loadScalar(dir + "speech_bias_factor.bin", speechBiasFactor_)) return false;
    LOG_INFO("TTS", "scaling=%.4f, bias=%.4f", speechScalingFactor_, speechBiasFactor_);

    if (type_ == ModelType::STREAMING_0_5B) {
        // TTS input types
        TTSInputTypesWeights ttsTypes;
        if (!loadTTSInputTypesWeights(dir + "tts_input_types.bin", ttsTypes)) return false;
        if (!ttsInputTypesGpu_.upload(ttsTypes.data)) return false;
        LOG_INFO("TTS", "tts_input_types [%u, %u] uploaded", ttsTypes.num_types, ttsTypes.embedding_dim);

        // EOS classifier
        BinaryClassifierWeights eos;
        if (!loadBinaryClassifierWeights(dir + "tts_eos_classifier.bin", eos)) return false;
        eosHiddenSize_ = eos.hidden_size;
        if (!eosFc1W_.upload(eos.fc1_weight)) return false;
        if (!eosFc1B_.upload(eos.fc1_bias)) return false;
        if (!eosFc2W_.upload(eos.fc2_weight)) return false;
        if (!eosFc2B_.upload(eos.fc2_bias)) return false;
        LOG_INFO("TTS", "tts_eos_classifier [%u] uploaded", eos.hidden_size);
    }

    if (type_ == ModelType::FULL_1_5B) {
        // Semantic connector
        ConnectorWeights semConn;
        if (!loadConnectorWeights(dir + "semantic_connector.bin", semConn)) return false;
        semConnInputDim_ = semConn.input_dim;
        semConnOutputDim_ = semConn.output_dim;
        if (!semanticConnFc1W_.upload(semConn.fc1_weight)) return false;
        if (!semanticConnFc1B_.upload(semConn.fc1_bias)) return false;
        if (!semanticConnNormW_.upload(semConn.norm_weight)) return false;
        if (!semanticConnFc2W_.upload(semConn.fc2_weight)) return false;
        if (!semanticConnFc2B_.upload(semConn.fc2_bias)) return false;
        LOG_INFO("TTS", "semantic_connector [%u -> %u] uploaded",
                 semConn.input_dim, semConn.output_dim);
    }

    return true;
}

bool TTSPipeline::loadVoicePresets() {
    if (type_ == ModelType::STREAMING_0_5B) {
        // Load .bin voice presets
        if (!fs::exists(cfg_.voicesDir)) {
            LOG_ERROR("TTS", "voices dir not found: %s", cfg_.voicesDir.c_str());
            return false;
        }
        for (auto& entry : fs::directory_iterator(cfg_.voicesDir)) {
            if (entry.path().extension() == ".bin") {
                std::string name = toLower(entry.path().stem().string());
                VoicePreset vp;
                if (loadVoicePreset(entry.path().string(), vp)) {
                    voicePresets_[name] = std::move(vp);
                    LOG_INFO("TTS", "loaded voice '%s' (%u groups)",
                             name.c_str(), voicePresets_[name].num_groups);
                }
            }
        }
        if (voicePresets_.empty()) {
            LOG_WARN("TTS", "no voice presets found");
        }
    } else {
        // Load .wav voice file paths
        if (!fs::exists(cfg_.voicesDir)) {
            LOG_ERROR("TTS", "voices dir not found: %s", cfg_.voicesDir.c_str());
            return false;
        }
        for (auto& entry : fs::directory_iterator(cfg_.voicesDir)) {
            if (entry.path().extension() == ".wav") {
                std::string name = toLower(entry.path().stem().string());
                voiceFiles_[name] = entry.path().string();
            }
        }
        LOG_INFO("TTS", "found %zu voice WAV files", voiceFiles_.size());
    }
    return true;
}

std::vector<std::string> TTSPipeline::availableVoices() const {
    std::vector<std::string> names;
    if (type_ == ModelType::STREAMING_0_5B) {
        for (auto& [name, _] : voicePresets_) names.push_back(name);
    } else {
        for (auto& [name, _] : voiceFiles_) names.push_back(name);
    }
    std::sort(names.begin(), names.end());
    return names;
}

std::string TTSPipeline::resolveVoice(const std::string& input) const {
    std::string voice = toLower(input);

    // Helper: get reference to the appropriate voice map
    auto getKeys = [&]() -> std::vector<std::string> {
        std::vector<std::string> keys;
        if (type_ == ModelType::STREAMING_0_5B) {
            for (auto& [k, _] : voicePresets_) keys.push_back(k);
        } else {
            for (auto& [k, _] : voiceFiles_) keys.push_back(k);
        }
        return keys;
    };

    auto keys = getKeys();

    // Stage 1: exact match (lowercase)
    for (auto& k : keys) {
        if (k == voice) return k;
    }

    // Stage 2: OpenAI alias mapping -> substring match with alias target
    static const std::unordered_map<std::string, std::string> aliasMap = {
        {"alloy", "carter"}, {"echo", "wayne"},
        {"fable", "carter"}, {"onyx", "wayne"},
        {"nova", "carter"},  {"shimmer", "wayne"},
    };
    auto ait = aliasMap.find(voice);
    if (ait != aliasMap.end()) {
        for (auto& k : keys) {
            if (k.find(ait->second) != std::string::npos)
                return k;
        }
    }

    // Stage 3: substring match (voice in key OR key in voice)
    for (auto& k : keys) {
        if (k.find(voice) != std::string::npos || voice.find(k) != std::string::npos)
            return k;
    }

    // Stage 4: first available voice (fallback)
    if (!keys.empty()) return keys.front();

    return "";  // no voices available
}

// ── Synthesis dispatch ──

TTSPipeline::Result TTSPipeline::synthesize(const Request& req) {
    {
        FILE* f = fopen("C:\\Users\\onion\\Desktop\\Workspace\\VibeVoiceWindowsApiServer\\synth_trace.txt", "w");
        if (f) {
            fprintf(f, "synthesize called: loaded=%d type=%d text=%s voice=%s\n",
                    (int)loaded_, (int)type_, req.text.c_str(), req.voice.c_str());
            fclose(f);
        }
    }
    if (!loaded_) {
        Result r;
        r.error = "Pipeline not loaded";
        return r;
    }
    if (type_ == ModelType::STREAMING_0_5B) {
        return synth05B(req);
    } else {
        return synth15B(req);
    }
}

// ── LM decode helper (handles fp16<->fp32 conversion for TRT engine I/O) ──

void TTSPipeline::runLmDecode(TRTEngine& engine, KVCache& kv,
                               const __half* inputGpu, __half* outputGpu,
                               int64_t positionId) {
    int H = meta_.hidden_size;

    // Convert input fp16 -> fp32 for TRT engine
    GpuOps::halfToFloat(inputGpu, stagingEmbedsF32_.as<float>(), H, stream_);

    // Set position ID (TRT engine expects int32, not int64)
    int32_t posId32 = (int32_t)positionId;
    positionIdsBuf_.copyFromHost(&posId32, sizeof(int32_t));

    // Configure and run engine (all I/O in fp32 except position_ids which is int32)
    engine.setInputShape("inputs_embeds", nvinfer1::Dims3{1, 1, H});
    engine.setInputShape("position_ids", nvinfer1::Dims2{1, 1});
    engine.setTensorAddress("inputs_embeds", stagingEmbedsF32_.data());
    engine.setTensorAddress("position_ids", positionIdsBuf_.data());
    engine.setTensorAddress("hidden_states", stagingHiddenF32_.data());
    bindKVCache(engine, kv);
    engine.enqueueV3(stream_);
    kv.advance();

    // Convert output fp32 -> fp16 for subsequent GPU ops
    GpuOps::floatToHalf(stagingHiddenF32_.as<float>(), outputGpu, H, stream_);
}

// ── 1.5B LM Prefill helper (FP16 ONNX engine) ──

void TTSPipeline::runLmPrefill(TRTEngine& engine, KVCache& kv,
                                float* embedsF32Gpu, int seqLen,
                                float* logitsF32Gpu, float* hiddenF32Gpu) {
    int H = meta_.hidden_size;

    // Build position_ids [1, seqLen]: 0, 1, 2, ..., seqLen-1
    std::vector<int32_t> posIds(seqLen);
    for (int i = 0; i < seqLen; ++i) posIds[i] = i;
    CudaBuffer posIdsBuf;
    posIdsBuf.resize(seqLen * sizeof(int32_t));
    posIdsBuf.copyFromHost(posIds.data(), seqLen * sizeof(int32_t));

    // Build 4D causal attention mask [1, 1, seqLen, seqLen] as float32 additive mask
    // mask[i][j] = 0.0f if j <= i (attend), -3.4e38f if j > i (block)
    constexpr float MASK_MIN = -3.4028235e+38f;
    std::vector<float> causalMask((size_t)seqLen * seqLen);
    for (int i = 0; i < seqLen; ++i) {
        for (int j = 0; j < seqLen; ++j) {
            causalMask[i * seqLen + j] = (j <= i) ? 0.0f : MASK_MIN;
        }
    }
    CudaBuffer causalMaskGpu;
    causalMaskGpu.resize((size_t)seqLen * seqLen * sizeof(float));
    causalMaskGpu.copyFromHost(causalMask.data(), (size_t)seqLen * seqLen * sizeof(float));

    // Set shapes and bind — FP32 ONNX: all I/O is fp32
    engine.setInputShape("inputs_embeds", nvinfer1::Dims3{1, seqLen, H});
    engine.setInputShape("position_ids", nvinfer1::Dims2{1, seqLen});
    nvinfer1::Dims4 maskShape{1, 1, seqLen, seqLen};
    engine.setInputShape("attention_mask", maskShape);
    engine.setTensorAddress("inputs_embeds", (void*)embedsF32Gpu);   // fp32
    engine.setTensorAddress("position_ids", posIdsBuf.data());
    engine.setTensorAddress("attention_mask", causalMaskGpu.data());  // fp32
    engine.setTensorAddress("logits", logitsF32Gpu);                  // fp32
    engine.setTensorAddress("hidden_states", (void*)hiddenF32Gpu);   // fp32

    // Bind KV cache (prefill mode: only present outputs, no past inputs)
    bindKVCache(engine, kv, /*prefillMode=*/true);

    engine.enqueueV3(stream_);

    // Advance KV cache after prefill: swap buffers once so present→past
    kv.advanceAfterPrefill(seqLen);
}

// ── 1.5B LM Decode with logits helper (FP32 ONNX engine) ──

void TTSPipeline::runLmDecodeWithLogits(TRTEngine& engine, KVCache& kv,
                                          const float* inputGpu, float* hiddenOutGpu,
                                          float* logitsF32Gpu, int64_t positionId) {
    int H = meta_.hidden_size;

    // Set position ID
    int32_t posId32 = (int32_t)positionId;
    positionIdsBuf_.copyFromHost(&posId32, sizeof(int32_t));

    // Build 4D decode attention mask [1, 1, 1, totalSeq] as float32 additive mask
    // All 0.0f: the single query token attends to all past tokens + itself
    int totalSeq = kv.seqLen() + 1;
    size_t maskBytes = (size_t)totalSeq * sizeof(float);
    decodeMaskBuf_.resize(maskBytes);
    cudaMemsetAsync(decodeMaskBuf_.data(), 0, maskBytes, stream_);

    // Configure engine — FP32 ONNX: all I/O is fp32
    engine.setInputShape("inputs_embeds", nvinfer1::Dims3{1, 1, H});
    engine.setInputShape("position_ids", nvinfer1::Dims2{1, 1});
    nvinfer1::Dims4 maskShape{1, 1, 1, totalSeq};
    engine.setInputShape("attention_mask", maskShape);
    engine.setTensorAddress("inputs_embeds", (void*)inputGpu);     // fp32
    engine.setTensorAddress("position_ids", positionIdsBuf_.data());
    engine.setTensorAddress("attention_mask", decodeMaskBuf_.data());
    engine.setTensorAddress("logits", logitsF32Gpu);                // fp32
    engine.setTensorAddress("hidden_states", (void*)hiddenOutGpu); // fp32
    bindKVCache(engine, kv);
    engine.enqueueV3(stream_);
    kv.advance();
}

// ── 0.5B Streaming Generation ──

TTSPipeline::Result TTSPipeline::synth05B(const Request& req) {
    Result result;
    result.sampleRate = meta_.sample_rate;

    // Find voice preset
    auto it = voicePresets_.find(req.voice);
    if (it == voicePresets_.end()) {
        result.error = "Voice not found: " + req.voice;
        return result;
    }
    const VoicePreset& vp = it->second;

    // Need 4 groups for CFG (lm, tts_lm, neg_lm, neg_tts_lm)
    if (vp.num_groups < 4) {
        result.error = "Voice preset needs 4 groups for CFG, got " + std::to_string(vp.num_groups);
        return result;
    }

    // 1. Tokenize (Python ref: tokenizer.encode(text.strip() + "\n"))
    std::string textInput = req.text;
    // Strip leading/trailing whitespace
    size_t start = textInput.find_first_not_of(" \t\r\n");
    size_t end2 = textInput.find_last_not_of(" \t\r\n");
    if (start != std::string::npos && end2 != std::string::npos) {
        textInput = textInput.substr(start, end2 - start + 1);
    }
    textInput += "\n";
    auto textIds = tokenizer_.encode(textInput);
    if (textIds.empty()) {
        result.error = "Failed to tokenize text";
        return result;
    }

    // 2. Load voice preset into 4 KV-caches
    if (!baseLmKV_.loadFromPreset(vp.groups[0], vp.hidden_size, vp.head_dim)) {
        result.error = "Failed to load base_lm KV-cache from preset";
        return result;
    }
    if (!ttsLmKV_.loadFromPreset(vp.groups[1], vp.hidden_size, vp.head_dim)) {
        result.error = "Failed to load tts_lm KV-cache from preset";
        return result;
    }
    if (!negBaseLmKV_.loadFromPreset(vp.groups[2], vp.hidden_size, vp.head_dim)) {
        result.error = "Failed to load neg_base_lm KV-cache from preset";
        return result;
    }
    if (!negTtsLmKV_.loadFromPreset(vp.groups[3], vp.hidden_size, vp.head_dim)) {
        result.error = "Failed to load neg_tts_lm KV-cache from preset";
        return result;
    }

    int H = meta_.hidden_size;
    int latentDim = meta_.diffusion.latent_size;

    // FIX: Separate position counters for each KV cache (pos/neg may differ in seq_len)
    int64_t baseLmPos    = (int64_t)vp.groups[0].seq_len;
    int64_t ttsLmPos     = (int64_t)vp.groups[1].seq_len;
    int64_t negBaseLmPos = (int64_t)vp.groups[2].seq_len;
    int64_t negTtsLmPos  = (int64_t)vp.groups[3].seq_len;
    LOG_DEBUG("DIAG", "baseLmPos=%lld, ttsLmPos=%lld, negBaseLmPos=%lld, negTtsLmPos=%lld, H=%d, latent=%d",
             baseLmPos, ttsLmPos, negBaseLmPos, negTtsLmPos, H, latentDim);
    LOG_DEBUG("DIAG", "textIds count=%d, first5: %d %d %d %d %d",
             (int)textIds.size(),
             textIds.size()>0 ? textIds[0] : -1,
             textIds.size()>1 ? textIds[1] : -1,
             textIds.size()>2 ? textIds[2] : -1,
             textIds.size()>3 ? textIds[3] : -1,
             textIds.size()>4 ? textIds[4] : -1);

    // Get tts_input_types pointers
    // type 0 = speech, type 1 = text
    const __half* ttsType0 = ttsInputTypesGpu_.as<__half>();             // speech
    const __half* ttsType1 = ttsInputTypesGpu_.as<__half>() + H;        // text

    // Collect all denormalized latents for batch decoding at the end
    std::vector<float> allDenormLatents;  // [N * 64] stored as [t0_d0..t0_d63, t1_d0..t1_d63, ...]

    // Initialize RNG once per synthesis
    std::random_device rd05;
    std::mt19937 synthRng05(rd05());

    bool finished = false;
    int textOffset = 0;
    int textWindowSize = 5;
    int speechWindowSize = 6;
    int totalSpeechTokens = 0;

    // Text-aware max speech token limit (matching 1.5B safety net)
    int hopLen05 = meta_.hop_length > 0 ? meta_.hop_length : 3200;
    int absMax = (meta_.sample_rate * 30) / hopLen05;  // 30s absolute max
    int textBasedMax = std::max(45, (int)textIds.size() * 4);
    int maxSpeechTokens = std::min(absMax, textBasedMax);
    LOG_INFO("TTS", "0.5B maxSpeechTokens: min(absMax=%d, textBased=%d [textTokens=%d]) = %d",
             absMax, textBasedMax, (int)textIds.size(), maxSpeechTokens);

    // Initialize streaming acoustic decoder cache (zero at request start)
    if (streaming05bAcDecoder_.isLoaded()) {
        cudaMemsetAsync(acDec05bCacheGpu_.data(), 0, acDec05bCacheSize_ * sizeof(float), stream_);
        cudaMemsetAsync(acDec05bCacheOutGpu_.data(), 0, acDec05bCacheSize_ * sizeof(float), stream_);
    }

    while (!finished && totalSpeechTokens < maxSpeechTokens &&
           (textOffset < (int)textIds.size() || !finished)) {
        // === Text Window ===
        int windowEnd = std::min(textOffset + textWindowSize, (int)textIds.size());
        for (int ti = textOffset; ti < windowEnd; ++ti) {
            int32_t tokenId = textIds[ti];

            // a. Embedding lookup
            inputIdBuf_.copyFromHost(&tokenId, sizeof(int32_t));
            GpuOps::embeddingLookup(embedTokensGpu_.as<__half>(),
                                    inputIdBuf_.as<int32_t>(), scratchEmbed_.as<__half>(),
                                    1, H, stream_);

            // b. Run base_lm_decode (positive + negative, each with own position)
            if (ti == textOffset) diagLogFp16("embed_lookup", scratchEmbed_.as<__half>(), H, stream_);
            runLmDecode(baseLmDecode_, baseLmKV_,
                        scratchEmbed_.as<__half>(), scratchHidden_.as<__half>(), baseLmPos);
            runLmDecode(baseLmDecode_, negBaseLmKV_,
                        scratchEmbed_.as<__half>(), scratchHidden2_.as<__half>(), negBaseLmPos);
            if (ti == textOffset) {
                diagLogFp16("base_lm_pos_out", scratchHidden_.as<__half>(), H, stream_);
                diagLogFp16("base_lm_neg_out", scratchHidden2_.as<__half>(), H, stream_);
            }
            baseLmPos++;
            negBaseLmPos++;

            // c. tts_input = hidden + ttsInputTypes[1] (text type)
            GpuOps::vectorAdd(scratchHidden_.as<__half>(), ttsType1,
                              scratchHidden_.as<__half>(), H, stream_);
            GpuOps::vectorAdd(scratchHidden2_.as<__half>(), ttsType1,
                              scratchHidden2_.as<__half>(), H, stream_);

            // d. Run tts_lm_decode (positive + negative, each with own position)
            runLmDecode(ttsLmDecode_, ttsLmKV_,
                        scratchHidden_.as<__half>(), scratchHidden_.as<__half>(), ttsLmPos);
            runLmDecode(ttsLmDecode_, negTtsLmKV_,
                        scratchHidden2_.as<__half>(), scratchHidden2_.as<__half>(), negTtsLmPos);
            if (ti == textOffset) {
                diagLogFp16("tts_lm_pos_out", scratchHidden_.as<__half>(), H, stream_);
                diagLogFp16("tts_lm_neg_out", scratchHidden2_.as<__half>(), H, stream_);
            }
            ttsLmPos++;
            negTtsLmPos++;
        }
        textOffset = windowEnd;

        // === Speech Window ===
        for (int si = 0; si < speechWindowSize; ++si) {
            // a. Sample speech token via diffusion
            if (si == 0) {
                diagLogFp16("diffusion_cond_pos", scratchHidden_.as<__half>(), H, stream_);
                diagLogFp16("diffusion_cond_neg", scratchHidden2_.as<__half>(), H, stream_);
            }
            // 0.5B path: convert FP16 hidden to FP32 for sampleSpeechTokens
            GpuOps::halfToFloat(scratchHidden_.as<__half>(), stagingEmbedsF32_.as<float>(), H, stream_);
            GpuOps::halfToFloat(scratchHidden2_.as<__half>(), stagingHiddenF32_.as<float>(), H, stream_);
            std::vector<float> latent = sampleSpeechTokens(
                stagingEmbedsF32_.as<float>(), stagingHiddenF32_.as<float>(),
                H, cfg_.cfgScale, synthRng05);

            if (si == 0) {
                float lMin = *std::min_element(latent.begin(), latent.end());
                float lMax = *std::max_element(latent.begin(), latent.end());
                LOG_DEBUG("DIAG", "diffusion_latent [%d]: min=%.4f max=%.4f", (int)latent.size(), lMin, lMax);
            }

            // b. Denormalize on CPU: latent / scaling_factor - bias_factor
            //    Collect for batch decoding at end
            std::vector<float> denormLatent(latentDim);
            for (int i = 0; i < latentDim; ++i) {
                denormLatent[i] = latent[i] / speechScalingFactor_ - speechBiasFactor_;
            }
            allDenormLatents.insert(allDenormLatents.end(), denormLatent.begin(), denormLatent.end());
            totalSpeechTokens++;

            // c. Run connector: latent -> LM embedding (needs fp16 latent on GPU)
            scratchLatentF32_.copyFromHost(latent.data(), latent.size() * sizeof(float));
            GpuOps::floatToHalf(scratchLatentF32_.as<float>(), scratchLatent_.as<__half>(),
                                latentDim, stream_);

            runConnector(scratchLatent_.as<__half>(), scratchEmbed_.as<__half>(),
                         connFc1W_, connFc1B_, connNormW_, connFc2W_, connFc2B_,
                         connInputDim_, connOutputDim_);

            // d. Add tts_input_types[0] (speech type)
            GpuOps::vectorAdd(scratchEmbed_.as<__half>(), ttsType0,
                              scratchHidden_.as<__half>(), H, stream_);
            GpuOps::vectorAdd(scratchEmbed_.as<__half>(), ttsType0,
                              scratchHidden2_.as<__half>(), H, stream_);

            // e. Run tts_lm_decode for both paths (base_lm NOT called for speech tokens)
            runLmDecode(ttsLmDecode_, ttsLmKV_,
                        scratchHidden_.as<__half>(), scratchHidden_.as<__half>(), ttsLmPos);
            runLmDecode(ttsLmDecode_, negTtsLmKV_,
                        scratchHidden2_.as<__half>(), scratchHidden2_.as<__half>(), negTtsLmPos);
            ttsLmPos++;
            negTtsLmPos++;

            // f. EOS check
            float eosScore = runEosClassifier(scratchHidden_.as<__half>());
            LOG_DEBUG("DIAG", "eos_score[si=%d]=%.6f", si, eosScore);
            if (eosScore > 0.5f) {
                finished = true;
                break;
            }
        }

        // If all text has been processed and we haven't gotten EOS,
        // continue generating speech tokens until EOS or maxSpeechTokens
        if (textOffset >= (int)textIds.size() && !finished) {
            // maxSpeechTokens check is in the while condition
        }
    }

    // === Acoustic Decoding ===
    int N = totalSpeechTokens;
    int hopLen = meta_.hop_length;

    if (N > 0 && streaming05bAcDecoder_.isLoaded()) {
        // === Streaming Acoustic Decoding (matches Python reference) ===
        // Decode per-frame: [1, 64, 1] + cache -> [1, 1, hop_length] + cache
        LOG_INFO("TTS", "streaming decoding %d latent frames -> %d audio samples", N, N * hopLen);

        result.audio.reserve(N * hopLen);
        for (int t = 0; t < N; ++t) {
            // Upload denormalized latent for this frame
            loop05bSingleLatGpu_.copyFromHost(
                allDenormLatents.data() + t * latentDim, latentDim * sizeof(float));

            streaming05bAcDecoder_.setInputShape("latent", nvinfer1::Dims3{1, latentDim, 1});
            nvinfer1::Dims2 cacheDims{1, acDec05bCacheSize_};
            streaming05bAcDecoder_.setInputShape("cache_in", cacheDims);
            streaming05bAcDecoder_.setTensorAddress("latent", loop05bSingleLatGpu_.data());
            streaming05bAcDecoder_.setTensorAddress("cache_in", acDec05bCacheGpu_.data());
            streaming05bAcDecoder_.setTensorAddress("audio", loop05bSingleAudioGpu_.data());
            streaming05bAcDecoder_.setTensorAddress("cache_out", acDec05bCacheOutGpu_.data());
            streaming05bAcDecoder_.enqueueV3(stream_);

            // Swap cache buffers (double-buffering)
            std::swap(acDec05bCacheGpu_, acDec05bCacheOutGpu_);

            // Copy audio chunk to result
            std::vector<float> chunk(hopLen);
            cudaStreamSynchronize(stream_);
            loop05bSingleAudioGpu_.copyToHost(chunk.data(), hopLen * sizeof(float));
            result.audio.insert(result.audio.end(), chunk.begin(), chunk.end());
        }

        float aMin = *std::min_element(result.audio.begin(), result.audio.end());
        float aMax = *std::max_element(result.audio.begin(), result.audio.end());
        LOG_INFO("TTS", "streaming decoded audio [%d samples]: min=%.4f max=%.4f",
                 (int)result.audio.size(), aMin, aMax);
    } else if (N > 0) {
        // === Fallback: Batch Acoustic Decoding ===
        // Decode all latents at once: [1, 64, N] -> [1, 1, N * hop_length]
        LOG_INFO("TTS", "batch decoding %d latent frames -> %d audio samples", N, N * hopLen);

        // Transpose from [N, 64] (row-major) to [64, N] for decoder input [1, 64, N]
        std::vector<float> batchLatent(64 * N);
        for (int t = 0; t < N; ++t) {
            for (int d = 0; d < 64; ++d) {
                batchLatent[d * N + t] = allDenormLatents[t * 64 + d];
            }
        }

        CudaBuffer batchLatentGpu;
        batchLatentGpu.resize(64 * N * sizeof(float));
        batchLatentGpu.copyFromHostAsync(batchLatent.data(), 64 * N * sizeof(float), stream_);

        CudaBuffer batchAudioGpu;
        batchAudioGpu.resize(1 * 1 * N * hopLen * sizeof(float));

        acousticDecoder_.setInputShape("latent", nvinfer1::Dims3{1, 64, N});
        acousticDecoder_.setTensorAddress("latent", batchLatentGpu.data());
        acousticDecoder_.setTensorAddress("audio", batchAudioGpu.data());
        acousticDecoder_.enqueueV3(stream_);

        result.audio.resize(N * hopLen);
        cudaStreamSynchronize(stream_);
        batchAudioGpu.copyToHost(result.audio.data(), N * hopLen * sizeof(float));

        float aMin = *std::min_element(result.audio.begin(), result.audio.end());
        float aMax = *std::max_element(result.audio.begin(), result.audio.end());
        LOG_INFO("TTS", "batch decoded audio [%d samples]: min=%.4f max=%.4f",
                 (int)result.audio.size(), aMin, aMax);
    }
    result.ok = !result.audio.empty();
    if (!result.ok) {
        result.error = "Generated empty audio";
    }
    return result;
}

// ── 1.5B Full Generation ──

TTSPipeline::Result TTSPipeline::synth15B(const Request& req) {
    Result result;
    result.sampleRate = meta_.sample_rate;

    int H = meta_.hidden_size;              // 1536
    int latentDim = meta_.diffusion.latent_size;  // 64
    int vocabSize = meta_.vocab_size;        // 151936
    int semDim = meta_.semantic_vae_dim > 0 ? meta_.semantic_vae_dim : 128;
    int hopLen = meta_.hop_length > 0 ? meta_.hop_length : 3200;
    int numLmLayers = meta_.num_hidden_layers; // 28
    int numKvHeads = meta_.num_key_value_heads; // 2
    int headDim = meta_.head_dim;            // 128

    // ── 1. Find voice WAV file ──
    auto it = voiceFiles_.find(req.voice);
    if (it == voiceFiles_.end()) {
        result.error = "Voice not found: " + req.voice;
        return result;
    }
    std::string voicePath = it->second;

    // ── 2. Load and prepare voice audio ──
    std::vector<float> voiceAudio;
    if (!AudioIO::loadAndPrepare(voicePath, voiceAudio, meta_.sample_rate, -25.0f)) {
        result.error = "Failed to load voice WAV: " + voicePath;
        return result;
    }
    LOG_INFO("TTS", "voice audio loaded: %zu samples (%.2f sec)",
             voiceAudio.size(), (float)voiceAudio.size() / meta_.sample_rate);

    // ── 3. Run acoustic encoder: audio -> latent_mean ──
    // Input: [1, 1, N] fp32, Output: [1, T, 64] fp32
    int audioLen = (int)voiceAudio.size();
    CudaBuffer voiceAudioGpu;
    voiceAudioGpu.resize(audioLen * sizeof(float));
    voiceAudioGpu.copyFromHost(voiceAudio.data(), audioLen * sizeof(float));

    // T = number of latent frames (depends on model hop; typically audioLen / 3200)
    int voiceLatentFrames = audioLen / hopLen;
    if (voiceLatentFrames < 1) voiceLatentFrames = 1;

    CudaBuffer voiceLatentGpu;
    voiceLatentGpu.resize((size_t)voiceLatentFrames * latentDim * sizeof(float));

    acousticEncoder_.setInputShape("audio", nvinfer1::Dims3{1, 1, audioLen});
    acousticEncoder_.setTensorAddress("audio", voiceAudioGpu.data());
    acousticEncoder_.setTensorAddress("latent_mean", voiceLatentGpu.data());
    acousticEncoder_.enqueueV3(stream_);
    cudaStreamSynchronize(stream_);

    LOG_INFO("TTS", "acoustic encoder: %d frames of latent dim %d", voiceLatentFrames, latentDim);

    // ── 4. Normalize voice latents: (latent + bias) * scale ──
    // Then run acoustic_connector for each frame: [64] -> [H]
    // Build voice embeddings [T, H] in fp32
    std::vector<float> voiceLatentCpu(voiceLatentFrames * latentDim);
    voiceLatentGpu.copyToHost(voiceLatentCpu.data(), voiceLatentFrames * latentDim * sizeof(float));

    // DIAG: Log raw voice latent stats BEFORE normalization
    {
        float vMin = 1e30f, vMax = -1e30f, vSum = 0;
        int vN = voiceLatentFrames * latentDim;
        for (int i = 0; i < vN; ++i) {
            if (voiceLatentCpu[i] < vMin) vMin = voiceLatentCpu[i];
            if (voiceLatentCpu[i] > vMax) vMax = voiceLatentCpu[i];
            vSum += voiceLatentCpu[i];
        }
        float vNorm = 0;
        for (int i = 0; i < latentDim; ++i) vNorm += voiceLatentCpu[i] * voiceLatentCpu[i];
        vNorm = sqrtf(vNorm);
        LOG_INFO("TTS", "DIAG voice raw latent: min=%.4f max=%.4f avg=%.4f frame0_norm=%.4f",
                 vMin, vMax, vSum / vN, vNorm);
    }

    // DIAG: Test acoustic decoder with voice latents (first 3 frames)
    {
        int testN = std::min(3, voiceLatentFrames);
        std::vector<float> testLatent((size_t)latentDim * testN);
        for (int t = 0; t < testN; ++t) {
            for (int d = 0; d < latentDim; ++d) {
                testLatent[d * testN + t] = voiceLatentCpu[t * latentDim + d];
            }
        }
        CudaBuffer testLatGpu;
        testLatGpu.resize(latentDim * testN * sizeof(float));
        testLatGpu.copyFromHost(testLatent.data(), latentDim * testN * sizeof(float));
        int testAudioLen = testN * hopLen;
        CudaBuffer testAudioGpu;
        testAudioGpu.resize(testAudioLen * sizeof(float));
        acousticDecoder_.setInputShape("latent", nvinfer1::Dims3{1, latentDim, testN});
        acousticDecoder_.setTensorAddress("latent", testLatGpu.data());
        acousticDecoder_.setTensorAddress("audio", testAudioGpu.data());
        acousticDecoder_.enqueueV3(stream_);
        cudaStreamSynchronize(stream_);
        std::vector<float> testAudio(testAudioLen);
        testAudioGpu.copyToHost(testAudio.data(), testAudioLen * sizeof(float));
        float aMin = *std::min_element(testAudio.begin(), testAudio.end());
        float aMax = *std::max_element(testAudio.begin(), testAudio.end());
        LOG_INFO("TTS", "DIAG decoder test with voice latents [%d frames]: min=%.4f max=%.4f",
                 testN, aMin, aMax);
    }

    // Normalize: (latent + bias_factor) * scaling_factor
    for (int i = 0; i < voiceLatentFrames * latentDim; ++i) {
        voiceLatentCpu[i] = (voiceLatentCpu[i] + speechBiasFactor_) * speechScalingFactor_;
    }

    // Run acoustic connector for each frame
    std::vector<float> voiceEmbedsCpu(voiceLatentFrames * H);
    for (int t = 0; t < voiceLatentFrames; ++t) {
        // Upload normalized latent [64] as fp16
        std::vector<float> frameLat(voiceLatentCpu.begin() + t * latentDim,
                                     voiceLatentCpu.begin() + (t + 1) * latentDim);
        scratchLatentF32_.copyFromHost(frameLat.data(), latentDim * sizeof(float));
        GpuOps::floatToHalf(scratchLatentF32_.as<float>(), scratchLatent_.as<__half>(),
                            latentDim, stream_);

        runConnector(scratchLatent_.as<__half>(), scratchEmbed_.as<__half>(),
                     connFc1W_, connFc1B_, connNormW_, connFc2W_, connFc2B_,
                     connInputDim_, connOutputDim_);

        // Download embedding [H] as fp16 -> fp32
        GpuOps::halfToFloat(scratchEmbed_.as<__half>(), stagingEmbedsF32_.as<float>(), H, stream_);
        cudaStreamSynchronize(stream_);
        stagingEmbedsF32_.copyToHost(voiceEmbedsCpu.data() + t * H, H * sizeof(float));
    }

    LOG_INFO("TTS", "voice embeddings: %d frames x %d dim", voiceLatentFrames, H);

    // ── 5. Build full prompt token sequence (matching Python reference) ──
    // Format: system_tokens + voice_section + text_section + output_section
    //
    // System: " Transform the text provided by various speakers into speech output,
    //          utilizing the distinct voice of each respective speaker.\n"
    // Voice:  " Voice input:\n Speaker 1:" + [speech_start] + [speech_diffusion]*T + [speech_end] + "\n"
    // Text:   " Text input:\n Speaker 1:{text}\n"
    // Output: " Speech output:\n" + [speech_start]

    // Strip text
    std::string textInput = req.text;
    {
        size_t start = textInput.find_first_not_of(" \t\r\n");
        size_t end2 = textInput.find_last_not_of(" \t\r\n");
        if (start != std::string::npos && end2 != std::string::npos) {
            textInput = textInput.substr(start, end2 - start + 1);
        }
    }

    // Tokenize each section (matching Python reference tokenization boundaries)
    auto systemTokens = tokenizer_.encode(
        " Transform the text provided by various speakers into speech output, "
        "utilizing the distinct voice of each respective speaker.\n");
    auto voiceInputTokens = tokenizer_.encode(" Voice input:\n");
    auto speakerVoicePrefix = tokenizer_.encode(" Speaker 0:");
    auto voiceNewline = tokenizer_.encode("\n");
    auto textInputPrefix = tokenizer_.encode(" Text input:\n");
    auto textLineTokens = tokenizer_.encode(" Speaker 0:" + textInput + "\n");
    auto outputSectionTokens = tokenizer_.encode(" Speech output:\n");

    if (systemTokens.empty() || textLineTokens.empty()) {
        result.error = "Failed to tokenize prompt";
        return result;
    }

    LOG_INFO("TTS", "prompt tokens: system=%zu voiceInput=%zu speakerVoice=%zu textPrefix=%zu textLine=%zu output=%zu voice_frames=%d",
             systemTokens.size(), voiceInputTokens.size(), speakerVoicePrefix.size(),
             textInputPrefix.size(), textLineTokens.size(),
             outputSectionTokens.size(), voiceLatentFrames);

    // Build full token ID sequence (voice placeholders = speech_diffusion)
    // Matching Python: system + voiceInput + speakerVoice + speech_start + diff*T + speech_end + \n
    //                  + textInputPrefix + textLine + outputSection + speech_start
    std::vector<int32_t> fullTokens;
    std::vector<bool> isVoiceSlot;

    // System tokens
    for (int32_t t : systemTokens) { fullTokens.push_back(t); isVoiceSlot.push_back(false); }
    // Voice input prefix: " Voice input:\n"
    for (int32_t t : voiceInputTokens) { fullTokens.push_back(t); isVoiceSlot.push_back(false); }
    // Speaker voice prefix: " Speaker 0:"
    for (int32_t t : speakerVoicePrefix) { fullTokens.push_back(t); isVoiceSlot.push_back(false); }
    // speech_start
    fullTokens.push_back(specialTokens_.speech_start); isVoiceSlot.push_back(false);
    // VAE placeholders (will be replaced with voice embeddings)
    for (int t = 0; t < voiceLatentFrames; ++t) {
        fullTokens.push_back(specialTokens_.speech_diffusion); isVoiceSlot.push_back(true);
    }
    // speech_end
    fullTokens.push_back(specialTokens_.speech_end); isVoiceSlot.push_back(false);
    // newline after voice section
    for (int32_t t : voiceNewline) { fullTokens.push_back(t); isVoiceSlot.push_back(false); }
    // Text input prefix: " Text input:\n"
    for (int32_t t : textInputPrefix) { fullTokens.push_back(t); isVoiceSlot.push_back(false); }
    // Text line: " Speaker 0:{text}\n"
    for (int32_t t : textLineTokens) { fullTokens.push_back(t); isVoiceSlot.push_back(false); }
    // Output section: " Speech output:\n"
    for (int32_t t : outputSectionTokens) { fullTokens.push_back(t); isVoiceSlot.push_back(false); }
    // Final speech_start (generation begins after this)
    fullTokens.push_back(specialTokens_.speech_start); isVoiceSlot.push_back(false);

    int promptLen = (int)fullTokens.size();

    // ── 6. Build prompt embedding sequence ──
    // Get token embedding helper
    auto getTokenEmbed = [&](int32_t tokenId, float* outF32) {
        inputIdBuf_.copyFromHost(&tokenId, sizeof(int32_t));
        GpuOps::embeddingLookup(embedTokensGpu_.as<__half>(),
                                inputIdBuf_.as<int32_t>(), scratchEmbed_.as<__half>(),
                                1, H, stream_);
        GpuOps::halfToFloat(scratchEmbed_.as<__half>(), stagingEmbedsF32_.as<float>(), H, stream_);
        cudaStreamSynchronize(stream_);
        stagingEmbedsF32_.copyToHost(outF32, H * sizeof(float));
    };

    // Build prompt embeddings in CPU fp32
    std::vector<float> promptEmbeds(promptLen * H);
    int voiceIdx = 0;
    for (int i = 0; i < promptLen; ++i) {
        if (isVoiceSlot[i]) {
            // Replace with actual voice acoustic embedding
            memcpy(promptEmbeds.data() + (size_t)i * H,
                   voiceEmbedsCpu.data() + (size_t)voiceIdx * H,
                   H * sizeof(float));
            voiceIdx++;
        } else {
            getTokenEmbed(fullTokens[i], promptEmbeds.data() + (size_t)i * H);
        }
    }

    LOG_INFO("TTS", "positive prompt: %d tokens (%zu sys + voice[%d] + %zu text + output)",
             promptLen, systemTokens.size(), voiceLatentFrames, textLineTokens.size());

    // Debug: dump first and last token IDs
    {
        FILE* tokDbg = fopen("C:\\Users\\onion\\Desktop\\Workspace\\VibeVoiceWindowsApiServer\\gen_debug.txt", "w");
        if (tokDbg) {
            fprintf(tokDbg, "=== Prompt Token IDs (total=%d) ===\n", promptLen);
            for (int i = 0; i < promptLen; ++i) {
                fprintf(tokDbg, "  [%d] id=%d voice=%d\n", i, fullTokens[i], isVoiceSlot[i] ? 1 : 0);
            }
            fprintf(tokDbg, "\n");
            fclose(tokDbg);
        }
    }

    // ── 7. Build negative prompt: just [speech_start_embed] ──
    std::vector<float> negPromptEmbeds(1 * H);
    getTokenEmbed(specialTokens_.speech_start, negPromptEmbeds.data());

    // ── 8. Init KV caches (pos + neg) — FP32 for 1.5B FP32 ONNX engines ──
    int maxSeqLen = 4096;
    if (!lmKV_.init(numLmLayers, numKvHeads, headDim, maxSeqLen, /*useFp16=*/false)) {
        result.error = "Failed to init positive LM KV cache";
        return result;
    }
    if (!negLmKV_.init(numLmLayers, numKvHeads, headDim, maxSeqLen, /*useFp16=*/false)) {
        result.error = "Failed to init negative LM KV cache";
        return result;
    }

    // ── 9. Prefill positive path (FP32 embeds, FP32 hidden, FP32 logits) ──
    // Upload fp32 embeddings directly to GPU (no fp16 conversion needed)
    CudaBuffer prefillEmbedsF32Gpu;
    prefillEmbedsF32Gpu.resize((size_t)promptLen * H * sizeof(float));
    prefillEmbedsF32Gpu.copyFromHost(promptEmbeds.data(), (size_t)promptLen * H * sizeof(float));

    CudaBuffer prefillLogitsGpu;
    prefillLogitsGpu.resize((size_t)promptLen * vocabSize * sizeof(float));
    CudaBuffer prefillHiddenF32Gpu;
    prefillHiddenF32Gpu.resize((size_t)promptLen * H * sizeof(float));

    runLmPrefill(lmPrefill_, lmKV_,
                 prefillEmbedsF32Gpu.as<float>(), promptLen,
                 prefillLogitsGpu.as<float>(), prefillHiddenF32Gpu.as<float>());
    cudaStreamSynchronize(stream_);
    LOG_INFO("TTS", "positive prefill done (seq=%d)", promptLen);

    // Get last-position logits (fp32) from prefill
    std::vector<float> lastLogits(vocabSize);
    cudaMemcpy(lastLogits.data(),
               (float*)prefillLogitsGpu.data() + (size_t)(promptLen - 1) * vocabSize,
               vocabSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Extract last-position hidden (fp32) directly to scratchHidden_ on GPU
    cudaMemcpy(scratchHidden_.data(),
               (float*)prefillHiddenF32Gpu.data() + (size_t)(promptLen - 1) * H,
               H * sizeof(float), cudaMemcpyDeviceToDevice);

    prefillEmbedsF32Gpu.free();
    prefillLogitsGpu.free();
    prefillHiddenF32Gpu.free();

    // ── 10. Prefill negative path (FP32 embeds, FP32 hidden, FP32 logits) ──
    CudaBuffer negEmbedsF32Gpu;
    negEmbedsF32Gpu.resize(1 * H * sizeof(float));
    negEmbedsF32Gpu.copyFromHost(negPromptEmbeds.data(), 1 * H * sizeof(float));

    CudaBuffer negPrefillLogitsGpu;
    negPrefillLogitsGpu.resize(1 * vocabSize * sizeof(float));
    CudaBuffer negPrefillHiddenF32Gpu;
    negPrefillHiddenF32Gpu.resize(1 * H * sizeof(float));

    runLmPrefill(lmPrefill_, negLmKV_,
                 negEmbedsF32Gpu.as<float>(), 1,
                 negPrefillLogitsGpu.as<float>(), negPrefillHiddenF32Gpu.as<float>());
    cudaStreamSynchronize(stream_);
    LOG_INFO("TTS", "negative prefill done (seq=1)");

    // ── 11. Autoregressive generation loop ──
    int64_t posPos = (int64_t)promptLen;
    int64_t negPos = 1;

    std::vector<float> allDenormLatents; // collected (kept for diagnostics)
    std::vector<std::vector<float>> audioChunks; // streaming decoded audio chunks
    int totalSpeechTokens = 0;

    // Initialize RNG once per synthesis (matches Python's global torch RNG behavior)
    std::random_device rd;
    std::mt19937 synthRng(rd());

    // Initialize streaming caches to zero (equivalent to Python's zero-initialized cache)
    // Zero both in and out buffers to prevent stale data from previous requests
    cudaMemsetAsync(semCacheGpu_.data(), 0, semCacheSize_ * sizeof(float), stream_);
    cudaMemsetAsync(semCacheOutGpu_.data(), 0, semCacheSize_ * sizeof(float), stream_);
    cudaMemsetAsync(acDecCacheGpu_.data(), 0, acDecCacheSize_ * sizeof(float), stream_);
    cudaMemsetAsync(acDecCacheOutGpu_.data(), 0, acDecCacheSize_ * sizeof(float), stream_);
    // Max speech tokens: text-aware safety limit
    // On GTX 1660 (Python/BF16), LM runs in effective FP32 -> stable logit convergence
    // On RTX 3080 (C++/TRT FP16), logit convergence is less reliable
    // Safety net: cap max speech tokens based on text length to prevent extreme over-generation
    // Empirical ratio: ~2.5 speech tokens per text token (0.33s audio per text token)
    // Use 4x safety factor -> ~10 speech tokens per text token
    int absMax = (meta_.sample_rate * 30) / hopLen; // 30 seconds absolute max
    int textTokenCount = (int)textLineTokens.size();
    int textBasedMax = std::max(45, textTokenCount * 4); // min 6 seconds, 4x estimated duration
    int maxSpeechTokens = std::min(absMax, textBasedMax);
    LOG_INFO("TTS", "maxSpeechTokens: min(absMax=%d, textBased=%d [textTokens=%d]) = %d",
             absMax, textBasedMax, textTokenCount, maxSpeechTokens);
    bool finished = false;

    // Positive hidden is already in scratchHidden_ (extracted from prefill above)

    // Extract negative hidden (fp32) directly to scratchHidden2_ on GPU
    cudaMemcpy(scratchHidden2_.data(), negPrefillHiddenF32Gpu.data(),
               H * sizeof(float), cudaMemcpyDeviceToDevice);

    // Free negative prefill buffers
    negEmbedsF32Gpu.free();
    negPrefillLogitsGpu.free();
    negPrefillHiddenF32Gpu.free();

    // Use persistent negative logits buffer (class member, avoids per-request alloc/free)

    LOG_INFO("TTS", "starting autoregressive generation (maxTokens=%d)", maxSpeechTokens);

    // Debug: open a log file for generation diagnostics (append to token dump)
    FILE* dbgLog = fopen("C:\\Users\\onion\\Desktop\\Workspace\\VibeVoiceWindowsApiServer\\gen_debug.txt", "a");
    if (dbgLog) {
        fprintf(dbgLog, "=== Generation Debug ===\n");
        fprintf(dbgLog, "vocabSize=%d maxSpeechTokens=%d\n", vocabSize, maxSpeechTokens);
        fprintf(dbgLog, "speech_diffusion=%d speech_end=%d eos=%d\n",
                specialTokens_.speech_diffusion, specialTokens_.speech_end, specialTokens_.eos);
        fprintf(dbgLog, "Initial logits: diff=%.6f end=%.6f eos=%.6f\n",
                lastLogits[specialTokens_.speech_diffusion],
                lastLogits[specialTokens_.speech_end],
                lastLogits[specialTokens_.eos]);
        fflush(dbgLog);
    }

    while (!finished && totalSpeechTokens < maxSpeechTokens) {
        // a. Apply token constraint: only allow {speech_start, speech_end, speech_diffusion, eos}
        int32_t allowedTokens[] = {
            specialTokens_.speech_start,
            specialTokens_.speech_end,
            specialTokens_.speech_diffusion,
            specialTokens_.eos
        };
        for (int i = 0; i < vocabSize; ++i) {
            bool allowed = false;
            for (int32_t at : allowedTokens) {
                if (i == at) { allowed = true; break; }
            }
            if (!allowed) lastLogits[i] = -1e30f;
        }

        // b. Argmax -> next_token
        int32_t nextToken = 0;
        float maxLogit = lastLogits[0];
        for (int i = 1; i < vocabSize; ++i) {
            if (lastLogits[i] > maxLogit) {
                maxLogit = lastLogits[i];
                nextToken = i;
            }
        }

        if (dbgLog) {
            fprintf(dbgLog, "step %d: token=%d logits[diff=%.6f end=%.6f eos=%.6f]\n",
                    totalSpeechTokens, nextToken,
                    lastLogits[specialTokens_.speech_diffusion],
                    lastLogits[specialTokens_.speech_end],
                    lastLogits[specialTokens_.eos]);
            fflush(dbgLog);
        }

        if (nextToken == specialTokens_.eos) {
            LOG_INFO("TTS", "generation stopped at eos (step %d)", totalSpeechTokens);
            finished = true;
            break;
        }

        if (nextToken == specialTokens_.speech_end) {
            LOG_INFO("TTS", "speech_end at step %d, stopping generation", totalSpeechTokens);
            finished = true;
            break;
        }

        if (nextToken == specialTokens_.speech_start) {
            // Feed speech_start embedding to both LM paths
            LOG_INFO("TTS", "speech_start at step %d", totalSpeechTokens);
            // Get embedding fp16 on GPU, then convert to fp32 for LM
            int32_t tokId = specialTokens_.speech_start;
            inputIdBuf_.copyFromHost(&tokId, sizeof(int32_t));
            GpuOps::embeddingLookup(embedTokensGpu_.as<__half>(),
                                    inputIdBuf_.as<int32_t>(), scratchEmbed_.as<__half>(),
                                    1, H, stream_);
            GpuOps::halfToFloat(scratchEmbed_.as<__half>(), scratchEmbedF32_.as<float>(), H, stream_);

            runLmDecodeWithLogits(lmDecode_, negLmKV_,
                                  scratchEmbedF32_.as<float>(), scratchHidden2_.as<float>(),
                                  loopNegLogitsF32_.as<float>(), negPos);
            negPos++;
            // Sync: ensure negative decode completes before reusing shared
            // TRT context, positionIdsBuf_, and decodeMaskBuf_
            cudaStreamSynchronize(stream_);
            runLmDecodeWithLogits(lmDecode_, lmKV_,
                                  scratchEmbedF32_.as<float>(), scratchHidden_.as<float>(),
                                  stagingLogitsF32_.as<float>(), posPos);
            posPos++;

            cudaStreamSynchronize(stream_);
            stagingLogitsF32_.copyToHost(lastLogits.data(), vocabSize * sizeof(float));
            continue;
        }

        if (nextToken == specialTokens_.speech_diffusion) {
            // c. Diffusion sampling using positive and negative hidden states (FP32)
            std::vector<float> latent = sampleSpeechTokens(
                scratchHidden_.as<float>(), scratchHidden2_.as<float>(),
                H, cfg_.cfgScale, synthRng);

            // d. Denormalize for acoustic decoder: latent / scaling - bias
            std::vector<float> denormLatent(latentDim);
            for (int i = 0; i < latentDim; ++i) {
                denormLatent[i] = latent[i] / speechScalingFactor_ - speechBiasFactor_;
            }
            allDenormLatents.insert(allDenormLatents.end(), denormLatent.begin(), denormLatent.end());
            totalSpeechTokens++;

            // e. Compute next_embed = acoustic_embed + semantic_embed

            // e1. acoustic_connector: normalized latent [64] -> [H]
            scratchLatentF32_.copyFromHost(latent.data(), latentDim * sizeof(float));
            GpuOps::floatToHalf(scratchLatentF32_.as<float>(), scratchLatent_.as<__half>(),
                                latentDim, stream_);
            runConnector(scratchLatent_.as<__half>(), scratchEmbed_.as<__half>(),
                         connFc1W_, connFc1B_, connNormW_, connFc2W_, connFc2B_,
                         connInputDim_, connOutputDim_);
            // scratchEmbed_ now has acoustic embedding [H] fp16

            // e2. Streaming acoustic decode: latent [1,64,1] + cache → audio [1,1,3200] + cache
            // (matches Python's streaming decoder with per-layer SConv1d/SConvTranspose1d cache)
            loopSingleLatGpu_.copyFromHost(denormLatent.data(), latentDim * sizeof(float));

            streamingAcDecoder_.setInputShape("latent", nvinfer1::Dims3{1, latentDim, 1});
            nvinfer1::Dims2 acCacheDims{1, acDecCacheSize_};
            streamingAcDecoder_.setInputShape("cache_in", acCacheDims);
            streamingAcDecoder_.setTensorAddress("latent", loopSingleLatGpu_.data());
            streamingAcDecoder_.setTensorAddress("cache_in", acDecCacheGpu_.data());
            streamingAcDecoder_.setTensorAddress("audio", loopSingleAudioGpu_.data());
            streamingAcDecoder_.setTensorAddress("cache_out", acDecCacheOutGpu_.data());
            streamingAcDecoder_.enqueueV3(stream_);

            // Swap cache buffers for next iteration (double-buffering)
            std::swap(acDecCacheGpu_, acDecCacheOutGpu_);

            // Save audio chunk for final output (streaming decode produces final audio directly)
            {
                std::vector<float> audioChunk(hopLen);
                cudaStreamSynchronize(stream_);
                loopSingleAudioGpu_.copyToHost(audioChunk.data(), hopLen * sizeof(float));
                audioChunks.push_back(std::move(audioChunk));
            }

            // e3. Streaming semantic encode: audio [1,1,3200] + cache → features [1,1,128] + cache
            // (matches Python's streaming encoder with per-layer SConv1d cache)
            streamingSemEncoder_.setInputShape("audio", nvinfer1::Dims3{1, 1, hopLen});
            nvinfer1::Dims2 semCacheDims{1, semCacheSize_};
            streamingSemEncoder_.setInputShape("cache_in", semCacheDims);
            streamingSemEncoder_.setTensorAddress("audio", loopSingleAudioGpu_.data());
            streamingSemEncoder_.setTensorAddress("cache_in", semCacheGpu_.data());
            streamingSemEncoder_.setTensorAddress("semantic_mean", loopSemOutGpu_.data());
            streamingSemEncoder_.setTensorAddress("cache_out", semCacheOutGpu_.data());
            streamingSemEncoder_.enqueueV3(stream_);

            // Swap cache buffers for next iteration
            std::swap(semCacheGpu_, semCacheOutGpu_);

            if (dbgLog) {
                auto semOutShape = streamingSemEncoder_.getOutputShape("semantic_mean");
                fprintf(dbgLog, "  streaming_sem_encoder: audio=%d -> shape[%d]={",
                        hopLen, semOutShape.nbDims);
                for (int di = 0; di < semOutShape.nbDims; ++di)
                    fprintf(dbgLog, "%s%d", di > 0 ? "," : "", (int)semOutShape.d[di]);
                fprintf(dbgLog, "}\n");
                fflush(dbgLog);
            }

            // e4. semantic_connector: features [1,1,semDim] -> [H]
            // Output from streaming encoder is [1, 1, semDim] (batch, time=1, features)
            // Copy semantic output to staging buffer and convert to fp16 for connector
            cudaStreamSynchronize(stream_);
            cudaMemcpyAsync(loopSemFrameGpu_.data(), loopSemOutGpu_.data(),
                            semDim * sizeof(float), cudaMemcpyDeviceToDevice, stream_);
            GpuOps::floatToHalf(loopSemFrameGpu_.as<float>(), loopSemLatFp16_.as<__half>(), semDim, stream_);

            runConnector(loopSemLatFp16_.as<__half>(), scratchSemanticEmbed_.as<__half>(),
                         semanticConnFc1W_, semanticConnFc1B_, semanticConnNormW_,
                         semanticConnFc2W_, semanticConnFc2B_,
                         semConnInputDim_, semConnOutputDim_);

            // e5. next_embed = acoustic_embed + semantic_embed
            // No blending needed — streaming cache ensures smooth temporal continuity
            GpuOps::vectorAdd(scratchEmbed_.as<__half>(), scratchSemanticEmbed_.as<__half>(),
                              scratchEmbed_.as<__half>(), H, stream_);

            // e6. Convert combined fp16 embedding to fp32 for LM input
            GpuOps::halfToFloat(scratchEmbed_.as<__half>(), scratchEmbedF32_.as<float>(), H, stream_);

            // Diagnostic: log intermediate values
            if (dbgLog) {
                cudaStreamSynchronize(stream_);
                float latNorm = 0;
                for (int i = 0; i < latentDim; ++i) latNorm += latent[i] * latent[i];
                latNorm = sqrtf(latNorm);
                float dlNorm = 0;
                for (int i = 0; i < latentDim; ++i) dlNorm += denormLatent[i] * denormLatent[i];
                dlNorm = sqrtf(dlNorm);

                // Check combined embedding for NaN (now fp32)
                std::vector<float> acEmb(H);
                cudaStreamSynchronize(stream_);
                scratchEmbedF32_.copyToHost(acEmb.data(), H * sizeof(float));
                float acNorm = 0;
                int acNan = 0;
                for (int i = 0; i < H; ++i) {
                    if (std::isnan(acEmb[i]) || std::isinf(acEmb[i])) acNan++;
                    else acNorm += acEmb[i] * acEmb[i];
                }
                acNorm = sqrtf(acNorm);

                // Single-frame audio stats
                const auto& lastChunk = audioChunks.back();
                float sfMin = 1e30f, sfMax = -1e30f;
                for (int i = 0; i < hopLen; ++i) {
                    if (lastChunk[i] < sfMin) sfMin = lastChunk[i];
                    if (lastChunk[i] > sfMax) sfMax = lastChunk[i];
                }

                fprintf(dbgLog, "  diag[%d]: latNorm=%.4f dlNorm=%.4f acEmbNorm=%.4f acEmbNaN=%d frameAudio[%.4f,%.4f]\n",
                        totalSpeechTokens, latNorm, dlNorm, acNorm, acNan, sfMin, sfMax);
                fflush(dbgLog);
            }

            // f. Feed next_embed (fp32) to negative LM first (discard logits), then positive
            // Sync between neg/pos: ensures TRT context, positionIdsBuf_, and
            // decodeMaskBuf_ are not reused while still in flight
            runLmDecodeWithLogits(lmDecode_, negLmKV_,
                                  scratchEmbedF32_.as<float>(), scratchHidden2_.as<float>(),
                                  loopNegLogitsF32_.as<float>(), negPos);
            negPos++;
            cudaStreamSynchronize(stream_);

            // Run positive decode last so its logits stay in stagingLogitsF32_
            runLmDecodeWithLogits(lmDecode_, lmKV_,
                                  scratchEmbedF32_.as<float>(), scratchHidden_.as<float>(),
                                  stagingLogitsF32_.as<float>(), posPos);
            posPos++;

        } else {
            // Shouldn't reach here due to token constraints, but handle gracefully
            LOG_WARN("TTS", "unexpected token %d at step %d", nextToken, totalSpeechTokens);
            finished = true;
        }

        // Download positive logits for next iteration
        cudaStreamSynchronize(stream_);
        stagingLogitsF32_.copyToHost(lastLogits.data(), vocabSize * sizeof(float));
        if (dbgLog) {
            fprintf(dbgLog, "  post-decode logits: diff=%.6f end=%.6f eos=%.6f\n",
                    lastLogits[specialTokens_.speech_diffusion],
                    lastLogits[specialTokens_.speech_end],
                    lastLogits[specialTokens_.eos]);
            fflush(dbgLog);
        }
    }

    if (dbgLog) {
        fprintf(dbgLog, "generation done: %d latent frames\n", totalSpeechTokens);
        fclose(dbgLog);
        dbgLog = nullptr;
    }

    // ── 12. Concatenate streaming audio chunks ──
    // (Streaming decode already produced final audio during generation loop)
    int N = totalSpeechTokens;
    LOG_INFO("TTS", "concatenating %d streaming audio chunks (%d samples total)",
             N, N * hopLen);

    if (N > 0) {
        // DIAG: Log denormalized latent stats
        {
            float dMin = 1e30f, dMax = -1e30f, dSum = 0;
            for (int i = 0; i < N * 64; ++i) {
                if (allDenormLatents[i] < dMin) dMin = allDenormLatents[i];
                if (allDenormLatents[i] > dMax) dMax = allDenormLatents[i];
                dSum += allDenormLatents[i];
            }
            float dNorm0 = 0;
            for (int d = 0; d < 64; ++d) dNorm0 += allDenormLatents[d] * allDenormLatents[d];
            dNorm0 = sqrtf(dNorm0);
            LOG_INFO("TTS", "DIAG denorm latent: min=%.4f max=%.4f avg=%.4f frame0_norm=%.4f",
                     dMin, dMax, dSum / (N * 64), dNorm0);
        }

        // Concatenate all streaming audio chunks
        result.audio.reserve(N * hopLen);
        for (const auto& chunk : audioChunks) {
            result.audio.insert(result.audio.end(), chunk.begin(), chunk.end());
        }

        float aMin = *std::min_element(result.audio.begin(), result.audio.end());
        float aMax = *std::max_element(result.audio.begin(), result.audio.end());
        LOG_INFO("TTS", "streaming decoded audio [%d samples]: min=%.4f max=%.4f",
                 (int)result.audio.size(), aMin, aMax);
    }

    result.ok = !result.audio.empty();
    if (!result.ok) {
        result.error = "Generated empty audio";
    }
    return result;
}

// ── Diffusion sampling with CFG ──

std::vector<float> TTSPipeline::sampleSpeechTokens(
    const float* posCondF32Gpu, const float* negCondF32Gpu,
    int hiddenSize, float cfgScale, std::mt19937& rng)
{
    int latentDim = meta_.diffusion.latent_size;  // 64

    // 1. Generate random noise (fp32, CPU) using per-synthesis RNG
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> sample(latentDim);
    for (int i = 0; i < latentDim; ++i) {
        sample[i] = dist(rng);
    }

    // 2. Set up DPM solver
    dpm_.setTimesteps(cfg_.inferenceSteps);

    // 3. Prepare GPU condition buffer in fp32: [pos_cond, neg_cond] = [2, H]
    // Hidden states are already fp32 — copy directly
    cudaMemcpyAsync(stagingCondF32_.as<float>(), posCondF32Gpu,
                    hiddenSize * sizeof(float), cudaMemcpyDeviceToDevice, stream_);
    cudaMemcpyAsync(stagingCondF32_.as<float>() + hiddenSize, negCondF32Gpu,
                    hiddenSize * sizeof(float), cudaMemcpyDeviceToDevice, stream_);

    // 4. Diffusion loop
    std::vector<float> vpredF32(2 * latentDim);
    std::vector<float> prevSample(latentDim);

    for (int step = 0; step < dpm_.numSteps(); ++step) {
        int t = dpm_.timestep(step);

        // Upload sample duplicated for batch=2, directly as fp32 (engine expects fp32)
        std::vector<float> sampleDup(2 * latentDim);
        std::copy(sample.begin(), sample.end(), sampleDup.begin());
        std::copy(sample.begin(), sample.end(), sampleDup.begin() + latentDim);

        scratchSampleF32_.copyFromHostAsync(sampleDup.data(), sampleDup.size() * sizeof(float), stream_);

        // Timesteps: [t, t] int32 (TRT engine expects int32, not int64)
        int32_t timesteps[2] = {(int32_t)t, (int32_t)t};
        scratchTimesteps_.copyFromHostAsync(timesteps, 2 * sizeof(int32_t), stream_);

        // Run diffusion head: B=2 (all I/O in fp32 except timesteps=int32)
        diffusionHead_.setInputShape("noisy_images", nvinfer1::Dims2{2, latentDim});
        nvinfer1::Dims tsDims; tsDims.nbDims = 1; tsDims.d[0] = 2;
        diffusionHead_.setInputShape("timesteps", tsDims);
        diffusionHead_.setInputShape("condition", nvinfer1::Dims2{2, hiddenSize});
        diffusionHead_.setTensorAddress("noisy_images", scratchSampleF32_.data());
        diffusionHead_.setTensorAddress("timesteps", scratchTimesteps_.data());
        diffusionHead_.setTensorAddress("condition", stagingCondF32_.data());
        diffusionHead_.setTensorAddress("v_prediction", scratchVpredF32_.data());
        diffusionHead_.enqueueV3(stream_);

        // Download v_prediction [2, 64] fp32 directly (engine outputs fp32)
        cudaStreamSynchronize(stream_);
        scratchVpredF32_.copyToHost(vpredF32.data(), vpredF32.size() * sizeof(float));

        // CFG blend: blended = uncond + scale * (cond - uncond)
        std::vector<float> blended(latentDim);
        for (int i = 0; i < latentDim; ++i) {
            float cond_v = vpredF32[i];                    // first batch = positive
            float uncond_v = vpredF32[latentDim + i];      // second batch = negative
            blended[i] = uncond_v + cfgScale * (cond_v - uncond_v);
        }

        // DPM solver step
        dpm_.step(blended.data(), step, sample.data(), prevSample.data(), latentDim);
        sample = prevSample;
    }

    return sample;
}

// ── Connector: latent -> embedding ──

void TTSPipeline::runConnector(const __half* latentGpu, __half* outputGpu,
                               const CudaBuffer& fc1W, const CudaBuffer& fc1B,
                               const CudaBuffer& normW,
                               const CudaBuffer& fc2W, const CudaBuffer& fc2B,
                               int inputDim, int outputDim) {
    // fc1: [inputDim] -> [outputDim]
    GpuOps::linearForward(cublas_, latentGpu,
                          fc1W.as<__half>(), fc1B.as<__half>(),
                          connScratch_.as<__half>(), outputDim, inputDim, stream_);

    // RMSNorm
    GpuOps::rmsNorm(connScratch_.as<__half>(), normW.as<__half>(),
                     connScratch_.as<__half>(), outputDim,
                     (float)meta_.rms_norm_eps, stream_);

    // fc2: [outputDim] -> [outputDim]
    GpuOps::linearForward(cublas_, connScratch_.as<__half>(),
                          fc2W.as<__half>(), fc2B.as<__half>(),
                          outputGpu, outputDim, outputDim, stream_);
}

// ── EOS Classifier ──

float TTSPipeline::runEosClassifier(const __half* hiddenGpu) {
    int H = eosHiddenSize_;

    // fc1 + ReLU (reuse member buffers)
    scratchEosHidden_.resize(H * sizeof(uint16_t));
    GpuOps::linearForward(cublas_, hiddenGpu,
                          eosFc1W_.as<__half>(), eosFc1B_.as<__half>(),
                          scratchEosHidden_.as<__half>(), H, H, stream_);
    GpuOps::relu(scratchEosHidden_.as<__half>(), scratchEosHidden_.as<__half>(), H, stream_);

    // fc2 + sigmoid -> scalar
    scratchEosOut_.resize(1 * sizeof(uint16_t));
    GpuOps::linearForward(cublas_, scratchEosHidden_.as<__half>(),
                          eosFc2W_.as<__half>(), eosFc2B_.as<__half>(),
                          scratchEosOut_.as<__half>(), 1, H, stream_);

    scratchEosSigmoid_.resize(sizeof(float));
    GpuOps::sigmoid(scratchEosOut_.as<__half>(), scratchEosSigmoid_.as<float>(), 1, stream_);

    float score = 0.0f;
    cudaStreamSynchronize(stream_);
    scratchEosSigmoid_.copyToHost(&score, sizeof(float));
    return score;
}

// ── KV-cache binding helper ──

void TTSPipeline::bindKVCache(TRTEngine& engine, KVCache& kv, bool prefillMode) {
    int nLayers = kv.numLayers();
    int seqLen = kv.seqLen();

    for (int i = 0; i < nLayers; ++i) {
        std::string presKeyName = "present_key_" + std::to_string(i);
        std::string presValName = "present_value_" + std::to_string(i);

        if (!prefillMode) {
            // Decode: bind past KV as inputs + present KV as outputs
            std::string pastKeyName = "past_key_" + std::to_string(i);
            std::string pastValName = "past_value_" + std::to_string(i);

            nvinfer1::Dims4 pastShape{1, kv.numKVHeads(), seqLen, kv.headDim()};
            engine.setInputShape(pastKeyName, pastShape);
            engine.setInputShape(pastValName, pastShape);

            engine.setTensorAddress(pastKeyName, kv.pastKeyPtr(i));
            engine.setTensorAddress(pastValName, kv.pastValuePtr(i));
        }

        // Both prefill and decode: bind present KV as outputs
        engine.setTensorAddress(presKeyName, kv.presentKeyPtr(i));
        engine.setTensorAddress(presValName, kv.presentValuePtr(i));
    }
}
