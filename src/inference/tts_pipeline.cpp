#include "inference/tts_pipeline.h"
#include <algorithm>
#include "utils/logger.h"
#include <cstring>
#include <cmath>
#include <random>
#include <filesystem>

namespace fs = std::filesystem;

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
    } else {
        std::string dir = cfg.engineDir + "/";
        if (!lmPrefill_.loadFromFile(dir + "language_model_prefill.trt")) return false;
        if (!lmDecode_.loadFromFile(dir + "language_model_decode.trt")) return false;
        if (!acousticEncoder_.loadFromFile(dir + "acoustic_encoder.trt")) return false;
        if (!semanticEncoder_.loadFromFile(dir + "semantic_encoder.trt")) return false;
        if (!diffusionHead_.loadFromFile(dir + "diffusion_head.trt")) return false;
        if (!acousticDecoder_.loadFromFile(dir + "acoustic_decoder.trt")) return false;
        LOG_INFO("TTS", "loaded 6 TRT engines (1.5B)");
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
    scratchHidden_.resize(H * sizeof(uint16_t));
    scratchHidden2_.resize(H * sizeof(uint16_t));
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
                std::string name = entry.path().stem().string();
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
                std::string name = entry.path().stem().string();
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
                                __half* embedsFp16Gpu, int seqLen,
                                float* logitsF32Gpu, __half* hiddenFp16Gpu) {
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

    // Set shapes and bind — FP16 ONNX: embeds and hidden are fp16, logits and mask are fp32
    engine.setInputShape("inputs_embeds", nvinfer1::Dims3{1, seqLen, H});
    engine.setInputShape("position_ids", nvinfer1::Dims2{1, seqLen});
    nvinfer1::Dims4 maskShape{1, 1, seqLen, seqLen};
    engine.setInputShape("attention_mask", maskShape);
    engine.setTensorAddress("inputs_embeds", (void*)embedsFp16Gpu);  // fp16
    engine.setTensorAddress("position_ids", posIdsBuf.data());
    engine.setTensorAddress("attention_mask", causalMaskGpu.data());  // fp32
    engine.setTensorAddress("logits", logitsF32Gpu);                  // fp32
    engine.setTensorAddress("hidden_states", (void*)hiddenFp16Gpu);  // fp16

    // Bind KV cache (prefill mode: only present outputs, no past inputs)
    bindKVCache(engine, kv, /*prefillMode=*/true);

    engine.enqueueV3(stream_);

    // Advance KV cache after prefill: swap buffers once so present→past
    kv.advanceAfterPrefill(seqLen);
}

// ── 1.5B LM Decode with logits helper (FP16 ONNX engine) ──

void TTSPipeline::runLmDecodeWithLogits(TRTEngine& engine, KVCache& kv,
                                          const __half* inputGpu, __half* hiddenOutGpu,
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

    // Configure engine — FP16 ONNX: embeds and hidden are fp16, logits and mask are fp32
    engine.setInputShape("inputs_embeds", nvinfer1::Dims3{1, 1, H});
    engine.setInputShape("position_ids", nvinfer1::Dims2{1, 1});
    nvinfer1::Dims4 maskShape{1, 1, 1, totalSeq};
    engine.setInputShape("attention_mask", maskShape);
    engine.setTensorAddress("inputs_embeds", (void*)inputGpu);  // fp16 direct
    engine.setTensorAddress("position_ids", positionIdsBuf_.data());
    engine.setTensorAddress("attention_mask", decodeMaskBuf_.data());
    engine.setTensorAddress("logits", logitsF32Gpu);             // fp32
    engine.setTensorAddress("hidden_states", (void*)hiddenOutGpu); // fp16 direct
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

    bool finished = false;
    int textOffset = 0;
    int textWindowSize = 5;
    int speechWindowSize = 6;
    int totalSpeechTokens = 0;

    while (!finished && (textOffset < (int)textIds.size() || !finished)) {
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
            std::vector<float> latent = sampleSpeechTokens(
                scratchHidden_.as<__half>(), scratchHidden2_.as<__half>(),
                H, cfg_.cfgScale);

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
        // continue generating speech tokens until EOS or max length
        if (textOffset >= (int)textIds.size() && !finished) {
            int maxSamples = meta_.sample_rate * 30;
            int hopLen = meta_.hop_length;
            if (totalSpeechTokens * hopLen > maxSamples) {
                finished = true;
            }
        }
    }

    // === Batch Acoustic Decoding ===
    // Decode all latents at once: [1, 64, N] -> [1, 1, N * hop_length]
    // This allows convolutional layers to see the full temporal context.
    int N = totalSpeechTokens;
    int hopLen = meta_.hop_length;
    LOG_INFO("TTS", "batch decoding %d latent frames -> %d audio samples", N, N * hopLen);

    if (N > 0) {
        // Transpose from [N, 64] (row-major) to [64, N] for decoder input [1, 64, N]
        std::vector<float> batchLatent(64 * N);
        for (int t = 0; t < N; ++t) {
            for (int d = 0; d < 64; ++d) {
                batchLatent[d * N + t] = allDenormLatents[t * 64 + d];
            }
        }

        // Upload to GPU
        CudaBuffer batchLatentGpu;
        batchLatentGpu.resize(64 * N * sizeof(float));
        batchLatentGpu.copyFromHostAsync(batchLatent.data(), 64 * N * sizeof(float), stream_);

        // Allocate output buffer
        CudaBuffer batchAudioGpu;
        batchAudioGpu.resize(1 * 1 * N * hopLen * sizeof(float));

        // Run acoustic decoder with batch shape
        acousticDecoder_.setInputShape("latent", nvinfer1::Dims3{1, 64, N});
        acousticDecoder_.setTensorAddress("latent", batchLatentGpu.data());
        acousticDecoder_.setTensorAddress("audio", batchAudioGpu.data());
        acousticDecoder_.enqueueV3(stream_);

        // Download all audio
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

    // ── 8. Init KV caches (pos + neg) — FP16 for 1.5B FP16 ONNX engines ──
    int maxSeqLen = 4096;
    if (!lmKV_.init(numLmLayers, numKvHeads, headDim, maxSeqLen, /*useFp16=*/true)) {
        result.error = "Failed to init positive LM KV cache";
        return result;
    }
    if (!negLmKV_.init(numLmLayers, numKvHeads, headDim, maxSeqLen, /*useFp16=*/true)) {
        result.error = "Failed to init negative LM KV cache";
        return result;
    }

    // ── 9. Prefill positive path (FP16 embeds, FP16 hidden, FP32 logits) ──
    // Upload fp32 embeddings to GPU, then convert to fp16
    CudaBuffer prefillEmbedsF32Gpu;
    prefillEmbedsF32Gpu.resize((size_t)promptLen * H * sizeof(float));
    prefillEmbedsF32Gpu.copyFromHost(promptEmbeds.data(), (size_t)promptLen * H * sizeof(float));
    CudaBuffer prefillEmbedsFp16Gpu;
    prefillEmbedsFp16Gpu.resize((size_t)promptLen * H * sizeof(uint16_t));
    GpuOps::floatToHalf(prefillEmbedsF32Gpu.as<float>(), prefillEmbedsFp16Gpu.as<__half>(),
                         promptLen * H, stream_);
    prefillEmbedsF32Gpu.free();

    CudaBuffer prefillLogitsGpu;
    prefillLogitsGpu.resize((size_t)promptLen * vocabSize * sizeof(float));
    CudaBuffer prefillHiddenFp16Gpu;
    prefillHiddenFp16Gpu.resize((size_t)promptLen * H * sizeof(uint16_t));

    runLmPrefill(lmPrefill_, lmKV_,
                 prefillEmbedsFp16Gpu.as<__half>(), promptLen,
                 prefillLogitsGpu.as<float>(), prefillHiddenFp16Gpu.as<__half>());
    cudaStreamSynchronize(stream_);
    LOG_INFO("TTS", "positive prefill done (seq=%d)", promptLen);

    // Get last-position logits (fp32) from prefill
    std::vector<float> lastLogits(vocabSize);
    cudaMemcpy(lastLogits.data(),
               (float*)prefillLogitsGpu.data() + (size_t)(promptLen - 1) * vocabSize,
               vocabSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Extract last-position hidden (fp16) directly to scratchHidden_ on GPU
    cudaMemcpy(scratchHidden_.data(),
               (uint8_t*)prefillHiddenFp16Gpu.data() + (size_t)(promptLen - 1) * H * sizeof(uint16_t),
               H * sizeof(uint16_t), cudaMemcpyDeviceToDevice);

    prefillEmbedsFp16Gpu.free();
    prefillLogitsGpu.free();
    prefillHiddenFp16Gpu.free();

    // ── 10. Prefill negative path (FP16 embeds, FP16 hidden, FP32 logits) ──
    CudaBuffer negEmbedsF32Gpu;
    negEmbedsF32Gpu.resize(1 * H * sizeof(float));
    negEmbedsF32Gpu.copyFromHost(negPromptEmbeds.data(), 1 * H * sizeof(float));
    CudaBuffer negEmbedsFp16Gpu;
    negEmbedsFp16Gpu.resize(1 * H * sizeof(uint16_t));
    GpuOps::floatToHalf(negEmbedsF32Gpu.as<float>(), negEmbedsFp16Gpu.as<__half>(), H, stream_);
    negEmbedsF32Gpu.free();

    CudaBuffer negPrefillLogitsGpu;
    negPrefillLogitsGpu.resize(1 * vocabSize * sizeof(float));
    CudaBuffer negPrefillHiddenFp16Gpu;
    negPrefillHiddenFp16Gpu.resize(1 * H * sizeof(uint16_t));

    runLmPrefill(lmPrefill_, negLmKV_,
                 negEmbedsFp16Gpu.as<__half>(), 1,
                 negPrefillLogitsGpu.as<float>(), negPrefillHiddenFp16Gpu.as<__half>());
    cudaStreamSynchronize(stream_);
    LOG_INFO("TTS", "negative prefill done (seq=1)");

    // ── 11. Autoregressive generation loop ──
    int64_t posPos = (int64_t)promptLen;
    int64_t negPos = 1;

    std::vector<float> allDenormLatents; // collected for batch decode
    int totalSpeechTokens = 0;
    int maxSpeechTokens = (meta_.sample_rate * 30) / hopLen; // 30 seconds max
    bool finished = false;

    // Positive hidden is already in scratchHidden_ (extracted from prefill above)

    // Extract negative hidden (fp16) directly to scratchHidden2_ on GPU
    cudaMemcpy(scratchHidden2_.data(), negPrefillHiddenFp16Gpu.data(),
               H * sizeof(uint16_t), cudaMemcpyDeviceToDevice);

    // Free negative prefill buffers
    negEmbedsFp16Gpu.free();
    negPrefillLogitsGpu.free();
    negPrefillHiddenFp16Gpu.free();

    // Allocate separate logits buffer for negative path (we discard neg logits)
    CudaBuffer negLogitsF32;
    negLogitsF32.resize(vocabSize * sizeof(float));

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
            // Get embedding fp16 directly on GPU via embeddingLookup
            int32_t tokId = specialTokens_.speech_start;
            inputIdBuf_.copyFromHost(&tokId, sizeof(int32_t));
            GpuOps::embeddingLookup(embedTokensGpu_.as<__half>(),
                                    inputIdBuf_.as<int32_t>(), scratchEmbed_.as<__half>(),
                                    1, H, stream_);

            runLmDecodeWithLogits(lmDecode_, negLmKV_,
                                  scratchEmbed_.as<__half>(), scratchHidden2_.as<__half>(),
                                  negLogitsF32.as<float>(), negPos);
            negPos++;
            // Sync: ensure negative decode completes before reusing shared
            // TRT context, positionIdsBuf_, and decodeMaskBuf_
            cudaStreamSynchronize(stream_);
            runLmDecodeWithLogits(lmDecode_, lmKV_,
                                  scratchEmbed_.as<__half>(), scratchHidden_.as<__half>(),
                                  stagingLogitsF32_.as<float>(), posPos);
            posPos++;

            cudaStreamSynchronize(stream_);
            stagingLogitsF32_.copyToHost(lastLogits.data(), vocabSize * sizeof(float));
            continue;
        }

        if (nextToken == specialTokens_.speech_diffusion) {
            // c. Diffusion sampling using positive and negative hidden states
            std::vector<float> latent = sampleSpeechTokens(
                scratchHidden_.as<__half>(), scratchHidden2_.as<__half>(),
                H, cfg_.cfgScale);

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

            // e2. Single-frame decode for semantic feedback
            // (Python reference decodes frame-by-frame with streaming cache)
            CudaBuffer singleLatGpu;
            singleLatGpu.resize(latentDim * sizeof(float));
            singleLatGpu.copyFromHost(denormLatent.data(), latentDim * sizeof(float));

            int singleAudioLen = hopLen; // 3200 samples for 1 frame
            CudaBuffer singleAudioGpu;
            singleAudioGpu.resize(singleAudioLen * sizeof(float));

            acousticDecoder_.setInputShape("latent", nvinfer1::Dims3{1, latentDim, 1});
            acousticDecoder_.setTensorAddress("latent", singleLatGpu.data());
            acousticDecoder_.setTensorAddress("audio", singleAudioGpu.data());
            acousticDecoder_.enqueueV3(stream_);

            // e3. Run semantic encoder on decoded audio (padded to TRT minimum)
            int semAudioLen = 24000; // TRT min input size
            int semTMax = semAudioLen / hopLen + 2;

            CudaBuffer semAudioPadGpu;
            semAudioPadGpu.resize((size_t)semAudioLen * sizeof(float));
            cudaMemsetAsync(semAudioPadGpu.data(), 0, semAudioLen * sizeof(float), stream_);
            cudaMemcpyAsync(semAudioPadGpu.data(), singleAudioGpu.data(),
                            (size_t)singleAudioLen * sizeof(float), cudaMemcpyDeviceToDevice, stream_);

            CudaBuffer semOutGpu;
            semOutGpu.resize((size_t)semDim * semTMax * sizeof(float));
            cudaMemsetAsync(semOutGpu.data(), 0, (size_t)semDim * semTMax * sizeof(float), stream_);

            semanticEncoder_.setInputShape("audio", nvinfer1::Dims3{1, 1, semAudioLen});
            semanticEncoder_.setTensorAddress("audio", semAudioPadGpu.data());
            semanticEncoder_.setTensorAddress("semantic_mean", semOutGpu.data());
            semanticEncoder_.enqueueV3(stream_);

            // e4. semantic_connector: take first real frame [semDim] -> [H]
            // Output is [1, vae_dim, T] (channel-first); use frame 0 (covers real audio)
            auto semOutShape = semanticEncoder_.getOutputShape("semantic_mean");
            int actualSemFrames = (semOutShape.nbDims >= 3) ? (int)semOutShape.d[2] : 1;
            int semFrameIdx = 0; // first frame covers real audio (rest is zero-padded)
            if (dbgLog) {
                fprintf(dbgLog, "  sem_encoder: audio=%d -> shape[%d]={",
                        semAudioLen, semOutShape.nbDims);
                for (int di = 0; di < semOutShape.nbDims; ++di)
                    fprintf(dbgLog, "%s%d", di > 0 ? "," : "", (int)semOutShape.d[di]);
                fprintf(dbgLog, "} T=%d (frame=%d)\n", actualSemFrames, semFrameIdx);
                fflush(dbgLog);
            }

            // Extract frame from [1, semDim, T] layout (channel-first)
            int T_sem = actualSemFrames;
            std::vector<float> semFrameF32(semDim);
            {
                std::vector<float> semAllCpu(semDim * T_sem);
                cudaStreamSynchronize(stream_);
                semOutGpu.copyToHost(semAllCpu.data(), semDim * T_sem * sizeof(float));
                for (int c = 0; c < semDim; ++c) {
                    semFrameF32[c] = semAllCpu[c * T_sem + semFrameIdx];
                }
            }

            CudaBuffer semLatFp16;
            semLatFp16.resize(semDim * sizeof(uint16_t));
            CudaBuffer semFrameGpu;
            semFrameGpu.resize(semDim * sizeof(float));
            semFrameGpu.copyFromHost(semFrameF32.data(), semDim * sizeof(float));
            GpuOps::floatToHalf(semFrameGpu.as<float>(), semLatFp16.as<__half>(), semDim, stream_);

            runConnector(semLatFp16.as<__half>(), scratchSemanticEmbed_.as<__half>(),
                         semanticConnFc1W_, semanticConnFc1B_, semanticConnNormW_,
                         semanticConnFc2W_, semanticConnFc2B_,
                         semConnInputDim_, semConnOutputDim_);

            // e5. next_embed = acoustic_embed + semantic_embed
            GpuOps::vectorAdd(scratchEmbed_.as<__half>(), scratchSemanticEmbed_.as<__half>(),
                              scratchEmbed_.as<__half>(), H, stream_);

            // Diagnostic: log intermediate values
            if (dbgLog) {
                cudaStreamSynchronize(stream_);
                float latNorm = 0;
                for (int i = 0; i < latentDim; ++i) latNorm += latent[i] * latent[i];
                latNorm = sqrtf(latNorm);
                float dlNorm = 0;
                for (int i = 0; i < latentDim; ++i) dlNorm += denormLatent[i] * denormLatent[i];
                dlNorm = sqrtf(dlNorm);

                // Check combined embedding for NaN
                std::vector<float> acEmb(H);
                GpuOps::halfToFloat(scratchEmbed_.as<__half>(), stagingEmbedsF32_.as<float>(), H, stream_);
                cudaStreamSynchronize(stream_);
                stagingEmbedsF32_.copyToHost(acEmb.data(), H * sizeof(float));
                float acNorm = 0;
                int acNan = 0;
                for (int i = 0; i < H; ++i) {
                    if (std::isnan(acEmb[i]) || std::isinf(acEmb[i])) acNan++;
                    else acNorm += acEmb[i] * acEmb[i];
                }
                acNorm = sqrtf(acNorm);

                // Single-frame audio stats
                std::vector<float> sfAudioCpu(singleAudioLen);
                singleAudioGpu.copyToHost(sfAudioCpu.data(), singleAudioLen * sizeof(float));
                float sfMin = 1e30f, sfMax = -1e30f;
                for (int i = 0; i < singleAudioLen; ++i) {
                    if (sfAudioCpu[i] < sfMin) sfMin = sfAudioCpu[i];
                    if (sfAudioCpu[i] > sfMax) sfMax = sfAudioCpu[i];
                }

                fprintf(dbgLog, "  diag[%d]: latNorm=%.4f dlNorm=%.4f acEmbNorm=%.4f acEmbNaN=%d frameAudio[%.4f,%.4f]\n",
                        totalSpeechTokens, latNorm, dlNorm, acNorm, acNan, sfMin, sfMax);
                fflush(dbgLog);
            }

            // f. Feed next_embed to negative LM first (discard logits), then positive
            // Sync between neg/pos: ensures TRT context, positionIdsBuf_, and
            // decodeMaskBuf_ are not reused while still in flight
            runLmDecodeWithLogits(lmDecode_, negLmKV_,
                                  scratchEmbed_.as<__half>(), scratchHidden2_.as<__half>(),
                                  negLogitsF32.as<float>(), negPos);
            negPos++;
            cudaStreamSynchronize(stream_);

            // Run positive decode last so its logits stay in stagingLogitsF32_
            runLmDecodeWithLogits(lmDecode_, lmKV_,
                                  scratchEmbed_.as<__half>(), scratchHidden_.as<__half>(),
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

    // ── 12. Batch acoustic decode ──
    int N = totalSpeechTokens;
    LOG_INFO("TTS", "batch decoding %d latent frames -> %d audio samples", N, N * hopLen);

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

        // Transpose from [N, 64] to [64, N]
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

// ── Diffusion sampling with CFG ──

std::vector<float> TTSPipeline::sampleSpeechTokens(
    const __half* posCondGpu, const __half* negCondGpu,
    int hiddenSize, float cfgScale)
{
    int latentDim = meta_.diffusion.latent_size;  // 64

    // 1. Generate random noise (fp32, CPU)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> sample(latentDim);
    for (int i = 0; i < latentDim; ++i) {
        sample[i] = dist(gen);
    }

    // 2. Set up DPM solver
    dpm_.setTimesteps(cfg_.inferenceSteps);

    // 3. Prepare GPU condition buffer in fp32: [pos_cond, neg_cond] = [2, H]
    // Convert fp16 hidden states -> fp32 for TRT engine
    GpuOps::halfToFloat(posCondGpu, stagingCondF32_.as<float>(), hiddenSize, stream_);
    GpuOps::halfToFloat(negCondGpu, stagingCondF32_.as<float>() + hiddenSize, hiddenSize, stream_);

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
