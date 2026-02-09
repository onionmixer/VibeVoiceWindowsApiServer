#include "inference/tts_pipeline.h"
#include <algorithm>
#include "utils/logger.h"
#include <cstring>
#include <cmath>
#include <random>
#include <filesystem>

namespace fs = std::filesystem;

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
        if (!diffusionHead_.loadFromFile(dir + "diffusion_head.trt")) return false;
        if (!acousticDecoder_.loadFromFile(dir + "acoustic_decoder.trt")) return false;
        LOG_INFO("TTS", "loaded 5 TRT engines (1.5B)");
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

    scratchEmbed_.resize(H * sizeof(uint16_t));
    scratchHidden_.resize(H * sizeof(uint16_t));
    scratchHidden2_.resize(H * sizeof(uint16_t));
    scratchLatent_.resize(latentDim * sizeof(uint16_t));
    scratchLatentF32_.resize(latentDim * sizeof(float));
    scratchAudio_.resize(1 * 1 * 32000 * sizeof(uint16_t));  // ~10 chunks
    scratchCondition_.resize(2 * H * sizeof(uint16_t));
    scratchNoisy_.resize(2 * latentDim * sizeof(uint16_t));
    scratchVpred_.resize(2 * latentDim * sizeof(uint16_t));
    scratchTimesteps_.resize(2 * sizeof(int64_t));
    positionIdsBuf_.resize(sizeof(int64_t));
    inputIdBuf_.resize(sizeof(int32_t));
    connScratch_.resize(H * sizeof(uint16_t));

    // Per-request scratch buffers (pre-allocate to avoid cudaMalloc per request)
    int hopLen = meta_.hop_length > 0 ? meta_.hop_length : 3200;
    scratchDenormLatent_.resize(latentDim * sizeof(uint16_t));
    scratchDecoderInput_.resize(64 * sizeof(uint16_t));
    scratchAudioChunk_.resize(hopLen * sizeof(uint16_t));
    scratchAudioF32_.resize(hopLen * sizeof(float));
    scratchSampleF32_.resize(2 * latentDim * sizeof(float));
    scratchVpredF32_.resize(2 * latentDim * sizeof(float));
    if (eosHiddenSize_ > 0) {
        scratchEosHidden_.resize(eosHiddenSize_ * sizeof(uint16_t));
        scratchEosOut_.resize(1 * sizeof(uint16_t));
        scratchEosSigmoid_.resize(sizeof(float));
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

    // 1. Tokenize
    auto textIds = tokenizer_.encode(req.text);
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
    int headDim = meta_.head_dim;
    int latentDim = meta_.diffusion.latent_size;
    int pos = (int)vp.groups[0].seq_len;  // starting position after voice context

    // Get tts_input_types pointers
    // type 0 = speech, type 1 = text
    const __half* ttsType0 = ttsInputTypesGpu_.as<__half>();             // speech
    const __half* ttsType1 = ttsInputTypesGpu_.as<__half>() + H;        // text

    std::vector<float> allAudio;
    bool finished = false;
    int textOffset = 0;
    int textWindowSize = 5;
    int speechWindowSize = 6;

    while (!finished && (textOffset < (int)textIds.size() || !finished)) {
        // === Text Window ===
        int windowEnd = std::min(textOffset + textWindowSize, (int)textIds.size());
        for (int ti = textOffset; ti < windowEnd; ++ti) {
            int32_t tokenId = textIds[ti];

            // a. Embedding lookup
            GpuOps::embeddingLookup(embedTokensGpu_.as<__half>(),
                                    &tokenId, scratchEmbed_.as<__half>(),
                                    1, H, stream_);
            // Need tokenId on GPU
            inputIdBuf_.copyFromHost(&tokenId, sizeof(int32_t));
            GpuOps::embeddingLookup(embedTokensGpu_.as<__half>(),
                                    inputIdBuf_.as<int32_t>(), scratchEmbed_.as<__half>(),
                                    1, H, stream_);

            // b. Set position
            int64_t posVal = pos;
            positionIdsBuf_.copyFromHost(&posVal, sizeof(int64_t));

            // Run base_lm_decode for positive and negative paths
            // Positive path
            baseLmDecode_.setInputShape("inputs_embeds", nvinfer1::Dims3{1, 1, H});
            baseLmDecode_.setInputShape("position_ids", nvinfer1::Dims2{1, 1});
            baseLmDecode_.setTensorAddress("inputs_embeds", scratchEmbed_.data());
            baseLmDecode_.setTensorAddress("position_ids", positionIdsBuf_.data());
            baseLmDecode_.setTensorAddress("hidden_states", scratchHidden_.data());
            bindKVCache(baseLmDecode_, baseLmKV_);
            baseLmDecode_.enqueueV3(stream_);
            baseLmKV_.advance();

            // Negative path
            baseLmDecode_.setTensorAddress("hidden_states", scratchHidden2_.data());
            bindKVCache(baseLmDecode_, negBaseLmKV_);
            baseLmDecode_.enqueueV3(stream_);
            negBaseLmKV_.advance();

            // c. tts_input = hidden + ttsInputTypes[1] (text type)
            // Positive: scratchHidden_ += ttsType1
            GpuOps::vectorAdd(scratchHidden_.as<__half>(), ttsType1,
                              scratchHidden_.as<__half>(), H, stream_);
            // Negative: scratchHidden2_ += ttsType1
            GpuOps::vectorAdd(scratchHidden2_.as<__half>(), ttsType1,
                              scratchHidden2_.as<__half>(), H, stream_);

            // d. Run tts_lm_decode
            ttsLmDecode_.setInputShape("inputs_embeds", nvinfer1::Dims3{1, 1, H});
            ttsLmDecode_.setInputShape("position_ids", nvinfer1::Dims2{1, 1});
            ttsLmDecode_.setTensorAddress("inputs_embeds", scratchHidden_.data());
            ttsLmDecode_.setTensorAddress("position_ids", positionIdsBuf_.data());
            ttsLmDecode_.setTensorAddress("hidden_states", scratchHidden_.data());
            bindKVCache(ttsLmDecode_, ttsLmKV_);
            ttsLmDecode_.enqueueV3(stream_);
            ttsLmKV_.advance();

            // Negative tts_lm
            ttsLmDecode_.setTensorAddress("inputs_embeds", scratchHidden2_.data());
            ttsLmDecode_.setTensorAddress("hidden_states", scratchHidden2_.data());
            bindKVCache(ttsLmDecode_, negTtsLmKV_);
            ttsLmDecode_.enqueueV3(stream_);
            negTtsLmKV_.advance();

            pos++;
        }
        textOffset = windowEnd;

        // === Speech Window ===
        for (int si = 0; si < speechWindowSize; ++si) {
            // a. pos_condition = tts_hidden (positive), neg_condition = tts_hidden (negative)
            // scratchHidden_ has positive, scratchHidden2_ has negative

            // b. Sample speech token via diffusion
            std::vector<float> latent = sampleSpeechTokens(
                scratchHidden_.as<__half>(), scratchHidden2_.as<__half>(),
                H, cfg_.cfgScale);

            // c. Denormalize: latent / scaling_factor - bias_factor
            // Upload latent to GPU as fp16
            scratchLatentF32_.copyFromHost(latent.data(), latent.size() * sizeof(float));
            GpuOps::floatToHalf(scratchLatentF32_.as<float>(), scratchLatent_.as<__half>(),
                                latentDim, stream_);
            // scratchLatent_ now has raw latent in fp16

            // Denormalize for acoustic decoder (reuse member buffers)
            scratchDenormLatent_.resize(latentDim * sizeof(uint16_t));
            GpuOps::scaleAndBias(scratchLatent_.as<__half>(), speechScalingFactor_,
                                 speechBiasFactor_, scratchDenormLatent_.as<__half>(),
                                 latentDim, stream_);

            // d. Acoustic decoder: latent [1, 64, 1] -> audio [1, 1, 3200]
            scratchDecoderInput_.resize(64 * 1 * sizeof(uint16_t));
            // The decoder expects [1, 64, T] where T=1 for single token
            cudaMemcpyAsync(scratchDecoderInput_.data(), scratchDenormLatent_.data(),
                            64 * sizeof(uint16_t), cudaMemcpyDeviceToDevice, stream_);

            acousticDecoder_.setInputShape("latent", nvinfer1::Dims3{1, 64, 1});
            acousticDecoder_.setTensorAddress("latent", scratchDecoderInput_.data());
            // Output: audio [1, 1, hop_length]
            int hopLen = meta_.hop_length;  // 3200
            scratchAudioChunk_.resize(1 * 1 * hopLen * sizeof(uint16_t));
            acousticDecoder_.setTensorAddress("audio", scratchAudioChunk_.data());
            acousticDecoder_.enqueueV3(stream_);

            // Convert fp16 -> fp32 (reuse member buffer)
            std::vector<float> audioChunk(hopLen);
            scratchAudioF32_.resize(hopLen * sizeof(float));
            GpuOps::halfToFloat(scratchAudioChunk_.as<__half>(), scratchAudioF32_.as<float>(),
                                hopLen, stream_);
            cudaStreamSynchronize(stream_);
            scratchAudioF32_.copyToHost(audioChunk.data(), hopLen * sizeof(float));

            allAudio.insert(allAudio.end(), audioChunk.begin(), audioChunk.end());

            // f. Run connector: latent -> LM embedding
            runConnector(scratchLatent_.as<__half>(), scratchEmbed_.as<__half>(),
                         connFc1W_, connFc1B_, connNormW_, connFc2W_, connFc2B_,
                         connInputDim_, connOutputDim_);

            // g. Add tts_input_types[0] (speech type)
            GpuOps::vectorAdd(scratchEmbed_.as<__half>(), ttsType0,
                              scratchHidden_.as<__half>(), H, stream_);
            GpuOps::vectorAdd(scratchEmbed_.as<__half>(), ttsType0,
                              scratchHidden2_.as<__half>(), H, stream_);

            // h. Run tts_lm_decode for both paths
            int64_t posVal = pos;
            positionIdsBuf_.copyFromHost(&posVal, sizeof(int64_t));

            ttsLmDecode_.setInputShape("inputs_embeds", nvinfer1::Dims3{1, 1, H});
            ttsLmDecode_.setInputShape("position_ids", nvinfer1::Dims2{1, 1});
            ttsLmDecode_.setTensorAddress("inputs_embeds", scratchHidden_.data());
            ttsLmDecode_.setTensorAddress("position_ids", positionIdsBuf_.data());
            ttsLmDecode_.setTensorAddress("hidden_states", scratchHidden_.data());
            bindKVCache(ttsLmDecode_, ttsLmKV_);
            ttsLmDecode_.enqueueV3(stream_);
            ttsLmKV_.advance();

            ttsLmDecode_.setTensorAddress("inputs_embeds", scratchHidden2_.data());
            ttsLmDecode_.setTensorAddress("hidden_states", scratchHidden2_.data());
            bindKVCache(ttsLmDecode_, negTtsLmKV_);
            ttsLmDecode_.enqueueV3(stream_);
            negTtsLmKV_.advance();

            // i. EOS check
            float eosScore = runEosClassifier(scratchHidden_.as<__half>());
            if (eosScore > 0.5f) {
                finished = true;
                break;
            }

            pos++;
        }

        // If all text has been processed and we haven't gotten EOS,
        // continue generating speech tokens until EOS or max length
        if (textOffset >= (int)textIds.size() && !finished) {
            // Safety limit: max ~30 seconds of audio
            if (allAudio.size() > (size_t)meta_.sample_rate * 30) {
                finished = true;
            }
        }
    }

    result.audio = std::move(allAudio);
    result.ok = !result.audio.empty();
    if (!result.ok) {
        result.error = "Generated empty audio";
    }
    return result;
}

// ── 1.5B Full Generation (deferred - requires export_onnx.py fix + rebuild) ──

TTSPipeline::Result TTSPipeline::synth15B(const Request& req) {
    Result result;
    result.error = "1.5B pipeline not yet implemented (requires ONNX re-export with hidden_states)";
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

    // 3. Prepare GPU condition buffer: [pos_cond, neg_cond] = [2, H]
    cudaMemcpyAsync((uint8_t*)scratchCondition_.data(),
                    posCondGpu, hiddenSize * sizeof(uint16_t),
                    cudaMemcpyDeviceToDevice, stream_);
    cudaMemcpyAsync((uint8_t*)scratchCondition_.data() + hiddenSize * sizeof(uint16_t),
                    negCondGpu, hiddenSize * sizeof(uint16_t),
                    cudaMemcpyDeviceToDevice, stream_);

    // 4. Diffusion loop
    std::vector<float> vpredF32(2 * latentDim);
    std::vector<float> prevSample(latentDim);

    for (int step = 0; step < dpm_.numSteps(); ++step) {
        int t = dpm_.timestep(step);

        // Upload sample to GPU as fp16, duplicated for batch=2
        std::vector<float> sampleDup(2 * latentDim);
        std::copy(sample.begin(), sample.end(), sampleDup.begin());
        std::copy(sample.begin(), sample.end(), sampleDup.begin() + latentDim);

        scratchSampleF32_.resize(2 * latentDim * sizeof(float));
        scratchSampleF32_.copyFromHostAsync(sampleDup.data(), sampleDup.size() * sizeof(float), stream_);
        GpuOps::floatToHalf(scratchSampleF32_.as<float>(), scratchNoisy_.as<__half>(),
                            2 * latentDim, stream_);

        // Timesteps: [t, t] int64
        int64_t timesteps[2] = {(int64_t)t, (int64_t)t};
        scratchTimesteps_.copyFromHostAsync(timesteps, 2 * sizeof(int64_t), stream_);

        // Run diffusion head: B=2
        diffusionHead_.setInputShape("noisy_images", nvinfer1::Dims2{2, latentDim});
        nvinfer1::Dims tsDims; tsDims.nbDims = 1; tsDims.d[0] = 2;
        diffusionHead_.setInputShape("timesteps", tsDims);
        diffusionHead_.setInputShape("condition", nvinfer1::Dims2{2, hiddenSize});
        diffusionHead_.setTensorAddress("noisy_images", scratchNoisy_.data());
        diffusionHead_.setTensorAddress("timesteps", scratchTimesteps_.data());
        diffusionHead_.setTensorAddress("condition", scratchCondition_.data());
        diffusionHead_.setTensorAddress("v_prediction", scratchVpred_.data());
        diffusionHead_.enqueueV3(stream_);

        // Download v_prediction [2, 64] fp16 -> fp32
        scratchVpredF32_.resize(2 * latentDim * sizeof(float));
        GpuOps::halfToFloat(scratchVpred_.as<__half>(), scratchVpredF32_.as<float>(),
                            2 * latentDim, stream_);
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

void TTSPipeline::bindKVCache(TRTEngine& engine, KVCache& kv) {
    int nLayers = kv.numLayers();
    int seqLen = kv.seqLen();

    for (int i = 0; i < nLayers; ++i) {
        std::string pastKeyName = "past_key_" + std::to_string(i);
        std::string pastValName = "past_value_" + std::to_string(i);
        std::string presKeyName = "present_key_" + std::to_string(i);
        std::string presValName = "present_value_" + std::to_string(i);

        // Set input shapes for past KV: [1, numKVHeads, seqLen, headDim]
        nvinfer1::Dims4 pastShape{1, kv.numKVHeads(), seqLen, kv.headDim()};
        engine.setInputShape(pastKeyName, pastShape);
        engine.setInputShape(pastValName, pastShape);

        // Bind buffers
        engine.setTensorAddress(pastKeyName, kv.pastKeyPtr(i));
        engine.setTensorAddress(pastValName, kv.pastValuePtr(i));
        engine.setTensorAddress(presKeyName, kv.presentKeyPtr(i));
        engine.setTensorAddress(presValName, kv.presentValuePtr(i));
    }
}
