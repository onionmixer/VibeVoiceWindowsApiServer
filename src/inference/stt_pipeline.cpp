#include "inference/stt_pipeline.h"
#include <json.hpp>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <sstream>
#include <iomanip>
#include "utils/logger.h"

using json = nlohmann::json;

// ── Constants ──

static constexpr int kMaxSeqLen = 8192;
static constexpr int kChunkSeconds = 60;
static constexpr int kSampleRate = 24000;

static const char* kSystemPrompt =
    "You are a helpful assistant.";

// ── Constructor / Destructor ──

STTPipeline::STTPipeline() {}

STTPipeline::~STTPipeline() {
    if (cublas_) { cublasDestroy(cublas_); cublas_ = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
}

// ── Loading ──

bool STTPipeline::load(const Config& cfg) {
    cfg_ = cfg;

    // Load metadata
    if (!loadModelMetadata(cfg.metadataPath, meta_)) {
        LOG_ERROR("STT", "failed to load metadata from %s", cfg.metadataPath.c_str());
        return false;
    }
    LOG_INFO("STT", "model_type=%s, hidden=%d, layers=%d, kv_heads=%d",
             meta_.model_type.c_str(), meta_.hidden_size,
             meta_.num_hidden_layers, meta_.num_key_value_heads);

    // Create CUDA stream and cuBLAS handle
    cudaError_t cerr = cudaStreamCreate(&stream_);
    if (cerr != cudaSuccess) {
        LOG_ERROR("STT", "failed to create CUDA stream: %s", cudaGetErrorString(cerr));
        return false;
    }
    cublasStatus_t bstat = cublasCreate(&cublas_);
    if (bstat != CUBLAS_STATUS_SUCCESS) {
        LOG_ERROR("STT", "failed to create cuBLAS handle");
        return false;
    }
    cublasSetStream(cublas_, stream_);

    // Load tokenizer
    if (!tokenizer_.load(cfg.tokenizerPath)) {
        LOG_ERROR("STT", "failed to load tokenizer from %s", cfg.tokenizerPath.c_str());
        return false;
    }
    if (!loadASRSpecialTokens(cfg.specialTokensPath, specialTokens_)) {
        LOG_ERROR("STT", "failed to load special tokens from %s", cfg.specialTokensPath.c_str());
        return false;
    }
    LOG_INFO("STT", "tokenizer loaded (vocab=%zu), speech_start=%d, speech_end=%d, speech_pad=%d",
             tokenizer_.vocabSize(), specialTokens_.speech_start,
             specialTokens_.speech_end, specialTokens_.speech_pad);

    // Load TRT engines
    std::string dir = cfg.engineDir + "/";
    if (!acousticEncoder_.loadFromFile(dir + "acoustic_encoder.trt")) return false;
    if (!semanticEncoder_.loadFromFile(dir + "semantic_encoder.trt")) return false;
    if (!lmPrefill_.loadFromFile(dir + "language_model_prefill.trt")) return false;
    if (!lmDecode_.loadFromFile(dir + "language_model_decode.trt")) return false;
    LOG_INFO("STT", "loaded 4 TRT engines (ASR)");

    // Load weights
    if (!loadWeights()) return false;

    // Allocate scratch buffers
    int H = meta_.hidden_size;
    int V = meta_.vocab_size;
    connScratchGpu_.resize((size_t)kMaxSeqLen * H * sizeof(uint16_t));
    scratchEmbed_.resize((size_t)H * sizeof(uint16_t));
    logitsGpu_.resize((size_t)kMaxSeqLen * V * sizeof(uint16_t));
    scratchTokenId_.resize(sizeof(int32_t));

    loaded_ = true;
    LOG_INFO("STT", "ready");
    return true;
}

bool STTPipeline::loadWeights() {
    std::string dir = cfg_.weightsDir + "/";

    // Embedding tokens
    EmbeddingWeights emb;
    if (!loadEmbeddingWeights(dir + "embed_tokens.bin", emb)) return false;
    if (!embedTokensGpu_.upload(emb.data)) return false;
    LOG_INFO("STT", "embed_tokens [%u, %u] uploaded", emb.num_embeddings, emb.embedding_dim);

    // Acoustic connector (64 -> hidden_size)
    ConnectorWeights acConn;
    if (!loadConnectorWeights(dir + "acoustic_connector.bin", acConn)) return false;
    acConnInputDim_ = acConn.input_dim;
    acConnOutputDim_ = acConn.output_dim;
    if (!acConnFc1W_.upload(acConn.fc1_weight)) return false;
    if (!acConnFc1B_.upload(acConn.fc1_bias)) return false;
    if (!acConnNormW_.upload(acConn.norm_weight)) return false;
    if (!acConnFc2W_.upload(acConn.fc2_weight)) return false;
    if (!acConnFc2B_.upload(acConn.fc2_bias)) return false;
    LOG_INFO("STT", "acoustic_connector [%u -> %u] uploaded",
             acConn.input_dim, acConn.output_dim);

    // Semantic connector (128 -> hidden_size)
    ConnectorWeights semConn;
    if (!loadConnectorWeights(dir + "semantic_connector.bin", semConn)) return false;
    semConnInputDim_ = semConn.input_dim;
    semConnOutputDim_ = semConn.output_dim;
    if (!semConnFc1W_.upload(semConn.fc1_weight)) return false;
    if (!semConnFc1B_.upload(semConn.fc1_bias)) return false;
    if (!semConnNormW_.upload(semConn.norm_weight)) return false;
    if (!semConnFc2W_.upload(semConn.fc2_weight)) return false;
    if (!semConnFc2B_.upload(semConn.fc2_bias)) return false;
    LOG_INFO("STT", "semantic_connector [%u -> %u] uploaded",
             semConn.input_dim, semConn.output_dim);

    return true;
}

// ── Transcription ──

STTPipeline::Result STTPipeline::transcribe(const Request& req) {
    Result result;
    if (!loaded_) {
        result.error = "Pipeline not loaded";
        return result;
    }

    if (req.audio.empty()) {
        result.error = "Empty audio input";
        return result;
    }

    float audioDuration = (float)req.audio.size() / (float)kSampleRate;
    result.duration = audioDuration;

    // 1. Encode speech
    encodeSpeech(req.audio.data(), (int)req.audio.size());

    // Compute speech frame count (T) from acoustic encoder output
    // Approximate: ~50 frames per second at 24kHz
    int T = (int)req.audio.size() / 480;  // 24000/50 = 480 samples per frame
    if (T <= 0) T = 1;

    // 2. Build prompt tokens
    buildPromptTokens(audioDuration, T);

    int totalTokens = (int)inputIds_.size();

    // 3. Build input embeddings
    buildInputEmbeds(totalTokens, T);

    // 4. Prefill
    runPrefill(totalTokens);

    // 5. Decode
    auto generatedIds = runDecode(totalTokens);

    // 6. Decode tokens to text
    std::string rawText = tokenizer_.decode(generatedIds);
    result.text = rawText;

    // 7. Parse transcription segments
    result.segments = parseTranscription(rawText);

    // Combine segment text if no structured output
    if (result.segments.empty() && !rawText.empty()) {
        Segment seg;
        seg.startTime = 0.0f;
        seg.endTime = audioDuration;
        seg.text = rawText;
        result.segments.push_back(seg);
    }

    // Build plain text from segments
    if (!result.segments.empty() && result.text.empty()) {
        for (auto& seg : result.segments) {
            if (!result.text.empty()) result.text += " ";
            result.text += seg.text;
        }
    }

    result.language = req.language.empty() ? "en" : req.language;
    result.ok = true;
    return result;
}

// ── Speech Encoding ──

void STTPipeline::encodeSpeech(const float* audio, int numSamples) {
    int H = meta_.hidden_size;
    int chunkSamples = kChunkSeconds * kSampleRate;

    // Determine chunks
    int numChunks = (numSamples + chunkSamples - 1) / chunkSamples;
    if (numChunks < 1) numChunks = 1;

    // We'll collect acoustic and semantic latents from all chunks
    std::vector<CudaBuffer> acousticChunks(numChunks);
    std::vector<CudaBuffer> semanticChunks(numChunks);
    std::vector<int> chunkFrames(numChunks, 0);

    int totalFrames = 0;

    for (int c = 0; c < numChunks; ++c) {
        int offset = c * chunkSamples;
        int len = std::min(chunkSamples, numSamples - offset);

        // Upload audio chunk to GPU as fp16
        CudaBuffer audioF32Gpu, audioFp16Gpu;
        audioF32Gpu.resize(len * sizeof(float));
        audioF32Gpu.copyFromHost(audio + offset, len * sizeof(float));
        audioFp16Gpu.resize(len * sizeof(uint16_t));
        GpuOps::floatToHalf(audioF32Gpu.as<float>(), audioFp16Gpu.as<__half>(), len, stream_);

        // Acoustic encoder: audio [1,1,N] -> latent [1,T_c,64]
        acousticEncoder_.setInputShape("audio", nvinfer1::Dims3{1, 1, len});
        acousticEncoder_.setTensorAddress("audio", audioFp16Gpu.data());

        // We need to estimate output size. Typically ~N/480 frames
        int estFrames = len / 480;
        if (estFrames < 1) estFrames = 1;
        acousticChunks[c].resize((size_t)estFrames * 64 * sizeof(uint16_t));
        acousticEncoder_.setTensorAddress("latent", acousticChunks[c].data());
        acousticEncoder_.enqueueV3(stream_);

        // Semantic encoder: audio [1,1,N] -> semantic [1,T_c,128]
        semanticEncoder_.setInputShape("audio", nvinfer1::Dims3{1, 1, len});
        semanticEncoder_.setTensorAddress("audio", audioFp16Gpu.data());
        semanticChunks[c].resize((size_t)estFrames * 128 * sizeof(uint16_t));
        semanticEncoder_.setTensorAddress("semantic", semanticChunks[c].data());
        semanticEncoder_.enqueueV3(stream_);

        chunkFrames[c] = estFrames;
        totalFrames += estFrames;

        cudaStreamSynchronize(stream_);
    }

    // Run acoustic connector: latent [totalFrames, 64] -> [totalFrames, H]
    // First, concatenate chunks into one buffer
    CudaBuffer allAcoustic, allSemantic;
    allAcoustic.resize((size_t)totalFrames * 64 * sizeof(uint16_t));
    allSemantic.resize((size_t)totalFrames * 128 * sizeof(uint16_t));

    size_t acOffset = 0, semOffset = 0;
    for (int c = 0; c < numChunks; ++c) {
        size_t acBytes = (size_t)chunkFrames[c] * 64 * sizeof(uint16_t);
        size_t semBytes = (size_t)chunkFrames[c] * 128 * sizeof(uint16_t);
        cudaMemcpyAsync((uint8_t*)allAcoustic.data() + acOffset,
                        acousticChunks[c].data(), acBytes,
                        cudaMemcpyDeviceToDevice, stream_);
        cudaMemcpyAsync((uint8_t*)allSemantic.data() + semOffset,
                        semanticChunks[c].data(), semBytes,
                        cudaMemcpyDeviceToDevice, stream_);
        acOffset += acBytes;
        semOffset += semBytes;
    }

    // Acoustic connector: [T, 64] -> [T, H]
    CudaBuffer acousticEmbeds;
    acousticEmbeds.resize((size_t)totalFrames * H * sizeof(uint16_t));
    runConnectorBatched(allAcoustic.as<__half>(), acousticEmbeds.as<__half>(),
                        acConnFc1W_, acConnFc1B_, acConnNormW_,
                        acConnFc2W_, acConnFc2B_,
                        totalFrames, acConnInputDim_, acConnOutputDim_);

    // Semantic connector: [T, 128] -> [T, H]
    CudaBuffer semanticEmbeds;
    semanticEmbeds.resize((size_t)totalFrames * H * sizeof(uint16_t));
    runConnectorBatched(allSemantic.as<__half>(), semanticEmbeds.as<__half>(),
                        semConnFc1W_, semConnFc1B_, semConnNormW_,
                        semConnFc2W_, semConnFc2B_,
                        totalFrames, semConnInputDim_, semConnOutputDim_);

    // speechFeatures = acoustic + semantic (element-wise)
    speechFeaturesGpu_.resize((size_t)totalFrames * H * sizeof(uint16_t));
    GpuOps::vectorAdd(acousticEmbeds.as<__half>(), semanticEmbeds.as<__half>(),
                      speechFeaturesGpu_.as<__half>(), totalFrames * H, stream_);

    cudaStreamSynchronize(stream_);
}

// ── Prompt Construction ──

void STTPipeline::buildPromptTokens(float audioDuration, int speechFrames) {
    inputIds_.clear();
    maskIndices_.clear();

    // Qwen2 chat template tokens
    // <|im_start|> = 151644, <|im_end|> = 151645
    int32_t imStart = 151644;
    int32_t imEnd = 151645;
    int32_t newline = tokenizer_.tokenToId("\n");

    // system\n{SYSTEM_PROMPT}<|im_end|>\n
    auto systemTokens = tokenizer_.encode("system");
    auto systemPromptTokens = tokenizer_.encode(kSystemPrompt);

    inputIds_.push_back(imStart);
    inputIds_.insert(inputIds_.end(), systemTokens.begin(), systemTokens.end());
    inputIds_.push_back(newline);
    inputIds_.insert(inputIds_.end(), systemPromptTokens.begin(), systemPromptTokens.end());
    inputIds_.push_back(imEnd);
    inputIds_.push_back(newline);

    // <|im_start|>user\n
    auto userTokens = tokenizer_.encode("user");
    inputIds_.push_back(imStart);
    inputIds_.insert(inputIds_.end(), userTokens.begin(), userTokens.end());
    inputIds_.push_back(newline);

    // [speech_start][speech_pad]*T[speech_end]
    inputIds_.push_back(specialTokens_.speech_start);
    int maskStart = (int)inputIds_.size();
    for (int i = 0; i < speechFrames; ++i) {
        maskIndices_.push_back((int32_t)inputIds_.size());
        inputIds_.push_back(specialTokens_.speech_pad);
    }
    inputIds_.push_back(specialTokens_.speech_end);

    // \nThis is a X.XX seconds audio, please transcribe it...
    char durationStr[64];
    snprintf(durationStr, sizeof(durationStr), "%.2f", audioDuration);
    std::string instruction = std::string("\nThis is a ") + durationStr +
        " seconds audio, please transcribe it into text with timestamps and speaker identification.";
    auto instrTokens = tokenizer_.encode(instruction);
    inputIds_.insert(inputIds_.end(), instrTokens.begin(), instrTokens.end());

    // <|im_end|>\n<|im_start|>assistant\n
    inputIds_.push_back(imEnd);
    inputIds_.push_back(newline);

    auto assistantTokens = tokenizer_.encode("assistant");
    inputIds_.push_back(imStart);
    inputIds_.insert(inputIds_.end(), assistantTokens.begin(), assistantTokens.end());
    inputIds_.push_back(newline);

    (void)maskStart;  // maskIndices_ already recorded
}

// ── Input Embedding Construction ──

void STTPipeline::buildInputEmbeds(int totalTokens, int /*speechFrames*/) {
    int H = meta_.hidden_size;

    // Upload input IDs to GPU
    inputIdsGpu_.resize(totalTokens * sizeof(int32_t));
    inputIdsGpu_.copyFromHost(inputIds_.data(), totalTokens * sizeof(int32_t));

    // Embedding lookup
    embedsGpu_.resize((size_t)totalTokens * H * sizeof(uint16_t));
    GpuOps::embeddingLookup(embedTokensGpu_.as<__half>(),
                            inputIdsGpu_.as<int32_t>(),
                            embedsGpu_.as<__half>(),
                            totalTokens, H, stream_);

    // Upload mask indices
    int T = (int)maskIndices_.size();
    if (T > 0) {
        maskIndicesGpu_.resize(T * sizeof(int32_t));
        maskIndicesGpu_.copyFromHost(maskIndices_.data(), T * sizeof(int32_t));

        // Replace masked positions with speech features
        GpuOps::replaceMaskedEmbeds(embedsGpu_.as<__half>(),
                                     speechFeaturesGpu_.as<__half>(),
                                     maskIndicesGpu_.as<int32_t>(),
                                     T, H, stream_);
    }

    cudaStreamSynchronize(stream_);
}

// ── LM Prefill ──

void STTPipeline::runPrefill(int totalTokens) {
    int H = meta_.hidden_size;
    int V = meta_.vocab_size;
    int numLayers = meta_.num_hidden_layers;
    int numKVHeads = meta_.num_key_value_heads;
    int headDim = meta_.head_dim;

    // Init KV-cache
    lmKV_.init(numLayers, numKVHeads, headDim, kMaxSeqLen);

    // Build position IDs [1, S] where S = totalTokens
    std::vector<int64_t> posIds(totalTokens);
    for (int i = 0; i < totalTokens; ++i) posIds[i] = i;
    positionIdsGpu_.resize(totalTokens * sizeof(int64_t));
    positionIdsGpu_.copyFromHost(posIds.data(), totalTokens * sizeof(int64_t));

    // Set shapes
    lmPrefill_.setInputShape("inputs_embeds", nvinfer1::Dims3{1, totalTokens, H});
    lmPrefill_.setInputShape("position_ids", nvinfer1::Dims2{1, totalTokens});

    // Bind I/O
    lmPrefill_.setTensorAddress("inputs_embeds", embedsGpu_.data());
    lmPrefill_.setTensorAddress("position_ids", positionIdsGpu_.data());

    // Logits output: [1, totalTokens, vocab_size]
    logitsGpu_.resize((size_t)totalTokens * V * sizeof(uint16_t));
    lmPrefill_.setTensorAddress("logits", logitsGpu_.data());

    // Bind KV-cache present buffers (prefill: past is empty)
    for (int i = 0; i < numLayers; ++i) {
        std::string pastKeyName = "past_key_" + std::to_string(i);
        std::string pastValName = "past_value_" + std::to_string(i);
        std::string presKeyName = "present_key_" + std::to_string(i);
        std::string presValName = "present_value_" + std::to_string(i);

        // Past is empty: shape [1, numKVHeads, 0, headDim]
        nvinfer1::Dims4 emptyPast{1, numKVHeads, 0, headDim};
        lmPrefill_.setInputShape(pastKeyName, emptyPast);
        lmPrefill_.setInputShape(pastValName, emptyPast);

        // Use a dummy pointer for empty past (won't be read)
        lmPrefill_.setTensorAddress(pastKeyName, lmKV_.pastKeyPtr(i));
        lmPrefill_.setTensorAddress(pastValName, lmKV_.pastValuePtr(i));

        // Present outputs
        lmPrefill_.setTensorAddress(presKeyName, lmKV_.presentKeyPtr(i));
        lmPrefill_.setTensorAddress(presValName, lmKV_.presentValuePtr(i));
    }

    lmPrefill_.enqueueV3(stream_);
    cudaStreamSynchronize(stream_);

    // Set KV-cache sequence length and swap buffers
    // After prefill, seqLen = totalTokens (advance totalTokens times)
    for (int i = 0; i < totalTokens; ++i) {
        lmKV_.advance();
    }
}

// ── LM Decode ──

std::vector<int32_t> STTPipeline::runDecode(int prefillLen) {
    int H = meta_.hidden_size;
    int V = meta_.vocab_size;
    int maxNew = cfg_.maxNewTokens;

    std::vector<int32_t> generatedIds;

    // Get first token from prefill logits (last position)
    // Download logits for the last position
    size_t lastPosOffset = (size_t)(prefillLen - 1) * V * sizeof(uint16_t);
    CudaBuffer lastLogitsGpu;
    lastLogitsGpu.resize(V * sizeof(uint16_t));
    cudaMemcpyAsync(lastLogitsGpu.data(),
                    (uint8_t*)logitsGpu_.data() + lastPosOffset,
                    V * sizeof(uint16_t),
                    cudaMemcpyDeviceToDevice, stream_);

    // Convert to fp32 for argmax
    CudaBuffer logitsF32;
    logitsF32.resize(V * sizeof(float));
    GpuOps::halfToFloat(lastLogitsGpu.as<__half>(), logitsF32.as<float>(), V, stream_);
    cudaStreamSynchronize(stream_);

    std::vector<float> logitsHost(V);
    logitsF32.copyToHost(logitsHost.data(), V * sizeof(float));

    // Argmax
    int32_t nextToken = 0;
    float maxVal = logitsHost[0];
    for (int i = 1; i < V; ++i) {
        if (logitsHost[i] > maxVal) {
            maxVal = logitsHost[i];
            nextToken = i;
        }
    }

    int pos = prefillLen;

    // Decode loop
    CudaBuffer decEmbedGpu, decLogitsGpu, decPosGpu;
    decEmbedGpu.resize(H * sizeof(uint16_t));
    decLogitsGpu.resize(V * sizeof(uint16_t));
    decPosGpu.resize(sizeof(int64_t));

    for (int step = 0; step < maxNew; ++step) {
        // Check EOS
        if (nextToken == specialTokens_.eos) break;

        generatedIds.push_back(nextToken);

        // Embed the token (reuse member buffer)
        scratchTokenId_.copyFromHost(&nextToken, sizeof(int32_t));
        GpuOps::embeddingLookup(embedTokensGpu_.as<__half>(),
                                scratchTokenId_.as<int32_t>(),
                                decEmbedGpu.as<__half>(), 1, H, stream_);

        // Position
        int64_t posVal = pos;
        decPosGpu.copyFromHost(&posVal, sizeof(int64_t));

        // Set shapes
        lmDecode_.setInputShape("inputs_embeds", nvinfer1::Dims3{1, 1, H});
        lmDecode_.setInputShape("position_ids", nvinfer1::Dims2{1, 1});

        // Bind I/O
        lmDecode_.setTensorAddress("inputs_embeds", decEmbedGpu.data());
        lmDecode_.setTensorAddress("position_ids", decPosGpu.data());
        lmDecode_.setTensorAddress("logits", decLogitsGpu.data());

        // Bind KV-cache
        bindKVCache(lmDecode_, lmKV_);

        // Execute
        lmDecode_.enqueueV3(stream_);
        lmKV_.advance();
        pos++;

        // Get next token
        GpuOps::halfToFloat(decLogitsGpu.as<__half>(), logitsF32.as<float>(), V, stream_);
        cudaStreamSynchronize(stream_);
        logitsF32.copyToHost(logitsHost.data(), V * sizeof(float));

        nextToken = 0;
        maxVal = logitsHost[0];
        for (int i = 1; i < V; ++i) {
            if (logitsHost[i] > maxVal) {
                maxVal = logitsHost[i];
                nextToken = i;
            }
        }
    }

    return generatedIds;
}

// ── Connector (Batched) ──

void STTPipeline::runConnectorBatched(const __half* input, __half* output,
                                       const CudaBuffer& fc1W, const CudaBuffer& fc1B,
                                       const CudaBuffer& normW,
                                       const CudaBuffer& fc2W, const CudaBuffer& fc2B,
                                       int T, int inputDim, int outputDim) {
    // Ensure scratch is large enough
    size_t scratchNeeded = (size_t)T * outputDim * sizeof(uint16_t);
    if (connScratchGpu_.size() < scratchNeeded) {
        connScratchGpu_.resize(scratchNeeded);
    }

    // fc1: [T, inputDim] -> [T, outputDim]
    GpuOps::linearForwardBatch(cublas_, input,
                               fc1W.as<__half>(), fc1B.as<__half>(),
                               connScratchGpu_.as<__half>(),
                               T, outputDim, inputDim, stream_);

    // RMSNorm (per row)
    GpuOps::rmsNormBatched(connScratchGpu_.as<__half>(), normW.as<__half>(),
                           connScratchGpu_.as<__half>(),
                           T, outputDim, (float)meta_.rms_norm_eps, stream_);

    // fc2: [T, outputDim] -> [T, outputDim]
    GpuOps::linearForwardBatch(cublas_, connScratchGpu_.as<__half>(),
                               fc2W.as<__half>(), fc2B.as<__half>(),
                               output, T, outputDim, outputDim, stream_);
}

// ── KV-Cache Binding ──

void STTPipeline::bindKVCache(TRTEngine& engine, KVCache& kv) {
    int nLayers = kv.numLayers();
    int seqLen = kv.seqLen();

    for (int i = 0; i < nLayers; ++i) {
        std::string pastKeyName = "past_key_" + std::to_string(i);
        std::string pastValName = "past_value_" + std::to_string(i);
        std::string presKeyName = "present_key_" + std::to_string(i);
        std::string presValName = "present_value_" + std::to_string(i);

        nvinfer1::Dims4 pastShape{1, kv.numKVHeads(), seqLen, kv.headDim()};
        engine.setInputShape(pastKeyName, pastShape);
        engine.setInputShape(pastValName, pastShape);

        engine.setTensorAddress(pastKeyName, kv.pastKeyPtr(i));
        engine.setTensorAddress(pastValName, kv.pastValuePtr(i));
        engine.setTensorAddress(presKeyName, kv.presentKeyPtr(i));
        engine.setTensorAddress(presValName, kv.presentValuePtr(i));
    }
}

// ── JSON Transcription Parsing ──

std::vector<STTPipeline::Segment> STTPipeline::parseTranscription(const std::string& text) {
    std::vector<Segment> segments;

    // Find JSON array in the text (may be wrapped in ```json block)
    std::string jsonStr = text;

    // Strip ```json ... ``` wrapper if present
    auto codeStart = jsonStr.find("```json");
    if (codeStart != std::string::npos) {
        jsonStr = jsonStr.substr(codeStart + 7);
        auto codeEnd = jsonStr.find("```");
        if (codeEnd != std::string::npos) {
            jsonStr = jsonStr.substr(0, codeEnd);
        }
    }

    // Find the JSON array
    auto arrStart = jsonStr.find('[');
    if (arrStart == std::string::npos) return segments;
    auto arrEnd = jsonStr.rfind(']');
    if (arrEnd == std::string::npos || arrEnd <= arrStart) return segments;

    jsonStr = jsonStr.substr(arrStart, arrEnd - arrStart + 1);

    try {
        auto j = json::parse(jsonStr);
        if (!j.is_array()) return segments;

        for (auto& item : j) {
            Segment seg;

            // Parse "Start time": "0.00s"
            if (item.contains("Start time")) {
                std::string st = item["Start time"].get<std::string>();
                // Remove trailing 's'
                if (!st.empty() && st.back() == 's') st.pop_back();
                seg.startTime = std::stof(st);
            }
            if (item.contains("End time")) {
                std::string et = item["End time"].get<std::string>();
                if (!et.empty() && et.back() == 's') et.pop_back();
                seg.endTime = std::stof(et);
            }
            if (item.contains("Speaker ID")) {
                if (item["Speaker ID"].is_string()) {
                    seg.speakerId = item["Speaker ID"].get<std::string>();
                } else {
                    seg.speakerId = std::to_string(item["Speaker ID"].get<int>());
                }
            }
            if (item.contains("Content")) {
                seg.text = item["Content"].get<std::string>();
            }

            segments.push_back(std::move(seg));
        }
    } catch (const std::exception&) {
        // JSON parse failed, return empty
    }

    return segments;
}

// ── SRT Formatting ──

static std::string formatTimeSRT(float seconds) {
    int totalMs = (int)(seconds * 1000.0f + 0.5f);
    int h = totalMs / 3600000; totalMs %= 3600000;
    int m = totalMs / 60000;   totalMs %= 60000;
    int s = totalMs / 1000;
    int ms = totalMs % 1000;
    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d,%03d", h, m, s, ms);
    return buf;
}

std::string STTPipeline::formatSRT(const std::vector<Segment>& segments) {
    std::ostringstream oss;
    for (size_t i = 0; i < segments.size(); ++i) {
        oss << (i + 1) << "\n";
        oss << formatTimeSRT(segments[i].startTime) << " --> "
            << formatTimeSRT(segments[i].endTime) << "\n";
        oss << segments[i].text << "\n\n";
    }
    return oss.str();
}

// ── VTT Formatting ──

static std::string formatTimeVTT(float seconds) {
    int totalMs = (int)(seconds * 1000.0f + 0.5f);
    int h = totalMs / 3600000; totalMs %= 3600000;
    int m = totalMs / 60000;   totalMs %= 60000;
    int s = totalMs / 1000;
    int ms = totalMs % 1000;
    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d.%03d", h, m, s, ms);
    return buf;
}

std::string STTPipeline::formatVTT(const std::vector<Segment>& segments) {
    std::ostringstream oss;
    oss << "WEBVTT\n\n";
    for (size_t i = 0; i < segments.size(); ++i) {
        oss << formatTimeVTT(segments[i].startTime) << " --> "
            << formatTimeVTT(segments[i].endTime) << "\n";
        oss << segments[i].text << "\n\n";
    }
    return oss.str();
}
