#pragma once
#include "inference/trt_engine.h"
#include "inference/model_config.h"
#include "inference/kv_cache.h"
#include "inference/gpu_ops.h"
#include "tokenizer/bpe_tokenizer.h"
#include "tokenizer/special_tokens.h"
#include "utils/cuda_buffer.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <string>
#include <vector>

class STTPipeline {
public:
    struct Config {
        std::string engineDir;          // "engines/asr"
        std::string weightsDir;         // "onnx/asr/weights"
        std::string metadataPath;       // "onnx/asr/model_metadata.json"
        std::string tokenizerPath;      // "tokenizer/qwen2.5-7b/tokenizer.json"
        std::string specialTokensPath;  // "tokenizer/qwen2.5-7b/special_tokens.json"
        int maxNewTokens = 32768;
        float temperature = 0.0f;
    };

    struct Segment {
        float startTime = 0.0f;
        float endTime = 0.0f;
        std::string speakerId;
        std::string text;
    };

    struct Request {
        std::vector<float> audio;
        int sampleRate = 24000;
        std::string language;
        std::string prompt;
        float temperature = 0.0f;
        bool translate = false;
    };

    struct Result {
        std::string text;
        std::vector<Segment> segments;
        float duration = 0.0f;
        std::string language;
        bool ok = false;
        std::string error;
    };

    STTPipeline();
    ~STTPipeline();

    bool load(const Config& cfg);
    Result transcribe(const Request& req);
    bool isLoaded() const { return loaded_; }

    // Exposed for testing
    static std::vector<Segment> parseTranscription(const std::string& text);
    static std::string formatSRT(const std::vector<Segment>& segments);
    static std::string formatVTT(const std::vector<Segment>& segments);

private:
    // Encode speech audio into feature embeddings
    void encodeSpeech(const float* audio, int numSamples);

    // Build prompt token IDs for the LM
    void buildPromptTokens(float audioDuration, int speechFrames);

    // Build input embeddings and replace masked positions
    void buildInputEmbeds(int totalTokens, int speechFrames);

    // Run LM prefill pass
    void runPrefill(int totalTokens);

    // Run LM autoregressive decode
    std::vector<int32_t> runDecode(int prefillLen);

    // Run connector MLP: input[T, inDim] -> output[T, outDim]
    void runConnectorBatched(const __half* input, __half* output,
                             const CudaBuffer& fc1W, const CudaBuffer& fc1B,
                             const CudaBuffer& normW,
                             const CudaBuffer& fc2W, const CudaBuffer& fc2B,
                             int T, int inputDim, int outputDim);

    // Bind KV-cache to TRT engine
    void bindKVCache(TRTEngine& engine, KVCache& kv);

    // Load GPU weights
    bool loadWeights();

    Config cfg_;
    ModelMetadata meta_;
    bool loaded_ = false;

    // TRT engines
    TRTEngine acousticEncoder_;
    TRTEngine semanticEncoder_;
    TRTEngine lmPrefill_;
    TRTEngine lmDecode_;

    // KV-cache for the LM (28 layers)
    KVCache lmKV_;

    // GPU weight buffers
    CudaBuffer embedTokensGpu_;     // [vocab_size, hidden_size] fp16

    // Acoustic connector: 64 -> hidden_size
    CudaBuffer acConnFc1W_, acConnFc1B_, acConnNormW_;
    CudaBuffer acConnFc2W_, acConnFc2B_;
    int acConnInputDim_ = 0, acConnOutputDim_ = 0;

    // Semantic connector: 128 -> hidden_size
    CudaBuffer semConnFc1W_, semConnFc1B_, semConnNormW_;
    CudaBuffer semConnFc2W_, semConnFc2B_;
    int semConnInputDim_ = 0, semConnOutputDim_ = 0;

    // Tokenizer & special tokens
    BPETokenizer tokenizer_;
    ASRSpecialTokens specialTokens_;

    // CUDA resources
    cudaStream_t stream_ = nullptr;
    cublasHandle_t cublas_ = nullptr;

    // Prompt tokens (built per request)
    std::vector<int32_t> inputIds_;
    std::vector<int32_t> maskIndices_;   // positions of speech_pad tokens

    // GPU scratch buffers
    CudaBuffer speechFeaturesGpu_;   // [T, hidden_size] fp16  (acoustic + semantic combined)
    CudaBuffer embedsGpu_;          // [totalTokens, hidden_size] fp16
    CudaBuffer inputIdsGpu_;        // [totalTokens] int32
    CudaBuffer maskIndicesGpu_;     // [T] int32
    CudaBuffer logitsGpu_;          // [1, S, vocab_size] fp16
    CudaBuffer positionIdsGpu_;     // [1, S] int64
    CudaBuffer connScratchGpu_;     // scratch for connector
    CudaBuffer scratchEmbed_;       // [1, hidden_size] fp16
};
